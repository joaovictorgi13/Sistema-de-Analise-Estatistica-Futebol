"""
MÓDULO: api_client.py
OBJETIVO: Obter e organizar os dados reais do Campeonato Brasileiro Série A 2024.

FONTE DOS DADOS (gratuita, sem chave de API):
    Brasileirao_Dataset (Adão Duque) — https://github.com/adaoduque/Brasileirao_Dataset
    - campeonato-brasileiro-full.csv     -> partidas (rodada, data, hora, times, placar, técnicos, formação, arena)
    - campeonato-brasileiro-gols.csv     -> cada gol (clube, atleta, minuto, tipo: pênalti / gol contra)
    - campeonato-brasileiro-cartoes.csv  -> cada cartão (clube, atleta, posição, minuto, cor)

Na primeira execução os arquivos são baixados do GitHub e filtrados para 2024.
O recorte de 2024 fica salvo em data/, então o sistema funciona offline (por exemplo, na apresentação).

COMENTÁRIOS DA DISCIPLINA:
- População: todas as partidas do Brasileirão. Amostra analisada: as 380 partidas da temporada 2024.
- Unidade amostral (elemento): uma partida (ou um gol, ou um jogador, conforme a análise).
- Cada partida é a realização de um experimento aleatório com várias variáveis associadas:
  qualitativas (resultado, horário, formação tática, estado) e quantitativas
  discretas (gols, cartões) e contínuas (minuto do gol, tempo até o 1º gol).
"""

import os
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd

ANO = 2024
BASE_URL = "https://raw.githubusercontent.com/adaoduque/Brasileirao_Dataset/master"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

ARQ_PARTIDAS = os.path.join(DATA_DIR, f"brasileirao_{ANO}_partidas.csv")
ARQ_GOLS = os.path.join(DATA_DIR, f"brasileirao_{ANO}_gols.csv")
ARQ_CARTOES = os.path.join(DATA_DIR, f"brasileirao_{ANO}_cartoes.csv")

# Nomes dos clubes como aparecem na fonte -> nome de exibição
NOMES_CLUBES = {
    "Athletico-PR": "Athletico-PR",
    "Atletico-GO": "Atlético-GO",
    "Atletico-MG": "Atlético-MG",
    "Bahia": "Bahia",
    "Botafogo-RJ": "Botafogo",
    "Bragantino": "Bragantino",
    "Corinthians": "Corinthians",
    "Criciuma": "Criciúma",
    "Cruzeiro": "Cruzeiro",
    "Cuiaba": "Cuiabá",
    "Flamengo": "Flamengo",
    "Fluminense": "Fluminense",
    "Fortaleza": "Fortaleza",
    "Gremio": "Grêmio",
    "Internacional": "Internacional",
    "Juventude": "Juventude",
    "Palmeiras": "Palmeiras",
    "Sao Paulo": "São Paulo",
    "Vasco": "Vasco",
    "Vitoria": "Vitória",
}


def _nome_clube(nome: str) -> str:
    return NOMES_CLUBES.get(str(nome).strip(), str(nome).strip())


def _parse_minuto(valor) -> Tuple[int, int, int]:
    """
    Converte o minuto da fonte ('36', '45+4', '90+3') em (tempo, minuto_base, acrescimo).
    tempo = 1 (1º tempo) se o minuto base for até 45 (inclui acréscimos do 1º tempo), senão 2.
    """
    texto = str(valor).strip()
    m = re.match(r"^(\d+)(?:\+(\d+))?", texto)
    if not m:
        return (0, 0, 0)
    base = int(m.group(1))
    extra = int(m.group(2)) if m.group(2) else 0
    tempo = 1 if base <= 45 else 2
    return (tempo, base, extra)


class FootballDataAPI:
    """
    Cliente de dados do Brasileirão 2024.
    Mantém o nome da classe da versão anterior para não quebrar o restante do sistema.
    """

    def __init__(self, ano: int = ANO):
        self.ano = ano

    # ------------------------------------------------------------------
    # Download e cache
    # ------------------------------------------------------------------
    def _baixar_e_salvar(self) -> None:
        """Baixa os CSVs completos do GitHub, filtra a temporada e salva o recorte em data/."""
        print(f"[DADOS] Baixando o Brasileirão {self.ano} de {BASE_URL} ...")
        partidas = pd.read_csv(f"{BASE_URL}/campeonato-brasileiro-full.csv", encoding="utf-8")
        gols = pd.read_csv(f"{BASE_URL}/campeonato-brasileiro-gols.csv", encoding="utf-8")
        cartoes = pd.read_csv(f"{BASE_URL}/campeonato-brasileiro-cartoes.csv", encoding="utf-8")

        partidas["ano"] = pd.to_datetime(partidas["data"], dayfirst=True).dt.year
        # A temporada começa na rodada 1 do ano escolhido e vai até o fim do arquivo daquele campeonato
        inicio = partidas.loc[(partidas["ano"] == self.ano) & (partidas["rodata"] == 1), "ID"].min()
        partidas_ano = partidas[(partidas["ID"] >= inicio) & (partidas["ID"] < inicio + 380)].copy()
        ids = set(partidas_ano["ID"])

        os.makedirs(DATA_DIR, exist_ok=True)
        partidas_ano.drop(columns=["ano"]).to_csv(ARQ_PARTIDAS, index=False, encoding="utf-8")
        gols[gols["partida_id"].isin(ids)].to_csv(ARQ_GOLS, index=False, encoding="utf-8")
        cartoes[cartoes["partida_id"].isin(ids)].to_csv(ARQ_CARTOES, index=False, encoding="utf-8")
        print(f"[DADOS] {len(partidas_ano)} partidas salvas em {DATA_DIR}")

    def _carregar_brutos(self, force_refresh: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        faltando = not all(os.path.exists(p) for p in (ARQ_PARTIDAS, ARQ_GOLS, ARQ_CARTOES))
        if force_refresh or faltando:
            self._baixar_e_salvar()
        partidas = pd.read_csv(ARQ_PARTIDAS, encoding="utf-8")
        gols = pd.read_csv(ARQ_GOLS, encoding="utf-8")
        cartoes = pd.read_csv(ARQ_CARTOES, encoding="utf-8")
        return partidas, gols, cartoes

    # ------------------------------------------------------------------
    # Tabelas de eventos (um gol ou um cartão por linha)
    # ------------------------------------------------------------------
    def get_goals(self) -> pd.DataFrame:
        """Um gol por linha, com tempo de jogo (1T/2T) e minuto numérico."""
        partidas, gols, _ = self._carregar_brutos()
        g = gols.copy()
        g["clube"] = g["clube"].map(_nome_clube)
        partes = g["minuto"].apply(_parse_minuto)
        g["tempo"] = [p[0] for p in partes]
        g["minuto_base"] = [p[1] for p in partes]
        g["acrescimo"] = [p[2] for p in partes]
        # Minuto como variável contínua no intervalo [0, 90] (acréscimos contam no fim do tempo)
        g["minuto_jogo"] = g["minuto_base"].clip(upper=90).astype(float)
        g["tipo_de_gol"] = g["tipo_de_gol"].fillna("Normal")
        g = g.rename(columns={"rodata": "rodada"})
        return g[["partida_id", "rodada", "clube", "atleta", "minuto", "tempo", "minuto_base",
                  "acrescimo", "minuto_jogo", "tipo_de_gol"]]

    def get_cards(self) -> pd.DataFrame:
        """Um cartão por linha."""
        _, _, cartoes = self._carregar_brutos()
        c = cartoes.copy()
        c["clube"] = c["clube"].map(_nome_clube)
        partes = c["minuto"].apply(_parse_minuto)
        c["tempo"] = [p[0] for p in partes]
        c["minuto_jogo"] = [min(p[1], 90) for p in partes]
        c["posicao"] = c["posicao"].fillna("Não informada")
        c = c.rename(columns={"rodata": "rodada"})
        return c[["partida_id", "rodada", "clube", "cartao", "atleta", "num_camisa", "posicao",
                  "minuto", "tempo", "minuto_jogo"]]

    # ------------------------------------------------------------------
    # Tabela de partidas (uma partida por linha)
    # ------------------------------------------------------------------
    def get_dataset(self, force_refresh: bool = False) -> pd.DataFrame:
        partidas, _, _ = self._carregar_brutos(force_refresh)
        gols = self.get_goals()
        cartoes = self.get_cards()

        df = pd.DataFrame({
            "partida_id": partidas["ID"],
            "rodada": partidas["rodata"],
            "data": pd.to_datetime(partidas["data"], dayfirst=True).dt.strftime("%d/%m/%Y"),
            "hora": partidas["hora"],
            "mandante": partidas["mandante"].map(_nome_clube),
            "visitante": partidas["visitante"].map(_nome_clube),
            "estado_mandante": partidas["mandante_Estado"],
            "estado_visitante": partidas["visitante_Estado"],
            "arena": partidas["arena"].astype(str).str.strip(),
            "tecnico_mandante": partidas["tecnico_mandante"],
            "tecnico_visitante": partidas["tecnico_visitante"],
            "formacao_mandante": partidas["formacao_mandante"].fillna("Não informada"),
            "formacao_visitante": partidas["formacao_visitante"].fillna("Não informada"),
            "gols_mandante": partidas["mandante_Placar"].astype(int),
            "gols_visitante": partidas["visitante_Placar"].astype(int),
        }).reset_index(drop=True)

        df["total_gols"] = df["gols_mandante"] + df["gols_visitante"]
        df["resultado"] = np.select(
            [df["gols_mandante"] > df["gols_visitante"], df["gols_mandante"] == df["gols_visitante"]],
            ["Vitória do mandante", "Empate"],
            default="Vitória do visitante",
        )
        hora = df["hora"].astype(str).str.slice(0, 2).astype(int)
        df["faixa_horario"] = pd.cut(hora, bins=[0, 17, 19, 24], right=False,
                                     labels=["Tarde (até 16h59)", "Fim de tarde (17h–18h59)", "Noite (19h ou mais)"]).astype(str)

        # Gols por tempo, separados por mandante e visitante
        g = gols.merge(df[["partida_id", "mandante", "visitante"]], on="partida_id")
        g["lado"] = np.where(g["clube"] == g["mandante"], "mandante", "visitante")
        contagem = g.groupby(["partida_id", "lado", "tempo"]).size().unstack(["lado", "tempo"], fill_value=0)
        for lado in ("mandante", "visitante"):
            for tempo in (1, 2):
                col = f"gols_{tempo}t_{lado}"
                df[col] = df["partida_id"].map(contagem[(lado, tempo)] if (lado, tempo) in contagem.columns else {}).fillna(0).astype(int)
        df["total_gols_1t"] = df["gols_1t_mandante"] + df["gols_1t_visitante"]
        df["total_gols_2t"] = df["gols_2t_mandante"] + df["gols_2t_visitante"]

        # Primeiro gol da partida: minuto e quem marcou
        g_ord = g.sort_values(["partida_id", "tempo", "minuto_base", "acrescimo"])
        primeiro = g_ord.groupby("partida_id").first()
        df["minuto_primeiro_gol"] = df["partida_id"].map(primeiro["minuto_jogo"])
        lado_primeiro = df["partida_id"].map(primeiro["lado"])
        df["primeiro_gol"] = lado_primeiro.map({"mandante": "Mandante", "visitante": "Visitante"}).fillna("Sem gols")

        df["penaltis"] = df["partida_id"].map(g[g["tipo_de_gol"] == "Penalty"].groupby("partida_id").size()).fillna(0).astype(int)

        # Cartões por partida
        c = cartoes.merge(df[["partida_id", "mandante"]], on="partida_id")
        c["lado"] = np.where(c["clube"] == c["mandante"], "mandante", "visitante")
        for cor, nome in (("Amarelo", "amarelos"), ("Vermelho", "vermelhos")):
            cont = c[c["cartao"] == cor].groupby(["partida_id", "lado"]).size().unstack(fill_value=0)
            for lado in ("mandante", "visitante"):
                serie = cont[lado] if lado in cont.columns else pd.Series(dtype=int)
                df[f"{nome}_{lado}"] = df["partida_id"].map(serie).fillna(0).astype(int)
        df["total_amarelos"] = df["amarelos_mandante"] + df["amarelos_visitante"]
        df["total_vermelhos"] = df["vermelhos_mandante"] + df["vermelhos_visitante"]
        df["total_cartoes"] = df["total_amarelos"] + df["total_vermelhos"]
        return df

    # ------------------------------------------------------------------
    # Jogadores
    # ------------------------------------------------------------------
    def get_players_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Retorna:
        1. df_players: um registro por (jogador, clube) com gols, gols por tempo, pênaltis e cartões.
           Entram todos os atletas que marcaram ou receberam cartão em 2024.
           Gols contra não contam para o atleta (a fonte credita o gol ao clube beneficiado).
        2. df_logs: para cada artilheiro, as 38 partidas do seu clube, indicando se ele marcou
           e o resultado do clube (base para Probabilidade Condicional e Bayes).
        """
        df = self.get_dataset()
        gols = self.get_goals()
        cartoes = self.get_cards()

        gols_atleta = gols[gols["tipo_de_gol"] != "Gol Contra"]
        resumo_gols = gols_atleta.groupby(["atleta", "clube"]).agg(
            gols=("partida_id", "size"),
            jogos_com_gol=("partida_id", "nunique"),
            gols_1t=("tempo", lambda s: int((s == 1).sum())),
            gols_2t=("tempo", lambda s: int((s == 2).sum())),
            penaltis=("tipo_de_gol", lambda s: int((s == "Penalty").sum())),
            minuto_medio_gol=("minuto_jogo", "mean"),
        )
        resumo_cartoes = cartoes.groupby(["atleta", "clube"]).agg(
            amarelos=("cartao", lambda s: int((s == "Amarelo").sum())),
            vermelhos=("cartao", lambda s: int((s == "Vermelho").sum())),
            posicao=("posicao", lambda s: s.mode().iloc[0] if not s.mode().empty else "Não informada"),
        )
        players = resumo_gols.join(resumo_cartoes, how="outer").reset_index()
        players = players.rename(columns={"atleta": "jogador", "clube": "time"})
        for col in ("gols", "jogos_com_gol", "gols_1t", "gols_2t", "penaltis", "amarelos", "vermelhos"):
            players[col] = players[col].fillna(0).astype(int)
        players["posicao"] = players["posicao"].fillna("Não informada")
        players["minuto_medio_gol"] = players["minuto_medio_gol"].round(1)
        players["jogos_do_time"] = 38
        players["gols_por_jogo"] = (players["gols"] / players["jogos_do_time"]).round(3)
        players = players.sort_values(["gols", "amarelos"], ascending=False).reset_index(drop=True)

        # Histórico partida a partida dos artilheiros (somente quem marcou ao menos 1 gol)
        logs = []
        marcadores = players[players["gols"] > 0]
        gols_por_partida = gols_atleta.groupby(["atleta", "clube", "partida_id"]).size()
        for _, p in marcadores.iterrows():
            jogos_time = df[(df["mandante"] == p["time"]) | (df["visitante"] == p["time"])]
            for _, j in jogos_time.iterrows():
                em_casa = j["mandante"] == p["time"]
                gp = j["gols_mandante"] if em_casa else j["gols_visitante"]
                gc = j["gols_visitante"] if em_casa else j["gols_mandante"]
                resultado = "Vitória" if gp > gc else ("Empate" if gp == gc else "Derrota")
                logs.append({
                    "jogador": p["jogador"],
                    "time": p["time"],
                    "partida_id": j["partida_id"],
                    "rodada": j["rodada"],
                    "mando": "Casa" if em_casa else "Fora",
                    "gols": int(gols_por_partida.get((p["jogador"], p["time"], j["partida_id"]), 0)),
                    "resultado_time": resultado,
                    "vitoria_time": int(resultado == "Vitória"),
                })
        df_logs = pd.DataFrame(logs)
        return players, df_logs


if __name__ == "__main__":
    api = FootballDataAPI()
    partidas = api.get_dataset()
    jogadores, logs = api.get_players_data()
    print(partidas.head())
    print(f"{len(partidas)} partidas | {len(jogadores)} jogadores | {len(logs)} registros jogo a jogo")
