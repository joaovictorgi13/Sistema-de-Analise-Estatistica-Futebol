"""
MÓDULO: api_client.py
OBJETIVO: Gerenciamento da obtenção de dados de futebol via API e repositórios abertos.
COMENTÁRIOS DA DISCIPLINA:
- Os dados obtidos aqui representam o 'Espaço Amostral' das partidas observadas.
- Cada partida é um elemento amostral (evento simples) com diversas variáveis aleatórias associadas:
  Gols no 1º tempo, Gols no 2º tempo, Cartões Amarelos, Escanteios, Passes, Posse de Bola, etc.
"""

import os
import json
import requests
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CACHE_FILE = os.path.join(DATA_DIR, "matches_dataset.csv")
PLAYERS_CACHE_FILE = os.path.join(DATA_DIR, "players_dataset.csv")
PLAYER_LOGS_CACHE_FILE = os.path.join(DATA_DIR, "player_match_logs.csv")

class FootballDataAPI:
    """
    Cliente para integração com APIs de Futebol e fontes abertas.
    Suporta:
    1. Football-Data.org (API REST padrão com token gratuito opcional)
    2. Repositórios abertos de futebol (football-data.co.uk / github open-football)
    3. Gerador estocástico realista calibrado (fallback autônomo offline)
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FOOTBALL_API_KEY")
        self.base_url = "https://api.football-data.org/v4"
        self.headers = {"X-Auth-Token": self.api_key} if self.api_key else {}

    def fetch_matches_from_api(self, competition: str = "PL", season: int = 2023) -> Optional[List[Dict[str, Any]]]:
        """
        Consome a API Football-Data.org se a chave de API for fornecida.
        """
        if not self.api_key:
            return None

        url = f"{self.base_url}/competitions/{competition}/matches?season={season}"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get("matches", [])
            else:
                print(f"[API WARN] Resposta da API: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"[API ERROR] Falha na conexão com API Football-Data: {e}")
            return None

    def fetch_open_data_csv(self, league_code: str = "E0") -> Optional[pd.DataFrame]:
        """
        Consome dados detalhados reais abertos (Premier League E0) do football-data.co.uk.
        Contém estatísticas minuciosas: gols 1T, 2T, chutes, faltas, escanteios, amarelos.
        """
        url = f"https://www.football-data.co.uk/mmz4281/2324/{league_code}.csv"
        try:
            print(f"[API CLIENT] Tentando carregar dados abertos de {url}...")
            df = pd.read_csv(url, encoding="latin1")
            
            # Mapeamento para nomes padronizados em português
            processed = pd.DataFrame()
            processed["mandante"] = df["HomeTeam"]
            processed["visitante"] = df["AwayTeam"]
            processed["gols_mandante"] = df["FTHG"].fillna(0).astype(int)
            processed["gols_visitante"] = df["FTAG"].fillna(0).astype(int)
            processed["gols_1t_mandante"] = df["HTHG"].fillna(0).astype(int)
            processed["gols_1t_visitante"] = df["HTAG"].fillna(0).astype(int)
            processed["gols_2t_mandante"] = processed["gols_mandante"] - processed["gols_1t_mandante"]
            processed["gols_2t_visitante"] = processed["gols_visitante"] - processed["gols_1t_visitante"]
            
            processed["total_gols"] = processed["gols_mandante"] + processed["gols_visitante"]
            processed["total_gols_1t"] = processed["gols_1t_mandante"] + processed["gols_1t_visitante"]
            processed["total_gols_2t"] = processed["gols_2t_mandante"] + processed["gols_2t_visitante"]

            processed["chutes_mandante"] = df.get("HS", np.random.poisson(13, len(df)))
            processed["chutes_visitante"] = df.get("AS", np.random.poisson(11, len(df)))
            processed["chutes_gol_mandante"] = df.get("HST", np.random.poisson(5, len(df)))
            processed["chutes_gol_visitante"] = df.get("AST", np.random.poisson(4, len(df)))

            processed["escanteios_mandante"] = df.get("HC", np.random.poisson(6, len(df)))
            processed["escanteios_visitante"] = df.get("AC", np.random.poisson(4, len(df)))
            processed["total_escanteios"] = processed["escanteios_mandante"] + processed["escanteios_visitante"]

            processed["faltas_mandante"] = df.get("HF", np.random.poisson(11, len(df)))
            processed["faltas_visitante"] = df.get("AF", np.random.poisson(11, len(df)))

            processed["amarelos_mandante"] = df.get("HY", np.random.poisson(2, len(df)))
            processed["amarelos_visitante"] = df.get("AY", np.random.poisson(2, len(df)))
            processed["total_amarelos"] = processed["amarelos_mandante"] + processed["amarelos_visitante"]

            # Variáveis contínuas complementares: Posse de bola (%) e Minuto do 1º Gol
            np.random.seed(42)
            processed["posse_mandante"] = np.clip(np.random.normal(51.5, 9.0, len(df)), 25, 75).round(1)
            processed["posse_visitante"] = (100.0 - processed["posse_mandante"]).round(1)
            
            # Passes totais (distribuição normal correlacionada com posse)
            processed["passes_mandante"] = (processed["posse_mandante"] * 8.5 + np.random.normal(0, 35, len(df))).astype(int)
            processed["passes_visitante"] = (processed["posse_visitante"] * 8.5 + np.random.normal(0, 35, len(df))).astype(int)

            # Minuto do primeiro gol (distribuição exponencial condicional a ter gol)
            minutos = []
            for g in processed["total_gols"]:
                if g > 0:
                    # Se houve gol, tempo até o 1º gol segue aproximadamente Exp(lambda=0.03) truncada em 90
                    min_gol = min(int(np.random.exponential(scale=32.0)) + 1, 90)
                    minutos.append(min_gol)
                else:
                    minutos.append(None) # Sem gols
            processed["minuto_primeiro_gol"] = minutos

            return processed
        except Exception as e:
            print(f"[API CLIENT] Falha ao carregar fonte aberta online ({e}). Usando gerador estocástico calibrado.")
            return None

    def generate_calibrated_dataset(self, n_matches: int = 380) -> pd.DataFrame:
        """
        Gera um conjunto de dados calibrado estocasticamente com parâmetros reais de futebol
        (baseado nas estatísticas reais do Brasileirão / Premier League).
        Garante funcionamento imediato e 100% reproduzível para o trabalho acadêmico.
        """
        np.random.seed(123)

        teams = [
            "Flamengo", "Palmeiras", "Atlético-MG", "Botafogo", "São Paulo",
            "Grêmio", "Internacional", "Fluminense", "Corinthians", "Athletico-PR",
            "Fortaleza", "Cruzeiro", "Vasco", "Bahia", "Red Bull Bragantino",
            "Santos", "Cuiabá", "Vitória", "Criciúma", "Juventude"
        ]

        records = []
        for i in range(n_matches):
            home, away = np.random.choice(teams, size=2, replace=False)
            
            # Parâmetros de intensidade (taxa lambda de Poisson)
            # Mandantes em média marcam mais (vantagem do mando de campo)
            lambda_home_1t = 0.65
            lambda_home_2t = 0.85
            lambda_away_1t = 0.45
            lambda_away_2t = 0.65

            # Variáveis Aleatórias Discretas: Gols via Distribuição de Poisson
            gh_1t = np.random.poisson(lambda_home_1t)
            gh_2t = np.random.poisson(lambda_home_2t)
            ga_1t = np.random.poisson(lambda_away_1t)
            ga_2t = np.random.poisson(lambda_away_2t)

            gh_total = gh_1t + gh_2t
            ga_total = ga_1t + ga_2t
            total_gols = gh_total + ga_total

            # Finalizações e Chutes a gol (correlacionados com gols)
            chutes_h = np.random.poisson(13 + gh_total * 1.5)
            chutes_a = np.random.poisson(10 + ga_total * 1.5)
            chutes_gol_h = min(chutes_h, np.random.binomial(chutes_h, 0.35) + gh_total)
            chutes_gol_a = min(chutes_a, np.random.binomial(chutes_a, 0.33) + ga_total)

            # Escanteios (Poisson com lambda em torno de 5.5 e 4.5)
            esc_h = np.random.poisson(5.8)
            esc_a = np.random.poisson(4.4)

            # Faltas e Cartões Amarelos
            faltas_h = np.random.poisson(13.2)
            faltas_a = np.random.poisson(14.0)
            amarelos_h = np.random.poisson(2.1 + (faltas_h * 0.05))
            amarelos_a = np.random.poisson(2.4 + (faltas_a * 0.05))

            # Variáveis Contínuas: Posse de Bola e Passes
            posse_h = np.clip(np.random.normal(52.0, 8.5), 28.0, 74.0).round(1)
            posse_a = round(100.0 - posse_h, 1)

            passes_h = int(posse_h * 9.0 + np.random.normal(0, 30))
            passes_a = int(posse_a * 9.0 + np.random.normal(0, 30))

            # Minuto do primeiro gol (distribuição Exponencial truncada em [1, 90])
            if total_gols > 0:
                min_gol = min(int(np.random.exponential(scale=31.0)) + 1, 90)
            else:
                min_gol = None

            records.append({
                "mandante": home,
                "visitante": away,
                "gols_mandante": gh_total,
                "gols_visitante": ga_total,
                "gols_1t_mandante": gh_1t,
                "gols_1t_visitante": ga_1t,
                "gols_2t_mandante": gh_2t,
                "gols_2t_visitante": ga_2t,
                "total_gols": total_gols,
                "total_gols_1t": gh_1t + ga_1t,
                "total_gols_2t": gh_2t + ga_2t,
                "chutes_mandante": chutes_h,
                "chutes_visitante": chutes_a,
                "chutes_gol_mandante": chutes_gol_h,
                "chutes_gol_visitante": chutes_gol_a,
                "escanteios_mandante": esc_h,
                "escanteios_visitante": esc_a,
                "total_escanteios": esc_h + esc_a,
                "faltas_mandante": faltas_h,
                "faltas_visitante": faltas_a,
                "amarelos_mandante": amarelos_h,
                "amarelos_visitante": amarelos_a,
                "total_amarelos": amarelos_h + amarelos_a,
                "posse_mandante": posse_h,
                "posse_visitante": posse_a,
                "passes_mandante": passes_h,
                "passes_visitante": passes_a,
                "minuto_primeiro_gol": min_gol
            })

        df = pd.DataFrame(records)
        return df

    def get_dataset(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Retorna o dataset de partidas:
        1. Se existir em cache local e force_refresh for False, carrega do arquivo.
        2. Tenta baixar dados reais da internet via API / CSV aberto.
        3. Caso offline ou erro, utiliza o gerador calibrado estocástico.
        Salva o resultado em CSV local.
        """
        if not force_refresh and os.path.exists(CACHE_FILE):
            try:
                df = pd.read_csv(CACHE_FILE)
                if len(df) > 0:
                    return df
            except Exception:
                pass

        # Tentativa de busca em repositório aberto
        df = self.fetch_open_data_csv()
        if df is None or len(df) == 0:
            print("[DATA] Gerando conjunto de dados estruturado e calibrado estocasticamente...")
            df = self.generate_calibrated_dataset(380)

        # Salva em cache para execuções rápidas
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(CACHE_FILE, index=False)
        print(f"[DATA] Base de dados carregada com sucesso ({len(df)} partidas) e salva em: {CACHE_FILE}")
        return df

    def get_players_data(self, force_refresh: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Retorna:
        1. df_players: Perfil consolidado de cada jogador (médias, totais, taxas de conversão).
        2. df_logs: Histórico jogo a jogo por jogador para cálculos de variância, Poisson e Bayes.
        """
        if not force_refresh and os.path.exists(PLAYERS_CACHE_FILE) and os.path.exists(PLAYER_LOGS_CACHE_FILE):
            try:
                df_p = pd.read_csv(PLAYERS_CACHE_FILE)
                df_l = pd.read_csv(PLAYER_LOGS_CACHE_FILE)
                if len(df_p) > 0 and len(df_l) > 0:
                    return df_p, df_l
            except Exception:
                pass

        # Geração de base realista de jogadores dos 20 clubes
        np.random.seed(42)

        players_roster = [
            # Man City
            ("Erling Haaland", "Man City", "Atacante", 27, 24, 0.95, 3.8, 0.45, 0.12, 18),
            ("Kevin De Bruyne", "Man City", "Meio-campo", 6, 18, 0.25, 2.2, 0.84, 0.15, 62),
            ("Phil Foden", "Man City", "Atacante", 19, 23, 0.55, 3.1, 0.52, 0.10, 42),
            ("Rodri", "Man City", "Meio-campo", 8, 27, 0.24, 1.8, 0.92, 0.28, 95),
            ("Rúben Dias", "Man City", "Defensor", 1, 26, 0.03, 0.5, 0.93, 0.22, 85),
            # Arsenal
            ("Bukayo Saka", "Arsenal", "Atacante", 16, 22, 0.48, 2.9, 0.50, 0.14, 38),
            ("Martin Ødegaard", "Arsenal", "Meio-campo", 11, 25, 0.32, 2.4, 0.86, 0.12, 58),
            ("Kai Havertz", "Arsenal", "Atacante", 13, 24, 0.38, 2.3, 0.48, 0.25, 28),
            ("Gabriel Martinelli", "Arsenal", "Atacante", 8, 22, 0.28, 2.2, 0.49, 0.10, 32),
            ("Declan Rice", "Arsenal", "Meio-campo", 7, 25, 0.19, 1.4, 0.90, 0.20, 68),
            ("William Saliba", "Arsenal", "Defensor", 2, 23, 0.05, 0.4, 0.92, 0.18, 72),
            # Liverpool
            ("Mohamed Salah", "Liverpool", "Atacante", 18, 31, 0.60, 3.4, 0.51, 0.08, 35),
            ("Darwin Núñez", "Liverpool", "Atacante", 11, 24, 0.42, 3.7, 0.42, 0.22, 20),
            ("Luis Díaz", "Liverpool", "Atacante", 9, 27, 0.30, 2.5, 0.53, 0.14, 36),
            ("Dominik Szoboszlai", "Liverpool", "Meio-campo", 5, 23, 0.16, 2.0, 0.87, 0.15, 55),
            ("Virgil van Dijk", "Liverpool", "Defensor", 3, 32, 0.08, 0.9, 0.91, 0.16, 80),
            # Tottenham
            ("Son Heung-min", "Tottenham", "Atacante", 17, 31, 0.52, 2.8, 0.55, 0.06, 34),
            ("Richarlison", "Tottenham", "Atacante", 11, 26, 0.40, 2.5, 0.46, 0.24, 22),
            ("James Maddison", "Tottenham", "Meio-campo", 4, 27, 0.20, 2.3, 0.85, 0.18, 52),
            ("Cristian Romero", "Tottenham", "Defensor", 5, 25, 0.15, 0.7, 0.89, 0.35, 65),
            # Chelsea
            ("Cole Palmer", "Chelsea", "Atacante", 22, 21, 0.68, 3.2, 0.54, 0.20, 44),
            ("Nicolas Jackson", "Chelsea", "Atacante", 14, 22, 0.44, 2.6, 0.48, 0.32, 22),
            ("Enzo Fernández", "Chelsea", "Meio-campo", 5, 23, 0.18, 1.9, 0.88, 0.28, 70),
            ("Moisés Caicedo", "Chelsea", "Meio-campo", 2, 22, 0.07, 1.1, 0.89, 0.38, 62),
            # Man United
            ("Bruno Fernandes", "Man United", "Meio-campo", 10, 29, 0.30, 2.7, 0.79, 0.26, 56),
            ("Rasmus Højlund", "Man United", "Atacante", 10, 21, 0.36, 1.8, 0.49, 0.10, 18),
            ("Marcus Rashford", "Man United", "Atacante", 8, 26, 0.26, 2.4, 0.46, 0.15, 28),
            ("Alejandro Garnacho", "Man United", "Atacante", 7, 19, 0.24, 2.5, 0.47, 0.12, 25),
            ("Casemiro", "Man United", "Meio-campo", 3, 32, 0.12, 1.3, 0.83, 0.35, 54),
            # Aston Villa
            ("Ollie Watkins", "Aston Villa", "Atacante", 19, 28, 0.53, 3.0, 0.52, 0.16, 24),
            ("Douglas Luiz", "Aston Villa", "Meio-campo", 9, 25, 0.26, 1.6, 0.86, 0.32, 58),
            ("Leon Bailey", "Aston Villa", "Atacante", 10, 26, 0.34, 2.1, 0.53, 0.15, 30),
            # Newcastle
            ("Alexander Isak", "Newcastle", "Atacante", 21, 24, 0.70, 3.1, 0.55, 0.10, 22),
            ("Anthony Gordon", "Newcastle", "Atacante", 11, 23, 0.33, 2.2, 0.49, 0.28, 30),
            ("Bruno Guimarães", "Newcastle", "Meio-campo", 7, 26, 0.20, 1.5, 0.87, 0.36, 64),
            # West Ham
            ("Jarrod Bowen", "West Ham", "Atacante", 16, 27, 0.47, 2.6, 0.52, 0.12, 26),
            ("Mohammed Kudus", "West Ham", "Atacante", 8, 23, 0.28, 2.3, 0.55, 0.18, 30),
            ("Lucas Paquetá", "West Ham", "Meio-campo", 5, 26, 0.18, 1.7, 0.81, 0.38, 50),
            # Brighton
            ("João Pedro", "Brighton", "Atacante", 9, 22, 0.35, 2.1, 0.50, 0.22, 26),
            ("Kaoru Mitoma", "Brighton", "Atacante", 6, 26, 0.25, 2.0, 0.56, 0.10, 32),
            # Wolves
            ("Matheus Cunha", "Wolves", "Atacante", 12, 24, 0.40, 2.5, 0.49, 0.30, 28),
            ("Hwang Hee-chan", "Wolves", "Atacante", 12, 28, 0.43, 2.0, 0.52, 0.14, 22),
            # Fulham
            ("Rodrigo Muniz", "Fulham", "Atacante", 9, 22, 0.36, 2.4, 0.47, 0.16, 18),
            ("Andreas Pereira", "Fulham", "Meio-campo", 3, 28, 0.10, 1.8, 0.80, 0.22, 45),
            # Everton
            ("Dominic Calvert-Lewin", "Everton", "Atacante", 7, 27, 0.25, 2.3, 0.43, 0.14, 16),
            ("Abdoulaye Doucouré", "Everton", "Meio-campo", 7, 31, 0.24, 1.6, 0.79, 0.26, 38),
            # Brentford
            ("Ivan Toney", "Brentford", "Atacante", 6, 28, 0.38, 2.8, 0.48, 0.25, 24),
            ("Bryan Mbeumo", "Brentford", "Atacante", 9, 24, 0.38, 2.4, 0.50, 0.12, 32),
            # Crystal Palace
            ("Jean-Philippe Mateta", "Crystal Palace", "Atacante", 16, 26, 0.48, 2.4, 0.54, 0.15, 18),
            ("Eberechi Eze", "Crystal Palace", "Meio-campo", 11, 25, 0.38, 2.9, 0.83, 0.12, 42),
            # Bournemouth
            ("Dominic Solanke", "Bournemouth", "Atacante", 19, 26, 0.52, 3.0, 0.48, 0.14, 20),
            # Nott'm Forest
            ("Chris Wood", "Nott'm Forest", "Atacante", 14, 32, 0.48, 2.0, 0.50, 0.10, 15),
            ("Morgan Gibbs-White", "Nott'm Forest", "Meio-campo", 5, 24, 0.16, 1.9, 0.81, 0.26, 44),
            # Luton
            ("Elijah Adebayo", "Luton", "Atacante", 10, 26, 0.38, 1.8, 0.48, 0.12, 16),
            ("Carlton Morris", "Luton", "Atacante", 10, 28, 0.34, 2.1, 0.45, 0.22, 22),
            # Burnley
            ("Lyle Foster", "Burnley", "Atacante", 5, 23, 0.24, 1.6, 0.45, 0.14, 18),
            ("Zeki Amdouni", "Burnley", "Atacante", 5, 23, 0.20, 1.9, 0.46, 0.12, 24),
            # Sheffield United
            ("Cameron Archer", "Sheffield United", "Atacante", 4, 22, 0.18, 1.7, 0.42, 0.10, 16),
            ("Gustavo Hamer", "Sheffield United", "Meio-campo", 4, 26, 0.14, 1.8, 0.78, 0.38, 42)
        ]

        players_list = []
        logs_list = []

        n_rodadas = 38
        for name, team, pos, goals, age, l_goal, avg_shots, conv_p, card_p, avg_passes in players_roster:
            # Geração dos registros jogo a jogo para o jogador
            jogos_com_gol = 0
            vitorias_quando_marca = 0
            vitorias_quando_nao_marca = 0
            total_jogos_marca = 0
            total_jogos_nao_marca = 0

            player_total_goals = 0
            player_total_shots = 0
            player_total_passes = 0
            player_total_yellows = 0

            for round_num in range(1, n_rodadas + 1):
                # Simula se o jogador participou do jogo
                jogou = np.random.rand() < 0.90
                if not jogou:
                    continue

                minutos_jogados = int(np.clip(np.random.normal(82, 12), 45, 90))
                
                # Gols no jogo via Poisson individual
                gols_jogo = int(np.random.poisson(l_goal))
                gols_1t = int(np.random.binomial(gols_jogo, 0.45)) if gols_jogo > 0 else 0
                gols_2t = gols_jogo - gols_1t

                # Chutes no jogo
                chutes_jogo = max(gols_jogo, int(np.random.poisson(avg_shots)))
                chutes_alvo = min(chutes_jogo, gols_jogo + int(np.random.binomial(chutes_jogo, 0.4)))

                # Passes e Cartão Amarelo
                passes_jogo = int(np.clip(np.random.normal(avg_passes, avg_passes * 0.25), 5, 120))
                amarelo = 1 if np.random.rand() < card_p else 0

                # Resultado do time na rodada (correlacionado positivamente com gols do jogador!)
                prob_vitoria_time = 0.72 if gols_jogo > 0 else 0.42
                rand_res = np.random.rand()
                if rand_res < prob_vitoria_time:
                    res_time = "Vitória"
                    vitoria_flag = 1
                elif rand_res < (prob_vitoria_time + 0.25):
                    res_time = "Empate"
                    vitoria_flag = 0
                else:
                    res_time = "Derrota"
                    vitoria_flag = 0

                if gols_jogo > 0:
                    total_jogos_marca += 1
                    vitorias_quando_marca += vitoria_flag
                else:
                    total_jogos_nao_marca += 1
                    vitorias_quando_nao_marca += vitoria_flag

                player_total_goals += gols_jogo
                player_total_shots += chutes_jogo
                player_total_passes += passes_jogo
                player_total_yellows += amarelo

                logs_list.append({
                    "jogador": name,
                    "time": team,
                    "posicao": pos,
                    "rodada": round_num,
                    "minutos": minutos_jogados,
                    "gols": gols_jogo,
                    "gols_1t": gols_1t,
                    "gols_2t": gols_2t,
                    "chutes": chutes_jogo,
                    "chutes_alvo": chutes_alvo,
                    "passes": passes_jogo,
                    "amarelos": amarelo,
                    "resultado_time": res_time,
                    "vitoria_time": vitoria_flag
                })

            n_jogos_real = total_jogos_marca + total_jogos_nao_marca
            players_list.append({
                "jogador": name,
                "time": team,
                "posicao": pos,
                "idade": age,
                "jogos": n_jogos_real,
                "gols": player_total_goals,
                "gols_por_jogo": round(player_total_goals / n_jogos_real, 3) if n_jogos_real > 0 else 0,
                "chutes_por_jogo": round(player_total_shots / n_jogos_real, 2) if n_jogos_real > 0 else 0,
                "taxa_conversao_pct": round((player_total_goals / player_total_shots * 100), 1) if player_total_shots > 0 else 0,
                "passes_por_jogo": round(player_total_passes / n_jogos_real, 1) if n_jogos_real > 0 else 0,
                "amarelos": player_total_yellows,
                "lambda_gols": round(l_goal, 3),
                "prob_vitoria_quando_marca": round(vitorias_quando_marca / total_jogos_marca, 3) if total_jogos_marca > 0 else 0.70,
                "prob_vitoria_quando_nao_marca": round(vitorias_quando_nao_marca / total_jogos_nao_marca, 3) if total_jogos_nao_marca > 0 else 0.40
            })

        df_players = pd.DataFrame(players_list)
        df_logs = pd.DataFrame(logs_list)

        os.makedirs(DATA_DIR, exist_ok=True)
        df_players.to_csv(PLAYERS_CACHE_FILE, index=False)
        df_logs.to_csv(PLAYER_LOGS_CACHE_FILE, index=False)
        print(f"[DATA] Base de dados de jogadores criada ({len(df_players)} jogadores, {len(df_logs)} logs).")

        return df_players, df_logs


if __name__ == "__main__":
    client = FootballDataAPI()
    dataset = client.get_dataset()
    print("Amostra dos dados obtidos:")
    print(dataset.head())
    print("\nResumo das Colunas:")
    print(dataset.info())
