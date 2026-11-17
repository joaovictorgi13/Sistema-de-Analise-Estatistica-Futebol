"""
MÓDULO: prob_stats_engine.py
DISCIPLINA: Probabilidade e Estatística
OBJETIVO: Núcleo de cálculos estatísticos e probabilísticos aplicados ao Brasileirão 2024.

CONTEÚDOS DA DISCIPLINA IMPLEMENTADOS (na ordem do plano de curso):
1. Introdução à Estatística e conceitos básicos
   - População, amostra, unidade amostral; variáveis qualitativas (nominais/ordinais)
     e quantitativas (discretas/contínuas).
2. Apresentação de dados: tabelas de distribuição de frequências
   - Frequência absoluta (fi), relativa (fri), acumulada (Fi) e relativa acumulada (Fri);
     dados agrupados em classes pela regra de Sturges.
3. Representação gráfica e medidas de posição
   - Média, mediana e moda (dados brutos e dados agrupados em classes).
4. Medidas de dispersão
   - Amplitude, variância, desvio padrão, coeficiente de variação, quartis,
     diferença interquartílica, simetria/assimetria, valores extremos (outliers) e boxplot.
5. Teoria das Probabilidades
   - Experimento determinístico x aleatório, espaço amostral, evento, probabilidade clássica
     e frequentista; eventos mutuamente exclusivos e independentes; axiomas e teoremas;
     união, interseção e probabilidade condicional.
6. Teorema da Probabilidade Total e Teorema de Bayes.
7. Variáveis aleatórias e distribuições discretas (Bernoulli, Binomial e Poisson).
8. Distribuições de variáveis aleatórias contínuas (Uniforme, Exponencial e Normal).
"""

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


def _pct(x: float) -> float:
    return round(float(x) * 100, 2)


class ProbabilityStatsEngine:
    """Motor matemático: um método para cada conteúdo da disciplina."""

    def __init__(self, df: pd.DataFrame, df_players: Optional[pd.DataFrame] = None,
                 df_logs: Optional[pd.DataFrame] = None, df_goals: Optional[pd.DataFrame] = None,
                 df_cards: Optional[pd.DataFrame] = None):
        self.df = df.copy()
        self.df_players = df_players.copy() if df_players is not None else None
        self.df_logs = df_logs.copy() if df_logs is not None else None
        self.df_goals = df_goals.copy() if df_goals is not None else None
        self.df_cards = df_cards.copy() if df_cards is not None else None

    # =========================================================================
    # 1. INTRODUÇÃO À ESTATÍSTICA E CONCEITOS BÁSICOS
    # =========================================================================
    @staticmethod
    def variable_catalog() -> pd.DataFrame:
        """Classificação das variáveis da base (qualitativa x quantitativa)."""
        linhas = [
            ("Resultado da partida", "resultado", "Qualitativa nominal", "Vitória do mandante, Empate, Vitória do visitante"),
            ("Clube mandante", "mandante", "Qualitativa nominal", "Flamengo, Palmeiras, Bahia..."),
            ("Estado do mandante", "estado_mandante", "Qualitativa nominal", "SP, RJ, MG..."),
            ("Formação tática", "formacao_mandante", "Qualitativa nominal", "4-2-3-1, 4-3-3..."),
            ("Faixa de horário", "faixa_horario", "Qualitativa ordinal", "Tarde < Fim de tarde < Noite"),
            ("Rodada", "rodada", "Quantitativa discreta", "1, 2, ..., 38"),
            ("Gols na partida", "total_gols", "Quantitativa discreta", "0, 1, 2, 3..."),
            ("Gols no 1º / 2º tempo", "total_gols_1t", "Quantitativa discreta", "0, 1, 2..."),
            ("Cartões amarelos", "total_amarelos", "Quantitativa discreta", "0, 1, 2..."),
            ("Minuto do gol", "minuto_jogo", "Quantitativa contínua", "0 a 90 minutos"),
            ("Tempo até o 1º gol", "minuto_primeiro_gol", "Quantitativa contínua", "0 a 90 minutos"),
        ]
        return pd.DataFrame(linhas, columns=["Variável", "Coluna", "Tipo", "Exemplos de valores"])

    def overview(self) -> Dict[str, Any]:
        d = self.df
        return {
            "partidas": len(d),
            "clubes": int(d["mandante"].nunique()),
            "gols": int(d["total_gols"].sum()),
            "media_gols": round(float(d["total_gols"].mean()), 2),
            "media_amarelos": round(float(d["total_amarelos"].mean()), 2),
            "vermelhos": int(d["total_vermelhos"].sum()),
            "jogadores": len(self.df_players) if self.df_players is not None else 0,
            "artilheiros": int((self.df_players["gols"] > 0).sum()) if self.df_players is not None else 0,
        }

    def standings(self) -> pd.DataFrame:
        """Classificação final: 3 pontos por vitória, 1 por empate."""
        linhas = []
        for time in sorted(self.df["mandante"].unique()):
            casa = self.df[self.df["mandante"] == time]
            fora = self.df[self.df["visitante"] == time]
            gp = casa["gols_mandante"].sum() + fora["gols_visitante"].sum()
            gc = casa["gols_visitante"].sum() + fora["gols_mandante"].sum()
            v = (casa["gols_mandante"] > casa["gols_visitante"]).sum() + (fora["gols_visitante"] > fora["gols_mandante"]).sum()
            e = (casa["gols_mandante"] == casa["gols_visitante"]).sum() + (fora["gols_visitante"] == fora["gols_mandante"]).sum()
            dr = len(casa) + len(fora) - v - e
            linhas.append({"Clube": time, "Pontos": int(3 * v + e), "Jogos": len(casa) + len(fora), "V": int(v),
                           "E": int(e), "D": int(dr), "Gols pró": int(gp), "Gols contra": int(gc), "Saldo": int(gp - gc)})
        tabela = pd.DataFrame(linhas).sort_values(["Pontos", "V", "Saldo", "Gols pró"], ascending=False).reset_index(drop=True)
        tabela.index = tabela.index + 1
        tabela.index.name = "Pos."
        return tabela

    def series(self, nome: str) -> pd.Series:
        """Retorna uma variável pelo nome, buscando em partidas, gols ou cartões."""
        if nome == "minuto_jogo":
            return self.df_goals["minuto_jogo"].astype(float)
        if nome == "gols_por_rodada":
            return self.df.groupby("rodada")["total_gols"].sum().astype(float)
        if nome == "gols_marcados_time":
            return self.team_goals_per_match_long()["gols_marcados"].astype(float)
        return self.df[nome].dropna()

    # =========================================================================
    # 2. TABELAS DE DISTRIBUIÇÃO DE FREQUÊNCIAS
    # =========================================================================
    @staticmethod
    def frequency_table(serie: pd.Series, ordem: Optional[List[Any]] = None) -> pd.DataFrame:
        """
        Distribuição de frequências para variável qualitativa ou quantitativa discreta:
        fi = frequência absoluta; fri = fi / n; Fi = Σ fi (acumulada); Fri = Fi / n.
        """
        serie = serie.dropna()
        n = len(serie)
        cont = serie.value_counts()
        if ordem is not None:
            cont = cont.reindex(ordem, fill_value=0)
        elif pd.api.types.is_numeric_dtype(serie):
            cont = cont.sort_index()
        tabela = pd.DataFrame({"Valor": cont.index.astype(str), "fi": cont.values})
        tabela["fri (%)"] = (tabela["fi"] / n * 100).round(2)
        tabela["Fi"] = tabela["fi"].cumsum()
        tabela["Fri (%)"] = (tabela["Fi"] / n * 100).round(2)
        return tabela

    @staticmethod
    def class_frequency_table(serie: pd.Series, k: Optional[int] = None) -> Dict[str, Any]:
        """
        Distribuição de frequências com dados agrupados em classes (variável contínua).
        Regra de Sturges: k ≈ 1 + 3,322·log10(n); amplitude total AT = máx − mín;
        amplitude de classe h = AT / k (arredondada para cima). Classes [li, ls).
        """
        x = serie.dropna().astype(float).values
        n = len(x)
        minimo, maximo = float(np.min(x)), float(np.max(x))
        at = maximo - minimo
        k_sturges = 1 + 3.322 * math.log10(n)
        k = k or int(round(k_sturges))
        h = math.ceil(at / k) if at > 0 else 1
        k = int(at // h) + 1  # nº de classes necessário para cobrir o máximo com classes [li, ls)
        limites = [minimo + i * h for i in range(k + 1)]
        fi = np.histogram(x, bins=limites)[0]
        tabela = pd.DataFrame({
            "Classe": [f"{limites[i]:g} ⊢ {limites[i] + h:g}" for i in range(k)],
            "li": [limites[i] for i in range(k)],
            "ls": [limites[i] + h for i in range(k)],
            "xi (ponto médio)": [limites[i] + h / 2 for i in range(k)],
            "fi": fi,
        })
        tabela["fri (%)"] = (tabela["fi"] / n * 100).round(2)
        tabela["Fi"] = tabela["fi"].cumsum()
        tabela["Fri (%)"] = (tabela["Fi"] / n * 100).round(2)
        return {"tabela": tabela, "n": n, "minimo": minimo, "maximo": maximo, "amplitude_total": at,
                "k_sturges": round(k_sturges, 2), "k": k, "h": h}

    # =========================================================================
    # 3. MEDIDAS DE POSIÇÃO (TENDÊNCIA CENTRAL)
    # =========================================================================
    @staticmethod
    def position_measures(serie: pd.Series) -> Dict[str, Any]:
        """Média aritmética x̄ = Σxi/n, mediana (valor central) e moda (valor mais frequente)."""
        x = serie.dropna().astype(float)
        modas = x.mode().tolist()
        return {
            "n": len(x),
            "soma": round(float(x.sum()), 2),
            "media": round(float(x.mean()), 3),
            "mediana": round(float(x.median()), 3),
            "moda": [round(m, 2) for m in modas][:3],
            "multimodal": len(modas) > 1,
        }

    def grouped_measures(self, serie: pd.Series) -> Dict[str, Any]:
        """
        Medidas de posição para dados agrupados em classes:
        - Média: x̄ = Σ(xi·fi) / n, com xi = ponto médio da classe
        - Mediana: Md = li + [(n/2 − Fant) / fi] · h
        - Moda de Czuber: Mo = li + [d1 / (d1 + d2)] · h, com d1 = fi − fant e d2 = fi − fpost
        """
        info = self.class_frequency_table(serie)
        t, n, h = info["tabela"], info["n"], info["h"]
        media = float((t["xi (ponto médio)"] * t["fi"]).sum() / n)
        i_md = int(np.argmax(t["Fi"].values >= n / 2))
        f_ant = int(t["Fi"].iloc[i_md - 1]) if i_md > 0 else 0
        mediana = float(t["li"].iloc[i_md] + ((n / 2 - f_ant) / t["fi"].iloc[i_md]) * h)
        i_mo = int(np.argmax(t["fi"].values))
        d1 = t["fi"].iloc[i_mo] - (t["fi"].iloc[i_mo - 1] if i_mo > 0 else 0)
        d2 = t["fi"].iloc[i_mo] - (t["fi"].iloc[i_mo + 1] if i_mo < len(t) - 1 else 0)
        moda = float(t["li"].iloc[i_mo] + (d1 / (d1 + d2)) * h) if (d1 + d2) != 0 else float(t["xi (ponto médio)"].iloc[i_mo])
        return {"media_agrupada": round(media, 2), "mediana_agrupada": round(mediana, 2),
                "moda_czuber": round(moda, 2), "classe_mediana": t["Classe"].iloc[i_md],
                "classe_modal": t["Classe"].iloc[i_mo], **{k: v for k, v in info.items() if k != "tabela"}}

    # =========================================================================
    # 4. MEDIDAS DE DISPERSÃO, QUARTIS, ASSIMETRIA, OUTLIERS E BOXPLOT
    # =========================================================================
    @staticmethod
    def dispersion_measures(serie: pd.Series) -> Dict[str, Any]:
        """
        - Amplitude total: AT = máx − mín
        - Variância amostral: s² = Σ(xi − x̄)² / (n − 1); populacional: σ² = Σ(xi − x̄)² / n
        - Desvio padrão: s = √s²
        - Coeficiente de variação: CV = s / x̄ · 100%  (baixa < 15%, média 15–30%, alta > 30%)
        - Quartis Q1, Q2, Q3 e diferença interquartílica DIQ = Q3 − Q1
        - Outliers (Tukey): abaixo de Q1 − 1,5·DIQ ou acima de Q3 + 1,5·DIQ
        - Assimetria de Pearson: As = (x̄ − Mo) / s  e  As = 3(x̄ − Md) / s
          |As| < 0,15 simétrica; 0,15 a 1 moderada; > 1 forte. As > 0: cauda à direita.
        """
        x = serie.dropna().astype(float)
        n = len(x)
        media, mediana = float(x.mean()), float(x.median())
        moda = float(x.mode().iloc[0])
        s2 = float(x.var(ddof=1))
        sigma2 = float(x.var(ddof=0))
        s = math.sqrt(s2)
        cv = s / media * 100 if media != 0 else 0.0
        q1, q3 = float(x.quantile(0.25)), float(x.quantile(0.75))
        diq = q3 - q1
        li, ls = q1 - 1.5 * diq, q3 + 1.5 * diq
        outliers = x[(x < li) | (x > ls)]
        as_moda = (media - moda) / s if s > 0 else 0.0
        as_mediana = 3 * (media - mediana) / s if s > 0 else 0.0

        def classe_assimetria(a: float) -> str:
            if abs(a) < 0.15:
                return "Simétrica"
            lado = "positiva (cauda à direita)" if a > 0 else "negativa (cauda à esquerda)"
            forca = "moderada" if abs(a) <= 1 else "forte"
            return f"Assimétrica {lado}, {forca}"

        return {
            "n": n, "minimo": float(x.min()), "maximo": float(x.max()),
            "amplitude": float(x.max() - x.min()),
            "media": round(media, 3), "mediana": round(mediana, 3), "moda": round(moda, 3),
            "variancia_amostral": round(s2, 3), "variancia_populacional": round(sigma2, 3),
            "desvio_padrao": round(s, 3), "cv_pct": round(cv, 2),
            "cv_classe": "Baixa dispersão" if cv < 15 else ("Média dispersão" if cv <= 30 else "Alta dispersão"),
            "q1": round(q1, 3), "q2": round(mediana, 3), "q3": round(q3, 3), "diq": round(diq, 3),
            "limite_inferior": round(li, 3), "limite_superior": round(ls, 3),
            "qtd_outliers": int(len(outliers)), "valores_outliers": sorted(set(outliers.round(2).tolist()))[:10],
            "assimetria_pearson_moda": round(as_moda, 3), "assimetria_pearson_mediana": round(as_mediana, 3),
            "classe_assimetria": classe_assimetria(as_mediana),
        }

    def team_goals_per_match_long(self) -> pd.DataFrame:
        """Gols marcados por clube em cada partida (38 observações por clube)."""
        casa = self.df[["rodada", "mandante", "gols_mandante"]].rename(columns={"mandante": "time", "gols_mandante": "gols_marcados"})
        fora = self.df[["rodada", "visitante", "gols_visitante"]].rename(columns={"visitante": "time", "gols_visitante": "gols_marcados"})
        return pd.concat([casa, fora], ignore_index=True)

    def team_dispersion_table(self) -> pd.DataFrame:
        """Média, desvio padrão, CV e quartis dos gols marcados por jogo de cada clube."""
        longo = self.team_goals_per_match_long()
        linhas = []
        for time, g in longo.groupby("time"):
            x = g["gols_marcados"].astype(float)
            linhas.append({"Clube": time, "Média": round(x.mean(), 2), "Mediana": x.median(),
                           "Desvio padrão": round(x.std(ddof=1), 2), "CV (%)": round(x.std(ddof=1) / x.mean() * 100, 1),
                           "Q1": x.quantile(.25), "Q3": x.quantile(.75), "Amplitude": x.max() - x.min()})
        return pd.DataFrame(linhas).sort_values("Média", ascending=False).reset_index(drop=True)

    # =========================================================================
    # 5. TEORIA DAS PROBABILIDADES
    # =========================================================================
    def sample_space_results(self) -> Dict[str, Any]:
        """
        Experimento aleatório: observar o resultado de uma partida.
        Espaço amostral S = {Vitória do mandante, Empate, Vitória do visitante}.
        - Probabilidade clássica (resultados supostamente equiprováveis): P(E) = n(E) / n(S) = 1/3.
        - Probabilidade frequentista (empírica): P(E) ≈ nº de ocorrências / nº de partidas.
        """
        n = len(self.df)
        ordem = ["Vitória do mandante", "Empate", "Vitória do visitante"]
        cont = self.df["resultado"].value_counts().reindex(ordem, fill_value=0)
        placares = (self.df["gols_mandante"].astype(str) + " x " + self.df["gols_visitante"].astype(str))
        top = placares.value_counts().head(8)
        return {
            "n": n,
            "resultados": [{"Evento": e, "Clássica (%)": round(100 / 3, 2), "Ocorrências": int(cont[e]),
                            "Frequentista (%)": _pct(cont[e] / n)} for e in ordem],
            "placares_distintos": int(placares.nunique()),
            "placares_mais_comuns": [{"Placar": p, "Ocorrências": int(c), "Probabilidade (%)": _pct(c / n)} for p, c in top.items()],
            "p_0x0": _pct((placares == "0 x 0").mean()),
            "p_mais_2_gols": _pct((self.df["total_gols"] >= 3).mean()),
            "p_ate_2_gols": _pct((self.df["total_gols"] <= 2).mean()),
        }

    def events_and_axioms(self) -> Dict[str, Any]:
        """
        Axiomas: (1) 0 ≤ P(A) ≤ 1; (2) P(S) = 1; (3) se A ∩ B = ∅, P(A ∪ B) = P(A) + P(B).
        Teoremas: complementar P(Aᶜ) = 1 − P(A); união P(A ∪ B) = P(A) + P(B) − P(A ∩ B).
        """
        d = self.df
        n = len(d)
        pV = float((d["resultado"] == "Vitória do mandante").mean())
        pE = float((d["resultado"] == "Empate").mean())
        pD = float((d["resultado"] == "Vitória do visitante").mean())

        M1 = d["gols_1t_mandante"] > 0      # mandante marca no 1º tempo
        V1 = d["gols_1t_visitante"] > 0     # visitante marca no 1º tempo
        pM1, pV1, pM1V1 = float(M1.mean()), float(V1.mean()), float((M1 & V1).mean())
        uniao_formula = pM1 + pV1 - pM1V1
        uniao_direta = float((M1 | V1).mean())

        return {
            "n": n,
            "p_vitoria_mandante": _pct(pV), "p_empate": _pct(pE), "p_vitoria_visitante": _pct(pD),
            "soma_espaco_amostral": _pct(pV + pE + pD),
            "p_mandante_nao_perde": _pct(pV + pE),
            "p_complementar_vitoria_mandante": _pct(1 - pV),
            "p_m1": _pct(pM1), "p_v1": _pct(pV1), "p_m1_inter_v1": _pct(pM1V1),
            "p_m1_uniao_v1_formula": _pct(uniao_formula), "p_m1_uniao_v1_direta": _pct(uniao_direta),
            "p_nenhum_gol_1t": _pct(1 - uniao_direta),
            "n_m1": int(M1.sum()), "n_v1": int(V1.sum()), "n_m1_v1": int((M1 & V1).sum()),
        }

    def conditional_and_independence(self) -> Dict[str, Any]:
        """
        Probabilidade condicional: P(A | B) = P(A ∩ B) / P(B)  e regra do produto P(A ∩ B) = P(B)·P(A | B).
        Independência: A e B são independentes ⇔ P(A ∩ B) = P(A)·P(B) ⇔ P(A | B) = P(A).
        """
        d = self.df
        A = d["resultado"] == "Vitória do mandante"
        B = d["primeiro_gol"] == "Mandante"
        pA, pB, pAB = float(A.mean()), float(B.mean()), float((A & B).mean())

        M1 = d["gols_1t_mandante"] > 0
        M2 = d["gols_2t_mandante"] > 0
        pM1, pM2, pM1M2 = float(M1.mean()), float(M2.mean()), float((M1 & M2).mean())
        p_m2_dado_m1 = pM1M2 / pM1 if pM1 > 0 else 0.0
        diferenca = abs(pM1M2 - pM1 * pM2)

        return {
            "p_a": _pct(pA), "p_b": _pct(pB), "p_a_inter_b": _pct(pAB),
            "n_a": int(A.sum()), "n_b": int(B.sum()), "n_a_inter_b": int((A & B).sum()),
            "p_a_dado_b": _pct(pAB / pB), "p_b_dado_a": _pct(pAB / pA),
            "produto_pa_pb": _pct(pA * pB),
            "a_b_independentes": abs(pAB - pA * pB) < 0.03,
            "p_m1": _pct(pM1), "p_m2": _pct(pM2), "p_m1_inter_m2": _pct(pM1M2),
            "produto_m1_m2": _pct(pM1 * pM2), "p_m2_dado_m1": _pct(p_m2_dado_m1),
            "diferenca_independencia": round(diferenca * 100, 2),
            "m1_m2_independentes": diferenca < 0.03,
        }

    # =========================================================================
    # 6. TEOREMA DA PROBABILIDADE TOTAL E TEOREMA DE BAYES
    # =========================================================================
    def total_probability_bayes(self, team: Optional[str] = None) -> Dict[str, Any]:
        """
        Partição do espaço amostral pelo primeiro gol:
            B1 = "marcou primeiro", B2 = "o adversário marcou primeiro", B3 = "jogo sem gols".
        Evento A = "vitória" (do mandante, ou do clube escolhido).
        - Probabilidade Total: P(A) = Σ P(A | Bi) · P(Bi)
        - Bayes: P(Bi | A) = P(A | Bi) · P(Bi) / P(A)
        """
        d = self.df
        if team is None:
            primeiro = d["primeiro_gol"].map({"Mandante": "B1", "Visitante": "B2", "Sem gols": "B3"})
            vitoria = d["resultado"] == "Vitória do mandante"
            rotulos = {"B1": "Mandante marcou primeiro", "B2": "Visitante marcou primeiro", "B3": "Jogo sem gols"}
            evento = "Vitória do mandante"
        else:
            jogos = d[(d["mandante"] == team) | (d["visitante"] == team)]
            em_casa = jogos["mandante"] == team
            lado_time = np.where(em_casa, "Mandante", "Visitante")
            primeiro = pd.Series(np.where(jogos["primeiro_gol"] == "Sem gols", "B3",
                                          np.where(jogos["primeiro_gol"] == lado_time, "B1", "B2")), index=jogos.index)
            gp = np.where(em_casa, jogos["gols_mandante"], jogos["gols_visitante"])
            gc = np.where(em_casa, jogos["gols_visitante"], jogos["gols_mandante"])
            vitoria = pd.Series(gp > gc, index=jogos.index)
            rotulos = {"B1": f"{team} marcou primeiro", "B2": "Adversário marcou primeiro", "B3": "Jogo sem gols"}
            evento = f"Vitória do {team}"

        n = len(primeiro)
        p_a = float(vitoria.mean())
        linhas, soma = [], 0.0
        for b in ("B1", "B2", "B3"):
            mask = primeiro == b
            p_b = float(mask.mean())
            p_a_b = float(vitoria[mask].mean()) if mask.sum() > 0 else 0.0
            contrib = p_a_b * p_b
            soma += contrib
            linhas.append({"Evento": b, "Descrição": rotulos[b], "Jogos": int(mask.sum()),
                           "P(Bi) (%)": _pct(p_b), "P(A|Bi) (%)": _pct(p_a_b), "P(A|Bi)·P(Bi) (%)": _pct(contrib),
                           "Bayes P(Bi|A) (%)": _pct(contrib / p_a) if p_a > 0 else 0.0})
        return {"evento": evento, "n": n, "linhas": linhas, "p_a_total": _pct(soma), "p_a_direta": _pct(p_a),
                "vitorias": int(vitoria.sum())}

    # =========================================================================
    # 7. VARIÁVEIS ALEATÓRIAS E DISTRIBUIÇÕES DISCRETAS
    # =========================================================================
    def discrete_random_variable(self, col: str) -> Dict[str, Any]:
        """
        V.A. discreta X: função de probabilidade p(x) = P(X = x), função de distribuição
        F(x) = P(X ≤ x), esperança E[X] = Σ x·p(x) e variância Var(X) = E[X²] − (E[X])².
        """
        x = self.series(col).astype(int)
        p = x.value_counts(normalize=True).sort_index()
        xs = p.index.values.astype(float)
        e = float((xs * p.values).sum())
        e2 = float(((xs ** 2) * p.values).sum())
        var = e2 - e ** 2
        tabela = pd.DataFrame({"x": p.index, "P(X = x)": p.values.round(4), "F(x) = P(X ≤ x)": np.cumsum(p.values).round(4),
                               "x·P(X = x)": (xs * p.values).round(4)})
        return {"tabela": tabela, "esperanca": round(e, 4), "e_x2": round(e2, 4), "variancia": round(var, 4),
                "desvio": round(math.sqrt(var), 4), "n": len(x)}

    def bernoulli_home_win(self, team: Optional[str] = None) -> Dict[str, Any]:
        """X ~ Bernoulli(p): X = 1 se o mandante vence. E[X] = p, Var(X) = p(1 − p)."""
        d = self.df if team is None else self.df[self.df["mandante"] == team]
        p = float((d["resultado"] == "Vitória do mandante").mean())
        return {"p": round(p, 4), "q": round(1 - p, 4), "esperanca": round(p, 4), "variancia": round(p * (1 - p), 4),
                "jogos": len(d), "vitorias": int((d["resultado"] == "Vitória do mandante").sum())}

    @staticmethod
    def binomial(n: int, p: float) -> Dict[str, Any]:
        """X ~ Bin(n, p): P(X = k) = C(n, k) · p^k · (1 − p)^(n − k); E[X] = np; Var(X) = np(1 − p)."""
        k = np.arange(0, n + 1)
        pmf = stats.binom.pmf(k, n, p)
        tabela = pd.DataFrame({"k": k, "P(X = k)": pmf.round(4), "P(X ≤ k)": np.cumsum(pmf).round(4)})
        return {"n": n, "p": round(p, 4), "tabela": tabela, "esperanca": round(n * p, 3),
                "variancia": round(n * p * (1 - p), 3), "desvio": round(math.sqrt(n * p * (1 - p)), 3),
                "p_nenhum": _pct(pmf[0]), "p_pelo_menos_1": _pct(1 - pmf[0]),
                "p_mais_da_metade": _pct(pmf[k > n / 2].sum())}

    def poisson_fit(self, col: str = "total_gols") -> Dict[str, Any]:
        """
        X ~ Poisson(λ): P(X = k) = e^(−λ) · λ^k / k!; E[X] = Var(X) = λ.
        λ é estimado pela média amostral; compara frequências observadas e esperadas (n·P(X = k)).
        """
        x = self.series(col).astype(int)
        n, lam = len(x), float(x.mean())
        kmax = int(x.max())
        k = np.arange(0, kmax + 1)
        obs = x.value_counts().reindex(k, fill_value=0).values
        pmf = stats.poisson.pmf(k, lam)
        tabela = pd.DataFrame({"k": k, "Observado (fi)": obs, "P(X = k) Poisson": pmf.round(4),
                               "Esperado (n·P)": (n * pmf).round(1)})
        return {"lambda": round(lam, 4), "variancia_amostral": round(float(x.var(ddof=1)), 4), "n": n, "tabela": tabela,
                "p_zero": _pct(pmf[0]), "p_zero_obs": _pct((x == 0).mean()),
                "p_3_ou_mais": _pct(1 - stats.poisson.cdf(2, lam)), "p_3_ou_mais_obs": _pct((x >= 3).mean())}

    def poisson_by_half(self, team: Optional[str] = None) -> Dict[str, Any]:
        """
        Gols do mandante por tempo: X1T ~ Poisson(λ1T), X2T ~ Poisson(λ2T).
        P(marcar ao menos 1) = 1 − e^(−λ). Se independentes, X1T + X2T ~ Poisson(λ1T + λ2T).
        """
        d = self.df if team is None else self.df[self.df["mandante"] == team]
        l1, l2 = float(d["gols_1t_mandante"].mean()), float(d["gols_2t_mandante"].mean())
        return {"time": team or "Todos os mandantes", "jogos": len(d), "lambda_1t": round(l1, 3), "lambda_2t": round(l2, 3),
                "lambda_total": round(l1 + l2, 3),
                "p_gol_1t": _pct(1 - math.exp(-l1)), "p_gol_1t_obs": _pct((d["gols_1t_mandante"] > 0).mean()),
                "p_gol_2t": _pct(1 - math.exp(-l2)), "p_gol_2t_obs": _pct((d["gols_2t_mandante"] > 0).mean()),
                "p_gol_jogo": _pct(1 - math.exp(-(l1 + l2))), "p_gol_jogo_obs": _pct((d["gols_mandante"] > 0).mean())}

    def simulate_match(self, home: str, away: str, max_gols: int = 7) -> Dict[str, Any]:
        """
        Simulador de partida com duas Poisson independentes:
        λ_casa = média de (gols que o mandante marca em casa, gols que o visitante sofre fora)
        λ_fora = média de (gols que o visitante marca fora, gols que o mandante sofre em casa)
        P(placar i x j) = P(H = i) · P(A = j).
        """
        casa = self.df[self.df["mandante"] == home]
        fora = self.df[self.df["visitante"] == away]
        lam_h = (casa["gols_mandante"].mean() + fora["gols_mandante"].mean()) / 2
        lam_a = (fora["gols_visitante"].mean() + casa["gols_visitante"].mean()) / 2
        i = np.arange(0, max_gols + 1)
        ph, pa = stats.poisson.pmf(i, lam_h), stats.poisson.pmf(i, lam_a)
        matriz = np.outer(ph, pa)
        vit_h = float(np.tril(matriz, -1).sum())
        emp = float(np.trace(matriz))
        vit_a = float(np.triu(matriz, 1).sum())
        idx = np.unravel_index(np.argmax(matriz), matriz.shape)
        soma_gols = np.add.outer(i, i)
        return {"mandante": home, "visitante": away, "lambda_mandante": round(float(lam_h), 3),
                "lambda_visitante": round(float(lam_a), 3), "matriz": matriz,
                "p_vitoria_mandante": _pct(vit_h), "p_empate": _pct(emp), "p_vitoria_visitante": _pct(vit_a),
                "placar_provavel": f"{idx[0]} x {idx[1]}", "p_placar_provavel": _pct(matriz[idx]),
                "p_mais_2_gols": _pct(matriz[soma_gols >= 3].sum()), "massa_total": _pct(matriz.sum())}

    # =========================================================================
    # 8. DISTRIBUIÇÕES CONTÍNUAS: UNIFORME, EXPONENCIAL E NORMAL
    # =========================================================================
    def uniform_goal_minute(self, a: float = 0.0, b: float = 90.0) -> Dict[str, Any]:
        """
        X ~ U(a, b): f(x) = 1/(b − a); P(c ≤ X ≤ d) = (d − c)/(b − a);
        E[X] = (a + b)/2; Var(X) = (b − a)²/12. Aplicação: minuto em que sai cada gol.
        """
        x = self.df_goals["minuto_jogo"].astype(float)
        faixas = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 75), (75, 90)]
        linhas = []
        for c, d in faixas:
            obs = ((x > c) & (x <= d)).mean() if c > 0 else ((x >= c) & (x <= d)).mean()
            linhas.append({"Faixa (min)": f"{c}–{d}", "Gols": int(round(obs * len(x))), "Observado (%)": _pct(obs),
                           "Uniforme (%)": _pct((d - c) / (b - a))})
        return {"a": a, "b": b, "densidade": round(1 / (b - a), 5), "esperanca": (a + b) / 2,
                "variancia": round((b - a) ** 2 / 12, 2), "desvio": round(math.sqrt((b - a) ** 2 / 12), 2),
                "media_obs": round(float(x.mean()), 2), "desvio_obs": round(float(x.std()), 2),
                "n": len(x), "tabela": pd.DataFrame(linhas)}

    def exponential_first_goal(self) -> Dict[str, Any]:
        """
        T ~ Exp(λ): f(t) = λ·e^(−λt); F(t) = P(T ≤ t) = 1 − e^(−λt); E[T] = 1/λ; Var(T) = 1/λ².
        λ estimado por 1 / (tempo médio até o 1º gol). Propriedade da falta de memória:
        P(T > s + t | T > s) = P(T > t).
        """
        t = self.df["minuto_primeiro_gol"].dropna().astype(float)
        media = float(t.mean())
        lam = 1 / media
        cortes = [15, 30, 45, 60]
        linhas = [{"Evento": f"1º gol até {c} min", "Exponencial (%)": _pct(1 - math.exp(-lam * c)),
                   "Observado (%)": _pct((t <= c).mean())} for c in cortes]
        p_sem_ate_30 = math.exp(-lam * 30)
        p_sem_ate_45 = math.exp(-lam * 45)
        return {"n": len(t), "sem_gols": int(self.df["minuto_primeiro_gol"].isna().sum()), "media": round(media, 2),
                "lambda": round(lam, 5), "desvio_teorico": round(media, 2), "desvio_obs": round(float(t.std()), 2),
                "tabela": pd.DataFrame(linhas),
                "p_mais_15_dado_mais_30": _pct(math.exp(-lam * 45) / math.exp(-lam * 30)),
                "p_mais_15": _pct(math.exp(-lam * 15)), "p_sem_gol_ate_30": _pct(p_sem_ate_30),
                "p_sem_gol_ate_45": _pct(p_sem_ate_45)}

    def normal_goals_per_round(self, x1: float = 20, x2: float = 30) -> Dict[str, Any]:
        """
        X ~ N(μ, σ²): f(x) = 1/(σ√(2π)) · e^(−(x − μ)²/(2σ²)). Padronização Z = (X − μ)/σ.
        Aplicação: total de gols de cada rodada (soma de 10 partidas), estimando μ e σ pela amostra.
        """
        x = self.series("gols_por_rodada")
        mu, sigma = float(x.mean()), float(x.std(ddof=1))
        z1, z2 = (x1 - mu) / sigma, (x2 - mu) / sigma
        regra = {f"μ ± {k}σ": {"Teórico (%)": t, "Observado (%)": _pct(((x >= mu - k * sigma) & (x <= mu + k * sigma)).mean())}
                 for k, t in ((1, 68.27), (2, 95.45), (3, 99.73))}
        return {"n": len(x), "mu": round(mu, 2), "sigma": round(sigma, 2), "x1": x1, "x2": x2,
                "z1": round(z1, 3), "z2": round(z2, 3),
                "p_entre": _pct(stats.norm.cdf(z2) - stats.norm.cdf(z1)),
                "p_entre_obs": _pct(((x >= x1) & (x <= x2)).mean()),
                "p_acima_x2": _pct(1 - stats.norm.cdf(z2)), "p_acima_x2_obs": _pct((x > x2).mean()),
                "rodada_max": int(x.idxmax()), "gols_max": int(x.max()), "z_max": round(float((x.max() - mu) / sigma), 2),
                "rodada_min": int(x.idxmin()), "gols_min": int(x.min()), "z_min": round(float((x.min() - mu) / sigma), 2),
                "regra_empirica": regra, "valores": x}

    # =========================================================================
    # CONSULTA POR TIME E JOGADOR
    # =========================================================================
    def get_teams_list(self) -> List[str]:
        return sorted(self.df["mandante"].unique())

    def get_players_by_team(self, team: str) -> List[str]:
        sub = self.df_players[self.df_players["time"] == team].sort_values(["gols", "amarelos"], ascending=False)
        return sub["jogador"].tolist()

    def team_profile(self, team: str) -> Dict[str, Any]:
        tabela = self.standings()
        linha = tabela[tabela["Clube"] == team].iloc[0]
        casa = self.df[self.df["mandante"] == team]
        fora = self.df[self.df["visitante"] == team]
        marcados = pd.concat([casa["gols_mandante"], fora["gols_visitante"]]).astype(float)
        sofridos = pd.concat([casa["gols_visitante"], fora["gols_mandante"]]).astype(float)
        amarelos = pd.concat([casa["amarelos_mandante"], fora["amarelos_visitante"]]).astype(float)
        return {"posicao": int(tabela.index[tabela["Clube"] == team][0]), **linha.to_dict(),
                "aproveitamento": round(linha["Pontos"] / (3 * linha["Jogos"]) * 100, 1),
                "vitorias_casa": int((casa["gols_mandante"] > casa["gols_visitante"]).sum()),
                "vitorias_fora": int((fora["gols_visitante"] > fora["gols_mandante"]).sum()),
                "media_marcados": round(marcados.mean(), 2), "desvio_marcados": round(marcados.std(), 2),
                "media_sofridos": round(sofridos.mean(), 2), "media_amarelos": round(amarelos.mean(), 2),
                "marcados": marcados, "sofridos": sofridos}

    def _player_row(self, jogador: str, time: str) -> pd.Series:
        return self.df_players[(self.df_players["jogador"] == jogador) & (self.df_players["time"] == time)].iloc[0]

    def player_poisson(self, jogador: str, time: str) -> Dict[str, Any]:
        """Gols do atleta por partida do clube ~ Poisson(λ = gols / 38)."""
        r = self._player_row(jogador, time)
        jogos = float(r["jogos_do_time"])
        lam = float(r["gols"]) / jogos
        l1, l2 = float(r["gols_1t"]) / jogos, float(r["gols_2t"]) / jogos
        lam_am = float(r["amarelos"]) / jogos
        pmf = {k: round(float(stats.poisson.pmf(k, lam)), 4) for k in range(4)}
        return {"lambda": round(lam, 3), "lambda_1t": round(l1, 3), "lambda_2t": round(l2, 3),
                "p_marcar": _pct(1 - math.exp(-lam)), "p_marcar_1t": _pct(1 - math.exp(-l1)),
                "p_marcar_2t": _pct(1 - math.exp(-l2)), "p_amarelo": _pct(1 - math.exp(-lam_am)), "pmf": pmf,
                "p_marcar_obs": _pct(r["jogos_com_gol"] / r["jogos_do_time"])}

    def player_binomial(self, jogador: str, time: str, n: int) -> Dict[str, Any]:
        """Y = nº de jogos com gol do atleta nos próximos n jogos ~ Bin(n, p), p = jogos com gol / 38."""
        r = self._player_row(jogador, time)
        p = r["jogos_com_gol"] / r["jogos_do_time"]
        return {"p": round(float(p), 4), **self.binomial(n, float(p))}

    def player_bayes(self, jogador: str, time: str) -> Optional[Dict[str, Any]]:
        """
        P(V | M) e P(V | Mᶜ) a partir das 38 partidas do clube; Bayes: P(M | V) = P(V | M)·P(M) / P(V).
        Retorna None se o atleta não tem gols (P(V | M) não existe quando P(M) = 0).
        """
        if self.df_logs is None or self.df_logs.empty:
            return None
        logs = self.df_logs[(self.df_logs["jogador"] == jogador) & (self.df_logs["time"] == time)]
        if logs.empty or (logs["gols"] > 0).sum() == 0:
            return None
        marcou = logs["gols"] > 0
        p_m = float(marcou.mean())
        p_v = float(logs["vitoria_time"].mean())
        p_v_m = float(logs.loc[marcou, "vitoria_time"].mean())
        p_v_nm = float(logs.loc[~marcou, "vitoria_time"].mean()) if (~marcou).sum() > 0 else 0.0
        return {"jogos": len(logs), "jogos_marcou": int(marcou.sum()), "vitorias": int(logs["vitoria_time"].sum()),
                "vitorias_quando_marcou": int(logs.loc[marcou, "vitoria_time"].sum()),
                "p_m": _pct(p_m), "p_v": _pct(p_v), "p_v_dado_m": _pct(p_v_m), "p_v_dado_nao_m": _pct(p_v_nm),
                "diferenca_pp": round((p_v_m - p_v_nm) * 100, 1),
                "p_m_dado_v": _pct(p_v_m * p_m / p_v) if p_v > 0 else 0.0,
                "p_v_total": _pct(p_v_m * p_m + p_v_nm * (1 - p_m))}

    def player_goal_minutes(self, jogador: str, time: str) -> pd.DataFrame:
        g = self.df_goals[(self.df_goals["atleta"] == jogador) & (self.df_goals["clube"] == time) &
                          (self.df_goals["tipo_de_gol"] != "Gol Contra")]
        return g.merge(self.df[["partida_id", "mandante", "visitante", "gols_mandante", "gols_visitante"]], on="partida_id")
