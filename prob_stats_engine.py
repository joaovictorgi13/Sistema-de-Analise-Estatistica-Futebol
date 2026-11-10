"""
MÓDULO: prob_stats_engine.py
DISCIPLINA: Probabilidade e Estatística
OBJETIVO: Núcleo de Modelagem Estatística e Teoria das Probabilidades aplicado ao Futebol.

CONTEÚDOS ABORDADOS E IMPLEMENTADOS:
1. Análise Exploratória de Dados (AED):
   - Média, Mediana, Moda, Variância (amostral/populacional), Desvio Padrão, CV,
     Quartis (Q1, Q2, Q3), IQR, Limites de Outliers, Assimetria (Skewness) e Curtose (Kurtosis).
2. Teoria das Probabilidades:
   - Probabilidade Frequentista, Probabilidade Condicional P(A|B), Teorema de Bayes,
     Teste de Independência de Eventos: P(A ∩ B) vs P(A)*P(B).
3. Variáveis Aleatórias Discretas e Contínuas:
   - V.A. Discretas (Gols, Cartões, Escanteios): PMF, CDF, E[X], Var(X).
   - V.A. Contínuas (Minuto do 1º Gol, Posse de Bola %): PDF estimada, CDF, P(a <= X <= b).
4. Variáveis Aleatórias Bidimensionais:
   - Distribuição de Probabilidade Conjunta P(X=x, Y=y), Distribuições Marginais P(X) e P(Y),
     Probabilidade Condicional P(Y|X), Covariância Cov(X,Y), Correlações de Pearson e Spearman.
5. Distribuições de Probabilidades:
   - Discretas: Poisson (modelagem de gols por tempo, matriz de placares 1T/2T/Jogo todo),
     Binomial (conversão de chutes a gol em gols).
   - Contínuas: Exponencial (tempo até o primeiro gol), Normal (posse de bola, escores Z).
6. Leis dos Grandes Números e Função Geratriz de Momentos (FGM):
   - Lei Fraca dos Grandes Números: Simulação de convergência da média amostral para a esperança teórica.
   - Função Geratriz de Momentos M_X(t) = E[e^{tX}]: derivação teórica e cálculo numérico para Poisson.
"""

import math
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Tuple, List, Optional

class ProbabilityStatsEngine:
    """
    Motor matemático que implementa os conceitos teóricos de Probabilidade e Estatística.
    """

    def __init__(self, df: pd.DataFrame, df_players: Optional[pd.DataFrame] = None, df_logs: Optional[pd.DataFrame] = None):
        self.df = df.copy()
        self.df_players = df_players.copy() if df_players is not None else None
        self.df_logs = df_logs.copy() if df_logs is not None else None

    # =========================================================================
    # TÓPICO 1: ANÁLISE EXPLORATÓRIA DE DADOS (AED)
    # =========================================================================
    def exploratory_data_analysis(self, column: str) -> Dict[str, Any]:
        """
        Calcula as estatísticas descritivas completas de uma variável:
        - Tendência Central: Média (μ / x̄), Mediana (Q2), Moda.
        - Dispersão: Variância Amostral (s²), Desvio Padrão (s), Coeficiente de Variação (CV).
        - Separatrizes / Posição: Quartis (Q1, Q2, Q3) e Intervalo Interquartil (IQR).
        - Outliers: Limites de Tukey [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
        - Forma da Distribuição: Assimetria (Skewness de Fisher-Pearson) e Curtose (Kurtosis).
        """
        # Limpeza de valores nulos (ex: jogos sem gols para minuto_primeiro_gol)
        series = self.df[column].dropna().astype(float)
        n = len(series)

        if n == 0:
            return {"erro": "Coluna vazia ou sem dados numéricos"}

        media = float(series.mean())
        mediana = float(series.median())
        
        # Moda estatística
        mode_res = stats.mode(series, keepdims=False)
        moda = float(mode_res.mode) if hasattr(mode_res, "mode") else float(mode_res[0])

        variancia_amostral = float(series.var(ddof=1)) # divisor (n - 1)
        variancia_populacional = float(series.var(ddof=0)) # divisor n
        desvio_padrao = float(series.std(ddof=1))
        
        # Coeficiente de Variação: CV = (s / x̄) * 100%
        cv = float((desvio_padrao / media) * 100) if media != 0 else 0.0

        q1 = float(series.quantile(0.25))
        q2 = mediana
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1

        # Limites para detecção de outliers (Critério de John Tukey)
        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr
        outliers = series[(series < limite_inferior) | (series > limite_superior)].tolist()

        # Medidas de Forma:
        # Assimetria (Skewness):
        # > 0: Cauda à direita (positivamente assimétrica)
        # = 0: Simétrica
        # < 0: Cauda à esquerda (negativamente assimétrica)
        assimetria = float(stats.skew(series))

        # Curtose (Excess Kurtosis):
        # > 0: Leptocúrtica (caudas pesadas, pico acentuado)
        # = 0: Mesocúrtica (semelhante à Normal)
        # < 0: Platicúrtica (caudas leves, pico achatado)
        curtose = float(stats.kurtosis(series))

        return {
            "variavel": column,
            "tamanho_amostral_n": n,
            "media": round(media, 3),
            "mediana": round(mediana, 3),
            "moda": round(moda, 3),
            "variancia_amostral": round(variancia_amostral, 3),
            "variancia_populacional": round(variancia_populacional, 3),
            "desvio_padrao": round(desvio_padrao, 3),
            "coeficiente_variacao_pct": round(cv, 2),
            "q1": round(q1, 3),
            "q2": round(q2, 3),
            "q3": round(q3, 3),
            "iqr": round(iqr, 3),
            "limite_inferior_outlier": round(limite_inferior, 3),
            "limite_superior_outlier": round(limite_superior, 3),
            "qtd_outliers": len(outliers),
            "outliers_exemplos": outliers[:5],
            "assimetria": round(assimetria, 3),
            "curtose": round(curtose, 3),
            "minimo": float(series.min()),
            "maximo": float(series.max())
        }

    # =========================================================================
    # TÓPICO 2: TEORIA DAS PROBABILIDADES
    # =========================================================================
    def calculate_conditional_probabilities(self) -> Dict[str, Any]:
        """
        Aplica a Teoria Clássica e Frequentista das Probabilidades:
        1. Probabilidade Simples: P(A) = n(A) / n(S)
        2. Probabilidade Condicional:
           P(A | B) = P(A ∩ B) / P(B)
           Exemplo: P(Vitória do Mandante | Mandante marcou gol no 1º tempo)
        3. Teorema de Bayes:
           P(B | A) = [P(A | B) * P(B)] / P(A)
           Exemplo: P(Mandante marcou no 1T | Mandante venceu o jogo)
        4. Teste de Independência Estocástica:
           Eventos A e B são independentes se e somente se:
           P(A ∩ B) = P(A) * P(B)  ou  P(A | B) = P(A)
           Exemplo: 'Marcar no 1º tempo' é independente de 'Marcar no 2º tempo'?
        """
        n_total = len(self.df)

        # Definição dos eventos de interesse:
        # Evento A: Mandante vence a partida
        evento_vitoria_casa = self.df["gols_mandante"] > self.df["gols_visitante"]
        # Evento B: Mandante marcou gol no 1º tempo
        evento_gol_1t_casa = self.df["gols_1t_mandante"] > 0
        # Evento C: Mandante marcou gol no 2º tempo
        evento_gol_2t_casa = self.df["gols_2t_mandante"] > 0

        # Probabilidades marginais (P(A), P(B), P(C))
        p_vitoria_casa = float(evento_vitoria_casa.mean())
        p_gol_1t_casa = float(evento_gol_1t_casa.mean())
        p_gol_2t_casa = float(evento_gol_2t_casa.mean())

        # Interseção A ∩ B: Mandante fez gol no 1T E venceu
        intersecao_vitoria_e_gol1t = evento_vitoria_casa & evento_gol_1t_casa
        p_intersecao_vitoria_e_gol1t = float(intersecao_vitoria_e_gol1t.mean())

        # Probabilidade Condicional P(A | B) = P(A ∩ B) / P(B)
        p_vitoria_dado_gol1t = p_intersecao_vitoria_e_gol1t / p_gol_1t_casa if p_gol_1t_casa > 0 else 0.0

        # Teorema de Bayes: P(B | A) = [P(A | B) * P(B)] / P(A)
        # Probabilidade do mandante ter feito gol no 1T dado que venceu a partida
        p_bayes_gol1t_dado_vitoria = (p_vitoria_dado_gol1t * p_gol_1t_casa) / p_vitoria_casa if p_vitoria_casa > 0 else 0.0
        # Conferência empírica direta: P(B | A) = n(B ∩ A) / n(A)
        p_empirica_gol1t_dado_vitoria = float((intersecao_vitoria_e_gol1t.sum()) / (evento_vitoria_casa.sum()))

        # Teste de Independência entre Marcar no 1T (B) e Marcar no 2T (C):
        intersecao_gol1t_e_gol2t = evento_gol_1t_casa & evento_gol_2t_casa
        p_intersecao_1t_2t = float(intersecao_gol1t_e_gol2t.mean())
        p_produto_independente = p_gol_1t_casa * p_gol_2t_casa
        diferenca_independencia = abs(p_intersecao_1t_2t - p_produto_independente)
        sao_independentes = diferenca_independencia < 0.03 # tolerância amostral

        return {
            "total_partidas": n_total,
            "p_vitoria_mandante": round(p_vitoria_casa, 4),
            "p_gol_1t_mandante": round(p_gol_1t_casa, 4),
            "p_gol_2t_mandante": round(p_gol_2t_casa, 4),
            "p_intersecao_vitoria_e_gol1t": round(p_intersecao_vitoria_e_gol1t, 4),
            "prob_condicional_p_vitoria_dado_gol1t": round(p_vitoria_dado_gol1t, 4),
            "teorema_bayes_p_gol1t_dado_vitoria": round(p_bayes_gol1t_dado_vitoria, 4),
            "conferencia_empirica_bayes": round(p_empirica_gol1t_dado_vitoria, 4),
            "teste_independencia_1t_e_2t": {
                "P(Gol_1T_e_Gol_2T)": round(p_intersecao_1t_2t, 4),
                "P(Gol_1T) * P(Gol_2T)": round(p_produto_independente, 4),
                "diferenca_absoluta": round(diferenca_independencia, 4),
                "conclusao": "Eventos aproximadamente Independentes" if sao_independentes else "Eventos Dependentes"
            }
        }

    # =========================================================================
    # TÓPICO 3: VARIÁVEIS ALEATÓRIAS DISCRETAS E CONTÍNUAS
    # =========================================================================
    def discrete_random_variable_analysis(self, column: str = "total_gols") -> Dict[str, Any]:
        """
        Analisa uma Variável Aleatória Discreta X:
        - Espaço Amostral e Suporte da V.A.: {x_1, x_2, ..., x_k}
        - Função de Massa de Probabilidade (PMF): P(X = x)
        - Função de Distribuição Acumulada (CDF): F(x) = P(X <= x)
        - Esperança Matemática: E[X] = ∑ x * P(X = x)
        - Segundo Momento: E[X²] = ∑ x² * P(X = x)
        - Variância Teórica Discreta: Var(X) = E[X²] - (E[X])²
        """
        series = self.df[column].dropna().astype(int)
        val_counts = series.value_counts(normalize=True).sort_index()

        suporte = val_counts.index.tolist()
        pmf = val_counts.values.tolist()
        cdf = np.cumsum(pmf).tolist()

        # E[X] = ∑ x * P(X=x)
        esperanca = sum(x * p for x, p in zip(suporte, pmf))
        # E[X²] = ∑ x² * P(X=x)
        esperanca_quadrado = sum((x ** 2) * p for x, p in zip(suporte, pmf))
        # Var(X) = E[X²] - (E[X])²
        variancia = esperanca_quadrado - (esperanca ** 2)
        desvio_padrao = math.sqrt(variancia)

        pmf_dict = {int(x): round(p, 4) for x, p in zip(suporte, pmf)}
        cdf_dict = {int(x): round(c, 4) for x, c in zip(suporte, cdf)}

        return {
            "variavel": column,
            "tipo": "Discreta",
            "suporte": suporte,
            "pmf": pmf_dict,
            "cdf": cdf_dict,
            "esperanca_matematica_E_X": round(esperanca, 4),
            "esperanca_segundo_momento_E_X2": round(esperanca_quadrado, 4),
            "variancia_Var_X": round(variancia, 4),
            "desvio_padrao_sigma": round(desvio_padrao, 4)
        }

    def continuous_random_variable_analysis(self, column: str = "posse_mandante") -> Dict[str, Any]:
        """
        Analisa uma Variável Aleatória Contínua Y:
        - Suporte Contínuo Y ∈ [a, b]
        - Função de Densidade de Probabilidade (PDF) f(y) estimada por Kernel Density Estimation (KDE)
        - Função de Distribuição Acumulada Empírica (ECDF)
        - Cálculo de Probabilidade por Integração em Intervalo: P(y1 <= Y <= y2) = ∫ f(y) dy
        """
        series = self.df[column].dropna().astype(float)
        n = len(series)

        media = float(series.mean())
        desvio = float(series.std())

        # Ajuste paramétrico de distribuição Normal contínua de referência
        # P(a <= Y <= b) via CDF Normal padronizada
        def prob_intervalo_normal(a: float, b: float) -> float:
            cdf_b = stats.norm.cdf(b, loc=media, scale=desvio)
            cdf_a = stats.norm.cdf(a, loc=media, scale=desvio)
            return float(cdf_b - cdf_a)

        # Exemplo prático de interesse no futebol:
        # Probabilidade da posse de bola do mandante estar no intervalo de dominância [50%, 65%]
        p_dominio = prob_intervalo_normal(50.0, 65.0)
        p_superior_60 = 1.0 - stats.norm.cdf(60.0, loc=media, scale=desvio)

        # Cálculo de percentis contínuos
        p25 = float(np.percentile(series, 25))
        p50 = float(np.percentile(series, 50))
        p75 = float(np.percentile(series, 75))
        p90 = float(np.percentile(series, 90))

        return {
            "variavel": column,
            "tipo": "Contínua",
            "amostras_validas": n,
            "media_mu": round(media, 3),
            "desvio_padrao_sigma": round(desvio, 3),
            "intervalo_exemplo_50_a_65_pct": round(p_dominio, 4),
            "prob_posse_maior_60_pct": round(float(p_superior_60), 4),
            "percentis": {
                "P25": round(p25, 2),
                "P50_mediana": round(p50, 2),
                "P75": round(p75, 2),
                "P90": round(p90, 2)
            }
        }

    # =========================================================================
    # TÓPICO 4: VARIÁVEIS ALEATÓRIAS BIDIMENSIONAIS (CONJUNTAS)
    # =========================================================================
    def bivariate_random_variables(self, var_x: str = "chutes_mandante", var_y: str = "gols_mandante") -> Dict[str, Any]:
        """
        Analisa o vetor aleatório bidimensional (X, Y):
        - Função de Probabilidade Conjunta: P(X = x, Y = y)
        - Funções de Probabilidade Marginais: P(X = x) = ∑_y P(X=x, Y=y)
                                             P(Y = y) = ∑_x P(X=x, Y=y)
        - Distribuição Condicional: P(Y = y | X = x) = P(X=x, Y=y) / P(X=x)
        - Covariância Populacional/Amostral:
          Cov(X, Y) = E[XY] - E[X]*E[Y] = (1 / (n-1)) * ∑ (x_i - x̄)(y_i - ȳ)
        - Coeficiente de Correlação Linear de Pearson:
          ρ(X, Y) = Cov(X, Y) / (σ_X * σ_Y)  [-1, +1]
        - Coeficiente de Correlação de Spearman (não-linear monotônico).
        """
        df_sub = self.df[[var_x, var_y]].dropna()
        n = len(df_sub)

        x = df_sub[var_x].values
        y = df_sub[var_y].values

        mean_x = float(np.mean(x))
        mean_y = float(np.mean(y))
        var_x_val = float(np.var(x, ddof=1))
        var_y_val = float(np.var(y, ddof=1))
        std_x = math.sqrt(var_x_val)
        std_y = math.sqrt(var_y_val)

        # Covariância
        cov_xy = float(np.cov(x, y, ddof=1)[0, 1])

        # Correlação de Pearson
        pearson_corr, p_value_pearson = stats.pearsonr(x, y)
        # Correlação de Spearman
        spearman_corr, p_value_spearman = stats.spearmanr(x, y)

        # Construção da tabela de contingência conjunta empírica P(X, Y)
        # Para variáveis de suporte amplo, agrupamos X em classes (faixas) se necessário
        if len(np.unique(x)) > 8:
            # Agrupa x em faixas de quartis para exibir uma matriz legível
            x_binned = pd.qcut(x, q=4, duplicates="drop")
            crosstab_counts = pd.crosstab(x_binned, y)
        else:
            crosstab_counts = pd.crosstab(x, y)

        tabela_conjunta_prob = (crosstab_counts / n).round(4)
        marginal_x = (tabela_conjunta_prob.sum(axis=1)).to_dict()
        marginal_y = (tabela_conjunta_prob.sum(axis=0)).to_dict()

        return {
            "var_x": var_x,
            "var_y": var_y,
            "n": n,
            "media_X": round(mean_x, 3),
            "media_Y": round(mean_y, 3),
            "desvio_X": round(std_x, 3),
            "desvio_Y": round(std_y, 3),
            "covariancia_Cov_XY": round(cov_xy, 3),
            "correlacao_pearson_rho": round(float(pearson_corr), 4),
            "p_valor_pearson": round(float(p_value_pearson), 6),
            "correlacao_spearman": round(float(spearman_corr), 4),
            "interpretacao_correlacao": (
                "Forte correlação positiva" if pearson_corr > 0.6 else
                "Moderada correlação positiva" if pearson_corr > 0.3 else
                "Fraca ou nula correlação" if abs(pearson_corr) <= 0.3 else
                "Correlação negativa"
            ),
            "marginais_X": {str(k): round(v, 4) for k, v in marginal_x.items()},
            "marginais_Y": {str(k): round(v, 4) for k, v in marginal_y.items()}
        }

    # =========================================================================
    # TÓPICO 5: DISTRIBUIÇÕES DE PROBABILIDADE TEÓRICAS (DISCRETAS E CONTÍNUAS)
    # =========================================================================
    def poisson_goals_distribution(self, team: Optional[str] = None, is_home: bool = True) -> Dict[str, Any]:
        """
        Modela a contagem de gols por meio da Distribuição de Poisson:
        P(X = k) = (e^(-λ) * λ^k) / k!

        Na literatura esportiva e estatística, gols em futebol satisfazem os axiomas
        de um Processo de Poisson homogêneo (independência e homogeneidade temporal).
        Calcula probabilidades para:
        - 1º Tempo (λ_1T)
        - 2º Tempo (λ_2T)
        - Jogo Completo (λ_Total = λ_1T + λ_2T, pela propriedade reprodutiva de Poisson!)
        """
        if team:
            if is_home:
                df_team = self.df[self.df["mandante"] == team]
                col_1t = "gols_1t_mandante"
                col_2t = "gols_2t_mandante"
                col_tot = "gols_mandante"
            else:
                df_team = self.df[self.df["visitante"] == team]
                col_1t = "gols_1t_visitante"
                col_2t = "gols_2t_visitante"
                col_tot = "gols_visitante"
        else:
            df_team = self.df
            col_1t = "gols_1t_mandante" if is_home else "gols_1t_visitante"
            col_2t = "gols_2t_mandante" if is_home else "gols_2t_visitante"
            col_tot = "gols_mandante" if is_home else "gols_visitante"

        lambda_1t = float(df_team[col_1t].mean())
        lambda_2t = float(df_team[col_2t].mean())
        lambda_tot = float(df_team[col_tot].mean())

        # Função de Poisson para k = 0 até 6 gols
        k_values = list(range(0, 7))
        pmf_1t = {k: round(float(stats.poisson.pmf(k, lambda_1t)), 4) for k in k_values}
        pmf_2t = {k: round(float(stats.poisson.pmf(k, lambda_2t)), 4) for k in k_values}
        pmf_tot = {k: round(float(stats.poisson.pmf(k, lambda_tot)), 4) for k in k_values}

        # Probabilidade de marcar pelo menos 1 gol (P(X >= 1) = 1 - P(X = 0))
        p_pelo_menos_um_1t = 1.0 - stats.poisson.pmf(0, lambda_1t)
        p_pelo_menos_um_2t = 1.0 - stats.poisson.pmf(0, lambda_2t)
        p_pelo_menos_um_tot = 1.0 - stats.poisson.pmf(0, lambda_tot)

        return {
            "time": team or "Média Geral da Liga",
            "condicao": "Mandante" if is_home else "Visitante",
            "lambda_1t": round(lambda_1t, 3),
            "lambda_2t": round(lambda_2t, 3),
            "lambda_total": round(lambda_tot, 3),
            "propriedade_reprodutiva_soma_lambdas": round(lambda_1t + lambda_2t, 3),
            "pmf_poisson_1t": pmf_1t,
            "pmf_poisson_2t": pmf_2t,
            "pmf_poisson_total": pmf_tot,
            "prob_marcar_pelo_menos_um_1t": round(float(p_pelo_menos_um_1t), 4),
            "prob_marcar_pelo_menos_um_2t": round(float(p_pelo_menos_um_2t), 4),
            "prob_marcar_pelo_menos_um_total": round(float(p_pelo_menos_um_tot), 4)
        }

    def simulate_match_poisson_bivariate(self, home_team: str, away_team: str, max_goals: int = 5) -> Dict[str, Any]:
        """
        Modela o confronto direto entre Mandante e Visitante usando Distribuição Bivariada de Poisson.
        Sob hipótese de independência condicional de Poisson:
        P(Mandante = i, Visitante = j) = P(Mandante = i) * P(Visitante = j)

        Gera:
        - Matriz de probabilidades de todos os placares (0x0 até 5x5).
        - Probabilidade de Vitória Mandante (1), Empate (X), Vitória Visitante (2).
        - Probabilidade de Over/Under 2.5 gols.
        - Probabilidade de Ambos Marcam (BTTS - Both Teams To Score).
        """
        # Estima taxas lambda de cada equipe
        df_h = self.df[self.df["mandante"] == home_team]
        df_a = self.df[self.df["visitante"] == away_team]

        # Se não houver amostra suficiente para o time específico, usa a média geral ponderada
        lambda_h = float(df_h["gols_mandante"].mean()) if len(df_h) > 0 else float(self.df["gols_mandante"].mean())
        lambda_a = float(df_a["gols_visitante"].mean()) if len(df_a) > 0 else float(self.df["gols_visitante"].mean())

        score_matrix = np.zeros((max_goals + 1, max_goals + 1))
        prob_home_win = 0.0
        prob_draw = 0.0
        prob_away_win = 0.0
        prob_over_25 = 0.0
        prob_btts = 0.0

        for i in range(max_goals + 1):
            p_i = stats.poisson.pmf(i, lambda_h)
            for j in range(max_goals + 1):
                p_j = stats.poisson.pmf(j, lambda_a)
                p_ij = p_i * p_j
                score_matrix[i, j] = p_ij

                # Acumula probabilidades dos cenários
                if i > j:
                    prob_home_win += p_ij
                elif i == j:
                    prob_draw += p_ij
                else:
                    prob_away_win += p_ij

                if (i + j) > 2.5:
                    prob_over_25 += p_ij

                if i > 0 and j > 0:
                    prob_btts += p_ij

        # Placar mais provável (Moda da distribuição bivariada)
        idx_max = np.unravel_index(np.argmax(score_matrix, axis=None), score_matrix.shape)
        placar_mais_provavel = f"{idx_max[0]} x {idx_max[1]}"
        prob_placar_mais_provavel = float(score_matrix[idx_max])

        return {
            "mandante": home_team,
            "visitante": away_team,
            "lambda_mandante": float(round(lambda_h, 3)),
            "lambda_visitante": float(round(lambda_a, 3)),
            "prob_vitoria_mandante": float(round(prob_home_win, 4)),
            "prob_empate": float(round(prob_draw, 4)),
            "prob_vitoria_visitante": float(round(prob_away_win, 4)),
            "prob_over_2_5_gols": float(round(prob_over_25, 4)),
            "prob_under_2_5_gols": float(round(1.0 - prob_over_25, 4)),
            "prob_ambos_marcam_sim": float(round(prob_btts, 4)),
            "prob_ambos_marcam_nao": float(round(1.0 - prob_btts, 4)),
            "placar_mais_provavel": placar_mais_provavel,
            "prob_placar_mais_provavel": float(round(prob_placar_mais_provavel, 4)),
            "matriz_placares": score_matrix.tolist()
        }

    def binomial_shot_conversion(self, n_chutes: int, p_conversao: Optional[float] = None) -> Dict[str, Any]:
        """
        Modela a taxa de conversão de chutes em gols como uma Distribuição Binomial:
        X ~ Bin(n, p)
        P(X = k) = (n escolhe k) * p^k * (1 - p)^(n - k)

        Aplicação: Se um time ou jogador realiza 'n' finalizações e possui probabilidade 'p'
        de converter cada chute, qual a probabilidade de converter 'k' gols?
        """
        if p_conversao is None:
            # Estimação empírica da taxa média de conversão da liga: Gols / Chutes
            total_gols = self.df["total_gols"].sum()
            total_chutes = (self.df["chutes_mandante"] + self.df["chutes_visitante"]).sum()
            p_conversao = float(total_gols / total_chutes) if total_chutes > 0 else 0.10

        suporte_k = list(range(0, n_chutes + 1))
        pmf_bin = {k: round(float(stats.binom.pmf(k, n_chutes, p_conversao)), 4) for k in suporte_k}
        
        esperanca = n_chutes * p_conversao
        variancia = n_chutes * p_conversao * (1 - p_conversao)

        return {
            "distribuicao": "Binomial",
            "n_tentativas_chutes": n_chutes,
            "p_sucesso_conversao": round(p_conversao, 4),
            "pmf": pmf_bin,
            "esperanca_E_X": round(esperanca, 3),
            "variancia_Var_X": round(variancia, 3),
            "desvio_padrao": round(math.sqrt(variancia), 3),
            "prob_marcar_ao_menos_1_gol": round(1.0 - stats.binom.pmf(0, n_chutes, p_conversao), 4)
        }

    def exponential_first_goal_distribution(self) -> Dict[str, Any]:
        """
        Modela o tempo até o primeiro gol através da Distribuição Exponencial Contínua:
        T ~ Exp(λ)
        PDF: f(t) = λ * e^(-λt)  para t >= 0
        CDF: F(t) = P(T <= t) = 1 - e^(-λt)

        A distribuição exponencial é a distribuição de tempo contínuo contínua
        canônica associada a um processo de Poisson ('Tempo entre chegadas').
        """
        minutos = self.df["minuto_primeiro_gol"].dropna().astype(float)
        n = len(minutos)

        # Na distribuição Exponencial: E[T] = 1 / λ  =>  λ_estimado = 1 / tempo_medio
        tempo_medio = float(minutos.mean())
        lambda_rate = 1.0 / tempo_medio if tempo_medio > 0 else 0.03

        # Probabilidade de sair gol nos intervalos canônicos do futebol:
        # P(T <= 15 min): Gol no início
        # P(T <= 45 min): Gol no 1º tempo
        # P(T > 45 min): Sem gols no 1º tempo
        p_ate_15 = 1.0 - math.exp(-lambda_rate * 15)
        p_ate_30 = 1.0 - math.exp(-lambda_rate * 30)
        p_ate_45 = 1.0 - math.exp(-lambda_rate * 45)
        p_ate_90 = 1.0 - math.exp(-lambda_rate * 90)

        return {
            "distribuicao": "Exponencial",
            "jogos_com_gol_analisados": n,
            "tempo_medio_primeiro_gol_min": round(tempo_medio, 2),
            "parametro_taxa_lambda": round(lambda_rate, 5),
            "prob_primeiro_gol_ate_15_min": round(p_ate_15, 4),
            "prob_primeiro_gol_ate_30_min": round(p_ate_30, 4),
            "prob_primeiro_gol_ate_45_min_1T": round(p_ate_45, 4),
            "prob_primeiro_gol_apenas_no_2T": round(1.0 - p_ate_45, 4),
            "propriedade_perda_de_memoria": "P(T > s + t | T > s) = P(T > t)"
        }

    def normal_distribution_passes_possession(self) -> Dict[str, Any]:
        """
        Modela variáveis contínuas através da Distribuição Normal / Gaussiana:
        X ~ N(μ, σ²)
        PDF: f(x) = (1 / (σ * √(2π))) * e^(-(x-μ)² / (2σ²))

        Pelo Teorema Central do Limite (TCL), grandezas que resultam do acúmulo de muitas
        ações individuais (como passes e posse de bola ao longo de 90 minutos)
        aproximam-se de uma Distribuição Normal.
        """
        posse = self.df["posse_mandante"].dropna()
        mu = float(posse.mean())
        sigma = float(posse.std())

        # Teste de Normalidade de Shapiro-Wilk ou D'Agostino-Pearson
        stat, p_val = stats.normaltest(posse)

        # Regra Empírica dos 3-Sigmas (68-95-99.7)
        pct_1_sigma = float(((posse >= mu - sigma) & (posse <= mu + sigma)).mean() * 100)
        pct_2_sigma = float(((posse >= mu - 2*sigma) & (posse <= mu + 2*sigma)).mean() * 100)
        pct_3_sigma = float(((posse >= mu - 3*sigma) & (posse <= mu + 3*sigma)).mean() * 100)

        return {
            "distribuicao": "Normal (Gaussiana)",
            "variavel": "Posse de Bola do Mandante (%)",
            "media_mu": round(mu, 2),
            "desvio_padrao_sigma": round(sigma, 2),
            "teste_normalidade_p_valor": round(float(p_val), 4),
            "aderencia_regra_empirica": {
                "intervalo_1_sigma_teorico_68pct": round(pct_1_sigma, 2),
                "intervalo_2_sigma_teorico_95pct": round(pct_2_sigma, 2),
                "intervalo_3_sigma_teorico_99_7pct": round(pct_3_sigma, 2)
            }
        }

    # =========================================================================
    # TÓPICO 6: LEIS DOS GRANDES NÚMEROS E FUNÇÃO GERATRIZ DE MOMENTOS (FGM)
    # =========================================================================
    def law_of_large_numbers_simulation(self, column: str = "total_gols", max_steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Demonstração da Lei Fraca dos Grandes Números (LLN):
        Seja X_1, X_2, ..., X_n uma sequência de variáveis aleatórias i.i.d. com E[X_i] = μ.
        Então, para todo ε > 0:
        lim_{n -> ∞} P(|X̄_n - μ| >= ε) = 0

        Ou seja, a média amostral acumulada X̄_n = (1/n) * ∑_{i=1}^n X_i converge quase certamente
        para a esperança matemática da população μ conforme n aumenta.
        """
        series = self.df[column].dropna().values
        if max_steps:
            series = series[:max_steps]
        
        n_total = len(series)
        media_esperada_populacional = float(np.mean(series))

        # Cálculo da média amostral cumulativa passo a passo
        somas_cumulativas = np.cumsum(series)
        indices_n = np.arange(1, n_total + 1)
        medias_cumulativas = somas_cumulativas / indices_n

        # Desvio absoluto acumulado |X̄_n - μ|
        desvios_absolutos = np.abs(medias_cumulativas - media_esperada_populacional)

        return {
            "variavel": column,
            "n_amostras": n_total,
            "esperanca_teorica_mu": round(media_esperada_populacional, 4),
            "primeiras_10_medias_cumulativas": [round(float(m), 4) for m in medias_cumulativas[:10]],
            "ultimas_10_medias_cumulativas": [round(float(m), 4) for m in medias_cumulativas[-10:]],
            "desvio_final_amostral": round(float(desvios_absolutos[-1]), 5),
            "medias_cumulativas_array": medias_cumulativas.tolist(),
            "conclusao": "A convergência estocástica confirma a Lei dos Grandes Números (LLN)."
        }

    def moment_generating_function_poisson(self, column: str = "total_gols", t_values: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Função Geratriz de Momentos (FGM) / Moment-Generating Function (MGF):
        Definição: M_X(t) = E[e^(tX)]

        Para a Distribuição de Poisson com parâmetro λ:
        M_X(t) = ∑_{k=0}^∞ e^(tk) * [e^(-λ) * λ^k / k!]
               = e^(-λ) * ∑_{k=0}^∞ (λ * e^t)^k / k!
               = e^(-λ) * e^(λ * e^t)
               = exp(λ * (e^t - 1))

        Propriedade Fundamental para Obtenção dos Momentos via Derivação:
        - Primeiro Momento (Esperança E[X]):
          M'_X(0) = d/dt [exp(λ(e^t - 1))]|_{t=0} = λ * e^t * exp(λ(e^t - 1))|_{t=0} = λ
        - Segundo Momento (E[X²]):
          M''_X(0) = d²/dt² [M_X(t)]|_{t=0} = (λ * e^t + λ² * e^(2t)) * exp(λ(e^t - 1))|_{t=0} = λ + λ²
        - Variância:
          Var(X) = E[X²] - (E[X])² = (λ + λ²) - λ² = λ!

        O método calcula:
        1. Os momentos teóricos derivados da FGM analítica.
        2. A FGM empírica calculada diretamente dos dados reais: M̂_n(t) = (1/n) * ∑ e^(t * x_i).
        3. A comparação exata entre teoria e empirismo.
        """
        series = self.df[column].dropna().values
        n = len(series)
        lambda_empirico = float(np.mean(series))

        # Obtenção dos momentos pela FGM Teórica
        primeiro_momento_fgm = lambda_empirico # M'(0) = λ
        segundo_momento_fgm = lambda_empirico + (lambda_empirico ** 2) # M''(0) = λ + λ²
        variancia_fgm = segundo_momento_fgm - (primeiro_momento_fgm ** 2) # Var = λ

        # Momentos amostrais diretos dos dados
        media_direta = float(np.mean(series))
        segundo_momento_direto = float(np.mean(series ** 2))
        variancia_direta = float(np.var(series, ddof=0))

        # Comparação da FGM Teórica vs FGM Empírica em uma grade de valores de t
        if t_values is None:
            t_values = [-0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5]

        fgm_teorica_dict = {}
        fgm_empirica_dict = {}

        for t in t_values:
            # FGM Teórica de Poisson: exp(λ * (e^t - 1))
            m_teorico = math.exp(lambda_empirico * (math.exp(t) - 1.0))
            # FGM Empírica: (1/n) * ∑ e^(t * x_i)
            m_empirico = float(np.mean(np.exp(t * series)))

            fgm_teorica_dict[t] = round(m_teorico, 4)
            fgm_empirica_dict[t] = round(m_empirico, 4)

        return {
            "distribuicao": "Poisson",
            "variavel": column,
            "lambda_estimado": round(lambda_empirico, 4),
            "formula_fgm_poisson": "M_X(t) = exp(lambda * (e^t - 1))",
            "derivadas_no_ponto_zero": {
                "primeiro_momento_M_prime_0 (E[X])": round(primeiro_momento_fgm, 4),
                "segundo_momento_M_double_prime_0 (E[X^2])": round(segundo_momento_fgm, 4),
                "variancia_via_FGM (M''(0) - [M'(0)]^2)": round(variancia_fgm, 4)
            },
            "comparacao_dados_amostrais": {
                "media_amostral": round(media_direta, 4),
                "segundo_momento_amostral": round(segundo_momento_direto, 4),
                "variancia_amostral": round(variancia_direta, 4)
            },
            "tabela_fgm_teorica_vs_empirica": {
                str(t): {"Teorica_M_t": fgm_teorica_dict[t], "Empirica_Mn_t": fgm_empirica_dict[t]}
                for t in t_values
            }
        }

    # =========================================================================
    # SEÇÃO COMPLEMENTAR: MODELAGEM ESTATÍSTICA E PROBABILÍSTICA POR JOGADOR
    # =========================================================================
    def get_teams_list(self) -> List[str]:
        """Retorna a lista ordenada de todos os clubes cadastrados."""
        if self.df_players is not None:
            return sorted(list(self.df_players["time"].unique()))
        return sorted(list(self.df["mandante"].unique()))

    def get_players_by_team(self, team_name: str) -> List[str]:
        """Retorna os jogadores disponíveis para um clube específico."""
        if self.df_players is None:
            return []
        sub = self.df_players[self.df_players["time"] == team_name]
        return sorted(list(sub["jogador"].unique()))

    def player_exploratory_analysis(self, player_name: str, metric: str = "chutes") -> Dict[str, Any]:
        """
        Calcula as estatísticas descritivas (AED) da performance jogo a jogo de um jogador.
        """
        if self.df_logs is None:
            return {"erro": "Base de dados jogo a jogo dos jogadores não informada"}

        logs = self.df_logs[self.df_logs["jogador"] == player_name]
        if len(logs) == 0:
            return {"erro": f"Jogador '{player_name}' não encontrado"}

        series = logs[metric].dropna().astype(float)
        n = len(series)
        media = float(series.mean())
        mediana = float(series.median())
        desvio = float(series.std(ddof=1)) if n > 1 else 0.0
        var_amostral = float(series.var(ddof=1)) if n > 1 else 0.0
        cv = (desvio / media * 100) if media > 0 else 0.0

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))

        assimetria = float(stats.skew(series)) if n > 2 else 0.0
        curtose = float(stats.kurtosis(series)) if n > 3 else 0.0

        return {
            "jogador": player_name,
            "metrica": metric,
            "jogos_amostrados": n,
            "media_por_jogo": round(media, 3),
            "mediana": round(mediana, 3),
            "desvio_padrao": round(desvio, 3),
            "variancia": round(var_amostral, 3),
            "coeficiente_variacao_pct": round(cv, 2),
            "q1": round(q1, 3),
            "q3": round(q3, 3),
            "iqr": round(q3 - q1, 3),
            "assimetria": round(assimetria, 3),
            "curtose": round(curtose, 3),
            "minimo": float(series.min()),
            "maximo": float(series.max())
        }

    def player_goal_probabilities_poisson(self, player_name: str) -> Dict[str, Any]:
        """
        Calcula as probabilidades de gols e cartões do jogador para a próxima partida
        usando Distribuição de Poisson: P(X = k) = (e^-λ * λ^k) / k!
        """
        if self.df_players is None or self.df_logs is None:
            return {"erro": "Dados de jogadores não carregados"}

        player_row = self.df_players[self.df_players["jogador"] == player_name]
        if len(player_row) == 0:
            return {"erro": f"Jogador '{player_name}' não encontrado"}

        row = player_row.iloc[0]
        logs = self.df_logs[self.df_logs["jogador"] == player_name]

        lambda_gols = float(logs["gols"].mean()) if len(logs) > 0 else float(row["lambda_gols"])
        lambda_gols_1t = float(logs["gols_1t"].mean()) if len(logs) > 0 else lambda_gols * 0.45
        lambda_gols_2t = float(logs["gols_2t"].mean()) if len(logs) > 0 else lambda_gols * 0.55
        lambda_amarelo = float(logs["amarelos"].mean()) if len(logs) > 0 else 0.15

        # PMF de gols no jogo (k = 0, 1, 2, 3+)
        pmf_gols = {k: round(float(stats.poisson.pmf(k, lambda_gols)), 4) for k in range(4)}
        prob_marcar_ao_menos_1 = 1.0 - stats.poisson.pmf(0, lambda_gols)
        prob_marcar_1t = 1.0 - stats.poisson.pmf(0, lambda_gols_1t)
        prob_marcar_2t = 1.0 - stats.poisson.pmf(0, lambda_gols_2t)
        prob_amarelo = 1.0 - stats.poisson.pmf(0, lambda_amarelo)

        return {
            "jogador": player_name,
            "time": row["time"],
            "posicao": row["posicao"],
            "idade": int(row["idade"]),
            "lambda_gols_jogo": round(lambda_gols, 3),
            "lambda_gols_1t": round(lambda_gols_1t, 3),
            "lambda_gols_2t": round(lambda_gols_2t, 3),
            "lambda_amarelo": round(lambda_amarelo, 3),
            "prob_marcar_ao_menos_1_gol": round(float(prob_marcar_ao_menos_1), 4),
            "prob_marcar_no_1t": round(float(prob_marcar_1t), 4),
            "prob_marcar_no_2t": round(float(prob_marcar_2t), 4),
            "prob_levar_amarelo": round(float(prob_amarelo), 4),
            "pmf_gols_jogo": pmf_gols
        }

    def player_bayes_impact(self, player_name: str) -> Dict[str, Any]:
        """
        Aplica o Teorema de Bayes para calcular o impacto do jogador no resultado da equipe:
        - P(Vitória do time | Jogador marcou)
        - P(Vitória do time | Jogador não marcou)
        - P(Jogador marcou | Time venceu) via Bayes
        """
        if self.df_logs is None:
            return {"erro": "Base de dados jogo a jogo não informada"}

        logs = self.df_logs[self.df_logs["jogador"] == player_name]
        if len(logs) == 0:
            return {"erro": f"Jogador '{player_name}' não encontrado"}

        total_jogos = len(logs)
        jogos_marcou = logs[logs["gols"] > 0]
        jogos_nao_marcou = logs[logs["gols"] == 0]
        jogos_vitoria = logs[logs["vitoria_time"] == 1]

        p_marcou = len(jogos_marcou) / total_jogos if total_jogos > 0 else 0.0
        p_vitoria_geral = len(jogos_vitoria) / total_jogos if total_jogos > 0 else 0.0

        p_vit_dado_marcou = float(jogos_marcou["vitoria_time"].mean()) if len(jogos_marcou) > 0 else 0.70
        p_vit_dado_nao_marcou = float(jogos_nao_marcou["vitoria_time"].mean()) if len(jogos_nao_marcou) > 0 else 0.40

        # Bayes: P(Marcou | Vitória) = [P(Vitória | Marcou) * P(Marcou)] / P(Vitória)
        p_bayes_marcou_dado_vit = (p_vit_dado_marcou * p_marcou) / p_vitoria_geral if p_vitoria_geral > 0 else 0.0

        return {
            "jogador": player_name,
            "total_jogos_analisados": total_jogos,
            "prob_jogador_marcar": round(p_marcou, 4),
            "prob_vitoria_geral_time": round(p_vitoria_geral, 4),
            "prob_vitoria_dado_jogador_marcou": round(p_vit_dado_marcou, 4),
            "prob_vitoria_dado_jogador_nao_marcou": round(p_vit_dado_nao_marcou, 4),
            "impacto_positivo_vitoria_pct": round((p_vit_dado_marcou - p_vit_dado_nao_marcou) * 100, 2),
            "teorema_bayes_p_marcou_dado_vitoria": round(p_bayes_marcou_dado_vit, 4)
        }
