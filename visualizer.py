"""
MÓDULO: visualizer.py
OBJETIVO: Geração de gráficos estatísticos e probabilísticos profissionais
para a disciplina de Probabilidade e Estatística.
Suporta exportação para PNG (pasta graficos/) e retorno de figuras matplotlib para o Streamlit.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Optional, Tuple, Dict, Any

GRAFICOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graficos")
os.makedirs(GRAFICOS_DIR, exist_ok=True)

# Configuração de estilo visual limpo e acadêmico
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

class FootballStatsVisualizer:
    """
    Gerador de gráficos temáticos para cada conteúdo da disciplina.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    # =========================================================================
    # 1. GRÁFICO AED: Histograma com Densidade e Boxplot
    # =========================================================================
    def plot_aed_distribution(self, column: str = "total_gols", save_filename: Optional[str] = "1_aed_distribuicao.png") -> plt.Figure:
        series = self.df[column].dropna()
        media = series.mean()
        mediana = series.median()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        fig, (ax_box, ax_hist) = plt.subplots(
            2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [0.25, 0.75]}
        )

        sns.boxplot(x=series, ax=ax_box, color="#4A90E2", flierprops={"marker": "o", "color": "red", "markersize": 6})
        ax_box.axvline(media, color="red", linestyle="--", linewidth=1.5, label=f"Média: {media:.2f}")
        ax_box.axvline(mediana, color="green", linestyle="-", linewidth=1.5, label=f"Mediana: {mediana:.2f}")
        ax_box.set(xlabel="")
        ax_box.legend(loc="upper right", fontsize=9)
        ax_box.set_title(f"Análise Exploratória de Dados: {column}", fontsize=12, fontweight="bold")

        bins = np.arange(series.min() - 0.5, series.max() + 1.5, 1) if series.dtype in [int, "int64"] else 20
        sns.histplot(series, bins=bins, kde=True, ax=ax_hist, color="#34495E", stat="density", alpha=0.6)
        ax_hist.axvline(media, color="red", linestyle="--", linewidth=1.8, label=f"Média = {media:.2f}")
        ax_hist.axvline(mediana, color="green", linestyle="-", linewidth=1.8, label=f"Mediana = {mediana:.2f}")
        ax_hist.axvline(q1, color="orange", linestyle=":", label=f"Q1 = {q1:.2f}")
        ax_hist.axvline(q3, color="orange", linestyle=":", label=f"Q3 = {q3:.2f}")

        ax_hist.set_xlabel(column.replace("_", " ").title(), fontsize=11)
        ax_hist.set_ylabel("Densidade Amostral", fontsize=11)
        ax_hist.legend(loc="upper right", fontsize=9)

        plt.tight_layout()
        if save_filename:
            fig.savefig(os.path.join(GRAFICOS_DIR, save_filename), dpi=150)
        return fig

    # =========================================================================
    # 2. GRÁFICO PROBABILIDADES CONDICIONAIS E BAYES
    # =========================================================================
    def plot_conditional_and_bayes(self, save_filename: Optional[str] = "2_teoria_probabilidades.png") -> plt.Figure:
        vitoria_casa = self.df["gols_mandante"] > self.df["gols_visitante"]
        gol1t = self.df["gols_1t_mandante"] > 0

        p_vit = vitoria_casa.mean()
        p_gol1t = gol1t.mean()
        p_vit_dado_gol1t = (vitoria_casa & gol1t).sum() / gol1t.sum()
        p_gol1t_dado_vit = (vitoria_casa & gol1t).sum() / vitoria_casa.sum()

        categorias = [
            "P(Vitória Mandante)\n[Marginal]",
            "P(Gol 1T Mandante)\n[Marginal]",
            "P(Vitória | Gol 1T)\n[Condicional]",
            "P(Gol 1T | Vitória)\n[Teorema de Bayes]"
        ]
        valores = [p_vit, p_gol1t, p_vit_dado_gol1t, p_gol1t_dado_vit]
        cores = ["#7F8C8D", "#95A5A6", "#2980B9", "#27AE60"]

        fig, ax = plt.subplots(figsize=(8, 5))
        barras = ax.bar(categorias, valores, color=cores, width=0.55, edgecolor="black")

        for b, v in zip(barras, valores):
            ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v*100:.1f}%", ha="center", fontsize=11, fontweight="bold")

        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Probabilidade P(E)", fontsize=11)
        ax.set_title("Teoria das Probabilidades: Marginais, Condicionais e Bayes", fontsize=12, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        if save_filename:
            fig.savefig(os.path.join(GRAFICOS_DIR, save_filename), dpi=150)
        return fig

    # =========================================================================
    # 3. GRÁFICO V.A. DISCRETAS (PMF e CDF de Gols / Cartões)
    # =========================================================================
    def plot_discrete_random_variable(self, column: str = "total_gols", save_filename: Optional[str] = "3_va_discreta_pmf_cdf.png") -> plt.Figure:
        series = self.df[column].dropna().astype(int)
        counts = series.value_counts(normalize=True).sort_index()
        x = counts.index.values
        pmf = counts.values
        cdf = np.cumsum(pmf)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

        ax1.stem(x, pmf, linefmt="b-", markerfmt="bo", basefmt="k-")
        ax1.set_title(f"Função de Massa de Probabilidade (PMF)\nP(X = k) para {column}", fontsize=11, fontweight="bold")
        ax1.set_xlabel("k (Contagem de Eventos)", fontsize=10)
        ax1.set_ylabel("P(X = k)", fontsize=10)
        ax1.set_xticks(x)
        for xi, pi in zip(x, pmf):
            ax1.text(xi, pi + 0.01, f"{pi:.2f}", ha="center", fontsize=8)

        ax2.step(np.append(x, x[-1]+1), np.append([0], cdf), where="pre", color="#C0392B", linewidth=2.0)
        ax2.plot(x, cdf, "ro", markersize=5)
        ax2.set_title("Distribuição Acumulada (CDF)\nF(x) = P(X <= x)", fontsize=11, fontweight="bold")
        ax2.set_xlabel("x", fontsize=10)
        ax2.set_ylabel("F(x)", fontsize=10)
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_xticks(x)

        plt.tight_layout()
        if save_filename:
            fig.savefig(os.path.join(GRAFICOS_DIR, save_filename), dpi=150)
        return fig

    # =========================================================================
    # 4. GRÁFICO V.A. CONTÍNUAS: Ajustes Exponencial e Normal
    # =========================================================================
    def plot_continuous_distributions(self, save_filename: Optional[str] = "4_distribuicoes_continuas.png") -> plt.Figure:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

        minutos = self.df["minuto_primeiro_gol"].dropna().values
        lambda_exp = 1.0 / np.mean(minutos)
        t_grid = np.linspace(0, 90, 200)
        pdf_exp = lambda_exp * np.exp(-lambda_exp * t_grid)

        sns.histplot(minutos, stat="density", bins=15, ax=ax1, color="#2ECC71", alpha=0.5, label="Empírico (Jogos)")
        ax1.plot(t_grid, pdf_exp, color="#27AE60", linewidth=2.2, label=f"Ajuste Exp(λ={lambda_exp:.4f})")
        ax1.axvline(45, color="red", linestyle="--", label="Fim do 1º Tempo (45 min)")
        ax1.set_title("V.A. Contínua: Tempo até o 1º Gol\nDistribuição Exponencial", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Minutos de Jogo", fontsize=10)
        ax1.set_ylabel("Densidade f(t)", fontsize=10)
        ax1.legend(fontsize=8, loc="upper right")

        posse = self.df["posse_mandante"].dropna().values
        mu = np.mean(posse)
        sigma = np.std(posse)
        p_grid = np.linspace(mu - 3.5*sigma, mu + 3.5*sigma, 200)
        pdf_norm = stats.norm.pdf(p_grid, mu, sigma)

        sns.histplot(posse, stat="density", bins=15, ax=ax2, color="#9B59B6", alpha=0.5, label="Empírico (Mandantes)")
        ax2.plot(p_grid, pdf_norm, color="#8E44AD", linewidth=2.2, label=f"Gaussiana N(μ={mu:.1f}, σ={sigma:.1f})")
        ax2.axvline(mu, color="black", linestyle="--", label=f"Média μ={mu:.1f}%")
        ax2.set_title("V.A. Contínua: Posse de Bola (%)\nDistribuição Normal (Gaussiana)", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Posse de Bola Mandante (%)", fontsize=10)
        ax2.set_ylabel("Densidade f(x)", fontsize=10)
        ax2.legend(fontsize=8, loc="upper right")

        plt.tight_layout()
        if save_filename:
            fig.savefig(os.path.join(GRAFICOS_DIR, save_filename), dpi=150)
        return fig

    # =========================================================================
    # 5. GRÁFICO BIDIMENSIONAL: Dispersão, Regressão e Correlação
    # =========================================================================
    def plot_bivariate_relationship(self, var_x: str = "chutes_mandante", var_y: str = "gols_mandante", save_filename: Optional[str] = "5_va_bidimensional.png") -> plt.Figure:
        x = self.df[var_x].values
        y = self.df[var_y].values
        corr_pearson, _ = stats.pearsonr(x, y)
        cov_val = np.cov(x, y)[0, 1]

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.regplot(x=x, y=y, ax=ax, color="#2980B9", scatter_kws={"alpha": 0.5, "s": 35}, line_kws={"color": "red", "linewidth": 2})

        ax.set_title(f"Variáveis Aleatórias Bidimensionais (X, Y)\nCorrelação de Pearson ρ = {corr_pearson:.3f} | Cov(X,Y) = {cov_val:.2f}", fontsize=12, fontweight="bold")
        ax.set_xlabel(var_x.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel(var_y.replace("_", " ").title(), fontsize=11)

        plt.tight_layout()
        if save_filename:
            fig.savefig(os.path.join(GRAFICOS_DIR, save_filename), dpi=150)
        return fig

    # =========================================================================
    # 6. GRÁFICO POISSON BIVARIADA: Heatmap de Placares Prováveis
    # =========================================================================
    def plot_poisson_score_matrix(self, score_matrix: np.ndarray, home_team: str, away_team: str, save_filename: Optional[str] = "6_poisson_matriz_placares.png") -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7, 6))
        pct_matrix = score_matrix * 100.0

        sns.heatmap(
            pct_matrix,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            cbar_kws={"label": "Probabilidade Conjunta P(H=i, A=j) (%)"},
            ax=ax,
            linewidths=0.5
        )
        ax.set_xlabel(f"Gols Visitante: {away_team}", fontsize=11, fontweight="bold")
        ax.set_ylabel(f"Gols Mandante: {home_team}", fontsize=11, fontweight="bold")
        ax.set_title(f"Distribuição Bivariada de Poisson: {home_team} x {away_team}\nMatriz de Placares Exatos (%)", fontsize=12, fontweight="bold")

        plt.tight_layout()
        if save_filename:
            fig.savefig(os.path.join(GRAFICOS_DIR, save_filename), dpi=150)
        return fig

    # =========================================================================
    # 7. GRÁFICO LEI DOS GRANDES NÚMEROS (LLN)
    # =========================================================================
    def plot_law_of_large_numbers(self, column: str = "total_gols", save_filename: Optional[str] = "7_lei_grandes_numeros.png") -> plt.Figure:
        series = self.df[column].dropna().values
        n = len(series)
        media_pop = np.mean(series)
        medias_cum = np.cumsum(series) / np.arange(1, n + 1)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(np.arange(1, n + 1), medias_cum, color="#2C3E50", linewidth=1.8, label="Média Amostral Acumulada (X̄_n)")
        ax.axhline(media_pop, color="red", linestyle="--", linewidth=2.0, label=f"Esperança Teórica E[X] = {media_pop:.3f}")

        epsilon = 0.15
        ax.fill_between(np.arange(1, n + 1), media_pop - epsilon, media_pop + epsilon, color="red", alpha=0.1, label=f"Vizinhança ε = ±{epsilon}")

        ax.set_title("Lei Fraca dos Grandes Números (LLN)\nConvergência Estocástica da Média de Gols por Partida", fontsize=12, fontweight="bold")
        ax.set_xlabel("Número de Partidas Amostradas (n)", fontsize=11)
        ax.set_ylabel("Média de Gols Acumulada (X̄_n)", fontsize=11)
        ax.legend(loc="lower right", fontsize=10)
        ax.set_ylim(media_pop - 1.2, media_pop + 1.2)

        plt.tight_layout()
        if save_filename:
            fig.savefig(os.path.join(GRAFICOS_DIR, save_filename), dpi=150)
        return fig

    # =========================================================================
    # 8. GRÁFICO FUNÇÃO GERATRIZ DE MOMENTOS (FGM)
    # =========================================================================
    def plot_moment_generating_function(self, column: str = "total_gols", save_filename: Optional[str] = "8_fgm_poisson.png") -> plt.Figure:
        series = self.df[column].dropna().values
        lambda_val = np.mean(series)
        t_vals = np.linspace(-0.6, 0.4, 50)

        fgm_teorica = np.exp(lambda_val * (np.exp(t_vals) - 1.0))
        fgm_empirica = [np.mean(np.exp(t * series)) for t in t_vals]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(t_vals, fgm_teorica, "r-", linewidth=2.5, label=f"FGM Teórica Poisson: exp({lambda_val:.2f}(e^t - 1))")
        ax.plot(t_vals, fgm_empirica, "b--", linewidth=2.0, label="FGM Empírica Amostral: (1/n)∑e^(tX)")
        ax.scatter([0], [1], color="black", zorder=5, s=60, label="M_X(0) = 1 (Axioma)")

        ax.set_title("Função Geratriz de Momentos (FGM)\nM_X(t) = E[e^(tX)] para a Distribuição de Gols", fontsize=12, fontweight="bold")
        ax.set_xlabel("Parâmetro t", fontsize=11)
        ax.set_ylabel("M_X(t)", fontsize=11)
        ax.legend(fontsize=10)

        plt.tight_layout()
        if save_filename:
            fig.savefig(os.path.join(GRAFICOS_DIR, save_filename), dpi=150)
        return fig

    # =========================================================================
    # 9. GRÁFICOS POR JOGADOR: PMF de Poisson e Impacto Decisivo (Bayes)
    # =========================================================================
    def plot_player_poisson_pmf(self, player_name: str, pmf_dict: Dict[int, float], save_filename: Optional[str] = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        k = list(pmf_dict.keys())
        p = [pmf_dict[ki] * 100 for ki in k]

        barras = ax.bar([f"{ki} Gols" for ki in k], p, color="#2980B9", edgecolor="black", width=0.55)
        for b, pi in zip(barras, p):
            ax.text(b.get_x() + b.get_width()/2, pi + 1.0, f"{pi:.1f}%", ha="center", fontsize=10, fontweight="bold")

        ax.set_ylim(0, max(p) + 12)
        ax.set_ylabel("Probabilidade P(X = k) (%)", fontsize=11)
        ax.set_title(f"Distribuição de Poisson para Gols: {player_name}\nProbabilidade de Marcar no Próximo Jogo", fontsize=12, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        if save_filename:
            fig.savefig(os.path.join(GRAFICOS_DIR, save_filename), dpi=150)
        return fig

    def plot_player_impact_bayes(self, player_name: str, bayes_dict: Dict[str, Any], save_filename: Optional[str] = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sobrenome = player_name.split()[-1]
        categorias = [
            f"Vitória com Gol de\n{sobrenome}",
            f"Vitória sem Gol de\n{sobrenome}",
            "Média Geral de\nVitória do Time"
        ]
        valores = [
            bayes_dict["prob_vitoria_dado_jogador_marcou"] * 100,
            bayes_dict["prob_vitoria_dado_jogador_nao_marcou"] * 100,
            bayes_dict["prob_vitoria_geral_time"] * 100
        ]
        cores = ["#27AE60", "#E74C3C", "#95A5A6"]

        barras = ax.bar(categorias, valores, color=cores, edgecolor="black", width=0.5)
        for b, vi in zip(barras, valores):
            ax.text(b.get_x() + b.get_width()/2, vi + 1.5, f"{vi:.1f}%", ha="center", fontsize=10, fontweight="bold")

        ax.set_ylim(0, 100)
        ax.set_ylabel("Probabilidade de Vitória (%)", fontsize=11)
        ax.set_title(f"Teorema de Bayes & Impacto Decisivo: {player_name}\nProbabilidade Condicional de Vitória da Equipe", fontsize=12, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        if save_filename:
            fig.savefig(os.path.join(GRAFICOS_DIR, save_filename), dpi=150)
        return fig
