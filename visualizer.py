"""
MÓDULO: visualizer.py
OBJETIVO: Gráficos estatísticos (representação gráfica dos dados) e probabilísticos.
Cada função devolve uma figura do matplotlib para o Streamlit e pode salvar o PNG em graficos/.

Tipos de gráfico usados na disciplina:
- Colunas/barras e setores (pizza): variáveis qualitativas e discretas.
- Histograma e polígono de frequências: dados agrupados em classes.
- Ogiva: frequência acumulada.
- Gráfico de linha: série ao longo do tempo (rodadas).
- Boxplot: quartis, diferença interquartílica e outliers.
"""

import os
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from scipy import stats

GRAFICOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graficos")
os.makedirs(GRAFICOS_DIR, exist_ok=True)

# Paleta: verde do gramado + amarelo de destaque + cinzas
VERDE = "#0B6E4F"
VERDE_CLARO = "#6FBF9B"
AMARELO = "#F2A900"
AZUL = "#1F5FAD"
VERMELHO = "#C0392B"
CINZA = "#6B7280"
PALETA = [VERDE, AMARELO, AZUL, VERMELHO, VERDE_CLARO, "#8E6C8A", "#3D5A80", "#E07A5F", "#81B29A", "#F4D35E"]

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#cccccc", "axes.titleweight": "bold",
                     "axes.titlesize": 12, "axes.labelsize": 10.5})


def _salvar(fig: plt.Figure, nome: Optional[str]) -> plt.Figure:
    fig.tight_layout()
    if nome:
        fig.savefig(os.path.join(GRAFICOS_DIR, nome), dpi=150, bbox_inches="tight")
    return fig


class FootballStatsVisualizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    # ------------------------------------------------------------------
    # Representação gráfica de distribuições de frequência
    # ------------------------------------------------------------------
    def plot_bar_frequency(self, tabela: pd.DataFrame, titulo: str, xlabel: str,
                           horizontal: bool = False, save: Optional[str] = None) -> plt.Figure:
        """Gráfico de colunas (ou barras) com a frequência absoluta e o percentual em cada barra."""
        fig, ax = plt.subplots(figsize=(9, 4.6))
        rotulos = tabela["Valor"].astype(str).tolist()
        fi = tabela["fi"].values
        cores = [PALETA[i % len(PALETA)] for i in range(len(fi))] if len(fi) <= 6 else VERDE
        if horizontal:
            barras = ax.barh(rotulos[::-1], fi[::-1], color=cores[::-1] if isinstance(cores, list) else cores)
            ax.set_xlabel("Frequência absoluta (fi)")
            ax.set_ylabel(xlabel)
            for b, pct in zip(barras, tabela["fri (%)"].values[::-1]):
                ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {int(b.get_width())} ({pct:.1f}%)", va="center", fontsize=9)
        else:
            barras = ax.bar(rotulos, fi, color=cores, edgecolor="white")
            ax.set_ylabel("Frequência absoluta (fi)")
            ax.set_xlabel(xlabel)
            for b, pct in zip(barras, tabela["fri (%)"].values):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{int(b.get_height())}\n{pct:.1f}%",
                        ha="center", va="bottom", fontsize=8.5)
            ax.set_ylim(0, fi.max() * 1.22)
            if max(len(r) for r in rotulos) > 10:
                plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        ax.set_title(titulo)
        return _salvar(fig, save)

    def plot_pie(self, tabela: pd.DataFrame, titulo: str, save: Optional[str] = None) -> plt.Figure:
        """Gráfico de setores: cada fatia é proporcional à frequência relativa."""
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        ax.pie(tabela["fi"], labels=tabela["Valor"], autopct="%1.1f%%", startangle=90,
               colors=PALETA[:len(tabela)], wedgeprops={"edgecolor": "white"}, textprops={"fontsize": 9})
        ax.set_title(titulo)
        ax.axis("equal")
        return _salvar(fig, save)

    def plot_histogram_classes(self, info: Dict[str, Any], titulo: str, xlabel: str,
                               save: Optional[str] = None) -> plt.Figure:
        """Histograma (barras justapostas por classe) com o polígono de frequências sobre os pontos médios."""
        t, h = info["tabela"], info["h"]
        fig, ax = plt.subplots(figsize=(9, 4.6))
        ax.bar(t["li"], t["fi"], width=h, align="edge", color=VERDE_CLARO, edgecolor=VERDE, label="Histograma (fi)")
        xs = [t["li"].iloc[0] - h / 2] + t["xi (ponto médio)"].tolist() + [t["ls"].iloc[-1] + h / 2]
        ys = [0] + t["fi"].tolist() + [0]
        ax.plot(xs, ys, "-o", color=AMARELO, linewidth=2, markersize=4, label="Polígono de frequências")
        ax.set_xticks(list(t["li"]) + [t["ls"].iloc[-1]])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequência absoluta (fi)")
        ax.set_title(titulo)
        ax.legend()
        return _salvar(fig, save)

    def plot_ogive(self, info: Dict[str, Any], titulo: str, xlabel: str, save: Optional[str] = None) -> plt.Figure:
        """Ogiva: frequência relativa acumulada (Fri) no limite superior de cada classe."""
        t = info["tabela"]
        fig, ax = plt.subplots(figsize=(9, 4.2))
        xs = [t["li"].iloc[0]] + t["ls"].tolist()
        ys = [0] + t["Fri (%)"].tolist()
        ax.plot(xs, ys, "-o", color=AZUL, linewidth=2)
        ax.axhline(50, color=CINZA, linestyle="--", linewidth=1, label="50% (mediana)")
        ax.set_xticks(xs)
        ax.set_ylim(0, 105)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Fri (%)")
        ax.set_title(titulo)
        ax.legend()
        return _salvar(fig, save)

    def plot_goals_per_round(self, save: Optional[str] = None) -> plt.Figure:
        """Gráfico de linha: total de gols em cada uma das 38 rodadas."""
        serie = self.df.groupby("rodada")["total_gols"].sum()
        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.plot(serie.index, serie.values, "-o", color=VERDE, markersize=4)
        ax.axhline(serie.mean(), color=AMARELO, linestyle="--", linewidth=2, label=f"Média = {serie.mean():.1f} gols")
        ax.set_xlabel("Rodada")
        ax.set_ylabel("Gols na rodada")
        ax.set_xticks(range(1, 39, 2))
        ax.set_title("Gols por rodada — Brasileirão 2024")
        ax.legend()
        return _salvar(fig, save)

    # ------------------------------------------------------------------
    # Medidas de posição e dispersão
    # ------------------------------------------------------------------
    def plot_position_measures(self, serie: pd.Series, medidas: Dict[str, Any], titulo: str, xlabel: str,
                               discreta: bool = True, save: Optional[str] = None) -> plt.Figure:
        """Distribuição com as linhas de média, mediana e moda."""
        x = serie.dropna().astype(float)
        fig, ax = plt.subplots(figsize=(9, 4.4))
        if discreta:
            cont = x.value_counts().sort_index()
            ax.bar(cont.index, cont.values, color=VERDE_CLARO, edgecolor=VERDE, width=0.8)
            ax.set_xticks(cont.index)
        else:
            ax.hist(x, bins=18, color=VERDE_CLARO, edgecolor=VERDE)
        ax.axvline(medidas["media"], color=VERMELHO, linewidth=2.2, label=f"Média = {medidas['media']:.2f}")
        ax.axvline(medidas["mediana"], color=AZUL, linewidth=2.2, linestyle="--", label=f"Mediana = {medidas['mediana']:g}")
        for i, m in enumerate(medidas["moda"]):
            ax.axvline(m, color=AMARELO, linewidth=2.2, linestyle=":", label=f"Moda = {m:g}" if i == 0 else None)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequência")
        ax.set_title(titulo)
        ax.legend()
        return _salvar(fig, save)

    def plot_boxplot(self, serie: pd.Series, disp: Dict[str, Any], titulo: str, xlabel: str,
                     save: Optional[str] = None) -> plt.Figure:
        """Boxplot horizontal com Q1, mediana, Q3, limites de Tukey e outliers anotados."""
        x = serie.dropna().astype(float)
        fig, ax = plt.subplots(figsize=(9, 3.4))
        ax.boxplot(x, vert=False, widths=0.5, patch_artist=True,
                   boxprops={"facecolor": VERDE_CLARO, "edgecolor": VERDE},
                   medianprops={"color": VERMELHO, "linewidth": 2.5},
                   flierprops={"marker": "o", "markerfacecolor": AMARELO, "markeredgecolor": "#8a6100", "markersize": 7})
        for nome, valor in (("Q1", disp["q1"]), ("Md", disp["q2"]), ("Q3", disp["q3"])):
            ax.annotate(f"{nome} = {valor:g}", (valor, 1.3), ha="center", fontsize=9, color="#333333")
        ax.axvline(disp["limite_superior"], color=CINZA, linestyle=":", linewidth=1)
        ax.axvline(disp["limite_inferior"], color=CINZA, linestyle=":", linewidth=1)
        ax.set_yticks([])
        ax.set_ylim(0.5, 1.5)
        ax.set_xlabel(xlabel)
        ax.set_title(titulo)
        return _salvar(fig, save)

    def plot_boxplot_teams(self, longo: pd.DataFrame, ordem: List[str], save: Optional[str] = None) -> plt.Figure:
        """Boxplots comparativos: gols marcados por jogo de cada clube."""
        dados = [longo.loc[longo["time"] == t, "gols_marcados"].values for t in ordem]
        fig, ax = plt.subplots(figsize=(11, 5))
        bp = ax.boxplot(dados, tick_labels=ordem, patch_artist=True, showmeans=True,
                        meanprops={"marker": "D", "markerfacecolor": AMARELO, "markeredgecolor": "#8a6100", "markersize": 5},
                        medianprops={"color": VERMELHO, "linewidth": 2})
        for caixa in bp["boxes"]:
            caixa.set(facecolor=VERDE_CLARO, edgecolor=VERDE)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_ylabel("Gols marcados por jogo")
        ax.set_title("Boxplot dos gols marcados por jogo, por clube (◆ = média)")
        return _salvar(fig, save)

    # ------------------------------------------------------------------
    # Probabilidade
    # ------------------------------------------------------------------
    def plot_classical_vs_frequentist(self, resultados: List[Dict[str, Any]], save: Optional[str] = None) -> plt.Figure:
        rot = [r["Evento"] for r in resultados]
        cl = [r["Clássica (%)"] for r in resultados]
        fr = [r["Frequentista (%)"] for r in resultados]
        x = np.arange(len(rot))
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        ax.bar(x - 0.2, cl, 0.4, color="#C9D3CF", label="Clássica (equiprovável)")
        b = ax.bar(x + 0.2, fr, 0.4, color=VERDE, label="Frequentista (2024)")
        for barra, v in zip(b, fr):
            ax.text(barra.get_x() + barra.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(rot)
        ax.set_ylabel("Probabilidade (%)")
        ax.set_ylim(0, max(fr) * 1.2)
        ax.set_title("Probabilidade clássica x frequentista do resultado")
        ax.legend()
        return _salvar(fig, save)

    def plot_venn(self, n_a: int, n_b: int, n_ab: int, n: int, rotulo_a: str, rotulo_b: str,
                  titulo: str, save: Optional[str] = None) -> plt.Figure:
        """Diagrama de Venn com contagens: A − B, A ∩ B, B − A e fora de A ∪ B."""
        fig, ax = plt.subplots(figsize=(7, 4.4))
        ax.add_patch(Circle((-0.55, 0), 1.0, color=VERDE, alpha=0.35))
        ax.add_patch(Circle((0.55, 0), 1.0, color=AMARELO, alpha=0.45))
        ax.text(-1.05, 0, f"{n_a - n_ab}\n({(n_a - n_ab) / n:.1%})", ha="center", va="center", fontsize=11)
        ax.text(0, 0, f"{n_ab}\n({n_ab / n:.1%})", ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(1.05, 0, f"{n_b - n_ab}\n({(n_b - n_ab) / n:.1%})", ha="center", va="center", fontsize=11)
        ax.text(-1.0, 1.12, rotulo_a, ha="center", fontsize=10, fontweight="bold", color=VERDE)
        ax.text(1.0, 1.12, rotulo_b, ha="center", fontsize=10, fontweight="bold", color="#8a6100")
        fora = n - (n_a + n_b - n_ab)
        ax.text(0, -1.3, f"Fora de A ∪ B: {fora} jogos ({fora / n:.1%})", ha="center", fontsize=10, color=CINZA)
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-1.5, 1.4)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(titulo)
        return _salvar(fig, save)

    def plot_probability_bars(self, rotulos: List[str], valores: List[float], titulo: str,
                              destaque: Optional[int] = None, save: Optional[str] = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        cores = [AMARELO if i == destaque else VERDE for i in range(len(valores))]
        b = ax.bar(rotulos, valores, color=cores)
        for barra, v in zip(b, valores):
            ax.text(barra.get_x() + barra.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.set_ylabel("Probabilidade (%)")
        ax.set_title(titulo)
        return _salvar(fig, save)

    def plot_probability_tree(self, res: Dict[str, Any], save: Optional[str] = None) -> plt.Figure:
        """Árvore de probabilidades: partição B1, B2, B3 e, em cada ramo, A e Aᶜ."""
        fig, ax = plt.subplots(figsize=(10, 5.2))
        ax.axis("off")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.text(0.3, 5, "Partida", fontsize=11, fontweight="bold", va="center")
        ys = [8.4, 5, 1.6]
        for linha, y in zip(res["linhas"], ys):
            ax.plot([1.6, 4.1], [5, y], color=CINZA, linewidth=1.5)
            ax.text(2.6, (5 + y) / 2 + 0.35, f"{linha['P(Bi) (%)']:.1f}%", fontsize=9.5, color=AZUL)
            ax.text(4.2, y, f"{linha['Evento']}: {linha['Descrição']}", fontsize=9.5, va="center",
                    bbox={"boxstyle": "round", "facecolor": "#EEF5F1", "edgecolor": VERDE})
            for dy, nome, p in ((0.9, "A (vitória)", linha["P(A|Bi) (%)"]), (-0.9, "Aᶜ", 100 - linha["P(A|Bi) (%)"])):
                ax.plot([7.6, 8.3], [y, y + dy], color=CINZA, linewidth=1.2)
                ax.text(8.35, y + dy, f"{nome}: {p:.1f}%", fontsize=9, va="center",
                        color=VERDE if nome.startswith("A ") else VERMELHO)
        ax.set_title(f"Árvore de probabilidades — P({res['evento']}) = Σ P(A|Bi)·P(Bi) = {res['p_a_total']:.2f}%")
        return _salvar(fig, save)

    # ------------------------------------------------------------------
    # Variáveis aleatórias discretas
    # ------------------------------------------------------------------
    def plot_pmf_cdf(self, tabela: pd.DataFrame, nome: str, save: Optional[str] = None) -> plt.Figure:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
        a1.vlines(tabela["x"], 0, tabela["P(X = x)"], color=VERDE, linewidth=3)
        a1.plot(tabela["x"], tabela["P(X = x)"], "o", color=VERDE)
        a1.set_title(f"Função de probabilidade p(x)\n{nome}")
        a1.set_xlabel("x")
        a1.set_ylabel("P(X = x)")
        a2.step(tabela["x"], tabela["F(x) = P(X ≤ x)"], where="post", color=AZUL, linewidth=2)
        a2.plot(tabela["x"], tabela["F(x) = P(X ≤ x)"], "o", color=AZUL)
        a2.set_title("Função de distribuição acumulada\nF(x) = P(X ≤ x)")
        a2.set_xlabel("x")
        a2.set_ylim(0, 1.05)
        return _salvar(fig, save)

    def plot_binomial(self, res: Dict[str, Any], titulo: str, save: Optional[str] = None) -> plt.Figure:
        t = res["tabela"]
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        b = ax.bar(t["k"], t["P(X = k)"], color=VERDE)
        for barra, v in zip(b, t["P(X = k)"]):
            if v >= 0.005:
                ax.text(barra.get_x() + barra.get_width() / 2, v, f"{v:.1%}", ha="center", va="bottom", fontsize=8.5)
        ax.axvline(res["esperanca"], color=AMARELO, linestyle="--", linewidth=2, label=f"E[X] = np = {res['esperanca']:g}")
        ax.set_xticks(t["k"])
        ax.set_xlabel("k (sucessos)")
        ax.set_ylabel("P(X = k)")
        ax.set_title(titulo)
        ax.legend()
        return _salvar(fig, save)

    def plot_poisson_fit(self, res: Dict[str, Any], xlabel: str, save: Optional[str] = None) -> plt.Figure:
        t = res["tabela"]
        fig, ax = plt.subplots(figsize=(9, 4.4))
        ax.bar(t["k"] - 0.2, t["Observado (fi)"], 0.4, color=VERDE, label="Observado (2024)")
        ax.bar(t["k"] + 0.2, t["Esperado (n·P)"], 0.4, color=AMARELO, label=f"Esperado Poisson (λ = {res['lambda']:.2f})")
        ax.set_xticks(t["k"])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Número de partidas")
        ax.set_title("Frequências observadas x esperadas pela Poisson")
        ax.legend()
        return _salvar(fig, save)

    def plot_score_matrix(self, matriz: np.ndarray, home: str, away: str, mostrar: int = 5,
                          save: Optional[str] = None) -> plt.Figure:
        m = matriz[:mostrar + 1, :mostrar + 1] * 100
        fig, ax = plt.subplots(figsize=(6.8, 5.6))
        im = ax.imshow(m, cmap="Greens")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m[i, j]:.1f}", ha="center", va="center", fontsize=9,
                        color="white" if m[i, j] > m.max() * 0.6 else "#14302a")
        ax.set_xticks(range(m.shape[1]))
        ax.set_yticks(range(m.shape[0]))
        ax.set_xlabel(f"Gols do {away} (visitante)")
        ax.set_ylabel(f"Gols do {home} (mandante)")
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="Probabilidade do placar (%)")
        ax.set_title(f"{home} x {away}: probabilidade de cada placar (%)")
        return _salvar(fig, save)

    # ------------------------------------------------------------------
    # Variáveis aleatórias contínuas
    # ------------------------------------------------------------------
    def plot_uniform(self, minutos: pd.Series, save: Optional[str] = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(9, 4.4))
        ax.hist(minutos, bins=np.arange(0, 91, 5), density=True, color=VERDE_CLARO, edgecolor=VERDE, label="Gols de 2024 (densidade)")
        ax.hlines(1 / 90, 0, 90, color=AMARELO, linewidth=3, label="Uniforme U(0, 90): f(x) = 1/90")
        ax.axvspan(40, 45, color=CINZA, alpha=0.12)
        ax.axvspan(85, 90, color=CINZA, alpha=0.12, label="Fim de cada tempo (acréscimos)")
        ax.set_xlabel("Minuto do gol")
        ax.set_ylabel("Densidade f(x)")
        ax.set_title("Minuto dos gols x distribuição Uniforme")
        ax.legend(fontsize=8.5)
        return _salvar(fig, save)

    def plot_exponential(self, tempos: pd.Series, lam: float, save: Optional[str] = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(9, 4.4))
        ax.hist(tempos, bins=np.arange(0, 91, 6), density=True, color=VERDE_CLARO, edgecolor=VERDE, label="Jogos de 2024 (densidade)")
        t = np.linspace(0, 90, 300)
        ax.plot(t, lam * np.exp(-lam * t), color=AMARELO, linewidth=2.5, label=f"Exponencial: f(t) = λe^(−λt), λ = {lam:.4f}")
        ax.fill_between(t[t <= 15], 0, lam * np.exp(-lam * t[t <= 15]), color=AMARELO, alpha=0.25, label="P(T ≤ 15) = área")
        ax.set_xlabel("Minuto do primeiro gol (T)")
        ax.set_ylabel("Densidade f(t)")
        ax.set_title("Tempo até o 1º gol x distribuição Exponencial")
        ax.legend(fontsize=8.5)
        return _salvar(fig, save)

    def plot_normal(self, valores: pd.Series, mu: float, sigma: float, x1: float, x2: float,
                    save: Optional[str] = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(9, 4.4))
        ax.hist(valores, bins=np.arange(valores.min() - 0.5, valores.max() + 1.5, 2), density=True,
                color=VERDE_CLARO, edgecolor=VERDE, label="38 rodadas (densidade)")
        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 400)
        y = stats.norm.pdf(x, mu, sigma)
        ax.plot(x, y, color=AZUL, linewidth=2.5, label=f"Normal N(μ = {mu:.1f}; σ = {sigma:.1f})")
        mask = (x >= x1) & (x <= x2)
        ax.fill_between(x[mask], 0, y[mask], color=AMARELO, alpha=0.35, label=f"P({x1:g} ≤ X ≤ {x2:g})")
        ax.axvline(mu, color=CINZA, linestyle="--", linewidth=1)
        ax.set_xlabel("Gols na rodada (X)")
        ax.set_ylabel("Densidade f(x)")
        ax.set_title("Gols por rodada x distribuição Normal")
        ax.legend(fontsize=8.5)
        return _salvar(fig, save)

    # ------------------------------------------------------------------
    # Jogador
    # ------------------------------------------------------------------
    def plot_player_minutes(self, minutos: pd.Series, jogador: str, save: Optional[str] = None) -> plt.Figure:
        faixas = pd.cut(minutos, bins=[0, 15, 30, 45, 60, 75, 90], include_lowest=True,
                        labels=["0–15", "15–30", "30–45", "45–60", "60–75", "75–90"]).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(8.5, 3.8))
        b = ax.bar(faixas.index.astype(str), faixas.values, color=[VERDE] * 3 + [AZUL] * 3)
        for barra, v in zip(b, faixas.values):
            ax.text(barra.get_x() + barra.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=10)
        ax.set_xlabel("Faixa de minutos (verde = 1º tempo, azul = 2º tempo)")
        ax.set_ylabel("Gols")
        ax.set_ylim(0, max(faixas.max(), 1) * 1.25)
        ax.set_title(f"Distribuição dos gols de {jogador} por faixa de minutos")
        return _salvar(fig, save)

    def plot_player_poisson(self, pmf: Dict[int, float], jogador: str, lam: float, save: Optional[str] = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ks = list(pmf.keys())
        vs = [v * 100 for v in pmf.values()]
        b = ax.bar([str(k) if k < 3 else "3" for k in ks], vs, color=[CINZA, VERDE, VERDE, VERDE])
        for barra, v in zip(b, vs):
            ax.text(barra.get_x() + barra.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontsize=10)
        ax.set_ylim(0, 105)
        ax.set_xlabel("Gols na partida (k)")
        ax.set_ylabel("P(X = k) (%)")
        ax.set_title(f"Poisson(λ = {lam:.3f}): gols de {jogador} por jogo")
        return _salvar(fig, save)
