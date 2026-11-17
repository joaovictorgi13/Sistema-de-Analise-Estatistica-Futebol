"""
PROGRAMA PRINCIPAL: main.py
PROJETO: Estatística e Probabilidade aplicadas ao Brasileirão Série A 2024
DISCIPLINA: Probabilidade e Estatística

Executa no terminal todas as análises do sistema e salva os gráficos na pasta graficos/.
Uso: python main.py
"""

import matplotlib.pyplot as plt

from api_client import FootballDataAPI
from prob_stats_engine import ProbabilityStatsEngine
from visualizer import FootballStatsVisualizer


def titulo(texto: str) -> None:
    print("\n" + "=" * 78)
    print(f" {texto.upper()}")
    print("=" * 78)


def salvar(fig) -> None:
    plt.close(fig)


def main() -> None:
    print("\n  ESTATÍSTICA E PROBABILIDADE NO BRASILEIRÃO 2024\n")
    api = FootballDataAPI()
    df = api.get_dataset()
    jogadores, logs = api.get_players_data()
    gols, cartoes = api.get_goals(), api.get_cards()
    e = ProbabilityStatsEngine(df, jogadores, logs, gols, cartoes)
    v = FootballStatsVisualizer(df)

    titulo("1. Introdução à Estatística e conceitos básicos")
    ov = e.overview()
    print(f"Amostra: {ov['partidas']} partidas, {ov['clubes']} clubes, {ov['gols']} gols, {ov['jogadores']} jogadores")
    print(e.variable_catalog()[["Variável", "Tipo"]].to_string(index=False))
    print("\nClassificação final (5 primeiros):")
    print(e.standings().head(5).to_string())

    titulo("2. Tabelas de distribuição de frequências")
    tab_res = e.frequency_table(df["resultado"], ["Vitória do mandante", "Empate", "Vitória do visitante"])
    print(tab_res.to_string(index=False))
    tab_gols = e.frequency_table(df["total_gols"])
    print("\nGols por partida:\n" + tab_gols.to_string(index=False))
    classes = e.class_frequency_table(gols["minuto_jogo"])
    print(f"\nMinuto dos gols em classes (Sturges: k = {classes['k_sturges']} → {classes['k']}, h = {classes['h']}):")
    print(classes["tabela"][["Classe", "xi (ponto médio)", "fi", "fri (%)", "Fi", "Fri (%)"]].to_string(index=False))

    titulo("3. Representação gráfica e medidas de posição")
    pos = e.position_measures(df["total_gols"])
    print(f"Gols por partida: média = {pos['media']}, mediana = {pos['mediana']}, moda = {pos['moda']}")
    print(f"Minuto dos gols (agrupado): {e.grouped_measures(gols['minuto_jogo'])}")
    salvar(v.plot_bar_frequency(tab_res, "Resultado da partida", "Resultado", save="01_colunas_resultado.png"))
    salvar(v.plot_pie(tab_res, "Resultado da partida", save="02_setores_resultado.png"))
    salvar(v.plot_histogram_classes(classes, "Histograma e polígono: minuto dos gols", "Minuto", save="03_histograma_minutos.png"))
    salvar(v.plot_ogive(classes, "Ogiva: minuto dos gols", "Minuto", save="04_ogiva_minutos.png"))
    salvar(v.plot_goals_per_round(save="05_linha_gols_rodada.png"))
    salvar(v.plot_position_measures(df["total_gols"], pos, "Medidas de posição: gols por partida", "Gols", save="06_posicao_gols.png"))

    titulo("4. Medidas de dispersão, quartis, assimetria e boxplot")
    disp = e.dispersion_measures(df["total_gols"])
    for k in ("amplitude", "variancia_amostral", "desvio_padrao", "cv_pct", "q1", "q2", "q3", "diq",
              "limite_inferior", "limite_superior", "qtd_outliers", "assimetria_pearson_mediana", "classe_assimetria"):
        print(f" - {k}: {disp[k]}")
    salvar(v.plot_boxplot(df["total_gols"], disp, "Boxplot: gols por partida", "Gols", save="07_boxplot_gols.png"))
    tabela_times = e.team_dispersion_table()
    salvar(v.plot_boxplot_teams(e.team_goals_per_match_long(), tabela_times["Clube"].tolist(), save="08_boxplot_clubes.png"))

    titulo("5. Teoria das Probabilidades")
    s = e.sample_space_results()
    for r in s["resultados"]:
        print(f" {r['Evento']}: clássica {r['Clássica (%)']}% | frequentista {r['Frequentista (%)']}%")
    ea = e.events_and_axioms()
    print(f"União (gol de alguém no 1T): {ea['p_m1']}% + {ea['p_v1']}% − {ea['p_m1_inter_v1']}% = {ea['p_m1_uniao_v1_formula']}%")
    ci = e.conditional_and_independence()
    print(f"P(vitória do mandante | marcou primeiro) = {ci['p_a_dado_b']}% (P(vitória) = {ci['p_a']}%)")
    print(f"Independência gol 1T x gol 2T do mandante: P(C∩D) = {ci['p_m1_inter_m2']}% x P(C)P(D) = {ci['produto_m1_m2']}%")
    salvar(v.plot_classical_vs_frequentist(s["resultados"], save="09_classica_frequentista.png"))
    salvar(v.plot_venn(ea["n_m1"], ea["n_v1"], ea["n_m1_v1"], ea["n"], "A: mandante marca no 1T", "B: visitante marca no 1T",
                       "Diagrama de Venn", save="10_venn_gols_1t.png"))

    titulo("6. Probabilidade Total e Teorema de Bayes")
    tp = e.total_probability_bayes()
    for linha in tp["linhas"]:
        print(f" {linha['Evento']} {linha['Descrição']}: P(Bi) = {linha['P(Bi) (%)']}%, P(A|Bi) = {linha['P(A|Bi) (%)']}%, "
              f"Bayes P(Bi|A) = {linha['Bayes P(Bi|A) (%)']}%")
    print(f" P(A) pela Probabilidade Total = {tp['p_a_total']}% (direto: {tp['p_a_direta']}%)")
    salvar(v.plot_probability_tree(tp, save="11_arvore_probabilidade_total.png"))

    titulo("7. Variáveis aleatórias discretas: Bernoulli, Binomial e Poisson")
    va = e.discrete_random_variable("total_amarelos")
    print(f"Amarelos por partida: E[X] = {va['esperanca']}, Var(X) = {va['variancia']}")
    be = e.bernoulli_home_win()
    bi = e.binomial(5, be["p"])
    print(f"Bernoulli (vitória em casa): p = {be['p']} | Binomial n = 5: E = {bi['esperanca']}, P(≥1) = {bi['p_pelo_menos_1']}%")
    pf = e.poisson_fit("total_gols")
    print(f"Poisson (gols): λ = {pf['lambda']}, variância amostral = {pf['variancia_amostral']}")
    salvar(v.plot_pmf_cdf(va["tabela"], "Cartões amarelos", save="12_pmf_cdf_amarelos.png"))
    salvar(v.plot_binomial(bi, "Binomial: vitórias em 5 jogos em casa", save="13_binomial_vitorias.png"))
    salvar(v.plot_poisson_fit(pf, "Gols por partida", save="14_poisson_gols.png"))
    sim = e.simulate_match("Botafogo", "Palmeiras")
    salvar(v.plot_score_matrix(sim["matriz"], "Botafogo", "Palmeiras", save="15_matriz_placares.png"))

    titulo("8. Distribuições contínuas: Uniforme, Exponencial e Normal")
    u = e.uniform_goal_minute()
    print(u["tabela"].to_string(index=False))
    ex = e.exponential_first_goal()
    print(f"Exponencial: tempo médio até o 1º gol = {ex['media']} min, λ = {ex['lambda']}")
    nr = e.normal_goals_per_round()
    print(f"Normal (gols por rodada): μ = {nr['mu']}, σ = {nr['sigma']}, P(20 ≤ X ≤ 30) = {nr['p_entre']}%")
    salvar(v.plot_uniform(gols["minuto_jogo"], save="16_uniforme_minutos.png"))
    salvar(v.plot_exponential(df["minuto_primeiro_gol"].dropna(), ex["lambda"], save="17_exponencial_primeiro_gol.png"))
    salvar(v.plot_normal(nr["valores"], nr["mu"], nr["sigma"], 20, 30, save="18_normal_gols_rodada.png"))

    titulo("Consulta de jogador: artilheiro")
    top = jogadores.iloc[0]
    print(f"{top['jogador']} ({top['time']}): {top['gols']} gols | {e.player_poisson(top['jogador'], top['time'])}")
    print(e.player_bayes(top["jogador"], top["time"]))
    print("\nGráficos salvos em graficos/.")


if __name__ == "__main__":
    main()
