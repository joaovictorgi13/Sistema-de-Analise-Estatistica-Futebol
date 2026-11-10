"""
PROGRAMA PRINCIPAL: main.py
PROJETO: Sistema de Suporte a Profissionais do Futebol baseado em Probabilidade e Estatística
DISCIPLINA: Probabilidade e Estatística

Executa uma bateria completa de análises estatísticas e probabilísticas:
1. Análise de Partidas e Equipes (AED, Bayes, V.A. Discreta/Contínua, V.A. Bidimensional, Poisson, LLN, FGM)
2. Consulta e Modelagem Estocástica de Jogadores (Poisson individual, Conversão Binomial, Impacto de Vitória via Bayes)
Salva todos os gráficos na pasta 'graficos/'.
"""

import os
import json
import numpy as np
from api_client import FootballDataAPI
from prob_stats_engine import ProbabilityStatsEngine
from visualizer import FootballStatsVisualizer

def imprimir_cabecalho(titulo: str):
    print("\n" + "="*80)
    print(f" {titulo.upper()}")
    print("="*80)

def main():
    print("""
    ========================================================================
     SISTEMA DE ANALISE PROBABILISTICA E ESTATISTICA APLICADA AO FUTEBOL
     Trabalho Academico - Disciplina: Probabilidade e Estatistica
    ========================================================================
    """)

    # 1. Obtenção de dados via API / Repositório
    print("[ETAPA 1] Conectando a API e obtendo base de dados de futebol e jogadores...")
    api = FootballDataAPI()
    df = api.get_dataset()
    df_players, df_logs = api.get_players_data()
    print(f"-> Partidas catalogadas: {len(df)}")
    print(f"-> Jogadores catalogados: {len(df_players)} atletas em {len(df_players['time'].unique())} clubes\n")

    # Inicialização dos motores
    engine = ProbabilityStatsEngine(df, df_players, df_logs)
    viz = FootballStatsVisualizer(df)

    # -------------------------------------------------------------------------
    # CONTEÚDO 1: ANÁLISE EXPLORATÓRIA DE DADOS (AED)
    # -------------------------------------------------------------------------
    imprimir_cabecalho("1. Analise Exploratoria de Dados (AED)")
    variaveis_aed = ["total_gols", "total_escanteios", "total_amarelos", "posse_mandante"]
    for col in variaveis_aed:
        res = engine.exploratory_data_analysis(col)
        print(f"\n[Variavel: {col.upper()}]")
        print(f" - Media: {res['media']} | Mediana: {res['mediana']} | Moda: {res['moda']}")
        print(f" - Desvio Padrao: {res['desvio_padrao']} | Variancia: {res['variancia_amostral']} | CV: {res['coeficiente_variacao_pct']}%")
        print(f" - Quartis: Q1={res['q1']}, Q2={res['q2']}, Q3={res['q3']} | IQR={res['iqr']}")
        print(f" - Assimetria (Skewness): {res['assimetria']} | Curtose: {res['curtose']}")
        print(f" - Limites de Outlier: [{res['limite_inferior_outlier']}, {res['limite_superior_outlier']}] | Qtd Outliers: {res['qtd_outliers']}")

    print("\n-> Gerando Grafico 1: Histograma com Boxplot e Medidas de Tendencia Central...")
    viz.plot_aed_distribution("total_gols", "1_aed_distribuicao.png")

    # -------------------------------------------------------------------------
    # CONTEÚDO 2: TEORIA DAS PROBABILIDADES
    # -------------------------------------------------------------------------
    imprimir_cabecalho("2. Teoria das Probabilidades: Marginais, Condicionais e Bayes")
    prob_teorica = engine.calculate_conditional_probabilities()
    print(f" - Total de Partidas no Espaco Amostral: {prob_teorica['total_partidas']}")
    print(f" - P(Vitoria do Mandante) [Marginal]: {prob_teorica['p_vitoria_mandante'] * 100:.2f}%")
    print(f" - P(Gol do Mandante no 1o Tempo) [Marginal]: {prob_teorica['p_gol_1t_mandante'] * 100:.2f}%")
    print(f" - P(Gol do Mandante no 2o Tempo) [Marginal]: {prob_teorica['p_gol_2t_mandante'] * 100:.2f}%")
    print(f" - P(Vitoria do Mandante | Marcou no 1o Tempo) [Condicional]: {prob_teorica['prob_condicional_p_vitoria_dado_gol1t'] * 100:.2f}%")
    print(f" - Teorema de Bayes: P(Marcou no 1T | Venceu o Jogo) = {prob_teorica['teorema_bayes_p_gol1t_dado_vitoria'] * 100:.2f}%")
    print(f"   (Conferencia Empirica Direta n(A e B)/n(B) = {prob_teorica['conferencia_empirica_bayes'] * 100:.2f}%)")
    
    indep = prob_teorica["teste_independencia_1t_e_2t"]
    print(f" - Teste de Independencia de Eventos (Gol 1T e Gol 2T):")
    print(f"   * P(Gol 1T e Gol 2T) = {indep['P(Gol_1T_e_Gol_2T)']}")
    print(f"   * P(Gol 1T) * P(Gol 2T) = {indep['P(Gol_1T) * P(Gol_2T)']}")
    print(f"   * Diferenca: {indep['diferenca_absoluta']} => {indep['conclusao']}")

    print("\n-> Gerando Grafico 2: Probabilidades Condicionais e Teorema de Bayes...")
    viz.plot_conditional_and_bayes("2_teoria_probabilidades.png")

    # -------------------------------------------------------------------------
    # CONTEÚDO 3: CONSULTA POR TIME & JOGADORES (DESTAQUE NOVO)
    # -------------------------------------------------------------------------
    imprimir_cabecalho("3. Consulta Estatistica e Probabilistica por Time e Jogador")
    time_destaque = "Man City"
    jogador_destaque = "Erling Haaland"
    print(f"Time Selecionado: {time_destaque}")
    print(f"Jogadores Disponiveis no elenco: {engine.get_players_by_team(time_destaque)}")
    print(f"\n[Consulta Individual: {jogador_destaque}]")

    j_poisson = engine.player_goal_probabilities_poisson(jogador_destaque)
    j_bayes = engine.player_bayes_impact(jogador_destaque)
    j_aed = engine.player_exploratory_analysis(jogador_destaque, "chutes")

    print(f" - Posicao: {j_poisson['posicao']} | Idade: {j_poisson['idade']} anos")
    print(f" - Taxa Poisson de Gols por Jogo (lambda): {j_poisson['lambda_gols_jogo']}")
    print(f" - Probabilidade de Marcar ao Menos 1 Gol: {j_poisson['prob_marcar_ao_menos_1_gol'] * 100:.2f}%")
    print(f" - Probabilidade de Marcar no 1o Tempo: {j_poisson['prob_marcar_no_1t'] * 100:.2f}%")
    print(f" - Probabilidade de Marcar no 2o Tempo: {j_poisson['prob_marcar_no_2t'] * 100:.2f}%")
    print(f" - Probabilidade de Levar Cartao Amarelo: {j_poisson['prob_levar_amarelo'] * 100:.2f}%")
    print(" - PMF de Gols (0, 1, 2, 3):", j_poisson["pmf_gols_jogo"])

    print(f"\n[Impacto no Resultado da Equipe via Teorema de Bayes]")
    print(f" - P(Vitoria do {time_destaque} | {jogador_destaque} Marcou) = {j_bayes['prob_vitoria_dado_jogador_marcou'] * 100:.2f}%")
    print(f" - P(Vitoria do {time_destaque} | {jogador_destaque} NAO Marcou) = {j_bayes['prob_vitoria_dado_jogador_nao_marcou'] * 100:.2f}%")
    print(f" - Impacto Positivo de Vitoria: +{j_bayes['impacto_positivo_vitoria_pct']}%")
    print(f" - Teorema de Bayes: P(Haaland Marcou | Time Venceu) = {j_bayes['teorema_bayes_p_marcou_dado_vitoria'] * 100:.2f}%")

    print(f"\n[AED do Jogador - Finalizacoes por Jogo]")
    print(f" - Media de Chutes: {j_aed['media_por_jogo']} | Mediana: {j_aed['mediana']} | Desvio Padrao: {j_aed['desvio_padrao']}")
    print(f" - Variancia: {j_aed['variancia']} | CV: {j_aed['coeficiente_variacao_pct']}% | Assimetria: {j_aed['assimetria']}")

    print("\n-> Gerando Graficos do Jogador (PMF Poisson e Impacto Bayes)...")
    viz.plot_player_poisson_pmf(jogador_destaque, j_poisson["pmf_gols_jogo"], "9_jogador_poisson_pmf.png")
    viz.plot_player_impact_bayes(jogador_destaque, j_bayes, "10_jogador_impacto_bayes.png")

    # -------------------------------------------------------------------------
    # CONTEÚDO 4: VARIÁVEIS ALEATÓRIAS DISCRETAS E CONTÍNUAS
    # -------------------------------------------------------------------------
    imprimir_cabecalho("4. Variaveis Aleatorias Discretas e Continuas")
    va_disc = engine.discrete_random_variable_analysis("total_amarelos")
    print(f"[V.A. Discreta: Cartoes Amarelos por Jogo]")
    print(f" - Suporte X em {va_disc['suporte']}")
    print(f" - Esperanca Matematica E[X] = {va_disc['esperanca_matematica_E_X']}")
    print(f" - Variancia Teorica Var(X) = {va_disc['variancia_Var_X']} | Desvio Padrao sigma = {va_disc['desvio_padrao_sigma']}")
    print(" - Funcao de Massa de Probabilidade (PMF):", va_disc["pmf"])

    print("\n-> Gerando Grafico 3: PMF e CDF de Variavel Aleatoria Discreta...")
    viz.plot_discrete_random_variable("total_amarelos", "3_va_discreta_pmf_cdf.png")

    va_cont = engine.continuous_random_variable_analysis("posse_mandante")
    print(f"\n[V.A. Continua: Posse de Bola do Mandante (%)]")
    print(f" - Media Continua mu = {va_cont['media_mu']}% | Desvio Padrao sigma = {va_cont['desvio_padrao_sigma']}%")
    print(f" - Probabilidade de Posse no intervalo [50%, 65%]: {va_cont['intervalo_exemplo_50_a_65_pct'] * 100:.2f}%")
    print(f" - Probabilidade de Posse > 60%: {va_cont['prob_posse_maior_60_pct'] * 100:.2f}%")

    print("\n-> Gerando Grafico 4: Ajustes das V.A. Continuas (Exponencial e Normal)...")
    viz.plot_continuous_distributions("4_distribuicoes_continuas.png")

    # -------------------------------------------------------------------------
    # CONTEÚDO 5: VARIÁVEIS ALEATÓRIAS BIDIMENSIONAIS
    # -------------------------------------------------------------------------
    imprimir_cabecalho("5. Variaveis Aleatorias Bidimensionais (Vetor Aleatorio Conjunto)")
    bivar = engine.bivariate_random_variables("chutes_mandante", "gols_mandante")
    print(f"Analise Conjunta entre X = '{bivar['var_x']}' e Y = '{bivar['var_y']}':")
    print(f" - E[X] = {bivar['media_X']} | E[Y] = {bivar['media_Y']}")
    print(f" - Covariancia Cov(X, Y) = {bivar['covariancia_Cov_XY']}")
    print(f" - Coeficiente de Correlacao de Pearson rho = {bivar['correlacao_pearson_rho']} (p-valor: {bivar['p_valor_pearson']})")
    print(f" - Coeficiente de Correlacao de Spearman = {bivar['correlacao_spearman']}")
    print(f" - Interpretacao: {bivar['interpretacao_correlacao']}")

    print("\n-> Gerando Grafico 5: Grafico de Dispersao e Relacao Bivariada com Reta de Regressao...")
    viz.plot_bivariate_relationship("chutes_mandante", "gols_mandante", "5_va_bidimensional.png")

    # -------------------------------------------------------------------------
    # CONTEÚDO 6: DISTRIBUIÇÕES TEÓRICAS (POISSON, BINOMIAL, EXPONENCIAL, NORMAL)
    # -------------------------------------------------------------------------
    imprimir_cabecalho("6. Distribuicoes de Probabilidade: Poisson, Binomial, Exponencial e Normal")
    confronto = engine.simulate_match_poisson_bivariate("Arsenal", "Chelsea")
    print(f"[Projecao de Partida Bivariada de Poisson: Arsenal x Chelsea]")
    print(f" - Probabilidade Vitoria Arsenal: {confronto['prob_vitoria_mandante'] * 100:.2f}%")
    print(f" - Probabilidade Empate: {confronto['prob_empate'] * 100:.2f}%")
    print(f" - Probabilidade Vitoria Chelsea: {confronto['prob_vitoria_visitante'] * 100:.2f}%")
    print(f" - Over 2.5 Gols: {confronto['prob_over_2_5_gols'] * 100:.2f}% | Under 2.5: {confronto['prob_under_2_5_gols'] * 100:.2f}%")
    print(f" - Ambos Marcam (Sim): {confronto['prob_ambos_marcam_sim'] * 100:.2f}%")
    print(f" - Placar Mais Provavel: {confronto['placar_mais_provavel']} (Probabilidade: {confronto['prob_placar_mais_provavel'] * 100:.2f}%)")

    print("\n-> Gerando Grafico 6: Heatmap da Matriz de Placares Provaveis de Poisson...")
    viz.plot_poisson_score_matrix(np.array(confronto["matriz_placares"]), "Arsenal", "Chelsea", "6_poisson_matriz_placares.png")

    # -------------------------------------------------------------------------
    # CONTEÚDO 7: LEIS DOS GRANDES NÚMEROS E FUNÇÃO GERATRIZ DE MOMENTOS
    # -------------------------------------------------------------------------
    imprimir_cabecalho("7. Leis dos Grandes Numeros (LLN) e Funcao Geratriz de Momentos (FGM)")
    lln_res = engine.law_of_large_numbers_simulation("total_gols")
    print(f"[Lei dos Grandes Numeros - Media de Gols]")
    print(f" - Esperanca Populacional E[X] (mu) = {lln_res['esperanca_teorica_mu']}")
    print(f" - Primeiras 5 Medias Cumulativas: {lln_res['primeiras_10_medias_cumulativas'][:5]}")
    print(f" - Ultimas 5 Medias Cumulativas: {lln_res['ultimas_10_medias_cumulativas'][-5:]}")
    print(f" - Conclusao: {lln_res['conclusao']}")

    print("\n-> Gerando Grafico 7: Trajetoria de Convergencia da Lei dos Grandes Numeros...")
    viz.plot_law_of_large_numbers("total_gols", "7_lei_grandes_numeros.png")

    fgm_res = engine.moment_generating_function_poisson("total_gols")
    print(f"\n[Funcao Geratriz de Momentos (FGM)]")
    print(f" - Modelo Teorico: {fgm_res['formula_fgm_poisson']}")
    print(f" - Parametro lambda = {fgm_res['lambda_estimado']}")
    print(" - Momentos obtidos via Derivacao Analitica da FGM no ponto t = 0:")
    for k, v in fgm_res["derivadas_no_ponto_zero"].items():
        print(f"   * {k} = {v}")

    print("\n-> Gerando Grafico 8: Curvas da Funcao Geratriz de Momentos (Teorica vs Empirica)...")
    viz.plot_moment_generating_function("total_gols", "8_fgm_poisson.png")

    imprimir_cabecalho("EXECUCAO CONCLUIDA COM SUCESSO!")
    print("Todos os conteudos da disciplina foram aplicados para Equipes e Jogadores.")
    print(f"Os graficos foram salvos na pasta: {os.path.abspath('graficos')}\n")

if __name__ == "__main__":
    main()
