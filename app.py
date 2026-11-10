"""
APLICAÇÃO INTERATIVA: app.py
INTERFACE: Streamlit
DISCIPLINA: Probabilidade e Estatística Aplicada ao Futebol
EXECUÇÃO: streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from api_client import FootballDataAPI
from prob_stats_engine import ProbabilityStatsEngine
from visualizer import FootballStatsVisualizer

# Configuração da página Streamlit
st.set_page_config(
    page_title="Probabilidade & Estatística no Futebol",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS personalizada
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        color: #1E3A8A;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 25px;
    }
    .player-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #E2E8F0;
        margin-bottom: 20px;
    }
    .theory-box {
        background-color: #EFF6FF;
        border-radius: 8px;
        padding: 18px;
        border: 1px solid #BFDBFE;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def carregar_dados():
    api = FootballDataAPI()
    df_matches = api.get_dataset()
    df_players, df_logs = api.get_players_data()
    return df_matches, df_players, df_logs

# Carregamento dos dados
df, df_players, df_logs = carregar_dados()
engine = ProbabilityStatsEngine(df, df_players, df_logs)
viz = FootballStatsVisualizer(df)

# Barra Lateral
st.sidebar.image("https://img.icons8.com/color/96/000000/football2--v1.png", width=75)
st.sidebar.title("Probabilidade & Futebol")
st.sidebar.markdown("**Trabalho Acadêmico de Estatística**")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navegação:",
    [
        "🏠 1. Visão Geral & Dados",
        "👤 2. Consulta por Time & Jogador",
        "📊 3. Análise Exploratória (AED)",
        "🎲 4. Teoria das Probabilidades & Bayes",
        "🔢 5. Variáveis Aleatórias (Discretas/Contínuas)",
        "🔄 6. Variáveis Bidimensionais (Conjuntas)",
        "⚽ 7. Distribuições Teóricas (Poisson/Binomial)",
        "⚖️ 8. Lei dos Grandes Números & FGM"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Funcionalidades:**\n"
    "- Análise completa de equipes e partidas.\n"
    "- Consulta de dados estatísticos e probabilísticos por jogador.\n"
    "- Modelagem de gols no 1T e 2T, cartões, escanteios e passes."
)

# =============================================================================
# ABA 1: VISÃO GERAL
# =============================================================================
if menu == "🏠 1. Visão Geral & Dados":
    st.markdown('<div class="main-title">⚽ Sistema de Análise Estatística no Futebol</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Integração entre Teoria das Probabilidades, Métodos Estatísticos e Consulta de Times e Jogadores</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de Partidas", len(df))
    with col2:
        st.metric("Total de Jogadores Cadastrados", len(df_players))
    with col3:
        st.metric("Média de Gols / Jogo", f"{df['total_gols'].mean():.2f}")
    with col4:
        st.metric("Média de Cartões Amarelos", f"{df['total_amarelos'].mean():.2f}")

    st.markdown("### 📋 Conteúdos da Disciplina Implementados no Sistema")
    st.markdown("""
    - **Consulta Individualizada por Time & Jogador:** Escolha de equipes e atletas com perfil descritivo, probabilidades de gol por tempo (Poisson), conversão de finalizações (Binomial) e impacto de vitória (Bayes).
    - **Análise Exploratória de Dados (AED):** Média, mediana, moda, variância ($s^2$ e $\sigma^2$), desvio padrão, CV, quartis, IQR, assimetria, curtose e detecção de outliers.
    - **Teoria das Probabilidades:** Probabilidades marginais, condicionais $P(A|B)$, Teorema de Bayes e teste de independência estocástica.
    - **Variáveis Aleatórias Discretas e Contínuas:** Funções PMF, PDF, CDF, Esperança $E[X]$ e Variância $Var(X)$.
    - **Variáveis Aleatórias Bidimensionais:** Probabilidade conjunta, covariância $Cov(X, Y)$ e correlações de Pearson e Spearman.
    - **Distribuições de Probabilidades:** Poisson (gols em cada tempo e matriz de placares), Binomial, Exponencial (tempo do primeiro gol) e Normal (posse de bola).
    - **Leis dos Grandes Números & Função Geratriz de Momentos (FGM):** Simulação da convergência amostral e derivação teórica analítica dos momentos via $M_X(t) = E[e^{tX}]$.
    """)

    st.markdown("### 🔍 Amostra dos Dados Coletados via API / Open Data")
    st.dataframe(df.head(6), use_container_width=True)

# =============================================================================
# ABA 2: CONSULTA POR TIME & JOGADOR
# =============================================================================
elif menu == "👤 2. Consulta por Time & Jogador":
    st.markdown('<div class="main-title">👤 Consulta Estatística e Probabilística de Jogadores</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Selecione uma equipe e um jogador para consultar métricas, probabilidades de gol e impacto tático</div>', unsafe_allow_html=True)

    times = engine.get_teams_list()

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        time_selecionado = st.selectbox("Selecione o Time:", times, index=times.index("Man City") if "Man City" in times else 0)
    with col_t2:
        jogadores_do_time = engine.get_players_by_team(time_selecionado)
        jogador_selecionado = st.selectbox("Selecione o Jogador:", jogadores_do_time)

    if jogador_selecionado:
        p_poisson = engine.player_goal_probabilities_poisson(jogador_selecionado)
        p_bayes = engine.player_bayes_impact(jogador_selecionado)
        p_row = df_players[df_players["jogador"] == jogador_selecionado].iloc[0]

        # Cartão de Informações Gerais
        st.markdown(f"""
        <div class="player-card">
            <h3 style="margin-top:0; color:#1E3A8A;">⚽ {jogador_selecionado} — {p_poisson['time']}</h3>
            <p><b>Posição:</b> {p_poisson['posicao']} | <b>Idade:</b> {p_poisson['idade']} anos | <b>Jogos na Temporada:</b> {p_row['jogos']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Métricas Principais
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Gols Totais na Época", p_row['gols'], f"{p_row['gols_por_jogo']} / jogo")
        with m2:
            st.metric("Finalizações / Jogo", p_row['chutes_por_jogo'])
        with m3:
            st.metric("Taxa de Conversão (Chutes -> Gol)", f"{p_row['taxa_conversao_pct']}%")
        with m4:
            st.metric("Passes Médios / Jogo", p_row['passes_por_jogo'])

        st.markdown("---")

        tab_j_prob, tab_j_aed, tab_j_binom, tab_j_bayes = st.tabs([
            "🎯 Probabilidades para o Próximo Jogo (Poisson)",
            "📊 Análise Exploratória (AED do Jogador)",
            "🎲 Conversão de Finalizações (Binomial)",
            "🧠 Impacto Decisivo na Equipe (Bayes)"
        ])

        with tab_j_prob:
            st.subheader("Modelagem de Gols e Cartões via Distribuição de Poisson")
            st.markdown(f"""
            Assumindo que as oportunidades de gol do atleta ocorrem segundo um **Processo Estocástico de Poisson** com taxa média $\lambda = {p_poisson['lambda_gols_jogo']}$ gols/jogo:
            """)

            pj1, pj2, pj3, pj4 = st.columns(4)
            with pj1:
                st.metric("Prob. Marcar no Jogo", f"{p_poisson['prob_marcar_ao_menos_1_gol']*100:.1f}%")
            with pj2:
                st.metric("Prob. Marcar no 1º Tempo", f"{p_poisson['prob_marcar_no_1t']*100:.1f}%")
            with pj3:
                st.metric("Prob. Marcar no 2º Tempo", f"{p_poisson['prob_marcar_no_2t']*100:.1f}%")
            with pj4:
                st.metric("Prob. Levar Amarelo", f"{p_poisson['prob_levar_amarelo']*100:.1f}%")

            fig_pmf_j = viz.plot_player_poisson_pmf(jogador_selecionado, p_poisson["pmf_gols_jogo"])
            st.pyplot(fig_pmf_j)

        with tab_j_aed:
            st.subheader(f"Estatísticas Descritivas Amostrais de {jogador_selecionado}")
            metrica_escolhida = st.selectbox(
                "Métrica para análise detalhada jogo a jogo:",
                ["chutes", "passes", "gols", "minutos"],
                format_func=lambda x: {"chutes": "Chutes por Partida", "passes": "Passes por Partida", "gols": "Gols por Partida", "minutos": "Minutos Jogados"}[x]
            )
            aed_j = engine.player_exploratory_analysis(jogador_selecionado, metrica_escolhida)

            c_a1, c_a2, c_a3, c_a4 = st.columns(4)
            with c_a1:
                st.metric("Média Amostral (x̄)", aed_j["media_por_jogo"])
                st.metric("Primeiro Quartil (Q1)", aed_j["q1"])
            with c_a2:
                st.metric("Mediana (Q2)", aed_j["mediana"])
                st.metric("Terceiro Quartil (Q3)", aed_j["q3"])
            with c_a3:
                st.metric("Desvio Padrão (s)", aed_j["desvio_padrao"])
                st.metric("Intervalo Interquartil (IQR)", aed_j["iqr"])
            with c_a4:
                st.metric("Variância (s²)", aed_j["variancia"])
                st.metric("Coef. Variação (CV)", f"{aed_j['coeficiente_variacao_pct']}%")

            st.markdown(f"""
            - **Assimetria (Skewness):** `{aed_j['assimetria']}`  
            - **Curtose (Kurtosis):** `{aed_j['curtose']}`  
            - **Valor Mínimo Registrado:** `{aed_j['minimo']}` | **Máximo:** `{aed_j['maximo']}`
            """)

        with tab_j_binom:
            st.subheader("Distribuição Binomial das Finalizações")
            st.markdown(f"""
            Modela a probabilidade de conversão de gols dado que o jogador realiza $n$ finalizações na partida,
            com sua taxa histórica de conversão calibrada em $p = {p_row['taxa_conversao_pct']}\\%$:
            $$X \sim Bin(n, p) \implies P(X = k) = \\binom{{n}}{{k}} p^k (1 - p)^{{n - k}}$$
            """)

            n_sim = st.slider(f"Número de Finalizações estimadas para {jogador_selecionado}:", 1, 15, int(round(p_row['chutes_por_jogo'])) or 3)
            p_conv_real = (p_row['taxa_conversao_pct'] / 100.0) if p_row['taxa_conversao_pct'] > 0 else 0.15

            binom_j = engine.binomial_shot_conversion(n_sim, p_conv_real)

            cb1, cb2, cb3 = st.columns(3)
            with cb1:
                st.metric("Gols Esperados E[X] = n·p", binom_j["esperanca_E_X"])
            with cb2:
                st.metric("Variância Var(X)", binom_j["variancia_Var_X"])
            with cb3:
                st.metric("Prob. Marcar ao Menos 1 Gol", f"{binom_j['prob_marcar_ao_menos_1_gol']*100:.1f}%")

            df_b_table = pd.DataFrame({
                "Gols Convertidos (k)": list(binom_j["pmf"].keys()),
                "Probabilidade P(X = k)": [f"{v*100:.2f}%" for v in binom_j["pmf"].values()]
            })
            st.dataframe(df_b_table, use_container_width=True)

        with tab_j_bayes:
            st.subheader("Impacto Tático no Resultado: Teorema de Bayes")
            st.markdown(f"""
            Avalia a dependência da equipe em relação aos gols do jogador:
            - **$P(\\text{{Vitória}} \\mid \\text{{{jogador_selecionado.split()[-1]} marcou}}):$** `{p_bayes['prob_vitoria_dado_jogador_marcou']*100:.1f}%`
            - **$P(\\text{{Vitória}} \\mid \\text{{{jogador_selecionado.split()[-1]} NÃO marcou}}):$** `{p_bayes['prob_vitoria_dado_jogador_nao_marcou']*100:.1f}%`
            - **Impacto Adicional de Vitória:** `+{p_bayes['impacto_positivo_vitoria_pct']}%`
            - **Teorema de Bayes — $P(\\text{{Marcou}} \\mid \\text{{Vitória}}):$** `{p_bayes['teorema_bayes_p_marcou_dado_vitoria']*100:.1f}%`
            """)

            fig_bayes_j = viz.plot_player_impact_bayes(jogador_selecionado, p_bayes)
            st.pyplot(fig_bayes_j)

# =============================================================================
# ABA 3: AED (EQUIPES / GERAL)
# =============================================================================
elif menu == "📊 3. Análise Exploratória (AED)":
    st.markdown('<div class="main-title">📊 Análise Exploratória de Dados (AED)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Medidas de Posição, Tendência Central, Dispersão e Forma da Distribuição</div>', unsafe_allow_html=True)

    col_select = st.selectbox(
        "Selecione a Variável para Análise Descritiva:",
        [
            ("total_gols", "Total de Gols na Partida"),
            ("gols_1t_mandante", "Gols Mandante no 1º Tempo"),
            ("gols_2t_mandante", "Gols Mandante no 2º Tempo"),
            ("total_escanteios", "Total de Escanteios"),
            ("total_amarelos", "Total de Cartões Amarelos"),
            ("posse_mandante", "Posse de Bola Mandante (%)"),
            ("chutes_mandante", "Total de Chutes Mandante")
        ],
        format_func=lambda x: x[1]
    )[0]

    stats_res = engine.exploratory_data_analysis(col_select)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Média Aritmética (x̄)", stats_res["media"])
        st.metric("Primeiro Quartil (Q1)", stats_res["q1"])
    with c2:
        st.metric("Mediana (Q2)", stats_res["mediana"])
        st.metric("Terceiro Quartil (Q3)", stats_res["q3"])
    with c3:
        st.metric("Moda Estatística", stats_res["moda"])
        st.metric("Intervalo Interquartil (IQR)", stats_res["iqr"])
    with c4:
        st.metric("Desvio Padrão (s)", stats_res["desvio_padrao"])
        st.metric("Coef. Variação (CV)", f"{stats_res['coeficiente_variacao_pct']}%")

    st.markdown("---")
    col_g1, col_g2 = st.columns([2, 1])

    with col_g1:
        st.subheader("Visualização: Histograma + Boxplot de Tukey")
        fig_aed = viz.plot_aed_distribution(col_select, save_filename=None)
        st.pyplot(fig_aed)

    with col_g2:
        st.subheader("Interpretação Estatística")
        st.markdown(f"""
        - **Variância Amostral ($s^2$):** `{stats_res['variancia_amostral']}`
        - **Variância Populacional ($\sigma^2$):** `{stats_res['variancia_populacional']}`
        - **Assimetria (Skewness):** `{stats_res['assimetria']}`  
          *({ 'Assimetria Positiva (cauda à direita)' if stats_res['assimetria'] > 0 else 'Assimetria Negativa (cauda à esquerda)' if stats_res['assimetria'] < 0 else 'Simétrica'})*
        - **Curtose (Excess Kurtosis):** `{stats_res['curtose']}`  
          *({ 'Leptocúrtica (caudas pesadas)' if stats_res['curtose'] > 0 else 'Platicúrtica (achatada)' if stats_res['curtose'] < 0 else 'Mesocúrtica'})*
        - **Limites para Outliers (Tukey):**  
          Inferior: `[{stats_res['limite_inferior_outlier']}]`  
          Superior: `[{stats_res['limite_superior_outlier']}]`
        - **Outliers Identificados:** `{stats_res['qtd_outliers']}` ocorrências.
        """)

# =============================================================================
# ABA 4: TEORIA DAS PROBABILIDADES
# =============================================================================
elif menu == "🎲 4. Teoria das Probabilidades & Bayes":
    st.markdown('<div class="main-title">🎲 Teoria das Probabilidades</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Probabilidades Clássica/Frequentista, Condicionais, Teorema de Bayes e Teste de Independência</div>', unsafe_allow_html=True)

    prob_res = engine.calculate_conditional_probabilities()

    st.markdown("""
    <div class="theory-box">
        <h4>Fundamentação Teórica:</h4>
        <ul>
            <li><b>Probabilidade Condicional:</b> $P(A \\mid B) = \\frac{P(A \\cap B)}{P(B)}$ (atualização da probabilidade na presença de evidência).</li>
            <li><b>Teorema de Bayes:</b> $P(B \\mid A) = \\frac{P(A \\mid B) \\cdot P(B)}{P(A)}$ (inversão da probabilidade condicional).</li>
            <li><b>Independência Estocástica:</b> Eventos $A$ e $B$ são independentes $\\iff P(A \\cap B) = P(A) \\cdot P(B)$.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cenários Práticos Analisados")
        st.markdown(f"""
        - **Evento A (Vitória Mandante):** $P(A) = {prob_res['p_vitoria_mandante']*100:.2f}\\%$
        - **Evento B (Mandante fez Gol no 1º Tempo):** $P(B) = {prob_res['p_gol_1t_mandante']*100:.2f}\\%$
        - **Evento C (Mandante fez Gol no 2º Tempo):** $P(C) = {prob_res['p_gol_2t_mandante']*100:.2f}\\%$
        - **Interseção $A \\cap B$:** $P(A \\cap B) = {prob_res['p_intersecao_vitoria_e_gol1t']*100:.2f}\\%$
        - **Probabilidade Condicional:**  
          $P(\\text{{Vitória}} \\mid \\text{{Fez gol no 1T}}) = \\frac{{{prob_res['p_intersecao_vitoria_e_gol1t']}}}{{{prob_res['p_gol_1t_mandante']}}} = \\mathbf{{{prob_res['prob_condicional_p_vitoria_dado_gol1t']*100:.2f}\\%}}$
        - **Aplicação do Teorema de Bayes:**  
          $P(\\text{{Fez gol no 1T}} \\mid \\text{{Venceu}}) = \\frac{{P(\\text{{Vitória}} \\mid \\text{{Gol 1T}}) \\cdot P(\\text{{Gol 1T}})}}{{P(\\text{{Vitória}})}} = \\mathbf{{{prob_res['teorema_bayes_p_gol1t_dado_vitoria']*100:.2f}\\%}}$  
          *(Conferência empírica direta: `{prob_res['conferencia_empirica_bayes']*100:.2f}%`)*
        """)

        ind = prob_res["teste_independencia_1t_e_2t"]
        st.info(f"""
        **Teste de Independência de Eventos (Gol 1T e Gol 2T):**
        - $P(\\text{{Gol 1T}} \\cap \\text{{Gol 2T}}) = {ind['P(Gol_1T_e_Gol_2T)']}$
        - $P(\\text{{Gol 1T}}) \\cdot P(\\text{{Gol 2T}}) = {ind['P(Gol_1T) * P(Gol_2T)']}$
        - Diferença absoluta: `{ind['diferenca_absoluta']}`
        - **Conclusão:** **{ind['conclusao']}**
        """)

    with col2:
        st.subheader("Comparativo Gráfico das Probabilidades")
        fig_prob = viz.plot_conditional_and_bayes(save_filename=None)
        st.pyplot(fig_prob)

# =============================================================================
# ABA 5: V.A. DISCRETAS E CONTÍNUAS
# =============================================================================
elif menu == "🔢 5. Variáveis Aleatórias (Discretas/Contínuas)":
    st.markdown('<div class="main-title">🔢 Variáveis Aleatórias Discretas e Contínuas</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Modelagem de PMF, PDF, CDF, Esperança Matemática e Variância</div>', unsafe_allow_html=True)

    tab_disc, tab_cont = st.tabs(["Variável Aleatória Discreta", "Variável Aleatória Contínua"])

    with tab_disc:
        var_d = st.selectbox("Selecione a V.A. Discreta:", ["total_amarelos", "total_gols", "total_escanteios"])
        disc_res = engine.discrete_random_variable_analysis(var_d)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Esperança E[X]", disc_res["esperanca_matematica_E_X"])
        with c2:
            st.metric("Variância Var(X)", disc_res["variancia_Var_X"])
        with c3:
            st.metric("Desvio Padrão σ", disc_res["desvio_padrao_sigma"])

        fig_disc = viz.plot_discrete_random_variable(var_d, save_filename=None)
        st.pyplot(fig_disc)

        with st.expander("Ver Tabela Detalhada da PMF e CDF"):
            df_pmf = pd.DataFrame({
                "k (Valor)": list(disc_res["pmf"].keys()),
                "P(X = k) [PMF]": list(disc_res["pmf"].values()),
                "P(X <= k) [CDF]": list(disc_res["cdf"].values())
            })
            st.dataframe(df_pmf, use_container_width=True)

    with tab_cont:
        cont_res = engine.continuous_random_variable_analysis("posse_mandante")
        st.markdown(f"""
        **Variável Contínua Analisada:** Posse de Bola do Mandante (%)  
        - Média Amostral ($\\mu$): `{cont_res['media_mu']}%`  
        - Desvio Padrão ($\\sigma$): `{cont_res['desvio_padrao_sigma']}%`  
        - $P(50\\% \\le Y \\le 65\\%) = {cont_res['intervalo_exemplo_50_a_65_pct']*100:.2f}\\%$ (calculada por integral da densidade)  
        - $P(Y > 60\\%) = {cont_res['prob_posse_maior_60_pct']*100:.2f}\\%$
        """)

        fig_cont = viz.plot_continuous_distributions(save_filename=None)
        st.pyplot(fig_cont)

# =============================================================================
# ABA 6: V.A. BIDIMENSIONAIS
# =============================================================================
elif menu == "🔄 6. Variáveis Bidimensionais (Conjuntas)":
    st.markdown('<div class="main-title">🔄 Variáveis Aleatórias Bidimensionais</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Distribuição Conjunta, Covariância e Coeficientes de Correlação (Pearson e Spearman)</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("Variável X:", ["chutes_mandante", "posse_mandante", "faltas_mandante"], index=0)
    with col2:
        y_col = st.selectbox("Variável Y:", ["gols_mandante", "passes_mandante", "amarelos_mandante"], index=0)

    bivar = engine.bivariate_random_variables(x_col, y_col)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Covariância Cov(X, Y)", bivar["covariancia_Cov_XY"])
    with c2:
        st.metric("Pearson (ρ)", bivar["correlacao_pearson_rho"])
    with c3:
        st.metric("Spearman", bivar["correlacao_spearman"])
    with c4:
        st.metric("p-valor (Significância)", f"{bivar['p_valor_pearson']:.4f}")

    st.markdown(f"**Interpretação da Associação Linear:** `{bivar['interpretacao_correlacao']}`")

    fig_bivar = viz.plot_bivariate_relationship(x_col, y_col, save_filename=None)
    st.pyplot(fig_bivar)

# =============================================================================
# ABA 7: DISTRIBUIÇÕES TEÓRICAS (POISSON, BINOMIAL, ETC)
# =============================================================================
elif menu == "⚽ 7. Distribuições Teóricas (Poisson/Binomial)":
    st.markdown('<div class="main-title">⚽ Distribuições Teóricas de Probabilidade</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Distribuição de Poisson, Binomial, Exponencial e Normal</div>', unsafe_allow_html=True)

    sub_tab1, sub_tab2, sub_tab3 = st.tabs([
        "Modelagem de Partida (Poisson Bivariada)",
        "Gols por Tempo (Poisson 1T e 2T)",
        "Finalizações (Binomial) & Tempo 1º Gol (Exponencial)"
    ])

    with sub_tab1:
        st.subheader("Simulador de Partida: Distribuição Bivariada de Poisson")
        times_disponiveis = sorted(list(set(df["mandante"].unique())))
        
        c1, c2 = st.columns(2)
        with c1:
            time_h = st.selectbox("Time Mandante (Casa):", times_disponiveis, index=times_disponiveis.index("Arsenal") if "Arsenal" in times_disponiveis else 0)
        with c2:
            time_a = st.selectbox("Time Visitante (Fora):", times_disponiveis, index=times_disponiveis.index("Chelsea") if "Chelsea" in times_disponiveis else 1)

        confronto = engine.simulate_match_poisson_bivariate(time_h, time_a)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric(f"Vitória {time_h}", f"{confronto['prob_vitoria_mandante']*100:.1f}%")
        with m2:
            st.metric("Empate", f"{confronto['prob_empate']*100:.1f}%")
        with m3:
            st.metric(f"Vitória {time_a}", f"{confronto['prob_vitoria_visitante']*100:.1f}%")
        with m4:
            st.metric("Placar Provável", confronto['placar_mais_provavel'], f"{confronto['prob_placar_mais_provavel']*100:.1f}%")

        col_over, col_btts = st.columns(2)
        with col_over:
            st.info(f"⚽ **Over 2.5 Gols:** `{confronto['prob_over_2_5_gols']*100:.1f}%` | **Under 2.5:** `{confronto['prob_under_2_5_gols']*100:.1f}%`")
        with col_btts:
            st.info(f"🎯 **Ambos Marcam (BTTS Sim):** `{confronto['prob_ambos_marcam_sim']*100:.1f}%` | **Não:** `{confronto['prob_ambos_marcam_nao']*100:.1f}%`")

        fig_matriz = viz.plot_poisson_score_matrix(np.array(confronto["matriz_placares"]), time_h, time_a, save_filename=None)
        st.pyplot(fig_matriz)

    with sub_tab2:
        st.subheader("Taxa de Gols por Tempo de Jogo (Distribuição de Poisson)")
        time_tempo = st.selectbox("Selecione a Equipe:", ["Média Geral"] + times_disponiveis)
        p_tempo = engine.poisson_goals_distribution(None if time_tempo == "Média Geral" else time_tempo)

        st.markdown(f"""
        - **Taxa Média no 1º Tempo ($\lambda_{{1T}}$):** `{p_tempo['lambda_1t']}` gols/tempo
        - **Taxa Média no 2º Tempo ($\lambda_{{2T}}$):** `{p_tempo['lambda_2t']}` gols/tempo
        - **Taxa Média no Jogo ($\lambda_{{Total}}$):** `{p_tempo['lambda_total']}` gols/jogo
        - **Propriedade Reprodutiva de Poisson:** $\lambda_{{Total}} \approx \lambda_{{1T}} + \lambda_{{2T}} = {p_tempo['propriedade_reprodutiva_soma_lambdas']}$
        """)

        c_t1, c_t2, c_t3 = st.columns(3)
        with c_t1:
            st.metric("Prob. Gol no 1º Tempo", f"{p_tempo['prob_marcar_pelo_menos_um_1t']*100:.1f}%")
        with c_t2:
            st.metric("Prob. Gol no 2º Tempo", f"{p_tempo['prob_marcar_pelo_menos_um_2t']*100:.1f}%")
        with c_t3:
            st.metric("Prob. Marcar no Jogo", f"{p_tempo['prob_marcar_pelo_menos_um_total']*100:.1f}%")

    with sub_tab3:
        st.subheader("Distribuição Binomial: Conversão de Chutes em Gols")
        n_ch = st.slider("Número de Finalizações (n):", 1, 20, 10)
        p_conv = st.slider("Taxa Histórica de Conversão do Atacante (p):", 0.05, 0.40, 0.15, step=0.01)

        binom_r = engine.binomial_shot_conversion(n_ch, p_conv)
        st.markdown(f"""
        - **Esperança Matemática:** $E[X] = n \\cdot p = {binom_r['esperanca_E_X']}$ gols
        - **Variância:** $Var(X) = n \\cdot p \\cdot (1 - p) = {binom_r['variancia_Var_X']}$
        - **Probabilidade de marcar $\ge 1$ gol:** `{binom_r['prob_marcar_ao_menos_1_gol']*100:.2f}%`
        """)

# =============================================================================
# ABA 8: LLN & FGM
# =============================================================================
elif menu == "⚖️ 8. Lei dos Grandes Números & FGM":
    st.markdown('<div class="main-title">⚖️ Leis dos Grandes Números & Função Geratriz de Momentos</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Convergência Estocástica e Obtenção de Momentos Teóricos</div>', unsafe_allow_html=True)

    tab_lln, tab_fgm = st.tabs(["Lei dos Grandes Números (LLN)", "Função Geratriz de Momentos (FGM)"])

    with tab_lln:
        st.markdown("""
        **Teorema (Lei Fraca dos Grandes Números):**  
        Seja $X_1, X_2, \dots, X_n$ uma sequência de v.a. i.i.d. com média $\mu = E[X_i]$. Conforme $n \to \infty$:
        $$\lim_{n \\to \infty} P(|\\bar{X}_n - \mu| \ge \epsilon) = 0$$
        """)

        lln_data = engine.law_of_large_numbers_simulation("total_gols")
        st.metric("Esperança Matemática Populacional E[X]", lln_data["esperanca_teorica_mu"])

        fig_lln = viz.plot_law_of_large_numbers("total_gols", save_filename=None)
        st.pyplot(fig_lln)

    with tab_fgm:
        st.markdown("""
        **Definição da Função Geratriz de Momentos (FGM):**  
        $$M_X(t) = E[e^{tX}]$$
        Para uma variável aleatória de Poisson com parâmetro $\lambda$:
        $$M_X(t) = \exp(\lambda(e^t - 1))$$

        **Propriedade de Derivação no ponto $t = 0$:**
        - **1º Momento (Esperança):** $E[X] = M'_X(0) = \lambda$
        - **2º Momento:** $E[X^2] = M''_X(0) = \lambda + \lambda^2$
        - **Variância:** $Var(X) = M''_X(0) - [M'_X(0)]^2 = (\lambda + \lambda^2) - \lambda^2 = \lambda$
        """)

        fgm_data = engine.moment_generating_function_poisson("total_gols")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Parâmetro λ", fgm_data["lambda_estimado"])
        with c2:
            st.metric("1º Momento Teórico M'(0)", fgm_data["derivadas_no_ponto_zero"]["primeiro_momento_M_prime_0 (E[X])"])
        with c3:
            st.metric("Variância via FGM", fgm_data["derivadas_no_ponto_zero"]["variancia_via_FGM (M''(0) - [M'(0)]^2)"])

        fig_fgm = viz.plot_moment_generating_function("total_gols", save_filename=None)
        st.pyplot(fig_fgm)

        st.subheader("Comparativo da FGM Teórica vs FGM Empírica $M_n(t) = \\frac{1}{n} \\sum e^{t x_i}$")
        df_fgm = pd.DataFrame.from_dict(fgm_data["tabela_fgm_teorica_vs_empirica"], orient="index")
        df_fgm.index.name = "Parâmetro t"
        st.dataframe(df_fgm, use_container_width=True)
