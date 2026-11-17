"""
APLICAÇÃO INTERATIVA: app.py
INTERFACE: Streamlit
DISCIPLINA: Probabilidade e Estatística aplicada ao Brasileirão 2024
EXECUÇÃO: streamlit run app.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from api_client import FootballDataAPI
from prob_stats_engine import ProbabilityStatsEngine
from visualizer import FootballStatsVisualizer

st.set_page_config(page_title="Estatística no Brasileirão 2024", page_icon="⚽", layout="wide",
                   initial_sidebar_state="expanded")


@st.cache_data(show_spinner="Carregando o Brasileirão 2024...")
def carregar_dados():
    api = FootballDataAPI()
    df = api.get_dataset()
    jogadores, logs = api.get_players_data()
    return df, jogadores, logs, api.get_goals(), api.get_cards()


df, df_players, df_logs, df_goals, df_cards = carregar_dados()
engine = ProbabilityStatsEngine(df, df_players, df_logs, df_goals, df_cards)
viz = FootballStatsVisualizer(df)
TIMES = engine.get_teams_list()


def mostrar(fig):
    st.pyplot(fig, width="content")
    plt.close(fig)


def pct(v: float) -> str:
    return f"{v:.2f}%".replace(".", ",")


def num(v, casas: int = 2) -> str:
    return f"{v:.{casas}f}".replace(".", ",")


# Variáveis disponíveis nos seletores
VAR_DISCRETAS = {
    "total_gols": "Gols na partida",
    "gols_mandante": "Gols do mandante",
    "gols_visitante": "Gols do visitante",
    "total_gols_1t": "Gols no 1º tempo",
    "total_gols_2t": "Gols no 2º tempo",
    "total_amarelos": "Cartões amarelos na partida",
    "total_cartoes": "Cartões (amarelos + vermelhos) na partida",
}
VAR_QUALITATIVAS = {
    "resultado": ("Resultado da partida", ["Vitória do mandante", "Empate", "Vitória do visitante"]),
    "primeiro_gol": ("Quem marcou o 1º gol", ["Mandante", "Visitante", "Sem gols"]),
    "faixa_horario": ("Faixa de horário (ordinal)", ["Tarde (até 16h59)", "Fim de tarde (17h–18h59)", "Noite (19h ou mais)"]),
    "estado_mandante": ("Estado do mandante", None),
    "formacao_mandante": ("Formação tática do mandante", None),
}
VAR_CONTINUAS = {
    "minuto_jogo": "Minuto de cada gol (929 gols)",
    "minuto_primeiro_gol": "Minuto do 1º gol da partida",
    "gols_por_rodada": "Gols por rodada (38 rodadas)",
}
VAR_NUMERICAS = {**VAR_DISCRETAS, **VAR_CONTINUAS}

# ------------------------------------------------------------------
# Barra lateral
# ------------------------------------------------------------------
st.sidebar.title("⚽ Estatística no Brasileirão")
st.sidebar.caption("Trabalho de Probabilidade e Estatística · Série A 2024")
PAGINAS = [
    "🏠 1. Visão Geral & Conceitos Básicos",
    "👤 2. Consulta por Time & Jogador",
    "📋 3. Tabelas de Frequência",
    "📊 4. Gráficos & Medidas de Posição",
    "📏 5. Medidas de Dispersão & Boxplot",
    "🎲 6. Teoria das Probabilidades",
    "🌳 7. Probabilidade Total & Bayes",
    "🔢 8. V.A. Discretas",
    "📈 9. V.A. Contínuas",
]
pagina = st.sidebar.radio("Navegação:", PAGINAS)
st.sidebar.divider()
st.sidebar.info("**Fonte dos dados:** Brasileirao_Dataset (GitHub, gratuito). 380 partidas, 929 gols e "
                f"{len(df_cards)} cartões reais da temporada 2024, salvos em `data/` para uso offline.")

# =============================================================================
# 1. VISÃO GERAL E CONCEITOS BÁSICOS
# =============================================================================
if pagina == PAGINAS[0]:
    st.title("⚽ Estatística e Probabilidade no Brasileirão 2024")
    st.caption("Introdução à Estatística, conceitos básicos e apresentação da base de dados")
    ov = engine.overview()
    c = st.columns(5)
    c[0].metric("Partidas", ov["partidas"])
    c[1].metric("Clubes", ov["clubes"])
    c[2].metric("Gols", ov["gols"], f"{num(ov['media_gols'])} por jogo")
    c[3].metric("Amarelos por jogo", num(ov["media_amarelos"]))
    c[4].metric("Jogadores na base", ov["jogadores"], f"{ov['artilheiros']} marcaram")

    st.subheader("Conceitos básicos aplicados à base")
    col1, col2 = st.columns([1, 1.4])
    with col1:
        with st.container(border=True):
            st.markdown("""
- **Estatística:** coleta, organização, apresentação e análise de dados para tirar conclusões.
- **População:** todas as partidas do Campeonato Brasileiro Série A.
- **Amostra:** as **380 partidas da temporada 2024** (38 rodadas × 10 jogos).
- **Unidade amostral (elemento):** uma partida. Em algumas análises, o elemento é um **gol** ou um **jogador**.
- **Dado bruto × rol:** os registros como vieram da fonte × os mesmos valores ordenados.
- **Estatística descritiva:** resume os dados (tabelas, gráficos, medidas).
- **Probabilidade:** mede a chance de eventos em experimentos aleatórios.
""")
    with col2:
        with st.container(border=True):
            st.markdown("**Classificação das variáveis**")
            st.dataframe(engine.variable_catalog().drop(columns=["Coluna"]), hide_index=True, width="stretch")

    st.subheader("Classificação final do Brasileirão 2024")
    st.dataframe(engine.standings(), width="stretch", height=420)

    st.subheader("Amostra dos dados (primeiras partidas)")
    st.dataframe(df[["rodada", "data", "hora", "mandante", "visitante", "gols_mandante", "gols_visitante", "resultado",
                     "gols_1t_mandante", "gols_1t_visitante", "minuto_primeiro_gol", "total_amarelos", "arena"]].head(10),
                 hide_index=True, width="stretch")

# =============================================================================
# 2. CONSULTA POR TIME E JOGADOR
# =============================================================================
elif pagina == PAGINAS[1]:
    st.title("👤 Consulta por Time & Jogador")
    st.caption("Escolha um clube e um atleta para ver estatísticas descritivas e probabilidades")

    col_t, col_j = st.columns(2)
    with col_t:
        time = st.selectbox("Clube:", TIMES, index=TIMES.index("Botafogo"))
    perfil = engine.team_profile(time)
    c = st.columns(6)
    c[0].metric("Posição final", f"{perfil['posicao']}º")
    c[1].metric("Pontos", perfil["Pontos"], f"{num(perfil['aproveitamento'], 1)}% de aproveitamento")
    c[2].metric("V / E / D", f"{perfil['V']} / {perfil['E']} / {perfil['D']}")
    c[3].metric("Gols pró / contra", f"{perfil['Gols pró']} / {perfil['Gols contra']}")
    c[4].metric("Gols marcados por jogo", num(perfil["media_marcados"]), f"s = {num(perfil['desvio_marcados'])}")
    c[5].metric("Vitórias casa / fora", f"{perfil['vitorias_casa']} / {perfil['vitorias_fora']}")

    jogadores = engine.get_players_by_team(time)
    linhas_time = df_players[df_players["time"] == time].set_index("jogador")
    with col_j:
        jogador = st.selectbox("Jogador:", jogadores,
                               format_func=lambda j: f"{j} — {linhas_time.loc[j, 'gols']} gols, {linhas_time.loc[j, 'amarelos']} amarelos")
    st.caption(f"{len(jogadores)} atletas do {time} na base (todos que marcaram ou receberam cartão em 2024).")
    r = linhas_time.loc[jogador]

    st.divider()
    st.subheader(f"{jogador} · {time}")
    st.caption(f"Posição: {r['posicao']} (registrada nos cartões; “Não informada” para quem não recebeu cartão)")
    c = st.columns(5)
    c[0].metric("Gols", int(r["gols"]), f"{int(r['jogos_com_gol'])} jogos com gol")
    c[1].metric("Gols 1º / 2º tempo", f"{int(r['gols_1t'])} / {int(r['gols_2t'])}")
    c[2].metric("Pênaltis convertidos", int(r["penaltis"]))
    c[3].metric("Amarelos / vermelhos", f"{int(r['amarelos'])} / {int(r['vermelhos'])}")
    c[4].metric("Minuto médio dos gols", num(r["minuto_medio_gol"], 1) if r["gols"] > 0 else "—")

    abas = st.tabs(["📊 Estatística descritiva", "🎯 Poisson (próximo jogo)", "🎲 Binomial (próximos jogos)",
                    "🧠 Condicional & Bayes"])
    with abas[0]:
        if r["gols"] > 0:
            minutos = engine.player_goal_minutes(jogador, time)
            mostrar(viz.plot_player_minutes(minutos["minuto_jogo"], jogador))
            logs = df_logs[(df_logs["jogador"] == jogador) & (df_logs["time"] == time)]
            st.markdown("**Gols do atleta por partida do clube (38 jogos):** distribuição de frequências e medidas")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.dataframe(engine.frequency_table(logs["gols"]), hide_index=True, width="stretch")
            with col2:
                dm = engine.dispersion_measures(logs["gols"])
                st.markdown(f"""
- Média: **{num(dm['media'], 3)}** gol por jogo · Mediana: **{dm['mediana']:g}** · Moda: **{dm['moda']:g}**
- Desvio padrão: **{num(dm['desvio_padrao'], 3)}** · CV: **{num(dm['cv_pct'], 1)}%** ({dm['cv_classe'].lower()})
- Amplitude: **{dm['amplitude']:g}** gol(s)
""")
            st.markdown("**Lista dos gols**")
            st.dataframe(minutos[["rodada", "mandante", "gols_mandante", "gols_visitante", "visitante", "minuto", "tempo", "tipo_de_gol"]]
                         .rename(columns={"gols_mandante": "placar casa", "gols_visitante": "placar fora"}),
                         hide_index=True, width="stretch")
        else:
            st.info(f"{jogador} não marcou gols em 2024. Ele está na base pelos cartões recebidos "
                    f"({int(r['amarelos'])} amarelo(s) e {int(r['vermelhos'])} vermelho(s)).")
    with abas[1]:
        pp = engine.player_poisson(jogador, time)
        st.markdown("Supondo que os gols do atleta em cada partida sigam uma **distribuição de Poisson**, "
                    "com taxa estimada por λ = gols ÷ 38 jogos do clube:")
        st.latex(r"P(X = k) = \frac{e^{-\lambda}\lambda^k}{k!} \qquad P(X \geq 1) = 1 - e^{-\lambda}")
        c = st.columns(4)
        c[0].metric("λ (gols por jogo)", num(pp["lambda"], 3))
        c[1].metric("P(marcar no jogo)", pct(pp["p_marcar"]), f"observado: {pct(pp['p_marcar_obs'])}", delta_color="off")
        c[2].metric("P(marcar no 1º tempo)", pct(pp["p_marcar_1t"]))
        c[3].metric("P(marcar no 2º tempo)", pct(pp["p_marcar_2t"]))
        st.metric("P(receber cartão amarelo no jogo)", pct(pp["p_amarelo"]))
        if r["gols"] > 0:
            mostrar(viz.plot_player_poisson(pp["pmf"], jogador, pp["lambda"]))
    with abas[2]:
        n_jogos = st.slider("Número de próximos jogos (n):", 1, 10, 5)
        pb = engine.player_binomial(jogador, time, n_jogos)
        st.markdown(f"Cada jogo é um ensaio de Bernoulli: **sucesso = o atleta marca**. Com "
                    f"p = {int(r['jogos_com_gol'])}/38 = **{num(pb['p'] * 100, 1)}%**, o número de jogos com gol "
                    f"nos próximos {n_jogos} jogos é Y ~ Bin({n_jogos}; {num(pb['p'], 3)}).")
        st.latex(r"P(Y = k) = \binom{n}{k}\,p^k\,(1-p)^{n-k} \qquad E[Y] = np \qquad Var(Y) = np(1-p)")
        c = st.columns(4)
        c[0].metric("E[Y] = np", num(pb["esperanca"], 2))
        c[1].metric("Var(Y)", num(pb["variancia"], 3))
        c[2].metric("P(marcar em pelo menos 1 jogo)", pct(pb["p_pelo_menos_1"]))
        c[3].metric("P(não marcar em nenhum)", pct(pb["p_nenhum"]))
        if pb["p"] > 0:
            mostrar(viz.plot_binomial(pb, f"Y ~ Bin(n = {n_jogos}, p = {pb['p']:.3f}): jogos com gol de {jogador}"))
    with abas[3]:
        pbay = engine.player_bayes(jogador, time)
        if pbay is None:
            st.info("Sem gols na temporada: P(Vitória | marcou) não existe, porque o evento “marcou” tem probabilidade zero.")
        else:
            st.markdown(f"Eventos nas 38 partidas do {time}: **M** = {jogador} marcou; **V** = o {time} venceu.")
            st.latex(r"P(V \mid M) = \frac{P(V \cap M)}{P(M)} \qquad P(M \mid V) = \frac{P(V \mid M)\,P(M)}{P(V)}")
            c = st.columns(4)
            c[0].metric("P(M)", pct(pbay["p_m"]), f"{pbay['jogos_marcou']} de 38 jogos", delta_color="off")
            c[1].metric("P(V | M)", pct(pbay["p_v_dado_m"]), f"{pbay['vitorias_quando_marcou']} vitórias em {pbay['jogos_marcou']}", delta_color="off")
            c[2].metric("P(V | Mᶜ)", pct(pbay["p_v_dado_nao_m"]))
            c[3].metric("Bayes: P(M | V)", pct(pbay["p_m_dado_v"]))
            st.markdown(f"""
- Quando {jogador} marca, a chance de vitória do {time} muda em **{num(pbay['diferenca_pp'], 1)} pontos percentuais**.
- Probabilidade total: P(V) = P(V|M)·P(M) + P(V|Mᶜ)·P(Mᶜ) = **{pct(pbay['p_v_total'])}** (igual à frequência direta, {pct(pbay['p_v'])}).
- Bayes: entre as vitórias do {time}, **{pct(pbay['p_m_dado_v'])}** tiveram gol de {jogador}.
""")
            mostrar(viz.plot_probability_bars(["P(V)", "P(V | M)", "P(V | Mᶜ)", "P(M | V)"],
                                              [pbay["p_v"], pbay["p_v_dado_m"], pbay["p_v_dado_nao_m"], pbay["p_m_dado_v"]],
                                              f"Vitória do {time} e gols de {jogador}", destaque=1))

# =============================================================================
# 3. TABELAS DE DISTRIBUIÇÃO DE FREQUÊNCIAS
# =============================================================================
elif pagina == PAGINAS[2]:
    st.title("📋 Tabelas de Distribuição de Frequências")
    st.caption("Frequência absoluta (fi), relativa (fri), acumulada (Fi) e relativa acumulada (Fri)")
    st.latex(r"f_{ri} = \frac{f_i}{n} \qquad F_i = \sum_{j \leq i} f_j \qquad F_{ri} = \frac{F_i}{n}")

    abas = st.tabs(["Variável qualitativa", "Variável quantitativa discreta", "Variável contínua (dados em classes)"])
    with abas[0]:
        var = st.selectbox("Variável:", list(VAR_QUALITATIVAS), format_func=lambda v: VAR_QUALITATIVAS[v][0])
        nome, ordem = VAR_QUALITATIVAS[var]
        tab = engine.frequency_table(df[var], ordem)
        if ordem is None:
            tab = tab.sort_values("fi", ascending=False).reset_index(drop=True)
            tab["Fi"] = tab["fi"].cumsum()
            tab["Fri (%)"] = (tab["Fi"] / tab["fi"].sum() * 100).round(2)
        col1, col2 = st.columns([1.1, 1])
        with col1:
            st.dataframe(tab, hide_index=True, width="stretch")
            st.caption(f"n = {int(tab['fi'].sum())} partidas. Para variáveis nominais, Fi só faz sentido após escolher uma ordem.")
        with col2:
            mostrar(viz.plot_bar_frequency(tab.head(12), nome, nome, horizontal=len(tab) > 6))
    with abas[1]:
        var = st.selectbox("Variável:", list(VAR_DISCRETAS), format_func=VAR_DISCRETAS.get, key="disc_freq")
        tab = engine.frequency_table(df[var])
        col1, col2 = st.columns([1.1, 1])
        with col1:
            st.dataframe(tab, hide_index=True, width="stretch")
        with col2:
            mostrar(viz.plot_bar_frequency(tab, VAR_DISCRETAS[var], VAR_DISCRETAS[var]))
        ate2 = tab.loc[tab["Valor"].astype(int) <= 2, "fi"].sum() / tab["fi"].sum() * 100
        st.info(f"Leitura da tabela: o valor mais frequente é **{tab.loc[tab['fi'].idxmax(), 'Valor']}** "
                f"({pct(tab['fri (%)'].max())} das partidas). Pela coluna Fri, **{pct(ate2)}** das partidas tiveram até 2.")
    with abas[2]:
        var = st.selectbox("Variável:", list(VAR_CONTINUAS), format_func=VAR_CONTINUAS.get, key="cont_freq")
        info = engine.class_frequency_table(engine.series(var))
        c = st.columns(5)
        c[0].metric("n", info["n"])
        c[1].metric("Mínimo · Máximo", f"{info['minimo']:g} · {info['maximo']:g}")
        c[2].metric("Amplitude total (AT)", f"{info['amplitude_total']:g}")
        c[3].metric("Sturges: k = 1 + 3,322·log n", f"{num(info['k_sturges'])} → {info['k']} classes")
        c[4].metric("Amplitude de classe h", f"{info['h']:g}")
        st.dataframe(info["tabela"], hide_index=True, width="stretch")
        st.caption("Classes no formato li ⊢ ls: inclui o limite inferior e exclui o superior. xi é o ponto médio (li + ls)/2.")
        if var == "minuto_jogo":
            st.info("As classes que contêm os minutos 45 e 90 concentram mais gols porque a fonte registra os "
                    "acréscimos como 45+x e 90+x.")

# =============================================================================
# 4. REPRESENTAÇÃO GRÁFICA E MEDIDAS DE POSIÇÃO
# =============================================================================
elif pagina == PAGINAS[3]:
    st.title("📊 Representação Gráfica & Medidas de Posição")
    abas = st.tabs(["Representação gráfica", "Média, mediana e moda", "Dados agrupados em classes"])
    with abas[0]:
        tipo = st.radio("Tipo de gráfico:", ["Colunas", "Setores (pizza)", "Histograma + polígono", "Ogiva", "Linha (rodadas)"],
                        horizontal=True)
        if tipo in ("Colunas", "Setores (pizza)"):
            opcoes = {**{k: v[0] for k, v in VAR_QUALITATIVAS.items() if k != "formacao_mandante"}, **VAR_DISCRETAS}
            var = st.selectbox("Variável:", list(opcoes), format_func=opcoes.get)
            ordem = VAR_QUALITATIVAS[var][1] if var in VAR_QUALITATIVAS else None
            tab = engine.frequency_table(df[var], ordem)
            if tipo == "Colunas":
                mostrar(viz.plot_bar_frequency(tab, f"Gráfico de colunas: {opcoes[var]}", opcoes[var]))
            else:
                if len(tab) > 8:
                    tab = tab.sort_values("fi", ascending=False).head(8)
                    st.caption("Mostrando as 8 categorias mais frequentes.")
                mostrar(viz.plot_pie(tab, f"Gráfico de setores: {opcoes[var]}"))
            st.caption("Colunas e setores servem para variáveis qualitativas e discretas com poucos valores.")
        elif tipo in ("Histograma + polígono", "Ogiva"):
            var = st.selectbox("Variável contínua:", list(VAR_CONTINUAS), format_func=VAR_CONTINUAS.get)
            info = engine.class_frequency_table(engine.series(var))
            if tipo == "Histograma + polígono":
                mostrar(viz.plot_histogram_classes(info, f"Histograma e polígono de frequências: {VAR_CONTINUAS[var]}", VAR_CONTINUAS[var]))
                st.caption("No histograma as barras ficam juntas, porque as classes são intervalos contínuos. "
                           "O polígono liga os pontos médios das classes.")
            else:
                mostrar(viz.plot_ogive(info, f"Ogiva (Fri): {VAR_CONTINUAS[var]}", VAR_CONTINUAS[var]))
                st.caption("A ogiva mostra a frequência acumulada; onde ela cruza 50% está a mediana.")
        else:
            mostrar(viz.plot_goals_per_round())
            st.caption("O gráfico de linha mostra a evolução de uma variável ao longo do tempo (rodadas).")
    with abas[1]:
        var = st.selectbox("Variável:", list(VAR_NUMERICAS), format_func=VAR_NUMERICAS.get, key="pos_var")
        serie = engine.series(var)
        m = engine.position_measures(serie)
        st.latex(r"\bar{x} = \frac{\sum x_i}{n} \qquad Md = \text{valor central do rol} \qquad Mo = \text{valor de maior frequência}")
        c = st.columns(4)
        c[0].metric("n", m["n"])
        c[1].metric("Média x̄", num(m["media"], 3), f"soma = {m['soma']:g}", delta_color="off")
        c[2].metric("Mediana Md", f"{m['mediana']:g}")
        c[3].metric("Moda Mo", " e ".join(f"{v:g}" for v in m["moda"]), "multimodal" if m["multimodal"] else None, delta_color="off")
        discreta = var in VAR_DISCRETAS
        mostrar(viz.plot_position_measures(serie, m, f"Medidas de posição: {VAR_NUMERICAS[var]}", VAR_NUMERICAS[var], discreta))
        if m["media"] > m["mediana"]:
            st.info("A média é maior que a mediana: valores altos (cauda à direita) puxam a média para cima.")
        elif m["media"] < m["mediana"]:
            st.info("A média é menor que a mediana: valores baixos (cauda à esquerda) puxam a média para baixo.")
    with abas[2]:
        var = st.selectbox("Variável contínua:", list(VAR_CONTINUAS), format_func=VAR_CONTINUAS.get, key="grp_var")
        serie = engine.series(var)
        g = engine.grouped_measures(serie)
        bruto = engine.position_measures(serie)
        st.latex(r"\bar{x} = \frac{\sum x_i f_i}{n} \qquad Md = l_i + \frac{\frac{n}{2} - F_{ant}}{f_i}\cdot h \qquad "
                 r"Mo_{Czuber} = l_i + \frac{d_1}{d_1 + d_2}\cdot h")
        c = st.columns(3)
        c[0].metric("Média agrupada", num(g["media_agrupada"]), f"dados brutos: {num(bruto['media'])}", delta_color="off")
        c[1].metric("Mediana agrupada", num(g["mediana_agrupada"]), f"classe {g['classe_mediana']}", delta_color="off")
        c[2].metric("Moda de Czuber", num(g["moda_czuber"]), f"classe modal {g['classe_modal']}", delta_color="off")
        st.dataframe(engine.class_frequency_table(serie)["tabela"], hide_index=True, width="stretch")
        st.caption("Agrupar em classes perde um pouco de informação: por isso a média agrupada difere levemente da média dos dados brutos.")

# =============================================================================
# 5. MEDIDAS DE DISPERSÃO, QUARTIS, ASSIMETRIA, OUTLIERS E BOXPLOT
# =============================================================================
elif pagina == PAGINAS[4]:
    st.title("📏 Medidas de Dispersão, Quartis e Boxplot")
    abas = st.tabs(["Uma variável", "Comparação entre clubes"])
    with abas[0]:
        var = st.selectbox("Variável:", list(VAR_NUMERICAS), format_func=VAR_NUMERICAS.get)
        serie = engine.series(var)
        d = engine.dispersion_measures(serie)
        st.latex(r"AT = x_{máx} - x_{mín} \qquad s^2 = \frac{\sum (x_i - \bar{x})^2}{n-1} \qquad s = \sqrt{s^2} \qquad "
                 r"CV = \frac{s}{\bar{x}}\cdot 100\%")
        c = st.columns(5)
        c[0].metric("Amplitude total", f"{d['amplitude']:g}", f"{d['minimo']:g} a {d['maximo']:g}", delta_color="off")
        c[1].metric("Variância amostral s²", num(d["variancia_amostral"], 3), f"σ² (populacional) = {num(d['variancia_populacional'], 3)}", delta_color="off")
        c[2].metric("Desvio padrão s", num(d["desvio_padrao"], 3))
        c[3].metric("Coef. de variação", f"{num(d['cv_pct'], 1)}%", d["cv_classe"], delta_color="off")
        c[4].metric("Média x̄", num(d["media"], 3))
        st.subheader("Quartis, diferença interquartílica e valores extremos")
        st.latex(r"DIQ = Q_3 - Q_1 \qquad \text{outlier: } x < Q_1 - 1{,}5\cdot DIQ \;\text{ ou }\; x > Q_3 + 1{,}5\cdot DIQ")
        c = st.columns(5)
        c[0].metric("Q1 (25%)", f"{d['q1']:g}")
        c[1].metric("Q2 = mediana", f"{d['q2']:g}")
        c[2].metric("Q3 (75%)", f"{d['q3']:g}")
        c[3].metric("DIQ", f"{d['diq']:g}")
        c[4].metric("Outliers", d["qtd_outliers"], f"limites [{d['limite_inferior']:g}; {d['limite_superior']:g}]", delta_color="off")
        mostrar(viz.plot_boxplot(serie, d, f"Boxplot: {VAR_NUMERICAS[var]}", VAR_NUMERICAS[var]))
        if d["qtd_outliers"]:
            st.caption(f"Valores extremos encontrados: {', '.join(f'{v:g}' for v in d['valores_outliers'])}.")
        st.subheader("Simetria e assimetria")
        st.latex(r"As_1 = \frac{\bar{x} - Mo}{s} \qquad As_2 = \frac{3(\bar{x} - Md)}{s}")
        c = st.columns(3)
        c[0].metric("1º coef. de Pearson (moda)", num(d["assimetria_pearson_moda"], 3))
        c[1].metric("2º coef. de Pearson (mediana)", num(d["assimetria_pearson_mediana"], 3))
        c[2].metric("Classificação", d["classe_assimetria"].split(",")[0])
        st.caption("Critério: |As| < 0,15 simétrica; 0,15 ≤ |As| ≤ 1 assimetria moderada; |As| > 1 forte. "
                   "Em variáveis discretas com poucos valores a moda é instável, por isso a classificação usa o coeficiente com a mediana.")
    with abas[1]:
        st.markdown("**Gols marcados por jogo, por clube (38 jogos cada)**")
        tabela = engine.team_dispersion_table()
        mostrar(viz.plot_boxplot_teams(engine.team_goals_per_match_long(), tabela["Clube"].tolist()))
        st.dataframe(tabela, hide_index=True, width="stretch")
        mais, menos = tabela.loc[tabela["CV (%)"].idxmin()], tabela.loc[tabela["CV (%)"].idxmax()]
        st.info(f"Ataque mais regular (menor CV): **{mais['Clube']}** ({num(mais['CV (%)'], 1)}%). "
                f"Mais irregular (maior CV): **{menos['Clube']}** ({num(menos['CV (%)'], 1)}%).")

# =============================================================================
# 6. TEORIA DAS PROBABILIDADES
# =============================================================================
elif pagina == PAGINAS[5]:
    st.title("🎲 Teoria das Probabilidades")
    abas = st.tabs(["Experimento, espaço amostral e evento", "Axiomas, união e interseção", "Condicional e independência"])
    with abas[0]:
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("""
**Experimento determinístico:** repetido nas mesmas condições, dá sempre o mesmo resultado.
*Ex.: a duração regulamentar de uma partida é 90 minutos; 3 pontos por vitória.*

**Experimento aleatório:** o resultado não pode ser previsto com certeza, mas conhecemos os resultados possíveis.
*Ex.: o resultado, o placar ou o número de cartões de uma partida.*
""")
        with col2:
            with st.container(border=True):
                st.markdown("""
**Espaço amostral (S):** conjunto de todos os resultados possíveis.
*S = {Vitória do mandante, Empate, Vitória do visitante}*

**Evento:** qualquer subconjunto de S.
*A = {Vitória do mandante}; B = {o mandante não perde} = {Vitória do mandante, Empate}*
""")
        s = engine.sample_space_results()
        st.latex(r"\text{Clássica: } P(E) = \frac{n(E)}{n(S)} \qquad \text{Frequentista: } P(E) \approx \frac{\text{nº de ocorrências de } E}{\text{nº de partidas}}")
        col1, col2 = st.columns([1, 1.2])
        with col1:
            st.dataframe(pd.DataFrame(s["resultados"]), hide_index=True, width="stretch")
            st.markdown(f"""
- A probabilidade clássica supõe resultados **equiprováveis** (1/3 cada).
- Os dados mostram que isso não vale: o mandante vence **{pct(s['resultados'][0]['Frequentista (%)'])}** das vezes (vantagem de jogar em casa).
- O espaço amostral dos **placares** teve **{s['placares_distintos']}** resultados distintos em 2024.
""")
        with col2:
            mostrar(viz.plot_classical_vs_frequentist(s["resultados"]))
        st.markdown("**Placares mais frequentes (probabilidade frequentista)**")
        c = st.columns(3)
        c[0].metric("P(0 x 0)", pct(s["p_0x0"]))
        c[1].metric("P(3 ou mais gols)", pct(s["p_mais_2_gols"]))
        c[2].metric("P(até 2 gols)", pct(s["p_ate_2_gols"]), "evento complementar", delta_color="off")
        st.dataframe(pd.DataFrame(s["placares_mais_comuns"]), hide_index=True, width="stretch")
    with abas[1]:
        ea = engine.events_and_axioms()
        st.subheader("Axiomas")
        st.latex(r"\text{(1) } 0 \leq P(A) \leq 1 \qquad \text{(2) } P(S) = 1 \qquad \text{(3) } A \cap B = \varnothing \Rightarrow P(A \cup B) = P(A) + P(B)")
        c = st.columns(4)
        c[0].metric("P(Vitória do mandante)", pct(ea["p_vitoria_mandante"]))
        c[1].metric("P(Empate)", pct(ea["p_empate"]))
        c[2].metric("P(Vitória do visitante)", pct(ea["p_vitoria_visitante"]))
        c[3].metric("Soma = P(S)", pct(ea["soma_espaco_amostral"]), "axioma 2", delta_color="off")
        st.markdown(f"""
**Eventos mutuamente exclusivos:** “vitória do mandante” e “empate” não ocorrem juntos (A ∩ B = ∅). Pelo axioma 3:
P(mandante não perde) = P(vitória) + P(empate) = {pct(ea['p_vitoria_mandante'])} + {pct(ea['p_empate'])} = **{pct(ea['p_mandante_nao_perde'])}**.

**Teorema do complementar:** P(Aᶜ) = 1 − P(A) → P(o mandante **não** vence) = 1 − {pct(ea['p_vitoria_mandante'])} = **{pct(ea['p_complementar_vitoria_mandante'])}**.
""")
        st.subheader("União e interseção (eventos não exclusivos)")
        st.latex(r"P(A \cup B) = P(A) + P(B) - P(A \cap B)")
        st.markdown("A = **o mandante marca no 1º tempo**; B = **o visitante marca no 1º tempo**. Os dois podem acontecer no mesmo jogo.")
        col1, col2 = st.columns([1, 1.1])
        with col1:
            c = st.columns(2)
            c[0].metric("P(A)", pct(ea["p_m1"]), f"{ea['n_m1']} jogos", delta_color="off")
            c[1].metric("P(B)", pct(ea["p_v1"]), f"{ea['n_v1']} jogos", delta_color="off")
            c = st.columns(2)
            c[0].metric("P(A ∩ B)", pct(ea["p_m1_inter_v1"]), f"{ea['n_m1_v1']} jogos", delta_color="off")
            c[1].metric("P(A ∪ B)", pct(ea["p_m1_uniao_v1_formula"]), "pela fórmula", delta_color="off")
            st.markdown(f"""
- P(A ∪ B) = {pct(ea['p_m1'])} + {pct(ea['p_v1'])} − {pct(ea['p_m1_inter_v1'])} = **{pct(ea['p_m1_uniao_v1_formula'])}**
- Contagem direta de jogos com gol no 1º tempo: **{pct(ea['p_m1_uniao_v1_direta'])}** ✔
- Complementar: P(1º tempo sem gols) = **{pct(ea['p_nenhum_gol_1t'])}**
""")
        with col2:
            mostrar(viz.plot_venn(ea["n_m1"], ea["n_v1"], ea["n_m1_v1"], ea["n"], "A: mandante marca no 1T",
                                  "B: visitante marca no 1T", "Diagrama de Venn (nº de jogos)"))
    with abas[2]:
        ci = engine.conditional_and_independence()
        st.subheader("Probabilidade condicional")
        st.latex(r"P(A \mid B) = \frac{P(A \cap B)}{P(B)} \qquad P(A \cap B) = P(B)\cdot P(A \mid B)")
        st.markdown("A = **vitória do mandante**; B = **o mandante marcou o primeiro gol**.")
        c = st.columns(4)
        c[0].metric("P(A)", pct(ci["p_a"]), f"{ci['n_a']} jogos", delta_color="off")
        c[1].metric("P(B)", pct(ci["p_b"]), f"{ci['n_b']} jogos", delta_color="off")
        c[2].metric("P(A ∩ B)", pct(ci["p_a_inter_b"]), f"{ci['n_a_inter_b']} jogos", delta_color="off")
        c[3].metric("P(A | B)", pct(ci["p_a_dado_b"]), f"{ci['n_a_inter_b']}/{ci['n_b']}", delta_color="off")
        mostrar(viz.plot_probability_bars(["P(A)", "P(B)", "P(A ∩ B)", "P(A | B)"],
                                          [ci["p_a"], ci["p_b"], ci["p_a_inter_b"], ci["p_a_dado_b"]],
                                          "Saber que o mandante marcou primeiro muda a chance de vitória", destaque=3))
        st.info(f"Sem informação, o mandante vence {pct(ci['p_a'])} das vezes. Sabendo que ele marcou primeiro, a chance sobe para "
                f"**{pct(ci['p_a_dado_b'])}**. Como P(A | B) ≠ P(A), A e B **não são independentes**.")
        st.subheader("Eventos independentes")
        st.latex(r"A \text{ e } B \text{ independentes} \iff P(A \cap B) = P(A)\cdot P(B) \iff P(A \mid B) = P(A)")
        st.markdown("C = **o mandante marca no 1º tempo**; D = **o mandante marca no 2º tempo**.")
        c = st.columns(4)
        c[0].metric("P(D)", pct(ci["p_m2"]))
        c[1].metric("P(D | C)", pct(ci["p_m2_dado_m1"]))
        c[2].metric("P(C ∩ D)", pct(ci["p_m1_inter_m2"]))
        c[3].metric("P(C)·P(D)", pct(ci["produto_m1_m2"]))
        conclusao = "aproximadamente independentes" if ci["m1_m2_independentes"] else "dependentes"
        st.info(f"P(C ∩ D) e P(C)·P(D) diferem só {num(ci['diferenca_independencia'], 2)} p.p.: marcar no 1º tempo quase não muda a chance "
                f"de o mandante marcar no 2º. Os eventos são **{conclusao}**. Já A e B do item anterior: "
                f"P(A ∩ B) = {pct(ci['p_a_inter_b'])} contra P(A)·P(B) = {pct(ci['produto_pa_pb'])} → dependentes.")

# =============================================================================
# 7. TEOREMA DA PROBABILIDADE TOTAL E TEOREMA DE BAYES
# =============================================================================
elif pagina == PAGINAS[6]:
    st.title("🌳 Teorema da Probabilidade Total & Teorema de Bayes")
    escolha = st.selectbox("Analisar:", ["Todos os mandantes"] + TIMES)
    res = engine.total_probability_bayes(None if escolha == "Todos os mandantes" else escolha)
    st.markdown(f"""
O primeiro gol divide o espaço amostral em **três eventos mutuamente exclusivos e exaustivos** (uma partição):
**B1**, **B2** e **B3**. Queremos A = **{res['evento']}** ({res['n']} partidas).
""")
    col1, col2 = st.columns(2)
    with col1:
        st.latex(r"P(A) = \sum_{i=1}^{3} P(A \mid B_i)\,P(B_i)")
    with col2:
        st.latex(r"P(B_i \mid A) = \frac{P(A \mid B_i)\,P(B_i)}{P(A)}")
    mostrar(viz.plot_probability_tree(res))
    st.dataframe(pd.DataFrame(res["linhas"]), hide_index=True, width="stretch")
    l1, l2, l3 = res["linhas"]
    st.markdown(f"""
**Probabilidade total:** P(A) = {pct(l1['P(A|Bi) (%)'])}·{pct(l1['P(Bi) (%)'])} + {pct(l2['P(A|Bi) (%)'])}·{pct(l2['P(Bi) (%)'])} + {pct(l3['P(A|Bi) (%)'])}·{pct(l3['P(Bi) (%)'])} = **{pct(res['p_a_total'])}**
(conferência direta: {res['vitorias']} vitórias em {res['n']} jogos = {pct(res['p_a_direta'])} ✔)

**Bayes:** dado que houve vitória, a probabilidade de ter marcado primeiro é **{pct(l1['Bayes P(Bi|A) (%)'])}**;
de ter virado o jogo (adversário marcou primeiro) é **{pct(l2['Bayes P(Bi|A) (%)'])}**.
""")
    st.info("A Probabilidade Total vai “das causas para o efeito” (soma os ramos da árvore). "
            "Bayes faz o caminho inverso: sabendo o efeito (vitória), atualiza a probabilidade de cada causa.")

# =============================================================================
# 8. VARIÁVEIS ALEATÓRIAS DISCRETAS
# =============================================================================
elif pagina == PAGINAS[7]:
    st.title("🔢 Variáveis Aleatórias e Distribuições Discretas")
    abas = st.tabs(["Variável aleatória discreta", "Bernoulli e Binomial", "Poisson", "Simulador de partida (Poisson)"])
    with abas[0]:
        var = st.selectbox("Variável aleatória X:", list(VAR_DISCRETAS), format_func=VAR_DISCRETAS.get)
        va = engine.discrete_random_variable(var)
        st.latex(r"p(x) = P(X = x) \qquad F(x) = P(X \leq x) \qquad E[X] = \sum x\,p(x) \qquad Var(X) = E[X^2] - (E[X])^2")
        c = st.columns(4)
        c[0].metric("E[X]", num(va["esperanca"], 4))
        c[1].metric("E[X²]", num(va["e_x2"], 4))
        c[2].metric("Var(X)", num(va["variancia"], 4))
        c[3].metric("Desvio padrão σ", num(va["desvio"], 4))
        mostrar(viz.plot_pmf_cdf(va["tabela"], VAR_DISCRETAS[var]))
        st.dataframe(va["tabela"], hide_index=True, width="stretch")
        st.caption("As probabilidades vêm das frequências relativas de 2024. A soma de P(X = x) é 1 e F(x) termina em 1.")
    with abas[1]:
        time_b = st.selectbox("Clube mandante (ou todos):", ["Todos os mandantes"] + TIMES)
        be = engine.bernoulli_home_win(None if time_b == "Todos os mandantes" else time_b)
        st.markdown("**Bernoulli:** uma partida em casa, X = 1 se o mandante vence e X = 0 caso contrário.")
        st.latex(r"P(X = 1) = p \qquad P(X = 0) = 1 - p \qquad E[X] = p \qquad Var(X) = p(1-p)")
        c = st.columns(4)
        c[0].metric("p (vitória em casa)", pct(be["p"] * 100), f"{be['vitorias']} de {be['jogos']} jogos", delta_color="off")
        c[1].metric("q = 1 − p", pct(be["q"] * 100))
        c[2].metric("E[X]", num(be["esperanca"], 4))
        c[3].metric("Var(X)", num(be["variancia"], 4))
        st.markdown("**Binomial:** n partidas independentes em casa, cada uma com a mesma probabilidade p de vitória.")
        n_bin = st.slider("Número de jogos em casa (n):", 1, 19, 5)
        bi = engine.binomial(n_bin, be["p"])
        st.latex(r"X \sim Bin(n, p): \quad P(X = k) = \binom{n}{k} p^k (1-p)^{n-k} \qquad E[X] = np \qquad Var(X) = np(1-p)")
        c = st.columns(4)
        c[0].metric("E[X] = np", num(bi["esperanca"], 2))
        c[1].metric("Var(X)", num(bi["variancia"], 3))
        c[2].metric("P(pelo menos 1 vitória)", pct(bi["p_pelo_menos_1"]))
        c[3].metric("P(vencer mais da metade)", pct(bi["p_mais_da_metade"]))
        mostrar(viz.plot_binomial(bi, f"X ~ Bin(n = {n_bin}, p = {be['p']:.3f}): vitórias de {time_b.lower() if time_b.startswith('Todos') else time_b} em casa"))
        st.dataframe(bi["tabela"], hide_index=True, width="stretch")
    with abas[2]:
        var_p = st.selectbox("Contagem X:", ["total_gols", "gols_mandante", "gols_visitante", "total_amarelos"],
                             format_func=VAR_DISCRETAS.get)
        pf = engine.poisson_fit(var_p)
        st.latex(r"X \sim Poisson(\lambda): \quad P(X = k) = \frac{e^{-\lambda}\lambda^k}{k!} \qquad E[X] = Var(X) = \lambda")
        c = st.columns(4)
        c[0].metric("λ (média amostral)", num(pf["lambda"], 3))
        c[1].metric("Variância amostral", num(pf["variancia_amostral"], 3), "na Poisson, Var = λ", delta_color="off")
        c[2].metric("P(X = 0) Poisson", pct(pf["p_zero"]), f"observado {pct(pf['p_zero_obs'])}", delta_color="off")
        c[3].metric("P(X ≥ 3) Poisson", pct(pf["p_3_ou_mais"]), f"observado {pct(pf['p_3_ou_mais_obs'])}", delta_color="off")
        mostrar(viz.plot_poisson_fit(pf, VAR_DISCRETAS[var_p]))
        st.dataframe(pf["tabela"], hide_index=True, width="stretch")
        st.subheader("Gols do mandante por tempo")
        time_p = st.selectbox("Mandante:", ["Todos os mandantes"] + TIMES, key="poisson_tempo")
        ph = engine.poisson_by_half(None if time_p == "Todos os mandantes" else time_p)
        c = st.columns(3)
        c[0].metric(f"P(gol no 1º tempo) · λ = {num(ph['lambda_1t'], 3)}", pct(ph["p_gol_1t"]), f"observado {pct(ph['p_gol_1t_obs'])}", delta_color="off")
        c[1].metric(f"P(gol no 2º tempo) · λ = {num(ph['lambda_2t'], 3)}", pct(ph["p_gol_2t"]), f"observado {pct(ph['p_gol_2t_obs'])}", delta_color="off")
        c[2].metric(f"P(gol no jogo) · λ = {num(ph['lambda_total'], 3)}", pct(ph["p_gol_jogo"]), f"observado {pct(ph['p_gol_jogo_obs'])}", delta_color="off")
        st.caption("Soma de Poissons independentes: X₁ + X₂ ~ Poisson(λ₁ + λ₂). Por isso o λ do jogo é a soma dos λ de cada tempo.")
    with abas[3]:
        c1, c2 = st.columns(2)
        with c1:
            casa = st.selectbox("Mandante:", TIMES, index=TIMES.index("Botafogo"))
        with c2:
            fora = st.selectbox("Visitante:", TIMES, index=TIMES.index("Palmeiras"))
        sim = engine.simulate_match(casa, fora)
        st.latex(r"P(\text{placar } i \times j) = P(H = i)\cdot P(V = j) = \frac{e^{-\lambda_H}\lambda_H^{\,i}}{i!}\cdot\frac{e^{-\lambda_V}\lambda_V^{\,j}}{j!}")
        st.caption(f"λ do mandante = média entre os gols que o {casa} marca em casa e os que o {fora} sofre fora "
                   f"= {num(sim['lambda_mandante'], 3)}. λ do visitante = {num(sim['lambda_visitante'], 3)}.")
        c = st.columns(4)
        c[0].metric(f"Vitória do {casa}", pct(sim["p_vitoria_mandante"]))
        c[1].metric("Empate", pct(sim["p_empate"]))
        c[2].metric(f"Vitória do {fora}", pct(sim["p_vitoria_visitante"]))
        c[3].metric("Placar mais provável", sim["placar_provavel"], pct(sim["p_placar_provavel"]), delta_color="off")
        mostrar(viz.plot_score_matrix(sim["matriz"], casa, fora))
        st.caption(f"P(3 ou mais gols) = {pct(sim['p_mais_2_gols'])}. A matriz vai até 7 × 7 gols e cobre {pct(sim['massa_total'])} da probabilidade.")

# =============================================================================
# 9. VARIÁVEIS ALEATÓRIAS CONTÍNUAS
# =============================================================================
elif pagina == PAGINAS[8]:
    st.title("📈 Distribuições de Variáveis Aleatórias Contínuas")
    st.caption("Para variáveis contínuas, P(X = x) = 0: só intervalos têm probabilidade, dada pela área sob a densidade f(x).")
    abas = st.tabs(["Uniforme", "Exponencial", "Normal"])
    with abas[0]:
        u = engine.uniform_goal_minute()
        st.markdown("X = **minuto em que sai um gol**. Se os gols fossem igualmente prováveis em qualquer minuto, X ~ U(0, 90).")
        st.latex(r"f(x) = \frac{1}{b-a} \qquad P(c \leq X \leq d) = \frac{d-c}{b-a} \qquad E[X] = \frac{a+b}{2} \qquad Var(X) = \frac{(b-a)^2}{12}")
        c = st.columns(4)
        c[0].metric("f(x) = 1/90", num(u["densidade"], 5))
        c[1].metric("E[X] teórico", num(u["esperanca"], 1), f"média observada {num(u['media_obs'])}", delta_color="off")
        c[2].metric("σ teórico", num(u["desvio"]), f"observado {num(u['desvio_obs'])}", delta_color="off")
        c[3].metric("Gols analisados", u["n"])
        mostrar(viz.plot_uniform(df_goals["minuto_jogo"]))
        st.dataframe(u["tabela"], hide_index=True, width="stretch")
        st.info("A Uniforme dá 16,67% para cada faixa de 15 minutos. Os dados mostram menos gols no início e mais no fim "
                "de cada tempo (cansaço e acréscimos), então a Uniforme é só uma primeira aproximação.")
    with abas[1]:
        ex = engine.exponential_first_goal()
        st.markdown(f"T = **minuto do primeiro gol** ({ex['n']} partidas; {ex['sem_gols']} terminaram 0 x 0 e ficam de fora).")
        st.latex(r"f(t) = \lambda e^{-\lambda t} \qquad F(t) = P(T \leq t) = 1 - e^{-\lambda t} \qquad E[T] = \frac{1}{\lambda} \qquad Var(T) = \frac{1}{\lambda^2}")
        c = st.columns(4)
        c[0].metric("Tempo médio até o 1º gol", f"{num(ex['media'], 1)} min")
        c[1].metric("λ = 1 / média", num(ex["lambda"], 5))
        c[2].metric("σ teórico = 1/λ", f"{num(ex['desvio_teorico'], 1)} min", f"observado {num(ex['desvio_obs'], 1)}", delta_color="off")
        c[3].metric("P(sem gol até os 45’)", pct(ex["p_sem_gol_ate_45"]))
        mostrar(viz.plot_exponential(df["minuto_primeiro_gol"].dropna(), ex["lambda"]))
        st.dataframe(ex["tabela"], hide_index=True, width="stretch")
        st.markdown(f"**Falta de memória:** P(T > 45 | T > 30) = P(T > 15) = **{pct(ex['p_mais_15_dado_mais_30'])}**. "
                    "Se o jogo está 0 x 0 aos 30 minutos, a chance de continuar sem gols por mais 15 minutos é a mesma do início.")
    with abas[2]:
        st.markdown("X = **total de gols de uma rodada** (soma das 10 partidas). Somas de muitas parcelas tendem a uma distribuição Normal.")
        c1, c2 = st.columns(2)
        with c1:
            x1 = st.slider("Limite inferior x₁:", 10, 35, 20)
        with c2:
            x2 = st.slider("Limite superior x₂:", 15, 40, 30)
        if x2 <= x1:
            st.warning("Escolha x₂ maior que x₁.")
        else:
            nr = engine.normal_goals_per_round(x1, x2)
            st.latex(r"f(x) = \frac{1}{\sigma\sqrt{2\pi}}\,e^{-\frac{(x-\mu)^2}{2\sigma^2}} \qquad Z = \frac{X - \mu}{\sigma} \sim N(0, 1)")
            c = st.columns(4)
            c[0].metric("μ (média)", num(nr["mu"]))
            c[1].metric("σ (desvio padrão)", num(nr["sigma"]))
            c[2].metric(f"P({x1} ≤ X ≤ {x2})", pct(nr["p_entre"]), f"observado {pct(nr['p_entre_obs'])}", delta_color="off")
            c[3].metric(f"P(X > {x2})", pct(nr["p_acima_x2"]), f"observado {pct(nr['p_acima_x2_obs'])}", delta_color="off")
            st.markdown(f"Padronização: z₁ = ({x1} − {num(nr['mu'])}) / {num(nr['sigma'])} = **{num(nr['z1'], 3)}**; "
                        f"z₂ = ({x2} − {num(nr['mu'])}) / {num(nr['sigma'])} = **{num(nr['z2'], 3)}**. "
                        "A probabilidade é Φ(z₂) − Φ(z₁), consultada na tabela da Normal padrão.")
            mostrar(viz.plot_normal(nr["valores"], nr["mu"], nr["sigma"], x1, x2))
            regra = pd.DataFrame(nr["regra_empirica"]).T
            regra.index.name = "Intervalo"
            st.dataframe(regra, width="stretch")
            st.caption(f"Rodada com mais gols: {nr['rodada_max']}ª ({nr['gols_max']} gols, z = {num(nr['z_max'])}). "
                       f"Com menos: {nr['rodada_min']}ª ({nr['gols_min']} gols, z = {num(nr['z_min'])}).")
