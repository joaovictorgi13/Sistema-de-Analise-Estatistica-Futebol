# ⚽ Estatística e Probabilidade no Brasileirão 2024

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.50%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-Estatística-0054A6?style=for-the-badge&logo=scipy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Análise_de_Dados-150458?style=for-the-badge&logo=pandas&logoColor=white)

**Os conteúdos da disciplina de Probabilidade e Estatística aplicados aos dados reais do Campeonato Brasileiro Série A 2024.**

</div>

---

## 📌 Visão geral

O sistema usa as **380 partidas**, os **929 gols** (com autor e minuto) e os **2.096 cartões** (com jogador e posição) da temporada 2024 do Brasileirão para aplicar, passo a passo, os conteúdos da disciplina: da introdução à Estatística até as distribuições de variáveis aleatórias contínuas.

São **20 clubes** e **586 jogadores** consultáveis: todos os atletas que marcaram gol ou receberam cartão em 2024.

## 📊 Fonte dos dados

[Brasileirao_Dataset](https://github.com/adaoduque/Brasileirao_Dataset) (Adão Duque), repositório público e gratuito, **sem chave de API**:

| Arquivo da fonte | Conteúdo usado |
| :--- | :--- |
| `campeonato-brasileiro-full.csv` | rodada, data, hora, mandante, visitante, placar, formação, técnico, arena, estado |
| `campeonato-brasileiro-gols.csv` | clube, atleta, minuto e tipo de cada gol (normal, pênalti, gol contra) |
| `campeonato-brasileiro-cartoes.csv` | clube, atleta, posição, minuto e cor de cada cartão |

Na primeira execução, `api_client.py` baixa os arquivos, filtra a temporada 2024 e salva o recorte em `data/`. Esses CSVs já estão no repositório, então **o sistema roda offline** (por exemplo, durante a apresentação). Para baixar de novo, apague os arquivos de `data/`.

A partir dos eventos são derivados os gols de cada tempo, o minuto e o autor do primeiro gol, os cartões por partida, a classificação final e o histórico jogo a jogo de cada artilheiro. Os gols por clube batem com o placar oficial nas 380 partidas.

## 📚 Conteúdos da disciplina por página do app

| Página | Conteúdos | Aplicação no Brasileirão 2024 |
| :--- | :--- | :--- |
| **1. Visão Geral & Conceitos Básicos** | População, amostra, unidade amostral; variáveis qualitativas (nominais e ordinais) e quantitativas (discretas e contínuas) | Classificação de cada variável da base e tabela final do campeonato |
| **2. Consulta por Time & Jogador** | Estatística descritiva, Poisson, Binomial, probabilidade condicional e Bayes | Perfil de cada clube; para cada atleta: gols por tempo, P(marcar), P(marcar em k dos próximos n jogos), P(vitória \| marcou) |
| **3. Tabelas de Frequência** | fi, fri, Fi, Fri; dados agrupados em classes (regra de Sturges, amplitude de classe, ponto médio) | Resultado, horário, formação tática, gols por partida, minuto dos gols |
| **4. Gráficos & Medidas de Posição** | Colunas, setores, histograma, polígono de frequências, ogiva, linha; média, mediana e moda (dados brutos e agrupados, moda de Czuber) | Gols por rodada, minuto dos gols, cartões |
| **5. Medidas de Dispersão & Boxplot** | Amplitude, variância, desvio padrão, CV, quartis, diferença interquartílica, assimetria de Pearson, outliers de Tukey, boxplot | Gols por partida; comparação da regularidade do ataque dos 20 clubes |
| **6. Teoria das Probabilidades** | Experimento determinístico e aleatório, espaço amostral, evento, probabilidade clássica e frequentista, axiomas, complementar, eventos mutuamente exclusivos, união e interseção (Venn), probabilidade condicional, eventos independentes | P(vitória do mandante), P(gol no 1º tempo), P(vitória \| marcou primeiro), independência entre gols no 1º e no 2º tempo |
| **7. Probabilidade Total & Bayes** | Partição do espaço amostral, Teorema da Probabilidade Total, Teorema de Bayes, árvore de probabilidades | Partição pelo primeiro gol; para todos os mandantes ou por clube |
| **8. V.A. Discretas** | Função de probabilidade, função de distribuição, E[X], Var(X); Bernoulli, Binomial e Poisson | Cartões e gols por partida; vitórias em casa; gols observados × Poisson; simulador de placares |
| **9. V.A. Contínuas** | Densidade; Uniforme, Exponencial (falta de memória) e Normal (padronização Z, regra empírica) | Minuto dos gols, tempo até o 1º gol, gols por rodada |

## 🗂️ Estrutura do projeto

```
├── app.py                 # Interface Streamlit (9 páginas)
├── main.py                # Executa todas as análises no terminal e salva os gráficos
├── api_client.py          # Coleta, filtro de 2024, cache e tabelas derivadas
├── prob_stats_engine.py   # Motor de cálculos: um método por conteúdo da disciplina
├── visualizer.py          # Gráficos (matplotlib)
├── data/                  # Recorte do Brasileirão 2024 (partidas, gols, cartões)
└── graficos/              # PNGs gerados por main.py
```

## ▶️ Como executar

```bash
python -m venv venv
venv\Scripts\activate          # Windows (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

Para gerar o relatório no terminal e os gráficos em `graficos/`:

```bash
python main.py
```

## ⚠️ Limitações

- A fonte não traz finalizações, posse de bola, escanteios nem minutos jogados por atleta em 2024. Por isso, as taxas dos jogadores usam os 38 jogos do clube.
- A posição do atleta vem do registro de cartões. Para quem não recebeu cartão, fica “Não informada”.
- Acréscimos (45+x e 90+x) são contados no minuto 45 ou 90. Isso concentra gols no fim de cada tempo.
- O simulador de placares supõe que os gols dos dois times são independentes (duas Poisson).
