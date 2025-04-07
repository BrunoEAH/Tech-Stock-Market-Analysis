# Predição do Nasdaq-100 junto à Análise de Sentimentos de Títulos de Notícia Obtidas no Twitter

Este projeto tem como objetivo realizar a predição dos movimentos do Nasdaq-100 utilizando técnicas de análise de sentimentos aplicadas aos títulos de notícias extraídos do Twitter, integrando essas informações com dados históricos do índice.


**Autores:**  
- Bruno E A Hayek - RA: 10389776    
- Xuan Zhu - RA: 10401714

---

## Descrição dos Arquivos


### 1. Dataset de Títulos de Notícia do Twitter
- **Arquivo:** `twitter_financial_news/sent_train.csv`
- **Descrição:** Contém títulos de notícias financeiras obtidas a partir do Twitter, que serão utilizados para realizar a análise de sentimentos.
- **Colunas:**
  - **text:** Conteúdo textual do título da notícia.
  - **label:** Rótulo de sentimento associado ao tweet (Bullish, Bearish, Neutral).

### 2. Dataset do Nasdaq-100
- **Arquivo:** `NASDAQ_100_Data_From_2010.csv`
- **Descrição:** Contém os dados históricos do Nasdaq-100 desde 2010, utilizados para análise de séries temporais e predição do índice.
- **Colunas:**
  - **Date:** Data da observação (formato `yyyy-mm-dd`).
  - **Open:** Preço de abertura do índice.
  - **High:** Preço máximo do índice.
  - **Low:** Preço mínimo do índice.
  - **Close:** Preço de fechamento do índice.
  - **Adj_Close:** Preço de fechamento ajustado (levando em conta dividendos, splits, etc.).
  - **Volume:** Volume de negociação.
  - **Name:** Nome da empresa ou ativo (quando aplicável).

### 3. Notebook de Análise Exploratória
- **Arquivo:** `analise_exploratoria_dados.ipynb`
- **Descrição:** Contém o pré-processamento dos dados e a análise exploratória, servindo de base para a modelagem e predição.

---

## Objetivos do Projeto

- **Análise de Sentimentos:**  
  Aplicar técnicas de processamento de linguagem natural (NLP) para classificar os sentimentos (Bullish, Bearish, Neutral) presentes nos títulos de notícias obtidas do Twitter.

- **Predição do Nasdaq-100:**  
  Utilizar dados históricos do Nasdaq-100 para construir modelos de predição, integrando insights da análise de sentimentos.

- **Integração dos Dados:**  
  Combinar informações dos dois datasets para aprimorar a compreensão dos movimentos do índice e fornecer uma predição mais robusta.

---

## Metodologia

1. **Pré-processamento dos Dados:**
   - **Twitter:** Limpeza textual, tokenização e classificação de sentimentos dos títulos.
   - **Nasdaq-100:** Cálculo de indicadores técnicos (ex.: SMA, RSI, retornos diários) e normalização dos dados.

2. **Análise Exploratória:**
   - Visualização das distribuições dos dados.
   - Identificação de correlações e padrões entre variáveis.

