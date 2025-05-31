
"""
Sobre stock_fetching.py

Este arquivo contém as funções que realizam a requisição à API do Yahoo Finance, normalização desses dados, aplicação
do modelo LSTM aos dados e plotagem das previsões nos gráficos.


Autores:  
Bruno E A Hayek - RA: 10389776    
Xuan Zhu - RA: 10401714



"""





import yfinance as yf
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import talib
import joblib
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fetch_stock_data(symbol, period="1y"):


    stock = yf.Ticker(symbol)
    df = stock.history(period=period)
    
    df = df.reset_index()
    df['Name'] = symbol
    
    df = df.rename(columns={
        'Open': 'Open',
        'High': 'High', 
        'Low': 'Low',
        'Close': 'Close',
        'Volume': 'Volume'
    })
    
    df['Adj_Close'] = df['Close']
    
    df['SMA_15'] = talib.SMA(df['Adj_Close'], timeperiod=15)
    df['RSI'] = talib.RSI(df['Close'].values, timeperiod=14)
    df['Retorno'] = df['Adj_Close'].pct_change()
    df['Flutuacao_Diaria'] = (df['High'] - df['Low'])
    
    return df

def normaliza_novo(df,simbolo):
    
    numeric_features = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume','SMA_15','RSI','Retorno','Flutuacao_Diaria']

    df[numeric_features] = df[numeric_features].astype(float)

    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])    
    
    joblib.dump(scaler, f"pkls/scaler_{simbolo}.pkl")

    return df


def sequencia_novo(df,target_col,feature_cols,seq_len=60):
    recent_seq = df[feature_cols].values[-seq_len:]  # shape: (seq_len, features)
    X_input = torch.tensor(recent_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, seq_len, features)

    return X_input


def previsao_dias(data, simbolo, dias, model, X, target_col, feature_cols, seq_len):

    predictions = []
    input_seq = data[feature_cols].values[-seq_len:].tolist()  # Sequência inicial como lista
    target_idx = feature_cols.index(target_col)

    scaler = joblib.load(f"pkls/scaler_{simbolo}.pkl")

    for _ in range(dias):
        X_input = torch.tensor([input_seq[-seq_len:]], dtype=torch.float32).to(DEVICE)

        # Faz previsão
        with torch.no_grad():
            pred = model(X_input).item()

        predictions.append(pred)

        # Cria próxima entrada com previsão injetada no índice correto
        next_input = input_seq[-1].copy()
        next_input[target_idx] = pred
        input_seq.append(next_input)

    # Desnormaliza apenas os valores da coluna alvo
    dummy_input = np.zeros((dias, len(feature_cols)))
    dummy_input[:, target_idx] = predictions
    inversed = scaler.inverse_transform(dummy_input)
    desnormalized_preds = inversed[:, target_idx]

    print(f"Previsão para os próximos {dias} dias ({target_col}):", desnormalized_preds)
    return desnormalized_preds


def plot_previsao(data, predictions, target_col, dias, ticker):

    recent_actuals = data[target_col].values[-60:]

    combined = np.concatenate([recent_actuals, predictions])
    days = np.arange(len(combined))

    actual_days = days[:len(recent_actuals)]
    pred_days = days[len(recent_actuals):]

    plt.figure(figsize=(12, 6))
    plt.plot(actual_days, recent_actuals, label="Últimos Preços Reais", linewidth=2, color="blue")
    plt.plot(pred_days, predictions, label=f"Previsões Próximos {dias} Dias", linewidth=2, color='orange')

    plt.axvline(x=len(recent_actuals)-1, color='gray', linestyle=':', label='Divisão Reais/Previsões')

    plt.title(f"Previsão de Preços para {ticker}", fontsize=14)
    plt.xlabel("Dias", fontsize=12)
    plt.ylabel("Preço de Fechamento", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



def plot_previsao_apenas_previsoes(predictions, dias, ticker):

    days = np.arange(1, dias + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(days, predictions, label=f"Previsões Próximos {dias} Dias", linewidth=2, linestyle='--', color='orange')
    
    plt.title(f"Previsões Futuras para {ticker}", fontsize=14)
    plt.xlabel("Dias Futuros", fontsize=12)
    plt.ylabel("Preço de Fechamento Previsto", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
