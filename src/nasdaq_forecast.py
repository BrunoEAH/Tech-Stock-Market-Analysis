
"""
Sobre nasdaq_forecast.py

Este arquivo contém as funções que carregam os dados do dataset, avaliam o modelo, criam as sequências, instanciam classes dos modelos de LSTM
e modelos de treinamento.

Autores:  
Bruno E A Hayek - RA: 10389776    
Xuan Zhu - RA: 10401714


"""


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SCALERS = {}

def carregar_dados(file_path):
    df = pd.read_csv(file_path)
    
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    
    print("Informações do Dataset:")
    print(f"Shape: {df.shape}")
    print(f"Colunas: {df.columns.tolist()}")
    print(f"Intervalo de datas: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Número de empresas: {df['Name'].nunique()}")
    print(f"Empresas: {df['Name'].unique()[:10]}...")
    print(f"Valores nulos por coluna:\n{df.isnull().sum()}")
    
    return df

def evaluate_model(model, test_loader, scaler=None, return_predictions=False, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.eval()
    predictions, actuals = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(X_batch).cpu().numpy().reshape(-1)
            y_true = y_batch.cpu().numpy()

            predictions.extend(pred)
            actuals.extend(y_true)

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    if scaler is not None:
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        actuals = scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()

    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mse)

    print(f"Test Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    return predictions,actuals

def create_sequences(df,numeric_features,target_col,seq_len):
    X, y = [], []
    for name, group in df.groupby("Name"):
        group = group.sort_values("Date")
        values = group[numeric_features].values
        target = group[target_col].values
        for i in range(len(group) - seq_len):
            X.append(values[i:i+seq_len])
            y.append(target[i+seq_len])
    return np.array(X), np.array(y)

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_sequences(df, feature_cols, target_col, seq_len=60):
    X, y = [], []
    for name, group in df.groupby("Name"):
        group = group.sort_values("Date")
        data = group[feature_cols + [target_col]].values
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len, :-1])
            y.append(data[i+seq_len, -1])
    return np.array(X), np.array(y)


def plot_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss', linewidth=2)
    
    if 'val_loss' in history and history['val_loss']:
        plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)

    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.title('Treinamento')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_predictions(actuals, predictions, title="Verdadeiro vs Previsto"):
    plt.figure(figsize=(12, 6))

    # Line plot
    plt.subplot(1, 2, 1)
    plt.plot(actuals[:100], label='Verdadeiro', linewidth=2)
    plt.plot(predictions[:100], label='Previsto', linewidth=2)
    plt.xlabel('Tempo')
    plt.ylabel('Valor')
    plt.title(f'{title} (Primeiros 100 Pontos)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Scatter plot
    plt.subplot(1, 2, 2)
    plt.scatter(actuals, predictions, alpha=0.5)
    min_val = min(min(actuals), min(predictions))
    max_val = max(max(actuals), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    plt.xlabel('Verdadeiro')
    plt.ylabel('Previsto')
    plt.title('Verdadeiro vs Previsto')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()




class EnhancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        
        # Multi-layer LSTM with dropout
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Multiple fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, 1)
        
        # Activation function
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Use the last hidden state
        out = self.dropout(hn[-1])
        
        # Pass through fully connected layers
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out

def enhanced_train(model, train_loader, test_loader=None, epochs=50, initial_lr=1e-3, patience=10, weight_decay=1e-5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Optimize
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Loss function
    loss_fn = nn.MSELoss() 
    
    history = {
        "train_loss": [],
        "val_loss": [] if test_loader is not None else None,
        "learning_rates": []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer.zero_grad()
            out = model(xb).squeeze()
            loss = loss_fn(out, yb)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item() * len(xb)
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        history["train_loss"].append(avg_train_loss)
        history["learning_rates"].append(optimizer.param_groups[0]['lr'])
        
        if test_loader is not None:
            model.eval()
            test_loss = 0.0
            
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb).squeeze()
                    loss = loss_fn(out, yb)
                    test_loss += loss.item() * len(xb)
            
            avg_test_loss = test_loss / len(test_loader.dataset)
            history["val_loss"].append(avg_test_loss)
            
            scheduler.step(avg_test_loss)
            
            if avg_test_loss < best_val_loss:
                best_val_loss = avg_test_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            print(f"Época {epoch+1:3d} | Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {avg_test_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            if patience_counter >= patience:
                print(f"Parada de depois de {epoch+1} épocas")
                break
        else:
            print(f"Época {epoch+1:3d} | Train Loss: {avg_train_loss:.6f}")
    
    return history

