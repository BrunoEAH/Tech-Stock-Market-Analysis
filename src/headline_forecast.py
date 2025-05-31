"""
Sobre headline_forecast.py

Este arquivo contém as funções que realizam o preprocessamento das headlines , aplicam o modelo e plotam a matriz de confusão.
e modelos de treinamento.

Autores:  
Bruno E A Hayek - RA: 10389776    
Xuan Zhu - RA: 10401714


"""


import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import re
import torch

from nltk.corpus import stopwords

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


sentimentos_labels = {
    0: "Bearish", 
    1: "Bullish", 
    2: "Neutral"
}


def preprocessamento(text):

    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+','', text)
    text = re.sub(r'#','', text)
    text = re.sub(r'@\w+','', text)
    text = re.sub(r'[^\w\s]','', text)
    text = re.sub(r'\s+',' ', text).strip()
    stops = set(stopwords.words('english'))
    return " ".join(w for w in text.split() if w not in stops)



def plot_matriz_conf(matriz_conf):

    plt.figure(figsize=(6,5))
    
    sns.heatmap(matriz_conf, annot=True, fmt="d",
                xticklabels=["Bearish","Bullish","Neutral"],
                yticklabels=["Bearish","Bullish","Neutral"],
                cmap="Blues")

    plt.xlabel("Previsto")
    plt.ylabel("Verdadeiro")
    plt.title("Matriz de confusão")
    plt.show()

def prever_noticias(news, model, tokenizer):

    model.to(device) 

    cleaned = [preprocessamento(t) for t in news]

    enc = tokenizer(cleaned, padding=True, truncation=True, max_length=128, return_tensors="pt")
    enc = {k:v.to(device) for k,v in enc.items()}

    # 5.3 Predict
    model.eval()
    with torch.no_grad():
        out = model(**enc)
        preds = torch.argmax(out.logits, dim=1).cpu().tolist()

    for txt, p in zip(news, preds):
        print(f"Notícia: {txt} → Previsão {sentimentos_labels[p]}")

