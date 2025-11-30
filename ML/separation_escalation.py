import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Definindo o caminho para o arquivo limpo
CLEANED_DATA_PATH = '../data/breast_cancer_cleaned.csv'

## 1. Carregamento dos Dados
def load_data(path: str = CLEANED_DATA_PATH) -> pd.DataFrame:
    """Carrega o dataset limpo (após feature selection)."""
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado no caminho: {path}")
        return pd.DataFrame()

## 2. Divisão dos Dados (Split)
def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Separa X e y e divide em treino e teste com estratificação."""
    X = df.drop('target', axis=1)
    y = df['target']

    # Usa 80/20 (0.2) e estratifica para manter o balanceamento
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

## 3. Escalonamento dos Dados
def scale_data(X_train, X_test):
    """Aplica o StandardScaler, seguindo a Regra de Ouro."""
    scaler = StandardScaler()

    # FIT APENAS no treino
    X_train_scaled = scaler.fit_transform(X_train)
    
    # TRANSFORM nos dois, usando os parâmetros aprendidos no FIT do treino
    X_test_scaled = scaler.transform(X_test)
    
    # Retorna os arrays transformados como DataFrames para manter os nomes das colunas
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    return X_train_scaled, X_test_scaled, scaler