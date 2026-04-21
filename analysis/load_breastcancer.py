import pandas as pd
from sklearn.datasets import load_breast_cancer

def load_data() -> pd.DataFrame:    
    # Carrega o dataset
    cancer_data = load_breast_cancer()
    
    # Cria o DataFrame apenas com as features
    df = pd.DataFrame(
        data=cancer_data.data, 
        columns=cancer_data.feature_names
    )
    
    # Adiciona a coluna 'target'
    df['target'] = cancer_data.target

    return df