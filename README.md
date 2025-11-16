# Proj_ML_BreastCancer
O objetivo deste projeto é explorar e analisar o famoso dataset BreastCancer, aplicando técnicas de análise exploratória de dados (EDA) e visualizações, com o intuito de compreender melhor as características das amostras malignas (0) e benignas (1) e suas relações. O projeto também visa preparar o terreno para possíveis aplicações de modelos de aprendizado de máquina.

## ✅ Etapas de Inicialização

- Estruturação do projeto em pastas
- Criação do ambiente virtual
- Definição das bibliotecas principais (via `requirements.txt`)
- Configuração do `.gitignore`
- Primeiros arquivos adicionados ao controle de versão

## 📁 Estrutura Inicial de Pastas

```
Proj_ML_BreastCancer/
├── analysis/
├── data/
│   ├── raw/           # dados originais
│   ├── processed/     # dados tratados
├── analysis/
│   ├── eda/           # análises exploratórias
│   ├── reports/       # gráficos e relatórios gerados
├── ML/
│   ├── models/        # arquivos de modelos salvos
│   ├── notebooks/     # experimentos em Jupyter
├── venv/
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## 🛠 Tecnologias Utilizadas

Este projeto foi construído utilizando as seguintes ferramentas e bibliotecas:

* **Python 3.x**
* **Pandas:** Para manipulação e processamento de dados.
* **Numpy:** Para suporte a grandes arrays e matrizes multidimensionais, juntamente com uma coleção de funções matemáticas de alto nível para operar nesses arrays. 
* **Jupyter:** Para documentação passo a passo do projeto (EDA e Modelagem).
* **Plotly:** Para a geração de gráficos de alta qualidade e interativos.
* **Scikit-learn:** Para a fase de Modelagem de classificação (Regressão Logística, Random Forest, XGBoost), avaliação de métricas (Acurácia, Precisão, Recall, F1, AUC-ROC).

## ⚙️ Como Instalar e Rodar o Projeto
Para executar a aplicação em sua máquina local, siga os passos abaixo:

1. Clonagem e Configuração do Ambiente
```
# Clone o repositório
git clone [https://github.com/Antoniojrsales/Proj_BreastCancer_ML]
cd Proj_BreastCancer_ML

# Crie e ative o ambiente virtual
python -m venv venv
-nix/Linux: venv/bin/activate  
-Windows: venv\Scripts\activate

## Instale as dependências
pip install -r requirements.txt
```