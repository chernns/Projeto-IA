# Projeto IA — Diagnóstico de Câncer de Mama

Classificação de tumores de mama em **benignos** ou **malignos** com Machine Learning, a partir da base
*Breast Cancer Wisconsin (Diagnostic)*. Projeto da disciplina de Inteligência Artificial (3ª Entrega Parcial).

- **Autores:** Ailton Rodrigo da Silva Filho e Arthur Henrique da Silva Gomes
- **Professora:** Patricia Bellin Ribeiro

> **Aviso:** projeto acadêmico. Os resultados são experimentais e **não** representam validação clínica
> nem autorizam o uso dos modelos para diagnóstico de pacientes.

## Estrutura do repositório

| Arquivo / pasta | Descrição |
|---|---|
| `treinamento_classificacao_v3.py` | Script completo: pré-processamento, treinamento, avaliação e geração das figuras |
| `BASE_CANCER_MAMA.csv` | Base de dados original (569 registros, 30 variáveis preditoras) |
| `Relatorio_3_Entrega_v3_novos_modelos.pdf` | Relatório da 3ª entrega parcial |
| `figuras_v3/` | Figuras e `resultados_v3.json` gerados pelo script (criada ao executar) |
| `requirements.txt` | Bibliotecas necessárias |

## Base de dados

- **Fonte original:** WOLBERG, W. H.; STREET, W. N.; MANGASARIAN, O. L. *Breast Cancer Wisconsin (Diagnostic)*.
  UCI Machine Learning Repository, 1995. DOI: 10.24432/C5DW2B.
- **Obtida via Kaggle:** https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
- 569 registros: 357 benignos (62,7%) e 212 malignos (37,3%).
- O CSV está sem alterações. O tratamento (remoção de `id` e `Unnamed: 32`, conversão de `diagnosis` em
  M=1 e B=0) é feito pelo próprio script.

## Metodologia

- **Divisão:** hold-out 80/20 estratificado (`random_state=42`).
- **Algoritmos:** SVM (kernel RBF), Rede Neural MLP, k-NN e Gradient Boosting.
- **Hiperparâmetros:** busca em grade (`GridSearchCV`, F1-Score) usando somente o conjunto de treino.
- **Validação:** validação cruzada aninhada (externa 5-fold × 2 repetições; interna 3-fold).
- **Desbalanceamento:** cada algoritmo foi treinado sem tratamento e com **SMOTE**, aplicado apenas nos
  dados de treino dentro do `Pipeline`, sem vazamento para a avaliação.
- **Métricas:** acurácia, precisão, recall, F1-Score e matriz de confusão (classe positiva = maligno).
- **Critério do melhor modelo (definido antes dos experimentos):** maior recall médio na validação aninhada,
  com desempate pelo F1-Score.

## Principais resultados (validação cruzada aninhada)

| Modelo | Acurácia | Precisão | Recall | F1-Score |
|---|---|---|---|---|
| SVM (RBF) | 0,9745 | 0,9761 | 0,9552 | 0,9654 |
| SVM (RBF) + SMOTE | 0,9772 | 0,9787 | 0,9600 | 0,9690 |
| MLP | 0,9763 | 0,9703 | 0,9668 | 0,9679 |
| **MLP + SMOTE** | **0,9772** | 0,9640 | **0,9763** | **0,9696** |
| k-NN | 0,9684 | 0,9829 | 0,9316 | 0,9560 |
| k-NN + SMOTE | 0,9640 | 0,9563 | 0,9482 | 0,9515 |
| Gradient Boosting | 0,9561 | 0,9526 | 0,9293 | 0,9399 |
| Gradient Boosting + SMOTE | 0,9622 | 0,9492 | 0,9504 | 0,9491 |

Melhor modelo pelo critério adotado: **MLP com SMOTE**. A diferença para a MLP sem SMOTE e para a SVM com
SMOTE é pequena diante do desvio padrão entre as avaliações; detalhes e ressalvas estão no relatório.

## Como executar

Requer Python 3.9 ou superior.

```bash
# 1. (opcional) criar ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. instalar as dependências
pip install -r requirements.txt

# 3. executar (o CSV deve estar na mesma pasta do script)
python treinamento_classificacao_v3.py
```

O script imprime as métricas no terminal e salva as figuras e o arquivo `resultados_v3.json` em `figuras_v3/`.
A execução completa leva alguns minutos, principalmente por causa da validação aninhada.

**Reprodutibilidade:** todas as etapas usam `random_state=42`. Os resultados se reproduzem com as mesmas
versões das bibliotecas, mas podem variar levemente em versões diferentes do scikit-learn (principalmente na MLP).
