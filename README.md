# Projeto IA — Diagnóstico de Câncer de Mama

Classificação de tumores de mama em **benignos** ou **malignos** com Machine Learning, usando a base *Breast Cancer Wisconsin (Diagnostic)*.

**Disciplina:** Inteligência Artificial — Profa. Patricia Bellin Ribeiro
**Autores:** Ailton Rodrigo da Silva Filho e Arthur Henrique da Silva Gomes

## Conteúdo

| Arquivo | Descrição |
|---|---|
| `treinamento_classificacao.py` | Pré-processamento, treino, avaliação (hold-out 80/20 e validação cruzada 5-fold) e geração dos gráficos |
| `BASE_CANCER_MAMA.csv` | Base de dados ([Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)) |
| `figuras/` | Gráficos e matrizes de confusão gerados pelo script |
| `relatorio_3_entrega.pdf` | Relatório da 3ª entrega parcial |
| `requirements.txt` | Dependências |

## Modelos
- Regressão Logística (com `StandardScaler` em Pipeline)
- Random Forest (300 árvores)

## Como executar
```bash
pip install -r requirements.txt
python treinamento_classificacao.py
```
O script imprime as métricas no terminal e salva as figuras em `figuras/`.

## Resultados (teste 80/20, classe positiva = maligno)

| Modelo | Acurácia | Precisão | Recall | F1-Score |
|---|---|---|---|---|
| Regressão Logística | 0,9649 | 0,9750 | 0,9286 | 0,9512 |
| Random Forest | 0,9737 | 1,0000 | 0,9286 | 0,9630 |

Na validação cruzada (5-fold) a Regressão Logística obteve acurácia média de 0,9737 e o Random Forest de 0,9526. Detalhes no relatório.

> Projeto acadêmico. Os modelos não passaram por validação clínica e não devem ser usados para diagnóstico de pacientes.
