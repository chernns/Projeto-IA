# Projeto IA — Diagnóstico de Câncer de Mama

Classificação de tumores de mama em **benignos** ou **malignos** com Machine Learning, a partir das
características celulares da base *Breast Cancer Wisconsin (Diagnostic)*.

- **Autores:** Ailton Rodrigo da Silva Filho e Arthur Henrique da Silva Gomes
- **Instituição:** Faculdade de Tecnologia de Bauru
- **Disciplina:** Inteligência Artificial — Profa. Patricia Bellin Ribeiro
- **Entrega:** 4ª e última etapa (artigo científico)

> **Aviso importante:** este é um projeto acadêmico. Os modelos foram avaliados apenas como classificadores sobre
> uma base de dados pública e **não passaram por validação clínica**. Os resultados **não autorizam** o uso dos
> modelos para diagnóstico, triagem ou qualquer decisão médica sobre pacientes.

## Sumário

1. [Objetivo](#objetivo)
2. [Base de dados](#base-de-dados)
3. [Metodologia](#metodologia)
4. [Resultados](#resultados)
5. [Estrutura do repositório](#estrutura-do-repositório)
6. [Como executar](#como-executar)
7. [Limitações](#limitações)
8. [Referências](#referências)

## Objetivo

Comparar o desempenho de quatro algoritmos de classificação (**SVM com kernel RBF, rede neural MLP, k-NN e
Gradient Boosting**) na tarefa de classificar tumores de mama em benignos ou malignos, identificando o modelo de
melhor comportamento para o problema. Como objetivos específicos, o projeto busca:

- definir um protocolo de avaliação **sem vazamento de informação** (*data leakage*);
- ajustar hiperparâmetros e avaliar os modelos por divisão *hold-out* e por **validação cruzada aninhada**;
- medir o efeito da técnica **SMOTE** no tratamento do desbalanceamento de classes;
- comparar os modelos por matrizes de confusão, métricas de classificação, curvas ROC e teste estatístico;
- identificar as variáveis mais relevantes e analisar os tumores malignos não identificados.

Em um problema de diagnóstico, o erro mais grave é o **falso negativo** (tumor maligno classificado como benigno).
Por isso, o **recall** da classe maligna é a métrica prioritária na escolha do melhor modelo.

## Base de dados

- **Nome:** Breast Cancer Wisconsin (Diagnostic).
- **Origem:** criada por Wolberg, Street e Mangasarian e disponibilizada pelo UCI Machine Learning Repository (citação recomendada: Wolberg et al., 1993)
  (DOI: [10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B)); obtida para este trabalho por meio do
  [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data).
- **Conteúdo:** 569 registros — **357 benignos (62,7%)** e **212 malignos (37,3%)** — e 30 atributos numéricos
  extraídos de imagens digitalizadas de punção aspirativa por agulha fina (PAAF).
- **Atributos:** 10 características do núcleo celular (raio, textura, perímetro, área, suavidade, compacidade,
  concavidade, pontos côncavos, simetria e dimensão fractal), cada uma representada pela **média**, pelo
  **erro padrão** (`_se`) e pelo **pior valor** (`_worst`).
- **Arquivo:** `BASE_CANCER_MAMA.csv`, sem alterações em relação ao original. O tratamento é feito pelo script.

## Metodologia

### Pré-processamento
- Não há valores nulos nas variáveis preditoras.
- Remoção das colunas `id` e `Unnamed: 32` (sem informação preditiva).
- Conversão do alvo `diagnosis`: **M = 1** (maligno, classe positiva) e **B = 0** (benigno).
- Padronização (`StandardScaler`) para SVM, MLP e k-NN, dentro do `Pipeline`, ajustada somente com os dados de
  treinamento de cada divisão. O Gradient Boosting não exige padronização.

### Divisão e validação
- **Hold-out 80/20 estratificado** (455 registros para treino e 114 para teste), com `random_state=42`. O conjunto
  de teste não participa do treinamento nem da escolha dos hiperparâmetros.
- **Validação cruzada aninhada:** laço externo com 5-fold estratificado repetido 2 vezes (10 avaliações) e laço
  interno com 3-fold, no qual os hiperparâmetros são escolhidos por busca em grade (`GridSearchCV`, critério
  F1-Score). Assim, o desempenho estimado inclui o ajuste dos hiperparâmetros, sem que os dados de teste
  participem dessa escolha.

### Algoritmos
| Algoritmo | Descrição resumida | Hiperparâmetros testados |
|---|---|---|
| SVM (RBF) | Margem máxima com fronteira não linear (kernel RBF) | `C` ∈ {0,1; 1; 10}; `gamma` ∈ {scale; 0,01} |
| MLP | Rede neural com camadas ocultas (ReLU), otimizador Adam e regularização L2 | camadas ∈ {(16); (32, 16)}; `alpha` ∈ {0,0001; 0,01} |
| k-NN | Voto dos *k* vizinhos mais próximos (distância euclidiana) | `k` ∈ {3; 5; 9; 15}; pesos ∈ {uniformes; por distância} |
| Gradient Boosting | Árvores rasas em sequência, cada uma corrigindo os erros das anteriores | árvores ∈ {100; 200}; taxa ∈ {0,05; 0,1}; profundidade ∈ {2; 3} |

### Tratamento do desbalanceamento (SMOTE)
A base é moderadamente desbalanceada (cerca de 1,7 benigno para cada maligno). Um classificador que previsse sempre
"benigno" já teria 62,7% de acurácia, por isso a acurácia isolada pode enganar.

O **SMOTE** (*Synthetic Minority Over-sampling Technique*) cria exemplos sintéticos da classe minoritária
(maligna) por interpolação entre um caso real e seus vizinhos mais próximos, equalizando as classes no treino.
Cada algoritmo foi treinado em **duas versões**, com a mesma divisão, a mesma semente e os mesmos *folds*:
**sem tratamento** e **com SMOTE**.

O SMOTE é aplicado **apenas nos dados de treino** de cada divisão ou fold, dentro do `Pipeline` do
`imbalanced-learn`. Os dados de teste e de validação mantêm a distribuição original, o que evita vazamento de
informação.

### Métricas
Classe positiva = maligno. Foram utilizadas:

- **Acurácia:** proporção de acertos totais.
- **Precisão:** entre os previstos como malignos, quantos realmente eram.
- **Recall (sensibilidade):** entre os malignos reais, quantos foram identificados.
- **Especificidade:** entre os benignos reais, quantos foram reconhecidos.
- **F1-Score:** média harmônica entre precisão e recall.
- **AUC (área sob a curva ROC):** capacidade de ordenar os casos malignos acima dos benignos, independentemente
  do limiar de decisão.
- **Matriz de confusão:** no teste e agregada ao longo da validação aninhada (1.138 previsões).

**Critério do melhor modelo** (definido antes dos experimentos): maior recall médio na validação aninhada, com
desempate pelo F1-Score. As diferenças entre modelos foram verificadas pelo **teste t corrigido para amostragem
repetida** (Nadeau e Bengio, 2003).

## Resultados

### Validação cruzada aninhada (média das 10 avaliações)

Em negrito, o modelo escolhido e os valores em que ele lidera (recall e F1-Score).

| Modelo | Acurácia | Precisão | Recall | F1-Score | AUC |
|---|---|---|---|---|---|
| SVM (RBF) | 0,9745 | 0,9761 | 0,9552 | 0,9654 | 0,9954 |
| SVM (RBF) + SMOTE | 0,9772 | 0,9787 | 0,9600 | 0,9690 | 0,9951 |
| MLP | 0,9763 | 0,9703 | 0,9668 | 0,9679 | 0,9952 |
| **MLP + SMOTE** | 0,9772 | 0,9640 | **0,9763** | **0,9696** | 0,9952 |
| k-NN | 0,9684 | 0,9829 | 0,9316 | 0,9560 | 0,9837 |
| k-NN + SMOTE | 0,9640 | 0,9563 | 0,9482 | 0,9515 | 0,9830 |
| Gradient Boosting | 0,9561 | 0,9526 | 0,9293 | 0,9399 | 0,9922 |
| Gradient Boosting + SMOTE | 0,9622 | 0,9492 | 0,9504 | 0,9491 | 0,9922 |

### Conjunto de teste (hold-out 80/20, 114 registros)

| Modelo | Acurácia | Precisão | Recall | F1-Score | AUC | FP | FN |
|---|---|---|---|---|---|---|---|
| SVM (RBF) | 0,9737 | 1,0000 | 0,9286 | 0,9630 | 0,9927 | 0 | 3 |
| SVM (RBF) + SMOTE | 0,9737 | 1,0000 | 0,9286 | 0,9630 | 0,9977 | 0 | 3 |
| MLP | 0,9825 | 1,0000 | 0,9524 | 0,9756 | 0,9970 | 0 | 2 |
| **MLP + SMOTE** | 0,9825 | 1,0000 | 0,9524 | 0,9756 | 0,9944 | 0 | 2 |
| k-NN | 0,9386 | 0,9730 | 0,8571 | 0,9114 | 0,9825 | 1 | 6 |
| k-NN + SMOTE | 0,9649 | 0,9750 | 0,9286 | 0,9512 | 0,9960 | 1 | 3 |
| Gradient Boosting | 0,9649 | 1,0000 | 0,9048 | 0,9500 | 0,9934 | 0 | 4 |
| Gradient Boosting + SMOTE | 0,9737 | 1,0000 | 0,9286 | 0,9630 | 0,9977 | 0 | 3 |

FP: falsos positivos; FN: falsos negativos (o teste contém 42 tumores malignos).

### Como interpretar

- **Melhor modelo pelo critério adotado: MLP com SMOTE.** Teve o maior recall e o maior F1-Score na validação
  aninhada e, no teste, errou 2 dos 42 tumores malignos e nenhum benigno.
- **Todos os modelos** obtiveram acurácia média superior a 95% e AUC superior a 0,98, o que indica que as classes
  são bem separáveis nesta base.
- **Efeito do SMOTE:** aumentou o recall médio dos **quatro** algoritmos, em troca de **menor precisão** (mais falsos
  positivos) em parte deles. O F1-Score e a AUC praticamente não mudaram. Ou seja, o SMOTE desloca o ponto de
  operação do classificador no sentido de errar menos tumores malignos, sem melhorar a capacidade de separar as
  classes. É uma escolha coerente quando o falso negativo é considerado o erro mais grave.
- **Ressalva estatística:** o teste t corrigido **não** encontrou diferença significativa entre a MLP com SMOTE e a
  MLP sem SMOTE nem a SVM. Apenas o recall do k-NN sem SMOTE foi significativamente inferior (p = 0,010), e
  mesmo esse resultado exige cautela por não haver correção para múltiplas comparações. A SVM com SMOTE é uma
  alternativa defensável: tem menos falsos positivos e treina muito mais rápido.
- **Matrizes agregadas** (1.138 previsões, das quais 424 são de tumores malignos):

| Modelo | VN | FP | FN | VP |
|---|---|---|---|---|
| SVM (RBF) | 704 | 10 | 19 | 405 |
| SVM (RBF) + SMOTE | 705 | 9 | 17 | 407 |
| MLP | 701 | 13 | 14 | 410 |
| MLP + SMOTE | 698 | 16 | 10 | 414 |
| k-NN | 707 | 7 | 29 | 395 |
| k-NN + SMOTE | 695 | 19 | 22 | 402 |
| Gradient Boosting | 694 | 20 | 30 | 394 |
| Gradient Boosting + SMOTE | 692 | 22 | 21 | 403 |

- **Tumores não identificados:** três tumores malignos foram errados por quase todos os algoritmos e têm área,
  perímetro e concavidade próximos aos dos benignos, o que indica um limite dos atributos disponíveis.
- **Variáveis mais relevantes** (importância por permutação, MLP com SMOTE): `concavity_worst`, `symmetry_worst`,
  `area_worst` e `area_mean`.

Detalhes, figuras, curvas ROC e discussão completa estão no artigo (`Artigo_Cientifico_Final.pdf`).

## Estrutura do repositório

| Arquivo / pasta | Descrição |
|---|---|
| `treinamento_classificacao.py` | Script principal: pré-processamento, treinamento, avaliação (hold-out e validação aninhada) e figuras |
| `analises_complementares.py` | Análises do artigo: curvas ROC e AUC, matrizes agregadas, teste t corrigido e análise de erros |
| `BASE_CANCER_MAMA.csv` | Base de dados original |
| `Artigo_Cientifico_Final.pdf` | Artigo científico completo (4ª entrega) |
| `figuras/` | Figuras do artigo e arquivos `resultados.json` e `resultados_complementares.json` |
| `requirements.txt` | Dependências |

## Como executar

Requer **Python 3.9 ou superior**. Os dois scripts devem ser executados na mesma pasta que contém o
`BASE_CANCER_MAMA.csv`, **nesta ordem**: o segundo usa o `resultados.json` gerado pelo primeiro.

### Windows (Prompt de Comando ou PowerShell)

```bat
:: 1. (opcional) criar e ativar o ambiente virtual
py -m venv .venv
.venv\Scripts\activate

:: 2. instalar as dependências
pip install -r requirements.txt

:: 3. executar os scripts
python treinamento_classificacao.py
python analises_complementares.py
```

No PowerShell, se a ativação for bloqueada, execute antes
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` e use `.venv\Scripts\Activate.ps1`.

### Linux e macOS (terminal)

```bash
# 1. (opcional) criar e ativar o ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# 2. instalar as dependências
pip install -r requirements.txt

# 3. executar os scripts
python3 treinamento_classificacao.py
python3 analises_complementares.py
```

### Saídas
Os scripts imprimem as métricas no terminal e salvam as figuras e os arquivos `.json` em `figuras/`. A execução
completa leva alguns minutos, principalmente por causa da validação aninhada.

**Reprodutibilidade:** todas as etapas aleatórias usam `random_state=42`. Os resultados foram gerados com Python 3.12,
scikit-learn 1.9.1, imbalanced-learn 0.14.2, pandas 3.0.6, NumPy 2.5.4, SciPy 1.18.1 e Matplotlib 3.11.2. Com
versões diferentes (principalmente do scikit-learn), os valores podem variar levemente, em especial na MLP.

## Limitações

- Base pequena (569 registros, 212 malignos), de uma única fonte e **sem validação externa**.
- Os atributos já vêm extraídos das imagens; a etapa de análise de imagem não foi avaliada.
- O conjunto de teste tem apenas 42 tumores malignos e a validação aninhada usa 10 avaliações, o que limita o poder
  estatístico das comparações.
- Grades de hiperparâmetros enxutas e apenas uma técnica de tratamento do desbalanceamento (SMOTE).
- Não foram avaliadas a calibração das probabilidades nem a robustez a outros equipamentos ou populações.

## Referências

- WOLBERG, W.; MANGASARIAN, O.; STREET, N.; STREET, W. **Breast Cancer Wisconsin (Diagnostic)**. UCI Machine Learning
  Repository, 1993. Licença CC BY 4.0. DOI: [10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B).
- UCI MACHINE LEARNING. **Breast Cancer Wisconsin (Diagnostic) Data Set**. Kaggle. Disponível em:
  <https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data>.
- STREET, W. N.; WOLBERG, W. H.; MANGASARIAN, O. L. Nuclear feature extraction for breast tumor diagnosis. In: IS&T/SPIE
  International Symposium on Electronic Imaging: Science and Technology, 1993. v. 1905, p. 861–870.
- CHAWLA, N. V. et al. SMOTE: synthetic minority over-sampling technique. **Journal of Artificial Intelligence
  Research**, v. 16, p. 321–357, 2002.
- NADEAU, C.; BENGIO, Y. Inference for the generalization error. **Machine Learning**, v. 52, n. 3, p. 239–281, 2003.
- PEDREGOSA, F. et al. Scikit-learn: machine learning in Python. **Journal of Machine Learning Research**, v. 12,
  p. 2825–2830, 2011.
- LEMAÎTRE, G.; NOGUEIRA, F.; ARIDAS, C. K. Imbalanced-learn: a Python toolbox to tackle the curse of imbalanced
  datasets in machine learning. **Journal of Machine Learning Research**, v. 18, n. 17, p. 1–5, 2017.

A lista completa de referências está no artigo.
