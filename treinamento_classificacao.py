"""
Projeto IA - Diagnóstico de Câncer de Mama (3ª Entrega Parcial)
Classificação de tumores (Benigno x Maligno) - Breast Cancer Wisconsin (Diagnostic)

Algoritmos: SVM (kernel RBF), Rede Neural MLP, k-NN e Gradient Boosting.
Para cada algoritmo, duas versões: (a) sem tratamento do desbalanceamento e
(b) com SMOTE aplicado SOMENTE nos dados de treino de cada divisão (dentro do Pipeline).

Etapas
 1. Pré-processamento e divisão treino/teste (80/20 estratificada)
 2. Ajuste de hiperparâmetros por GridSearchCV (5-fold) apenas no treino; avaliação no teste
 3. Validação cruzada ANINHADA (externa: 5-fold x 2 repetições; interna: 3-fold com GridSearchCV),
    que estima o desempenho incluindo a escolha de hiperparâmetros, sem vazamento
 4. Escolha do melhor modelo (critério definido antes da execução: maior recall médio na
    validação aninhada, desempate pelo F1-Score) e importância das variáveis por permutação
Figuras em "figuras_v3/".
"""
import os, json, time, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import (train_test_split, StratifiedKFold, RepeatedStratifiedKFold,
                                     GridSearchCV, cross_validate)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
from sklearn.inspection import permutation_importance
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
warnings.filterwarnings("ignore")

SEED = 42
CSV = "BASE_CANCER_MAMA.csv"
OUT = "figuras_v3"
os.makedirs(OUT, exist_ok=True)

# ---------------- 1. Dados ----------------
df = pd.read_csv(CSV)
df = df.drop(columns=[c for c in ["id", "Unnamed: 32"] if c in df.columns])
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
X, y = df.drop(columns=["diagnosis"]), df["diagnosis"]
print("Registros:", len(df), "| Preditoras:", X.shape[1], "| Nulos:", int(df.isna().sum().sum()))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
print("Treino:", len(X_train), (y_train == 0).sum(), (y_train == 1).sum(),
      "| Teste:", len(X_test), (y_test == 0).sum(), (y_test == 1).sum())

# ---------------- 2. Modelos e grades ----------------
algoritmos = {
    "SVM (RBF)": (SVC(kernel="rbf", random_state=SEED),
                  {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01]}, True),
    "MLP": (MLPClassifier(max_iter=2000, random_state=SEED),
            {"clf__hidden_layer_sizes": [(16,), (32, 16)], "clf__alpha": [1e-4, 1e-2]}, True),
    "k-NN": (KNeighborsClassifier(),
             {"clf__n_neighbors": [3, 5, 9, 15], "clf__weights": ["uniform", "distance"]}, True),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=SEED),
                          {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1],
                           "clf__max_depth": [2, 3]}, False),
}
def montar(nome, smote):
    clf, grade, escala = algoritmos[nome]
    passos = []
    if escala: passos.append(("scaler", StandardScaler()))
    if smote:  passos.append(("smote", SMOTE(random_state=SEED)))
    passos.append(("clf", clf))
    return Pipeline(passos), grade

configs = [(f"{n} + SMOTE" if s else n, n, s) for n in algoritmos for s in (False, True)]
nomes = [c[0] for c in configs]
metricas = ["Acurácia", "Precisão", "Recall", "F1-Score"]
scoring = {"Acurácia": "accuracy", "Precisão": "precision", "Recall": "recall", "F1-Score": "f1"}

def busca(nome, smote, cv_interno):
    pipe, grade = montar(nome, smote)
    return GridSearchCV(pipe, grade, cv=cv_interno, scoring="f1", n_jobs=1)

# ---------------- 3. Hold-out com ajuste de hiperparâmetros ----------------
teste, mats, params, ajustados = {}, {}, {}, {}
for rot, nome, smote in configs:
    t0 = time.time()
    gs = busca(nome, smote, StratifiedKFold(5, shuffle=True, random_state=SEED)).fit(X_train, y_train)
    p = gs.predict(X_test)
    teste[rot] = {"Acurácia": accuracy_score(y_test, p), "Precisão": precision_score(y_test, p),
                  "Recall": recall_score(y_test, p), "F1-Score": f1_score(y_test, p)}
    mats[rot] = confusion_matrix(y_test, p)
    params[rot] = {k.replace("clf__", ""): (list(v) if isinstance(v, tuple) else v) for k, v in gs.best_params_.items()}
    ajustados[rot] = gs
    print(f"[teste] {rot:28s}", {k: round(v, 4) for k, v in teste[rot].items()}, params[rot],
          mats[rot].ravel().tolist(), f"{time.time()-t0:.0f}s", flush=True)

# ---------------- 4. Validação cruzada aninhada ----------------
externo = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
cv_res = {}
for rot, nome, smote in configs:
    t0 = time.time()
    gs = busca(nome, smote, StratifiedKFold(3, shuffle=True, random_state=SEED))
    r = cross_validate(gs, X, y, cv=externo, scoring=scoring, n_jobs=1)
    cv_res[rot] = {m: (r[f"test_{m}"].mean(), r[f"test_{m}"].std()) for m in metricas}
    print(f"[CV aninhada] {rot:28s}", " | ".join(f"{m} {mu:.4f}±{sd:.4f}" for m, (mu, sd) in cv_res[rot].items()),
          f"{time.time()-t0:.0f}s", flush=True)

# ---------------- 5. Melhor modelo (critério: recall médio; desempate: F1) ----------------
ranking = sorted(nomes, key=lambda n: (round(cv_res[n]["Recall"][0], 4), cv_res[n]["F1-Score"][0]), reverse=True)
melhor = ranking[0]
print("\nRanking (recall, F1):")
for n in ranking: print(f"  {n:28s} recall={cv_res[n]['Recall'][0]:.4f} F1={cv_res[n]['F1-Score'][0]:.4f}")
print("MELHOR:", melhor)

# importância por permutação no teste (modelo ajustado do hold-out)
pi = permutation_importance(ajustados[melhor].best_estimator_, X_test, y_test, scoring="f1",
                            n_repeats=30, random_state=SEED)
imp = pd.Series(pi.importances_mean, index=X.columns).sort_values()
print("Top variáveis (permutação):", imp.tail(10).index[::-1].tolist())

# ---------------- 6. Figuras ----------------
base = list(algoritmos)
c_sem, c_sm = "#6baed6", "#d94801"
def painel(dados, titulo, arq, erro=False):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    for ax, m in zip(axes, metricas):
        for k, (s, cor, lab) in enumerate([(False, c_sem, "Sem tratamento"), (True, c_sm, "Com SMOTE")]):
            vals = [dados[(f"{b} + SMOTE" if s else b)][m] for b in base]
            if erro:
                ax.bar(np.arange(4) + k * 0.38, [v[0] for v in vals], 0.38, yerr=[v[1] for v in vals],
                       capsize=2, color=cor, label=lab)
            else:
                ax.bar(np.arange(4) + k * 0.38, vals, 0.38, color=cor, label=lab)
        ax.set_xticks(np.arange(4) + 0.19); ax.set_xticklabels(base, rotation=25, ha="right", fontsize=8)
        ax.set_title(m, fontsize=10); ax.set_ylim(0.84, 1.03)
    axes[0].legend(fontsize=7.5, loc="lower left")
    fig.suptitle(titulo, fontsize=11); plt.tight_layout(); plt.savefig(f"{OUT}/{arq}", dpi=200); plt.close()
painel(teste, "Métricas no conjunto de teste (80/20)", "metricas_teste.png")
painel(cv_res, "Validação cruzada aninhada (5-fold x 2): média e desvio padrão", "metricas_cv_aninhada.png", erro=True)

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, s in enumerate((False, True)):
    for j, b in enumerate(base):
        rot = f"{b} + SMOTE" if s else b; ax = axes[i, j]; cm = mats[rot]; ax.imshow(cm, cmap="Blues")
        for (a, c), v in np.ndenumerate(cm):
            ax.text(c, a, v, ha="center", va="center", fontsize=13, color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Benigno", "Maligno"], fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Benigno", "Maligno"], fontsize=8)
        ax.set_title(rot, fontsize=9)
        if j == 0: ax.set_ylabel("Classe real", fontsize=8)
        if i == 1: ax.set_xlabel("Classe prevista", fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/matrizes_confusao.png", dpi=200); plt.close()

top = imp.tail(10)
fig, ax = plt.subplots(figsize=(6.5, 3.8)); ax.barh(top.index, top.values, color="#4c72b0")
ax.set_xlabel("Queda média do F1-Score ao permutar a variável")
ax.set_title(f"10 variáveis mais importantes (permutação) — {melhor}", fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/importancia_permutacao.png", dpi=200); plt.close()

with open(f"{OUT}/resultados_v3.json", "w", encoding="utf-8") as f:
    json.dump({"teste": teste, "params": params, "melhor": melhor, "ranking": ranking,
               "cv": {n: {m: list(v) for m, v in d.items()} for n, d in cv_res.items()},
               "matrizes": {n: m.tolist() for n, m in mats.items()},
               "top_vars": imp.tail(10).index[::-1].tolist(),
               "top_vals": imp.tail(10).values[::-1].tolist()}, f, ensure_ascii=False, indent=1)
print("Figuras em:", OUT)
