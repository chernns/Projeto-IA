"""
Projeto IA - Diagnóstico de Câncer de Mama (3ª Entrega Parcial)
Classificação de tumores (Benigno x Maligno) - Breast Cancer Wisconsin (Diagnostic)

Etapas: pré-processamento, divisão treino/teste (80/20 estratificada),
treinamento (Regressão Logística e Random Forest), avaliação no teste
(métricas + matrizes de confusão) e validação cruzada estratificada 5-fold.
Gera as figuras usadas no relatório na pasta "figuras/".
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

SEED = 42
CSV = "BASE_CANCER_MAMA.csv"
OUT = "figuras"
os.makedirs(OUT, exist_ok=True)

# ---------------- 1. Carga e pré-processamento ----------------
df = pd.read_csv(CSV)
df = df.drop(columns=[c for c in ["id", "Unnamed: 32"] if c in df.columns])
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

print("Registros:", len(df), "| Preditoras:", df.shape[1] - 1)
print("Nulos:", int(df.isna().sum().sum()))
print("Distribuição (0=Benigno, 1=Maligno):\n", df["diagnosis"].value_counts().to_dict())

X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

# ---------------- 2. Divisão treino/teste ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)
print("Treino:", len(X_train), "| Teste:", len(X_test))

# ---------------- 3. Modelos ----------------
modelos = {
    "Regressão Logística": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, random_state=SEED))]),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=SEED, n_jobs=-1),
}

# ---------------- 4. Avaliação no conjunto de teste ----------------
resultados, matrizes = {}, {}
for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    resultados[nome] = {
        "Acurácia": accuracy_score(y_test, y_pred),
        "Precisão": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)}
    matrizes[nome] = confusion_matrix(y_test, y_pred)
    print(f"\n===== {nome} (teste) =====")
    for k, v in resultados[nome].items():
        print(f"{k:9s}: {v:.4f}")
    print("Matriz de confusão:\n", matrizes[nome])
    print(classification_report(y_test, y_pred, target_names=["Benigno", "Maligno"]))

# ---------------- 5. Validação cruzada estratificada (5-fold) ----------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scoring = {"Acurácia": "accuracy", "Precisão": "precision",
           "Recall": "recall", "F1-Score": "f1"}
cv_res = {}
for nome, modelo in modelos.items():
    r = cross_validate(modelo, X, y, cv=skf, scoring=scoring)
    cv_res[nome] = {m: (r[f"test_{m}"].mean(), r[f"test_{m}"].std()) for m in scoring}
    print(f"\n===== {nome} (validação cruzada 5-fold) =====")
    for m, (mu, sd) in cv_res[nome].items():
        print(f"{m:9s}: {mu:.4f} +/- {sd:.4f}")

# ---------------- 6. Figuras ----------------
metricas = list(scoring.keys())
nomes = list(modelos.keys())

# 6.1 Barras - teste
fig, ax = plt.subplots(figsize=(7, 4))
w = 0.2
for i, m in enumerate(metricas):
    ax.bar(np.arange(len(nomes)) + i * w, [resultados[n][m] for n in nomes], w, label=m)
ax.set_xticks(np.arange(len(nomes)) + 1.5 * w); ax.set_xticklabels(nomes)
ax.set_ylim(0.85, 1.02); ax.set_ylabel("Valor")
ax.set_title("Métricas no conjunto de teste (80/20)"); ax.legend(loc="upper left", ncol=4, fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/metricas_teste.png", dpi=200); plt.close()

# 6.2 Barras - validação cruzada (com desvio padrão)
fig, ax = plt.subplots(figsize=(7, 4))
for i, m in enumerate(metricas):
    ax.bar(np.arange(len(nomes)) + i * w, [cv_res[n][m][0] for n in nomes], w,
           yerr=[cv_res[n][m][1] for n in nomes], capsize=3, label=m)
ax.set_xticks(np.arange(len(nomes)) + 1.5 * w); ax.set_xticklabels(nomes)
ax.set_ylim(0.85, 1.02); ax.set_ylabel("Média nos 5 folds")
ax.set_title("Validação cruzada estratificada (5-fold)"); ax.legend(loc="upper left", ncol=4, fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/metricas_cv.png", dpi=200); plt.close()

# 6.3 Matrizes de confusão lado a lado
fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
for ax, n in zip(axes, nomes):
    cm = matrizes[n]
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center", fontsize=14,
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Benigno (0)", "Maligno (1)"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Benigno (0)", "Maligno (1)"])
    ax.set_xlabel("Classe prevista"); ax.set_ylabel("Classe real"); ax.set_title(n)
plt.tight_layout(); plt.savefig(f"{OUT}/matrizes_confusao.png", dpi=200); plt.close()

# 6.4 Distribuição das classes
fig, ax = plt.subplots(figsize=(4, 3))
cont = y.value_counts().sort_index()
ax.bar(["Benigno (0)", "Maligno (1)"], cont.values, color=["#4c72b0", "#c44e52"])
for i, v in enumerate(cont.values): ax.text(i, v + 4, str(v), ha="center")
ax.set_ylabel("Registros"); ax.set_title("Distribuição das classes")
plt.tight_layout(); plt.savefig(f"{OUT}/distribuicao_classes.png", dpi=200); plt.close()

# 6.5 Importância das variáveis (Random Forest)
rf = modelos["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values().tail(10)
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.barh(imp.index, imp.values, color="#4c72b0")
ax.set_xlabel("Importância"); ax.set_title("10 variáveis mais importantes (Random Forest)")
plt.tight_layout(); plt.savefig(f"{OUT}/importancia_variaveis.png", dpi=200); plt.close()

with open(f"{OUT}/resultados.json", "w", encoding="utf-8") as f:
    json.dump({"teste": resultados,
               "cv": {n: {m: list(v) for m, v in d.items()} for n, d in cv_res.items()},
               "matrizes": {n: m.tolist() for n, m in matrizes.items()},
               "top_vars": list(imp.index[::-1])}, f, ensure_ascii=False, indent=1)
print("\nFiguras salvas em:", OUT)
