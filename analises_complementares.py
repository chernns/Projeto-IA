"""
Projeto IA - Diagnóstico de Câncer de Mama (4ª Entrega - Artigo)
Análises complementares ao treinamento_classificacao_v3.py:
  1. Curvas ROC e AUC no conjunto de teste (hold-out), com os hiperparâmetros já selecionados
  2. Validação cruzada aninhada refeita com registro por fold (mesmos folds e sementes do script principal):
     métricas, AUC, previsões por caso e tempo médio por fold
  3. Matrizes de confusão agregadas (todas as previsões da validação aninhada)
  4. Teste t corrigido para amostragem repetida (Nadeau e Bengio, 2003) entre a MLP + SMOTE e os demais modelos
  5. Análise dos casos malignos não identificados (falsos negativos) no conjunto de teste
Requer: BASE_CANCER_MAMA.csv e figuras_v3/resultados_v3.json (gerado pelo script principal).
Saídas em "figuras_v3/".
"""
import json, time, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
warnings.filterwarnings("ignore")

SEED = 42
OUT = "figuras_v3"
res = json.load(open(f"{OUT}/resultados_v3.json", encoding="utf-8"))

df = pd.read_csv("BASE_CANCER_MAMA.csv")
df = df.drop(columns=[c for c in ["id", "Unnamed: 32"] if c in df.columns])
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
X, y = df.drop(columns=["diagnosis"]), df["diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)

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
    passos = ([("scaler", StandardScaler())] if escala else []) + ([("smote", SMOTE(random_state=SEED))] if smote else []) + [("clf", clf)]
    return Pipeline(passos), grade
configs = [(f"{n} + SMOTE" if s else n, n, s) for n in algoritmos for s in (False, True)]
nomes = [c[0] for c in configs]
def escore(modelo, Xv):
    return modelo.decision_function(Xv) if hasattr(modelo, "decision_function") else modelo.predict_proba(Xv)[:, 1]

# ---------------- 1. ROC no hold-out (hiperparâmetros já selecionados) ----------------
roc, auc_teste, pred_teste = {}, {}, {}
for rot, nome, smote in configs:
    pipe, _ = montar(nome, smote)
    p = {f"clf__{k}": (tuple(v) if isinstance(v, list) else v) for k, v in res["params"][rot].items()}
    pipe.set_params(**p).fit(X_train, y_train)
    sc = escore(pipe, X_test); pred = pipe.predict(X_test)
    assert confusion_matrix(y_test, pred).tolist() == res["matrizes"][rot], rot   # confere com o script principal
    fpr, tpr, _ = roc_curve(y_test, sc)
    roc[rot] = (fpr, tpr); auc_teste[rot] = roc_auc_score(y_test, sc); pred_teste[rot] = pred
    print(f"[ROC teste] {rot:28s} AUC={auc_teste[rot]:.4f}", flush=True)

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
cores = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
for ax, smote, tit in zip(axes, (False, True), ("Sem tratamento", "Com SMOTE")):
    for (nome, cor, ls) in zip(algoritmos, cores, ("-", "--", "-.", ":")):
        rot = f"{nome} + SMOTE" if smote else nome
        ax.step(*roc[rot], where="post", color=cor, lw=1.8, ls=ls, label=f"{nome} (AUC = {auc_teste[rot]:.3f})")
    ax.set_xlim(-0.005, 0.25); ax.set_ylim(0.70, 1.01)
    ax.set_xlabel("Taxa de falsos positivos"); ax.set_title(tit, fontsize=11); ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Taxa de verdadeiros positivos (recall)")
plt.tight_layout(); plt.savefig(f"{OUT}/curvas_roc.png", dpi=200); plt.close()

# ---------------- 2. Validação aninhada com registro por fold ----------------
externo = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
splits = list(externo.split(X, y))
folds = {r: {"Acurácia": [], "Precisão": [], "Recall": [], "F1-Score": [], "AUC": [], "tempo": []} for r in nomes}
agreg = {r: np.zeros((2, 2), int) for r in nomes}
erros = {r: np.zeros(len(y), int) for r in nomes}       # nº de vezes que cada caso foi classificado errado
for rot, nome, smote in configs:
    pipe, grade = montar(nome, smote)
    for tr, te in splits:
        t0 = time.time()
        gs = GridSearchCV(pipe, grade, cv=StratifiedKFold(3, shuffle=True, random_state=SEED), scoring="f1", n_jobs=1)
        gs.fit(X.iloc[tr], y.iloc[tr])
        yp = gs.predict(X.iloc[te]); sc = escore(gs.best_estimator_, X.iloc[te]); yt = y.iloc[te]
        f = folds[rot]; f["tempo"].append(time.time() - t0)
        f["Acurácia"].append(accuracy_score(yt, yp)); f["Precisão"].append(precision_score(yt, yp))
        f["Recall"].append(recall_score(yt, yp)); f["F1-Score"].append(f1_score(yt, yp)); f["AUC"].append(roc_auc_score(yt, sc))
        agreg[rot] += confusion_matrix(yt, yp, labels=[0, 1])
        erros[rot][te] += (yp != yt.values).astype(int)
    ant = res["cv"][rot]
    ok = all(abs(np.mean(folds[rot][m]) - ant[m][0]) < 1e-9 for m in ("Acurácia", "Precisão", "Recall", "F1-Score"))
    print(f"[CV aninhada] {rot:28s} AUC={np.mean(folds[rot]['AUC']):.4f}±{np.std(folds[rot]['AUC']):.4f} "
          f"tempo/fold={np.mean(folds[rot]['tempo']):.1f}s | idêntico ao script principal: {ok}", flush=True)

# ---------------- 3. Teste t corrigido (Nadeau e Bengio, 2003) ----------------
k = len(splits); n_tr, n_te = len(splits[0][0]), len(splits[0][1])
def t_corrigido(a, b):
    d = np.array(a) - np.array(b); var = d.var(ddof=1)
    if var == 0: return float(d.mean()), np.nan, 1.0
    t = d.mean() / np.sqrt((1 / k + n_te / n_tr) * var)
    return float(d.mean()), float(t), float(2 * stats.t.sf(abs(t), k - 1))
ref = "MLP + SMOTE"; comparacao = {}
for rot in nomes:
    if rot == ref: continue
    comparacao[rot] = {m: dict(zip(("dif_media", "t", "p"), t_corrigido(folds[ref][m], folds[rot][m])))
                       for m in ("Recall", "F1-Score", "AUC")}
    print(f"[t corrigido] {ref} vs {rot:26s}", " | ".join(f"{m}: dif={v['dif_media']:+.4f} p={v['p']:.3f}" for m, v in comparacao[rot].items()))

# ---------------- 4. Falsos negativos do hold-out ----------------
mal_idx = np.where(y_test.values == 1)[0]
tab_fn = []
for i in mal_idx:
    errou = [r for r in nomes if pred_teste[r][i] == 0]
    if errou: tab_fn.append({"indice_df": int(y_test.index[i]), "n_modelos_erraram": len(errou), "modelos": errou})
tab_fn.sort(key=lambda d: -d["n_modelos_erraram"])
print("\nFalsos negativos no teste (caso, nº de configurações que erraram):")
for d in tab_fn: print(" ", d["indice_df"], d["n_modelos_erraram"], "/ 8")
mal_te = X_test[y_test == 1]
nunca = [d["indice_df"] for d in tab_fn]
cols = ["area_worst", "perimeter_worst", "concavity_worst", "concave points_worst", "radius_mean"]
carac = {"fn_persistentes": {c: float(X.loc[[d["indice_df"] for d in tab_fn if d["n_modelos_erraram"] >= 4], c].mean()) for c in cols},
         "malignos_teste_corretos": {c: float(mal_te.drop(index=[d["indice_df"] for d in tab_fn], errors="ignore")[c].mean()) for c in cols},
         "benignos_treino": {c: float(X_train[y_train == 0][c].mean()) for c in cols}}
print(json.dumps(carac, indent=1))

# casos errados com maior frequência na validação aninhada (MLP + SMOTE)
freq = pd.Series(erros[ref], index=y.index)
print("\nMLP + SMOTE: casos com erro em ambas as repetições:", int((freq == 2).sum()), "| em ao menos uma:", int((freq >= 1).sum()))

# ---------------- 5. Saída ----------------
saida = {"auc_teste": auc_teste,
         "cv_auc": {r: [float(np.mean(folds[r]["AUC"])), float(np.std(folds[r]["AUC"]))] for r in nomes},
         "tempo_por_fold_s": {r: float(np.mean(folds[r]["tempo"])) for r in nomes},
         "matriz_agregada": {r: agreg[r].tolist() for r in nomes},
         "comparacao_corrigida": comparacao, "fn_teste": tab_fn, "caracteristicas": carac,
         "folds": {r: {m: list(map(float, v)) for m, v in folds[r].items()} for r in nomes},
         "erro_ambas_repeticoes_MLP_SMOTE": int((freq == 2).sum()), "erro_ao_menos_uma_MLP_SMOTE": int((freq >= 1).sum())}
json.dump(saida, open(f"{OUT}/resultados_complementares.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("Concluído.")
