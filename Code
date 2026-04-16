# MachineLearning
# RF-XGB-ArbolDecisi-n_ARGEDUCATIVO.ipynb
# ============================================================
# Árbol de Decisión (binario) con Trayectorias + EPH
# Target: riesgo_bin (0=NoRiesgo, 1=Riesgo) o "NoRiesgo/Riesgo"
# Holdout = 2022 + LOO por región/año
# Requiere DF_PLUS con columnas:
# ["Región","Año","Promoción","Repitencia","acceso_internet",
#  "ipcf_mean","q1_share","q5_share","q5_q1_gap","q5_q1_ratio"]
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay,
                             make_scorer)
from sklearn.model_selection import StratifiedKFold, GridSearchCV, GroupKFold
from sklearn.tree import DecisionTreeClassifier

DFP = DF_PLUS.copy()

# --- columnas necesarias
NUM_COLS = ["Promoción","Repitencia","acceso_internet","ipcf_mean",
            "q1_share","q5_share","q5_q1_gap","q5_q1_ratio"]
CAT_COLS = ["Región","Año"]
for c in NUM_COLS + CAT_COLS:
    if c not in DFP.columns:
        raise ValueError(f"Falta la columna: {c}")

# --- target binario: acepta 0/1 o "Riesgo/NoRiesgo"
def to_bin(y_series):
    if pd.api.types.is_numeric_dtype(y_series):
        return y_series.astype(int)
    s = y_series.astype(str).str.strip().str.lower()
    return (s == "riesgo").astype(int)

if "riesgo_bin" in DFP.columns:
    y_all = to_bin(DFP["riesgo_bin"])
elif "riesgo" in DFP.columns:
    y_all = to_bin(DFP["riesgo"])
else:
    # si no existe, construir desde Abandono por terciles dentro de cada año
    if "Abandono" not in DFP.columns:
        raise ValueError("No existe 'riesgo_bin', 'riesgo' ni 'Abandono' para construir el target.")
    def tercil_por_anio(g):
        return pd.qcut(g["Abandono"].rank(method="first"), 3, labels=["bajo","medio","alto"])
    DFP["riesgo"] = DFP.groupby("Año", group_keys=False).apply(tercil_por_anio)
    y_all = to_bin(DFP["riesgo"])

print("Clases bin:\n", y_all.value_counts(dropna=False))

# --- features
X_all = DFP[NUM_COLS + CAT_COLS].copy()

# --- helpers OHE sin fuga
def fit_ohe(X_cat):
    if X_cat is None or X_cat.shape[1] == 0:
        return None
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc.fit(X_cat)
    return enc

def transform_features(X_df, enc, num_cols, cat_cols):
    X_num = X_df[num_cols].astype(float).to_numpy() if num_cols else np.empty((len(X_df),0))
    if cat_cols and enc is not None:
        X_cat = enc.transform(X_df[cat_cols])
        X_mat = np.hstack([X_num, X_cat])
        names = num_cols + list(enc.get_feature_names_out(cat_cols))
    else:
        X_mat, names = X_num, num_cols
    return X_mat, names

# =========================
# 1) Holdout: test = 2022
# =========================
mask_te = DFP["Año"] == 2022
X_tr_df, X_te_df = X_all.loc[~mask_te], X_all.loc[mask_te]
y_tr_bin, y_te_bin = y_all.loc[~mask_te], y_all.loc[mask_te]

enc = fit_ohe(X_tr_df[CAT_COLS] if CAT_COLS else None)
X_tr, feat_names = transform_features(X_tr_df, enc, NUM_COLS, CAT_COLS)
X_te, _          = transform_features(X_te_df, enc, NUM_COLS, CAT_COLS)

# --- grid y búsqueda (optimiza F1 de la clase positiva=1)
grid = {
    "max_depth": [2, 3, 4, None],
    "min_samples_leaf": [1, 2, 3],
    "min_samples_split": [2, 3, 4],
    "ccp_alpha": [0.0, 0.001, 0.01]
}
dt = DecisionTreeClassifier(random_state=42, class_weight="balanced")
cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
gs = GridSearchCV(
    dt, param_grid=grid,
    scoring=make_scorer(f1_score, pos_label=1),
    cv=cv_inner, n_jobs=-1
)
gs.fit(X_tr, y_tr_bin)

best_tree = gs.best_estimator_
print("Mejores hiperparámetros árbol:", gs.best_params_)
print("CV interno (F1 pos):", round(gs.best_score_, 3))

# --- evaluación holdout
pred = best_tree.predict(X_te)
# proba de la clase positiva (1)
pos_idx = np.where(best_tree.classes_ == 1)[0][0]
proba = best_tree.predict_proba(X_te)[:, pos_idx]

print("\n===== Árbol de Decisión — Holdout 2022 =====")
f1 = f1_score(y_te_bin, pred, pos_label=1)
print("F1 (bin, test):", round(f1, 3))
print(classification_report(
    np.where(y_te_bin==1,"Riesgo","NoRiesgo"),
    np.where(pred==1,"Riesgo","NoRiesgo"))
)

cm = confusion_matrix(y_te_bin, pred, labels=[0,1])
ConfusionMatrixDisplay(cm, display_labels=["NoRiesgo","Riesgo"]).plot(cmap="viridis")
plt.title("Matriz de confusión — Árbol (Holdout 2022)")
plt.show()

auc = roc_auc_score(y_te_bin, proba)
RocCurveDisplay.from_predictions(y_te_bin, proba)
plt.title(f"ROC — Árbol (Holdout 2022) | AUC={auc:.3f}")
plt.show()

# --- importancias
importances = pd.Series(best_tree.feature_importances_, index=feat_names)
imp_top = importances.sort_values(ascending=False).head(12)
imp_top.plot(kind="barh"); plt.gca().invert_yaxis()
plt.title("Importancias — Árbol (bin)"); plt.show()

# ============================================================
# 2) Robustez LOO por región y por año
# ============================================================
tree_params = {k: v for k, v in best_tree.get_params().items()
               if k in ["max_depth","min_samples_leaf","min_samples_split","ccp_alpha"]}

def group_cv_scores_tree(X_df, y_vec, groups, num_cols, cat_cols, tree_params):
    gkf = GroupKFold(n_splits=len(np.unique(groups)))
    scores = []
    for tr_idx, te_idx in gkf.split(X_df, y_vec, groups):
        Xt_df, Xv_df = X_df.iloc[tr_idx], X_df.iloc[te_idx]
        yt, yv = y_vec.iloc[tr_idx], y_vec.iloc[te_idx]
        enc_g = fit_ohe(Xt_df[cat_cols] if cat_cols else None)
        Xt, _ = transform_features(Xt_df, enc_g, num_cols, cat_cols)
        Xv, _ = transform_features(Xv_df, enc_g, num_cols, cat_cols)
        clf = DecisionTreeClassifier(random_state=42, class_weight="balanced", **tree_params)
        clf.fit(Xt, yt)
        pred = clf.predict(Xv)
        scores.append(f1_score(yv, pred, pos_label=1))
    return np.array(scores)

scores_reg = group_cv_scores_tree(X_all, y_all, DFP["Región"], NUM_COLS, CAT_COLS, tree_params)
print("\n=== Robustez por región (leave-one-region-out) — Árbol ===")
print(f"F1 bin prom={scores_reg.mean():.3f}  | desvío={scores_reg.std():.3f}")

scores_year = group_cv_scores_tree(X_all, y_all, DFP["Año"], NUM_COLS, CAT_COLS, tree_params)
print("\n=== Robustez por año (leave-one-year-out) — Árbol ===")
print(f"F1 bin prom={scores_year.mean():.3f}  | desvío={scores_year.std():.3f}")

# ================================
# XGBoost + EPH (target binario, compatible scikit-learn 1.0–1.5)
# ================================
!pip -q install xgboost

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             f1_score, roc_auc_score, RocCurveDisplay)
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# --------------------------------
# (1) CARGA / PREP DE DATOS
# --------------------------------
# Si YA tenés DF_PLUS en memoria, no hagas nada.
# Si no, descomentá para cargar:
# DF_PLUS = pd.read_excel("/content/base_te_ai.xlsx")

assert 'Región' in DF_PLUS.columns and 'Año' in DF_PLUS.columns, "Faltan columnas Región/Año en DF_PLUS"

def construir_riesgo_binario(df):
    df = df.copy()
    if 'riesgo' in df.columns:
        df['riesgo_bin'] = np.where(df['riesgo'].astype(str).str.lower()=='alto', 'Riesgo', 'NoRiesgo')
    elif 'Abandono' in df.columns:
        def tercil_por_anio(g):
            return pd.qcut(g['Abandono'].rank(method='first'), 3, labels=['bajo','medio','alto'])
        df['riesgo'] = df.groupby('Año', group_keys=False).apply(tercil_por_anio)
        df['riesgo_bin'] = np.where(df['riesgo'].astype(str).str.lower()=='alto', 'Riesgo', 'NoRiesgo')
    else:
        raise ValueError("No hay columna 'riesgo' ni 'Abandono' para construir el target.")
    return df

DFP = construir_riesgo_binario(DF_PLUS)
print("Clases bin:\n", DFP['riesgo_bin'].value_counts())

# Features
NUM_COLS_CAND = ["Promoción","Repitencia","acceso_internet","ipcf_mean",
                 "q1_share","q5_share","q5_q1_gap","q5_q1_ratio"]
CAT_COLS_CAND = ["Región","Año"]

NUM_COLS = [c for c in NUM_COLS_CAND if c in DFP.columns]
CAT_COLS = [c for c in CAT_COLS_CAND if c in DFP.columns]
assert len(NUM_COLS) > 0, "No hay features numéricas disponibles."

X = DFP[NUM_COLS + CAT_COLS].copy()
y = (DFP['riesgo_bin'] == "Riesgo").astype(int)  # 1 = Riesgo, 0 = NoRiesgo

# --------------------------------
# Helpers OHE compatibles
# --------------------------------
def fit_ohe(X_cat):
    """Entrena OneHotEncoder compatible (sparse_output o sparse según versión)."""
    if X_cat is None or X_cat.shape[1] == 0:
        return None
    try:
        enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    enc.fit(X_cat)
    return enc

def transform_features(X_part, enc, num_cols, cat_cols):
    Xn = X_part[num_cols].to_numpy().astype(float)
    if enc is None or len(cat_cols) == 0:
        return Xn, num_cols
    Xc = enc.transform(X_part[cat_cols])
    feat_names = num_cols + enc.get_feature_names_out(cat_cols).tolist()
    X_all = np.hstack([Xn, Xc])
    return X_all, feat_names

# --------------------------------
# (2) Holdout por año (2022)
# --------------------------------
TEST_YEAR = 2022 if 2022 in DFP["Año"].unique() else int(sorted(DFP["Año"].unique())[-1])
mask_tr = DFP["Año"] != TEST_YEAR
mask_te = DFP["Año"] == TEST_YEAR

X_tr_df, y_tr = X.loc[mask_tr], y.loc[mask_tr]
X_te_df, y_te = X.loc[mask_te], y.loc[mask_te]

enc_hold = fit_ohe(X_tr_df[CAT_COLS] if CAT_COLS else None)
X_tr, feat_names = transform_features(X_tr_df, enc_hold, NUM_COLS, CAT_COLS)
X_te, _          = transform_features(X_te_df, enc_hold, NUM_COLS, CAT_COLS)

xgb = XGBClassifier(
    n_estimators=700,
    max_depth=5,
    learning_rate=0.07,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    tree_method='hist'
)

xgb.fit(X_tr, y_tr)
y_pred = (xgb.predict_proba(X_te)[:,1] >= 0.5).astype(int)

print(f"\n=== XGBoost + EPH — Holdout {TEST_YEAR} ===")
print("F1 (bin, test):", round(f1_score(y_te, y_pred, pos_label=1), 3))
print(classification_report(y_te, y_pred, target_names=["NoRiesgo","Riesgo"], digits=3))

# Matriz de confusión
fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_te, y_pred, display_labels=["NoRiesgo","Riesgo"], ax=ax)
ax.set_title(f"Matriz de confusión — XGB (Holdout {TEST_YEAR})")
plt.show()

# ROC AUC
proba_te = xgb.predict_proba(X_te)[:,1]
auc = roc_auc_score(y_te, proba_te)
RocCurveDisplay.from_predictions(y_te, proba_te)
plt.title(f"ROC — XGB (Holdout {TEST_YEAR}) | AUC={auc:.3f}")
plt.show()

# --------------------------------
# (3) Robustez: LORO (región) y LOYO (año)
# --------------------------------
def group_cv_scores_xgb(X_df, y_vec, groups, num_cols, cat_cols, model_params):
    gkf = GroupKFold(n_splits=len(np.unique(groups)))
    scores = []
    for tr_idx, te_idx in gkf.split(X_df, y_vec, groups):
        Xt_df, yt = X_df.iloc[tr_idx], y_vec.iloc[tr_idx]
        Xv_df, yv = X_df.iloc[te_idx], y_vec.iloc[te_idx]
        enc = fit_ohe(Xt_df[cat_cols] if cat_cols else None)
        Xt, _ = transform_features(Xt_df, enc, num_cols, cat_cols)
        Xv, _ = transform_features(Xv_df, enc, num_cols, cat_cols)
        clf = XGBClassifier(**model_params)
        clf.fit(Xt, yt)
        pred = (clf.predict_proba(Xv)[:,1] >= 0.5).astype(int)
        scores.append(f1_score(yv, pred, pos_label=1))
    return np.array(scores)

params = dict(
    n_estimators=700, max_depth=5, learning_rate=0.07,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    objective='binary:logistic', eval_metric='logloss',
    random_state=42, tree_method='hist'
)

scores_reg = group_cv_scores_xgb(X, y, DFP["Región"], NUM_COLS, CAT_COLS, params)
print("\n=== Robustez por región (leave-one-region-out) — XGB ===")
print(f"F1 bin prom={scores_reg.mean():.3f}  | desvío={scores_reg.std():.3f}")

scores_year = group_cv_scores_xgb(X, y, DFP["Año"], NUM_COLS, CAT_COLS, params)
print("\n=== Robustez por año (leave-one-year-out) — XGB ===")
print(f"F1 bin prom={scores_year.mean():.3f} | desvío={scores_year.std():.3f}")

# --------------------------------
# (4) Importancia de variables (entrenado en todo el set)
# --------------------------------
enc_full = fit_ohe(X[CAT_COLS] if CAT_COLS else None)
X_full, feat_names = transform_features(X, enc_full, NUM_COLS, CAT_COLS)
xgb_full = XGBClassifier(**params).fit(X_full, y)
imps = pd.Series(xgb_full.feature_importances_, index=feat_names).sort_values(ascending=False)

print("\nTop importancias — XGB:\n", imps.head(12).round(4))
imps.head(12).iloc[::-1].plot(kind="barh")
plt.title("Importancia de variables — XGB (entrenado en todo el set)")
plt.xlabel("Gain-based importance")
plt.tight_layout()
plt.show()


# ================================
# Random Forest + EPH (target binario) DEFINITIVO
# ================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, roc_auc_score, RocCurveDisplay
)
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

# --------------------------------
# (1) CARGA DE DATOS
# --------------------------------
# Si YA tenés DF_PLUS en memoria, no hagas nada.
# Si no, podés descomentar estas líneas y cargar tu base base_te_ai.xlsx ya preparada:
# DF_PLUS = pd.read_excel("/content/base_te_ai.xlsx")  # Ajustá la ruta si hace falta

assert 'Región' in DF_PLUS.columns and 'Año' in DF_PLUS.columns, "Faltan columnas Región/Año en DF_PLUS"

# --------------------------------
# (2) Construcción de target binario
# --------------------------------
def construir_riesgo_binario(df):
    df = df.copy()
    if 'riesgo' in df.columns:
        # Mapear alto -> Riesgo ; medio/bajo -> NoRiesgo
        df['riesgo_bin'] = np.where(df['riesgo'].astype(str).str.lower()=='alto', 'Riesgo', 'NoRiesgo')
    elif 'Abandono' in df.columns:
        # Construir terciles por año y luego binarizar (alto = top tercil)
        def tercil_por_anio(g):
            return pd.qcut(g['Abandono'].rank(method='first'), 3, labels=['bajo','medio','alto'])
        df['riesgo'] = df.groupby('Año', group_keys=False).apply(tercil_por_anio)
        df['riesgo_bin'] = np.where(df['riesgo'].astype(str).str.lower()=='alto', 'Riesgo', 'NoRiesgo')
    else:
        raise ValueError("No hay columna 'riesgo' ni 'Abandono' para construir el target.")
    return df

DFP = construir_riesgo_binario(DF_PLUS)

print("Clases bin:", DFP['riesgo_bin'].value_counts())

# --------------------------------
# (3) Features
# --------------------------------
NUM_CANDIDATAS = ["Promoción","Repitencia","acceso_internet","ipcf_mean",
                  "q1_share","q5_share","q5_q1_gap","q5_q1_ratio"]
CAT_CANDIDATAS = ["Región","Año"]

# Usar solo las columnas que existan
NUM_COLS = [c for c in NUM_CANDIDATAS if c in DFP.columns]
CAT_COLS = [c for c in CAT_CANDIDATAS if c in DFP.columns]
assert len(NUM_COLS) > 0, "No hay features numéricas disponibles."

X = DFP[NUM_COLS + CAT_COLS].copy()
y = DFP['riesgo_bin'].astype(str)

# --------------------------------
# (4) Preprocesador + Modelo
# --------------------------------
pre = ColumnTransformer(
    transformers=[
        ("num", "passthrough", NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown='ignore'), CAT_COLS)
    ],
    remainder='drop'
)

rf = RandomForestClassifier(
    n_estimators=600,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

pipe_rf = Pipeline([("pre", pre), ("clf", rf)])

# --------------------------------
# (5) Holdout por año (2022)
# --------------------------------
TEST_YEAR = 2022
mask_tr = DFP["Año"] != TEST_YEAR
mask_te = DFP["Año"] == TEST_YEAR

X_tr, y_tr = X.loc[mask_tr], y.loc[mask_tr]
X_te, y_te = X.loc[mask_te], y.loc[mask_te]

pipe_rf.fit(X_tr, y_tr)
y_pred = pipe_rf.predict(X_te)

# F1 binario (Riesgo = clase positiva)
f1_bin = f1_score(y_te, y_pred, pos_label="Riesgo")
print("\n=== RF + EPH — Holdout", TEST_YEAR, "===")
print("F1 (bin, test):", round(f1_bin, 3))
print(classification_report(y_te, y_pred, digits=3))

# Confusion matrix
fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_te, y_pred, ax=ax)
ax.set_title(f"Matriz de confusión — RF (Holdout {TEST_YEAR})")
plt.show()

# ROC AUC
if hasattr(pipe_rf.named_steps['clf'], "predict_proba"):
    proba = pipe_rf.predict_proba(X_te)
    # encontrar índice de la clase 'Riesgo'
    idx_riesgo = list(pipe_rf.named_steps['clf'].classes_).index('Riesgo')
    auc = roc_auc_score((y_te=="Riesgo").astype(int), proba[:, idx_riesgo])
    RocCurveDisplay.from_predictions((y_te=="Riesgo").astype(int), proba[:, idx_riesgo])
    plt.title(f"ROC — RF (Holdout {TEST_YEAR}) | AUC={auc:.3f}")
    plt.show()

# --------------------------------
# (6) Robustez: Leave-One-Region-Out y Leave-One-Year-Out
# --------------------------------
f1_sc = make_scorer(f1_score, pos_label="Riesgo")

# LORO (grupos = Región)
gkf_region = GroupKFold(n_splits=DFP["Región"].nunique())
scores_reg = cross_val_score(pipe_rf, X, y, cv=gkf_region, groups=DFP["Región"], scoring=f1_sc)
print("\n=== Robustez por región (leave-one-region-out) ===")
print(f"RF — F1 bin prom={scores_reg.mean():.3f}  | desvío={scores_reg.std():.3f}")

# LOYO (grupos = Año)
gkf_year = GroupKFold(n_splits=DFP["Año"].nunique())
scores_year = cross_val_score(pipe_rf, X, y, cv=gkf_year, groups=DFP["Año"], scoring=f1_sc)
print("\n=== Robustez por año (leave-one-year-out) ===")
print(f"RF — F1 bin prom={scores_year.mean():.3f} | desvío={scores_year.std():.3f}")

# --------------------------------
# (7) Importancias de variables
# --------------------------------
# Ajustar en todo el dataset para obtener importancias (esto NO es para métricas)
pipe_rf.fit(X, y)
rf_fit = pipe_rf.named_steps['clf']
# Nombres de columnas transformadas
feat_names = []
# num
feat_names += NUM_COLS
# cat
if CAT_COLS:
    ohe = pipe_rf.named_steps['pre'].named_transformers_['cat']
    cat_names = ohe.get_feature_names_out(CAT_COLS).tolist()
    feat_names += cat_names

importances = pd.Series(rf_fit.feature_importances_, index=feat_names).sort_values(ascending=False)
print("\nTop importancias (RF):\n", importances.head(10).round(4))

# Plot
importances.head(12).iloc[::-1].plot(kind="barh")
plt.title("Importancia de variables — RF (entrenado en todo el set)")
plt.xlabel("Gini importance")
plt.tight_layout()
plt.show()
