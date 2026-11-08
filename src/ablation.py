"""
Ablaciones del MLP de scikit-learn que pide el enunciado: estrategia de
codificacion como factor experimental (pregunta P2 / hipotesis H2), dimension del
hashing, cruce de variables, profundidad del MLP y ponderacion de clases.

Diseno (declarado antes de entrenar):
- Un factor a la vez alrededor de la configuracion base, que es la del
  experimento principal: codificacion combinada (hashing 2^12 de todas las
  categoricas + frecuencia y target encoding de las de alta cardinalidad), cruce
  site_id x device_type, MLP (50,) con los hiperparametros afinados
  (artifacts/sklearn/best_params.json), sin ponderacion.
- Estrategias de codificacion aisladas (P2/H2), todas sobre el mismo conjunto de
  columnas (todas las categoricas + el cruce): solo hashing 2^12, solo frecuencia,
  solo target encoding out-of-fold. Se mide ademas la memoria de la matriz de
  entrenamiento (bytes de los arreglos de la matriz dispersa o densa).
- Dimension del hashing: 2^12 (base), 2^14 y 2^16, los tres niveles del
  enunciado, mas 2^10 como cota inferior.
- Variantes restantes: sin cruce; dos capas (50, 50) en vez de una (solo cambia
  la profundidad, no el ancho); ponderacion balanceada de clases.
- MLPClassifier de scikit-learn 1.5 no acepta class_weight ni sample_weight, asi
  que la ponderacion se emula sobremuestreando los positivos de train hasta
  igualar a los negativos, que en esperanza equivale a pesos balanceados. Los
  encoders se ajustan sobre el train original, no sobre el sobremuestreado.
  La variante "ponderacion_balanceada_val_limpia" corrige un defecto de la
  primera: separa la validacion de parada temprana ANTES de sobremuestrear (ver
  el comentario en CONFIGS).
- 5 semillas; en cada semilla todas las variantes usan la MISMA muestra de
  entrenamiento y la misma inicializacion, asi que el contraste contra la base
  es pareado por semilla (t pareada, TOST pareado, d_z, Holm sobre las 5).
- TRAIN_ROWS filas de train por semilla (dias 21-28), validacion y prueba fijas
  (dias 29 y 30). El muestreo usa subconsulta para que el WHERE se aplique
  antes del USING SAMPLE de DuckDB.
"""
from __future__ import annotations

import json
import logging

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.weightstats import ttost_paired

from config import (
    ARTIFACTS_DIR, DATA_PROCESSED_DIR, HIGH_CARDINALITY_CATEGORICALS, LABEL_COL,
    NUMERIC_LIKE_CATEGORICALS, SEEDS, TEST_DAY, TRAIN_DAYS, VALIDATION_DAY,
)
from metrics import choose_threshold_on_validation, full_metric_report
from preprocessing import FrequencyEncoder, HashingEncoder, OutOfFoldTargetEncoder, RareCategoryGrouper
from sklearn_pipeline import (
    CATEGORICAL_COLS, CROSS_A, CROSS_B, RARE_THRESHOLD, FittedPipeline, _build_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ablation")

ABL_DIR = ARTIFACTS_DIR / "ablation"
ABL_DIR.mkdir(parents=True, exist_ok=True)
RUNS_PATH = ABL_DIR / "runs.csv"
PARQUET_GLOB = (DATA_PROCESSED_DIR / "day=*" / "*.parquet").as_posix()

TRAIN_ROWS = 200_000
EVAL_ROWS = 200_000
ALPHA = 0.05
DELTA_AUC = 0.01
CROSS_COL = f"{CROSS_A}_x_{CROSS_B}"

def _cfg(encoding="combinada", hash_dim=2**12, cross=True, hidden="50", balance=False, extra=False):
    return {"encoding": encoding, "hash_dim": hash_dim, "cross": cross, "hidden": hidden,
            "balance": balance, "extra_cols": extra}


# Columnas que el experimento principal NO usa (omision detectada al redactar el
# libro): las 9 anonimizadas C* y la hora. La variante "con_C_y_hora" las agrega
# como categoricas dentro del hashing para cuantificar el efecto de esa omision.
EXTRA_COLS = NUMERIC_LIKE_CATEGORICALS + ["hour_of_day"]


CONFIGS = {
    "base":                   _cfg(),
    "solo_hashing":           _cfg(encoding="solo_hashing"),
    "solo_frecuencia":        _cfg(encoding="solo_frecuencia"),
    "solo_target_encoding":   _cfg(encoding="solo_target_encoding"),
    "hashing_2^10":           _cfg(hash_dim=2**10),
    "hashing_2^14":           _cfg(hash_dim=2**14),
    "hashing_2^16":           _cfg(hash_dim=2**16),
    "sin_cruce":              _cfg(cross=False),
    "dos_capas_50_50":        _cfg(hidden="50,50"),
    "ponderacion_balanceada": _cfg(balance=True),
    "con_C_y_hora":           _cfg(extra=True),
    # Correccion de "ponderacion_balanceada" (agregada al ver sus resultados): ahi
    # el sobremuestreo ocurre ANTES de la particion interna de parada temprana del
    # MLPClassifier, asi que copias del mismo positivo caen en entrenamiento y en
    # validacion; la validacion deja de detectar el sobreajuste (86,8 epocas frente
    # a 18,2 de la base). Aqui se separa primero una validacion limpia (10 %,
    # estratificada) y solo se sobremuestrea el 90 % restante; la regla de parada
    # replica la de scikit-learn (accuracy de validacion, tol, n_iter_no_change).
    "ponderacion_balanceada_val_limpia": _cfg(balance="val_limpia"),
    # Segunda correccion (agregada al ver la anterior): con entrenamiento
    # balanceado y validacion con la tasa base real, el accuracy con umbral 0,5
    # sigue subiendo mientras el modelo se corre hacia la clase mayoritaria, asi
    # que la parada temprana por accuracy no detecta el sobreajuste. Aqui la
    # parada usa el ROC AUC de validacion, que no depende de la tasa base.
    "ponderacion_balanceada_val_limpia_auc": _cfg(balance="val_limpia_auc"),
}


def _sample(con, days: list[int], n: int, seed: int) -> pd.DataFrame:
    days_sql = ",".join(map(str, days))
    return con.execute(
        f"SELECT * FROM (SELECT * FROM read_parquet('{PARQUET_GLOB}') WHERE day IN ({days_sql})) "
        f"USING SAMPLE {n} ROWS (reservoir, {seed})"
    ).df()


def _fit_dense(train_df, y_train, seed) -> tuple[FittedPipeline, np.ndarray]:
    max_day = max(TRAIN_DAYS)
    rare = RareCategoryGrouper(threshold=RARE_THRESHOLD, columns=CATEGORICAL_COLS).fit(train_df, max_day)
    pipe = FittedPipeline(rare, FrequencyEncoder(columns=HIGH_CARDINALITY_CATEGORICALS), None,
                          OutOfFoldTargetEncoder(columns=HIGH_CARDINALITY_CATEGORICALS), StandardScaler())
    dense_tr = pipe.dense_block(train_df, fit=True, y=y_train, max_train_day=max_day, seed=seed)
    return pipe, dense_tr


def _sparse(pipe: FittedPipeline, df: pd.DataFrame, hash_dim: int, cross: bool, extra: bool = False) -> csr_matrix:
    cols = CATEGORICAL_COLS + ([CROSS_COL] if cross else []) + (EXTRA_COLS if extra else [])
    return HashingEncoder(columns=cols, n_features=hash_dim).transform(pipe._prep_categoricals(df)).tocsr()


def _single_encoding(pipe: FittedPipeline, frames, y_train, seed: int, kind: str):
    """Frecuencia o target encoding de TODAS las categoricas + el cruce, escalado
    con un StandardScaler ajustado en train. Devuelve matrices densas train/val/test."""
    cols = CATEGORICAL_COLS + [CROSS_COL]
    prepped = [pipe._prep_categoricals(df) for df in frames]
    max_day = max(TRAIN_DAYS)
    if kind == "solo_frecuencia":
        enc = FrequencyEncoder(columns=cols).fit(prepped[0], max_day)
        outs = [enc.transform(p)[[f"{c}_freq" for c in cols]].values for p in prepped]
    else:
        enc = OutOfFoldTargetEncoder(columns=cols)
        tr = enc.fit_transform(prepped[0], pd.Series(y_train, index=prepped[0].index), max_day, seed)
        outs = [tr[[f"{c}_te" for c in cols]].values] + \
               [enc.transform(p)[[f"{c}_te" for c in cols]].values for p in prepped[1:]]
    scaler = StandardScaler().fit(outs[0])
    return [scaler.transform(o) for o in outs]


def _nbytes(X) -> int:
    if hasattr(X, "indptr"):
        return int(X.data.nbytes + X.indices.nbytes + X.indptr.nbytes)
    return int(X.nbytes)


def _oversample_positives(X, y: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    extra = rng.choice(pos, size=len(neg) - len(pos), replace=True)
    order = rng.permutation(len(y) + len(extra))
    X_bal = vstack([X, X[extra]]).tocsr()[order]
    y_bal = np.concatenate([y, y[extra]])[order]
    return X_bal, y_bal


def _fit_balanced_clean_val(X, y: np.ndarray, params: dict, seed: int, criterio: str = "accuracy"):
    """Sobremuestreo de positivos con validacion de parada temprana limpia: la
    validacion se separa antes de duplicar, de modo que ninguna copia de un
    positivo de entrenamiento aparece en ella. Replica la regla de scikit-learn
    1.5: se detiene cuando lleva mas de n_iter_no_change epocas sin superar el
    mejor puntaje en tol y restaura los pesos de la mejor epoca. El puntaje es el
    accuracy de validacion (criterio="accuracy", el de scikit-learn) o el ROC AUC
    de validacion (criterio="auc"), que no depende de la tasa base ni del umbral."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    tr_idx, va_idx = train_test_split(np.arange(len(y)), test_size=0.1, stratify=y, random_state=seed)
    X_in, y_in = _oversample_positives(X[tr_idx], y[tr_idx], seed)
    X_va, y_va = X[va_idx], y[va_idx]
    model = _build_model("mlp", params, seed)
    model.set_params(early_stopping=False)
    best_score, best_state, best_epoch, bad = -np.inf, None, 0, 0
    for epoch in range(1, model.max_iter + 1):
        model.partial_fit(X_in, y_in, classes=np.array([0, 1]))
        score = (roc_auc_score(y_va, model.predict_proba(X_va)[:, 1]) if criterio == "auc"
                 else model.score(X_va, y_va))
        bad = bad + 1 if score < best_score + model.tol else 0
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = ([c.copy() for c in model.coefs_], [b.copy() for b in model.intercepts_])
        if bad > model.n_iter_no_change:
            break
    model.coefs_, model.intercepts_ = best_state
    model.n_iter_ = epoch  # partial_fit lo reinicia a 1 en cada llamada
    return model, len(y_in), epoch, best_epoch


def _done() -> set[tuple[str, int]]:
    if not RUNS_PATH.exists():
        return set()
    d = pd.read_csv(RUNS_PATH)
    return {(r.config, int(r.seed)) for r in d.itertuples()}


def run_all() -> None:
    con = duckdb.connect()
    con.execute("SET memory_limit='10GB'")
    con.execute("SET threads TO 1")  # con un hilo el reservoir sampling es determinista (reanudable)
    val_df = _sample(con, [VALIDATION_DAY], EVAL_ROWS, SEEDS[0])
    test_df = _sample(con, [TEST_DAY], EVAL_ROWS, SEEDS[0])
    y_val, y_test = val_df[LABEL_COL].values, test_df[LABEL_COL].values
    best = json.loads((ARTIFACTS_DIR / "sklearn" / "best_params.json").read_text(encoding="utf-8"))["mlp"]
    done = _done()

    for seed in SEEDS:
        pending = [c for c in CONFIGS if (c, seed) not in done]
        if not pending:
            continue
        train_df = _sample(con, TRAIN_DAYS, TRAIN_ROWS, seed)
        y_train = train_df[LABEL_COL].values
        pipe, dense_tr = _fit_dense(train_df, train_df[LABEL_COL], seed)
        dense_val, dense_test = pipe.dense_block(val_df, fit=False), pipe.dense_block(test_df, fit=False)
        sparse_cache: dict[tuple[int, bool], tuple] = {}

        for name in pending:
            cfg = CONFIGS[name]
            if cfg["encoding"] in ("solo_frecuencia", "solo_target_encoding"):
                X_tr, X_val, X_test = _single_encoding(
                    pipe, (train_df, val_df, test_df), y_train, seed, cfg["encoding"])
            else:
                key = (cfg["hash_dim"], cfg["cross"], cfg["extra_cols"])
                if key not in sparse_cache:
                    sparse_cache[key] = tuple(_sparse(pipe, d, *key) for d in (train_df, val_df, test_df))
                s_tr, s_val, s_test = sparse_cache[key]
                if cfg["encoding"] == "solo_hashing":
                    X_tr, X_val, X_test = s_tr, s_val, s_test
                else:
                    X_tr = hstack([s_tr, csr_matrix(dense_tr)]).tocsr()
                    X_val = hstack([s_val, csr_matrix(dense_val)]).tocsr()
                    X_test = hstack([s_test, csr_matrix(dense_test)]).tocsr()
            mem_mb = _nbytes(X_tr) / 1024**2
            params = {**best, "hidden_layer_sizes": cfg["hidden"]}
            t0 = pd.Timestamp.now()
            if cfg["balance"] in ("val_limpia", "val_limpia_auc"):
                crit = "auc" if cfg["balance"] == "val_limpia_auc" else "accuracy"
                model, n_fit, _, _ = _fit_balanced_clean_val(X_tr, y_train, params, seed, criterio=crit)
            else:
                X_fit, y_fit = _oversample_positives(X_tr, y_train, seed) if cfg["balance"] else (X_tr, y_train)
                model = _build_model("mlp", params, seed)
                model.fit(X_fit, y_fit)
                n_fit = len(y_fit)
            fit_s = (pd.Timestamp.now() - t0).total_seconds()

            thr = choose_threshold_on_validation(y_val, model.predict_proba(X_val)[:, 1])
            report = full_metric_report(y_test, model.predict_proba(X_test)[:, 1], thr)
            row = {"config": name, "seed": seed, **cfg, "n_train": len(y_train), "n_fit": n_fit,
                   "n_features": X_tr.shape[1], "mem_X_train_mb": mem_mb,
                   "fit_time_s": fit_s, "n_iter": model.n_iter_, **report}
            pd.DataFrame([row]).to_csv(RUNS_PATH, mode="a", header=not RUNS_PATH.exists(), index=False)
            log.info("  seed=%d %-24s roc_auc=%.4f log_loss=%.4f fit=%.0fs", seed, name,
                     report["roc_auc"], report["log_loss"], fit_s)


def summarize() -> None:
    runs = pd.read_csv(RUNS_PATH)
    metrics = ["roc_auc", "pr_auc", "log_loss", "brier", "ece"]
    agg = runs.groupby("config")[metrics + ["fit_time_s", "mem_X_train_mb", "n_features"]].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg.reset_index().to_csv(ABL_DIR / "resumen_por_config.csv", index=False)

    # Contra la base, y ademas el contraste directo de H2: solo hashing vs solo frecuencia.
    contrasts = [(c, "base") for c in CONFIGS if c != "base"] + [("solo_hashing", "solo_frecuencia")]
    rows = []
    for name, ref_name in contrasts:
        base = runs[runs["config"] == ref_name].set_index("seed").sort_index()
        var = runs[runs["config"] == name].set_index("seed").sort_index()
        a, b = var["roc_auc"].values, base.loc[var.index, "roc_auc"].values
        diff = a - b
        t, p = stats.ttest_rel(a, b)
        p_tost = ttost_paired(a, b, low=-DELTA_AUC, upp=DELTA_AUC)[0]
        sd = diff.std(ddof=1)
        rows.append({"config": name, "referencia": ref_name,
                     "auc_media_variante": a.mean(), "auc_media_base": b.mean(),
                     "mem_mb_variante": var["mem_X_train_mb"].mean(), "mem_mb_referencia": base["mem_X_train_mb"].mean(),
                     "diferencia_media": diff.mean(), "de_diferencia": sd,
                     "ic95_low": diff.mean() - stats.t.ppf(0.975, len(diff) - 1) * sd / np.sqrt(len(diff)),
                     "ic95_high": diff.mean() + stats.t.ppf(0.975, len(diff) - 1) * sd / np.sqrt(len(diff)),
                     "d_z": diff.mean() / sd if sd > 0 else np.inf, "t_pareada": t,
                     "p_valor": p, "p_valor_tost": p_tost, "shapiro_p_diferencias": stats.shapiro(diff).pvalue,
                     "log_loss_dif_media": (var["log_loss"].values - base.loc[var.index, "log_loss"].values).mean(),
                     "n_semillas": len(diff)})
    comp = pd.DataFrame(rows)
    comp["p_valor_holm"] = multipletests(comp["p_valor"], alpha=ALPHA, method="holm")[1]
    comp["p_valor_tost_holm"] = multipletests(comp["p_valor_tost"], alpha=ALPHA, method="holm")[1]
    comp["conclusion"] = np.where(comp["p_valor_tost_holm"] < ALPHA, "equivalente dentro de delta",
                          np.where(comp["p_valor_holm"] < ALPHA, "diferencia significativa",
                                   "no concluyente"))
    comp.to_csv(ABL_DIR / "contrastes_vs_base.csv", index=False)
    log.info("Resumen de ablaciones listo en %s", ABL_DIR)


if __name__ == "__main__":
    run_all()
    summarize()
