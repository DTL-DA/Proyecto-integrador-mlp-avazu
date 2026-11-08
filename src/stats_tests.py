"""
Evaluacion estadistica comparativa entre modelos y entornos.

Decisiones a priori declaradas (regla no negociable de la skill inddl: se fijan
ANTES de ver los resultados finales):
- Metrica primaria: ROC AUC. Secundarias: PR AUC, log loss, Brier, ECE.
- Unidad de independencia para la comparacion entre modelos: la semilla
  (5 semillas fijas) -> 5 observaciones por modelo. Pruebas: Welch (diferencia),
  TOST (equivalencia), d de Cohen (efecto) y diferencia minima detectable con
  potencia 0.8. Shapiro-Wilk se reporta como chequeo de normalidad, aunque con
  n=5 tiene muy poca potencia.
- Comparacion a nivel de instancia, dentro de una misma corrida: DeLong pareado
  (con la covarianza entre ambas AUC, que es lo que distingue a DeLong de una
  comparacion de AUC independientes) y McNemar sobre las clasificaciones.
- alpha = 0.05. Margen de equivalencia delta = 0.01 en ROC AUC.
- Correccion por comparaciones multiples: Holm, dentro de cada familia de
  pruebas (seed-level, DeLong, McNemar). Se reportan p ajustado y sin ajustar.
- Wilcoxon no se usa: con 5 pares su p minimo bilateral es 0.0625. La ausencia
  de significancia se reporta como "no concluyente", nunca como equivalencia.

Alineacion de conjuntos de prueba (verificada empiricamente):
- sklearn evalua sus tres modelos sobre la MISMA muestra fija del dia 30
  (202.511 filas), asi que dentro de sklearn las predicciones estan alineadas.
- Spark evalua sobre el dia 30 completo (4.218.938 filas) en el orden de
  flat_cache/test.parquet (las etiquetas guardadas coinciden fila a fila).
- Entre entornos NO hay alineacion por posicion: la muestra de sklearn no se
  puede reproducir exactamente (el muestreo de DuckDB con varios hilos no es
  determinista). Para el DeLong/McNemar pareado sklearn-vs-Spark se re-evalua el
  MLP de referencia de sklearn (tier mayor, semilla 13, el unico persistido)
  sobre el dia 30 completo, que si queda alineado con Spark.

Umbrales para McNemar:
- sklearn: el umbral de cada corrida, elegido en validacion (runs.csv).
- Spark no persistio predicciones de validacion, asi que su umbral no se puede
  elegir sin mirar prueba; se usa una regla a priori que solo depende de train:
  la tasa base de click en los dias de entrenamiento. En la comparacion entre
  entornos se aplica esa misma regla a ambos modelos.

Advertencia de interpretacion: sklearn-MLP y Spark-MLP no difieren solo en el
entorno, tambien en configuracion (dimension de hashing 4096 vs 512, filas de
entrenamiento ~1.6M vs 300k; ver spark_pipeline.py). La comparacion mide
"entorno + configuracion viable en esta maquina", no el entorno aislado.
"""
from __future__ import annotations

import json
import logging

import duckdb
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.weightstats import ttost_ind

from config import ARTIFACTS_DIR, DATA_PROCESSED_DIR, LABEL_COL, SEEDS, TRAIN_DAYS
from metrics import bootstrap_ci, expected_calibration_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("stats_tests")

STATS_DIR = ARTIFACTS_DIR / "stats"
STATS_DIR.mkdir(parents=True, exist_ok=True)
SK_PRED = ARTIFACTS_DIR / "sklearn" / "predictions"
SP_PRED = ARTIFACTS_DIR / "spark" / "predictions"
SPARK_TEST_PARQUET = ARTIFACTS_DIR / "spark" / "flat_cache" / "test.parquet"

ALPHA = 0.05
DELTA_EQUIVALENCIA_AUC = 0.01
PRIMARY_METRIC = "roc_auc"
METRICS = ["roc_auc", "pr_auc", "log_loss", "brier", "ece"]
RESCORE_BATCH = 500_000


# ---------------------------------------------------------------- DeLong

def _structural_components(y: np.ndarray, s: np.ndarray):
    """Componentes estructurales de DeLong via midranks (Sun & Xu, 2014)."""
    pos, neg = s[y == 1], s[y == 0]
    m, n = len(pos), len(neg)
    tz = stats.rankdata(np.concatenate([pos, neg]))
    v10 = (tz[:m] - stats.rankdata(pos)) / n
    v01 = 1.0 - (tz[m:] - stats.rankdata(neg)) / m
    return v10, v01


def delong_paired(y: np.ndarray, s_a: np.ndarray, s_b: np.ndarray) -> dict:
    a10, a01 = _structural_components(y, s_a)
    b10, b01 = _structural_components(y, s_b)
    m, n = len(a10), len(a01)
    cov = np.cov(np.vstack([a10, b10])) / m + np.cov(np.vstack([a01, b01])) / n
    auc_a, auc_b = float(a10.mean()), float(b10.mean())
    var_diff = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    se = np.sqrt(var_diff)
    diff = auc_a - auc_b
    z = diff / se if se > 0 else 0.0
    return {
        "auc_a": auc_a, "auc_b": auc_b, "diferencia_auc": diff,
        "ic95_dif_low": diff - 1.96 * se, "ic95_dif_high": diff + 1.96 * se,
        "correlacion_auc": float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])),
        "z": float(z), "p_valor": float(2 * stats.norm.sf(abs(z))),
        "n_pos": m, "n_neg": n,
    }


# ---------------------------------------------------------------- McNemar

def mcnemar_paired(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    ok_a, ok_b = pred_a == y, pred_b == y
    b = int(np.sum(ok_a & ~ok_b))
    c = int(np.sum(~ok_a & ok_b))
    table = [[int(np.sum(ok_a & ok_b)), b], [c, int(np.sum(~ok_a & ~ok_b))]]
    res = mcnemar(table, exact=(b + c) < 25, correction=True)
    return {"solo_a_acierta": b, "solo_b_acierta": c,
            "acc_a": float(ok_a.mean()), "acc_b": float(ok_b.mean()),
            "estadistico": float(res.statistic), "p_valor": float(res.pvalue)}


# ---------------------------------------------------------------- seed-level

def seed_level_comparison(a: np.ndarray, b: np.ndarray, delta: float = DELTA_EQUIVALENCIA_AUC) -> dict:
    t_welch, p_welch = stats.ttest_ind(a, b, equal_var=False)
    sd_pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    d = (a.mean() - b.mean()) / sd_pooled if sd_pooled > 0 else np.inf
    p_tost, _, _ = ttost_ind(a, b, low=-delta, upp=delta, usevar="unequal")
    d_mde = TTestIndPower().solve_power(nobs1=len(a), alpha=ALPHA, power=0.8, ratio=len(b) / len(a))
    return {
        "media_a": float(a.mean()), "media_b": float(b.mean()),
        "diferencia_media": float(a.mean() - b.mean()),
        "t_welch": float(t_welch), "p_valor_welch": float(p_welch),
        "d_cohen": float(d), "p_valor_tost": float(p_tost), "delta": delta,
        "shapiro_p_a": float(stats.shapiro(a).pvalue), "shapiro_p_b": float(stats.shapiro(b).pvalue),
        "mde_d_potencia80": float(d_mde), "mde_en_auc": float(d_mde * sd_pooled),
    }


def _conclusion(p_welch_adj: float, p_tost: float) -> str:
    if p_tost < ALPHA and p_welch_adj < ALPHA:
        return "diferencia significativa pero menor que delta (equivalencia practica)"
    if p_tost < ALPHA:
        return "equivalente dentro de delta"
    if p_welch_adj < ALPHA:
        return "diferencia significativa"
    return "no concluyente (ni diferencia ni equivalencia con n=5)"


def add_holm(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[f"{col}_holm"] = multipletests(df[col].values, alpha=ALPHA, method="holm")[1]
    return df


# ---------------------------------------------------------------- utilidades

def threshold_free_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p_clip = np.clip(p, 1e-7, 1 - 1e-7)
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "log_loss": log_loss(y, p_clip, labels=[0, 1]), "brier": brier_score_loss(y, p),
            "ece": expected_calibration_error(y, p)}


def aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for metric in METRICS:
            vals = g[metric].values
            ci = bootstrap_ci(vals, seed=13)
            t_crit = stats.t.ppf(1 - ALPHA / 2, df=max(len(vals) - 1, 1))
            half = t_crit * ci["std"] / np.sqrt(len(vals))
            rows.append({**dict(zip(group_cols, keys)), "metrica": metric,
                         "media": ci["mean"], "de": ci["std"],
                         "ic95_t_low": ci["mean"] - half, "ic95_t_high": ci["mean"] + half,
                         "ic95_boot_low": ci["ci_low"], "ic95_boot_high": ci["ci_high"],
                         "n": len(vals)})
    return pd.DataFrame(rows)


def train_base_rate() -> float:
    glob = (DATA_PROCESSED_DIR / "day=*" / "*.parquet").as_posix()
    days = ",".join(map(str, TRAIN_DAYS))
    return float(duckdb.connect().execute(
        f"SELECT avg({LABEL_COL}) FROM read_parquet('{glob}') WHERE day IN ({days})"
    ).fetchone()[0])


def rescore_sklearn_mlp_on_full_test() -> np.ndarray:
    import __main__
    from sklearn_pipeline import FittedPipeline
    __main__.FittedPipeline = FittedPipeline
    ref = joblib.load(ARTIFACTS_DIR / "sklearn" / "models" / "mlp_reference.joblib")
    pipe, model = ref["pipeline"], ref["model"]
    probs = []
    for batch in pq.ParquetFile(SPARK_TEST_PARQUET).iter_batches(batch_size=RESCORE_BATCH):
        X, _ = pipe.full_features(batch.to_pandas(), fit=False)
        probs.append(model.predict_proba(X)[:, 1])
        log.info("  re-evaluadas %d filas", sum(len(p) for p in probs))
    return np.concatenate(probs)


# ---------------------------------------------------------------- main

def main() -> None:
    sk_runs = pd.read_csv(ARTIFACTS_DIR / "sklearn" / "runs.csv")
    sp_runs = pd.read_csv(ARTIFACTS_DIR / "spark" / "runs.csv")
    max_tier = int(sk_runs["tier"].max())
    base_rate = train_base_rate()
    log.info("Tasa base de click en train (umbral a priori para McNemar en Spark): %.5f", base_rate)

    # Metricas sin umbral de Spark, recalculadas desde las predicciones guardadas.
    sp_rows = []
    for _, r in sp_runs.iterrows():
        y = np.load(SP_PRED / f"{r['model']}_seed{r['seed']}_y_test.npy")
        p = np.load(SP_PRED / f"{r['model']}_seed{r['seed']}_p_test.npy")
        sp_rows.append({"model": r["model"], "seed": int(r["seed"]), "fit_time_s": r["fit_time_s"],
                        "n_test": len(y), **threshold_free_metrics(y, p)})
    sp_metrics = pd.DataFrame(sp_rows)
    sp_metrics.to_csv(STATS_DIR / "spark_metricas_por_corrida.csv", index=False)

    sk_n_test = len(np.load(SK_PRED / f"mlp_tier{max_tier}_seed{SEEDS[0]}_y_test.npy"))
    aggregate(sk_runs, ["model", "tier"]).to_csv(STATS_DIR / "sklearn_agregado.csv", index=False)
    aggregate(sp_metrics, ["model"]).to_csv(STATS_DIR / "spark_agregado.csv", index=False)
    log.info("Agregados listos (n_test sklearn=%d, n_test spark=%d)", sk_n_test, int(sp_metrics["n_test"].iloc[0]))

    # Familia 1: comparaciones a nivel de semilla (ROC AUC).
    def vals(df, model, **kw):
        q = df[df["model"] == model]
        for k, v in kw.items():
            q = q[q[k] == v]
        return q.sort_values("seed")[PRIMARY_METRIC].values

    pairs = [
        ("sklearn", "mlp", "sklearn", "histgb", vals(sk_runs, "mlp", tier=max_tier), vals(sk_runs, "histgb", tier=max_tier)),
        ("sklearn", "mlp", "sklearn", "sgd", vals(sk_runs, "mlp", tier=max_tier), vals(sk_runs, "sgd", tier=max_tier)),
        ("spark", "mlp", "spark", "lr", vals(sp_metrics, "mlp"), vals(sp_metrics, "lr")),
        ("spark", "mlp", "spark", "gbt", vals(sp_metrics, "mlp"), vals(sp_metrics, "gbt")),
        ("sklearn", "mlp", "spark", "mlp", vals(sk_runs, "mlp", tier=max_tier), vals(sp_metrics, "mlp")),
    ]
    seed_rows = [{"entorno_a": ea, "modelo_a": ma, "entorno_b": eb, "modelo_b": mb,
                  **seed_level_comparison(a, b)} for ea, ma, eb, mb, a, b in pairs]
    seed_df = add_holm(add_holm(pd.DataFrame(seed_rows), "p_valor_welch"), "p_valor_tost")
    seed_df["conclusion"] = [_conclusion(pw, pt) for pw, pt in
                             zip(seed_df["p_valor_welch_holm"], seed_df["p_valor_tost_holm"])]
    seed_df.to_csv(STATS_DIR / "comparaciones_por_semilla.csv", index=False)
    log.info("Comparaciones por semilla listas")

    # Familias 2 y 3: DeLong y McNemar pareados dentro de cada entorno.
    delong_rows, mcnemar_rows = [], []
    thr = sk_runs.set_index(["model", "tier", "seed"])["threshold"]
    for seed in SEEDS:
        y = np.load(SK_PRED / f"mlp_tier{max_tier}_seed{seed}_y_test.npy")
        p = {m: np.load(SK_PRED / f"{m}_tier{max_tier}_seed{seed}_p_test.npy") for m in ("mlp", "histgb", "sgd")}
        for other in ("histgb", "sgd"):
            meta = {"entorno": "sklearn", "modelo_a": "mlp", "modelo_b": other, "seed": seed}
            delong_rows.append({**meta, **delong_paired(y, p["mlp"], p[other])})
            pred_a = (p["mlp"] >= thr[("mlp", max_tier, seed)]).astype(int)
            pred_b = (p[other] >= thr[(other, max_tier, seed)]).astype(int)
            mcnemar_rows.append({**meta, "umbral": "validacion", **mcnemar_paired(y, pred_a, pred_b)})

        y = np.load(SP_PRED / f"mlp_seed{seed}_y_test.npy")
        p = {m: np.load(SP_PRED / f"{m}_seed{seed}_p_test.npy") for m in ("mlp", "lr", "gbt")}
        for other in ("lr", "gbt"):
            meta = {"entorno": "spark", "modelo_a": "mlp", "modelo_b": other, "seed": seed}
            delong_rows.append({**meta, **delong_paired(y, p["mlp"], p[other])})
            mcnemar_rows.append({**meta, "umbral": "tasa_base_train",
                                 **mcnemar_paired(y, (p["mlp"] >= base_rate).astype(int),
                                                  (p[other] >= base_rate).astype(int))})
        log.info("  DeLong/McNemar dentro de entorno, semilla %d listos", seed)

    # Entre entornos: MLP sklearn (referencia, semilla 13) re-evaluado sobre el dia 30 completo.
    log.info("Re-evaluando MLP de sklearn (semilla %d) sobre el dia 30 completo...", SEEDS[0])
    p_sk_full = rescore_sklearn_mlp_on_full_test()
    np.save(STATS_DIR / "mlp_sklearn_ref_p_test_dia30_completo.npy", p_sk_full)
    y_sp = np.load(SP_PRED / f"mlp_seed{SEEDS[0]}_y_test.npy")
    p_sp = np.load(SP_PRED / f"mlp_seed{SEEDS[0]}_p_test.npy")
    meta = {"entorno": "sklearn_vs_spark", "modelo_a": "mlp_sklearn", "modelo_b": "mlp_spark", "seed": SEEDS[0]}
    delong_rows.append({**meta, **delong_paired(y_sp, p_sk_full, p_sp)})
    mcnemar_rows.append({**meta, "umbral": "tasa_base_train",
                         **mcnemar_paired(y_sp, (p_sk_full >= base_rate).astype(int),
                                          (p_sp >= base_rate).astype(int))})

    add_holm(pd.DataFrame(delong_rows), "p_valor").to_csv(STATS_DIR / "delong_pareado.csv", index=False)
    add_holm(pd.DataFrame(mcnemar_rows), "p_valor").to_csv(STATS_DIR / "mcnemar_pareado.csv", index=False)

    (STATS_DIR / "decisiones_a_priori.json").write_text(json.dumps({
        "alpha": ALPHA, "metrica_primaria": PRIMARY_METRIC, "delta_equivalencia_auc": DELTA_EQUIVALENCIA_AUC,
        "unidad_independencia": "semilla (5)", "correccion_multiple": "Holm por familia",
        "umbral_mcnemar_spark": "tasa base de click en train", "tasa_base_train": base_rate,
        "n_test_sklearn": sk_n_test, "n_test_spark": int(len(y_sp)),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Pruebas estadisticas completas. Artifacts en %s", STATS_DIR)


if __name__ == "__main__":
    main()
