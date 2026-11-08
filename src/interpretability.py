"""
Interpretabilidad local con LIME, contrastada con SHAP y permutation importance.

Adaptacion declarada del protocolo del enunciado: con hashing trick, las
dimensiones del vector de features no son legibles por humanos (son buckets de
una funcion hash, no columnas con nombre). Por eso LIME/SHAP no corren sobre el
vector hasheado completo, sino sobre un "espacio interpretable reducido": el
bloque denso de frequency + target encoding, que SI tiene una columna por cada
variable categorica original con nombre (ej. site_id_freq, device_id_te). El
bloque disperso hasheado de cada instancia se mantiene fijo en su valor real
mientras LIME/SHAP perturban solo el bloque denso con nombre; el mapeo
"feature interpretable -> variable original" es la identidad (cada _freq/_te
viene de una sola columna), y queda documentado aqui. El bloque denso esta
estandarizado (StandardScaler ajustado en train), asi que los umbrales que
reporta LIME estan en desviaciones estandar, no en unidades originales.

SHAP usa PermutationExplainer (no KernelExplainer): el modelo es un MLP, asi
que TreeExplainer/LinearExplainer no aplican, y con solo 14 variables
interpretables el explainer por permutaciones es barato y cumple exactamente la
propiedad de eficiencia local (la suma de los valores SHAP reproduce
f(x) - E[f(X)]), cosa que KernelExplainer con pocas muestras no garantiza.
"""
from __future__ import annotations

import itertools
import json
import logging
import sys

import duckdb
import joblib
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import roc_auc_score

from config import ARTIFACTS_DIR, DATA_PROCESSED_DIR, LABEL_COL, SEEDS, TEST_DAY

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("interpretability")

INTERP_DIR = ARTIFACTS_DIR / "interpretability"
INTERP_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_GLOB = (DATA_PROCESSED_DIR / "day=*" / "*.parquet").as_posix()
TEST_SAMPLE_ROWS = 500_000
N_FP = 6  # falsos positivos mas confiados
N_FN = 6  # falsos negativos mas confiados
N_STABILITY_SEEDS = 5
LIME_TOP_K = 8
AGREEMENT_TOP_K = 5
N_SHAP_BACKGROUND = 100
SHAP_MAX_EVALS = 1000
PERM_SAMPLE_ROWS = 200_000
PERM_REPEATS = 5


def _repeat_row(row: csr_matrix, n: int) -> csr_matrix:
    return row[np.zeros(n, dtype=np.int64)]


class ReducedSpaceWrapper:
    """predict_proba que perturba solo el bloque denso (con nombre) y fija el
    bloque disperso hasheado de la instancia original."""

    def __init__(self, model, fixed_sparse_row: csr_matrix):
        self.model = model
        self.fixed_sparse_row = fixed_sparse_row

    def predict_proba(self, dense_batch: np.ndarray) -> np.ndarray:
        sparse_batch = _repeat_row(self.fixed_sparse_row, dense_batch.shape[0])
        return self.model.predict_proba(hstack([sparse_batch, csr_matrix(dense_batch)]).tocsr())

    def predict_positive(self, dense_batch: np.ndarray) -> np.ndarray:
        return self.predict_proba(dense_batch)[:, 1]


def jaccard(a, b) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def select_confident_errors(preds_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Falsos positivos y negativos mas confiados, balanceados entre ambos tipos."""
    fp = preds_df[(preds_df["y_true"] == 0) & (preds_df["p_pred"] >= threshold)]
    fn = preds_df[(preds_df["y_true"] == 1) & (preds_df["p_pred"] < threshold)]
    fp = fp.sort_values("p_pred", ascending=False).head(N_FP).assign(tipo_error="FP")
    fn = fn.sort_values("p_pred", ascending=True).head(N_FN).assign(tipo_error="FN")
    return pd.concat([fp, fn])


def run_lime(model, dense_test, sparse_block, cases, feature_names) -> pd.DataFrame:
    lime_rows = []
    for pos, case in cases.iterrows():
        wrapper = ReducedSpaceWrapper(model, sparse_block[pos])
        for seed in SEEDS[:N_STABILITY_SEEDS]:
            exp = LimeTabularExplainer(
                training_data=dense_test, feature_names=feature_names, mode="classification",
                discretize_continuous=True, random_state=seed,
            ).explain_instance(dense_test[pos], wrapper.predict_proba, num_features=LIME_TOP_K)
            weights = sorted(exp.as_map()[1], key=lambda t: abs(t[1]), reverse=True)
            lime_rows.append({
                "indice_instancia": int(pos), "tipo_error": case["tipo_error"], "seed": seed,
                "y_true": int(case["y_true"]), "p_pred": float(case["p_pred"]),
                "fidelidad_r2": float(exp.score),
                "top_features": json.dumps([feature_names[j] for j, _ in weights], ensure_ascii=False),
                "pesos": json.dumps([float(w) for _, w in weights]),
                "condiciones": json.dumps([c for c, _ in exp.as_list()], ensure_ascii=False),
            })
        log.info("  LIME instancia %d (%s) lista", pos, case["tipo_error"])
    return pd.DataFrame(lime_rows)


def lime_sensitivity(model, dense_test, sparse_block, cases, feature_names) -> pd.DataFrame:
    """Sensibilidad de LIME al ancho del kernel y al numero de muestras perturbadas.
    Factorial 3x3 (ancho 0.5x, 1x y 2x el valor por defecto de LIME, sqrt(p)*0.75;
    1000, 5000 y 10000 muestras) con 3 semillas por celda. Por celda se reporta la
    fidelidad R2, la estabilidad entre semillas (Jaccard del top-k) y la
    coincidencia del top-k con la configuracion por defecto (ancho 1x, 5000)."""
    kw0 = np.sqrt(dense_test.shape[1]) * 0.75
    widths = {"0.5x": 0.5 * kw0, "1x": kw0, "2x": 2 * kw0}
    samples = [1000, 5000, 10000]
    seeds = SEEDS[:3]
    rows = []
    for pos, case in cases.iterrows():
        wrapper = ReducedSpaceWrapper(model, sparse_block[pos])
        tops = {}
        for (wname, kw), ns in itertools.product(widths.items(), samples):
            for seed in seeds:
                exp = LimeTabularExplainer(
                    training_data=dense_test, feature_names=feature_names, mode="classification",
                    discretize_continuous=True, kernel_width=kw, random_state=seed,
                ).explain_instance(dense_test[pos], wrapper.predict_proba,
                                   num_features=LIME_TOP_K, num_samples=ns)
                w = sorted(exp.as_map()[1], key=lambda t: abs(t[1]), reverse=True)
                tops[(wname, ns, seed)] = [feature_names[j] for j, _ in w][:AGREEMENT_TOP_K]
                rows.append({"indice_instancia": int(pos), "tipo_error": case["tipo_error"],
                             "ancho_kernel": wname, "ancho_valor": kw, "num_muestras": ns,
                             "seed": seed, "fidelidad_r2": float(exp.score)})
        for r in rows[-len(widths) * len(samples) * len(seeds):]:
            key = (r["ancho_kernel"], r["num_muestras"], r["seed"])
            r["jaccard_vs_defecto"] = jaccard(tops[key], tops[("1x", 5000, r["seed"])])
            r["jaccard_entre_semillas"] = float(np.mean([
                jaccard(tops[(key[0], key[1], a)], tops[(key[0], key[1], b)])
                for a, b in itertools.combinations(seeds, 2)]))
        log.info("  sensibilidad LIME instancia %d lista", pos)
    return pd.DataFrame(rows)


def discretization_check(model, dense_test, sparse_block, cases, feature_names,
                         n_draws: int = 200) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Diagnostico de la discretizacion por cuartiles que usa LIME.
    (1) Por variable: cuartiles, numero de intervalos distintos y masa del valor
    mas frecuente; un punto de masa grande colapsa los cuartiles.
    (2) Por instancia y variable: probabilidad media cuando solo esa variable se
    reemplaza por valores de las filas que caen FUERA del intervalo de la
    instancia (lo que LIME contrasta), frente a la probabilidad original."""
    rows_q = []
    bounds = {}
    for j, name in enumerate(feature_names):
        col = dense_test[:, j]
        q = np.percentile(col, [25, 50, 75])
        uq = np.unique(q)
        vals, counts = np.unique(col, return_counts=True)
        bounds[name] = uq
        rows_q.append({"feature": name, "q25": q[0], "q50": q[1], "q75": q[2],
                       "n_intervalos": len(uq) + 1,
                       "masa_valor_modal": counts.max() / len(col),
                       "valor_modal": vals[counts.argmax()]})
    rng = np.random.default_rng(SEEDS[0])
    rows_e = []
    for pos, case in cases.iterrows():
        wrapper = ReducedSpaceWrapper(model, sparse_block[pos])
        x0 = dense_test[pos]
        p0 = float(wrapper.predict_positive(x0[None, :])[0])
        for j, name in enumerate(feature_names):
            col = dense_test[:, j]
            b = np.searchsorted(bounds[name], col, side="left")
            b0 = np.searchsorted(bounds[name], x0[j], side="left")
            outside = np.flatnonzero(b != b0)
            if len(outside) == 0:
                continue
            draws = rng.choice(outside, size=min(n_draws, len(outside)), replace=False)
            batch = np.repeat(x0[None, :], len(draws), axis=0)
            batch[:, j] = col[draws]
            p_out = wrapper.predict_positive(batch)
            rows_e.append({"indice_instancia": int(pos), "tipo_error": case["tipo_error"],
                           "feature": name, "p_original": p0,
                           "fraccion_fuera_intervalo": len(outside) / len(col),
                           "p_media_fuera_intervalo": float(p_out.mean()),
                           "cambio_p_medio": float(p_out.mean() - p0)})
    return pd.DataFrame(rows_q), pd.DataFrame(rows_e)


def lime_stability(lime_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pos, grp in lime_df.groupby("indice_instancia"):
        sets = [json.loads(s) for s in grp["top_features"]]
        pairwise = [jaccard(a, b) for a, b in itertools.combinations(sets, 2)]
        rows.append({
            "indice_instancia": int(pos), "tipo_error": grp["tipo_error"].iloc[0],
            "jaccard_promedio_entre_semillas": float(np.mean(pairwise)),
            "jaccard_min_entre_semillas": float(np.min(pairwise)),
            "fidelidad_r2_media": float(grp["fidelidad_r2"].mean()),
        })
    return pd.DataFrame(rows)


def run_shap(model, dense_test, sparse_block, cases, feature_names) -> pd.DataFrame:
    rng = np.random.default_rng(SEEDS[0])
    bg_idx = rng.choice(dense_test.shape[0], size=N_SHAP_BACKGROUND, replace=False)
    masker = shap.maskers.Independent(dense_test[bg_idx], max_samples=N_SHAP_BACKGROUND)
    rows = []
    for pos, case in cases.iterrows():
        wrapper = ReducedSpaceWrapper(model, sparse_block[pos])
        explainer = shap.PermutationExplainer(wrapper.predict_positive, masker, seed=SEEDS[0])
        sv = explainer(dense_test[pos][None, :], max_evals=SHAP_MAX_EVALS)
        row = {"indice_instancia": int(pos), "tipo_error": case["tipo_error"],
               "valor_base": float(sv.base_values[0]), "p_pred": float(case["p_pred"])}
        row.update({name: float(v) for name, v in zip(feature_names, sv.values[0])})
        rows.append(row)
        log.info("  SHAP instancia %d (%s) lista", pos, case["tipo_error"])
    return pd.DataFrame(rows)


def lime_shap_agreement(lime_df: pd.DataFrame, shap_df: pd.DataFrame, feature_names) -> pd.DataFrame:
    rows = []
    for _, srow in shap_df.iterrows():
        pos = srow["indice_instancia"]
        shap_top = sorted(feature_names, key=lambda f: abs(srow[f]), reverse=True)[:AGREEMENT_TOP_K]
        lime_sets = [json.loads(s)[:AGREEMENT_TOP_K] for s in
                     lime_df.loc[lime_df["indice_instancia"] == pos, "top_features"]]
        jac = [jaccard(ls, shap_top) for ls in lime_sets]
        rows.append({
            "indice_instancia": int(pos), "tipo_error": srow["tipo_error"],
            "shap_top": json.dumps(shap_top, ensure_ascii=False),
            "jaccard_lime_vs_shap_media": float(np.mean(jac)),
        })
    return pd.DataFrame(rows)


def permutation_importance_dense(model, sparse_block, dense_block, y_true, feature_names) -> pd.DataFrame:
    rng = np.random.default_rng(SEEDS[0])
    X_full = hstack([sparse_block, csr_matrix(dense_block)]).tocsr()
    base_auc = roc_auc_score(y_true, model.predict_proba(X_full)[:, 1])

    rows = []
    for j, name in enumerate(feature_names):
        drops = []
        for _ in range(PERM_REPEATS):
            permuted = dense_block.copy()
            permuted[:, j] = rng.permutation(permuted[:, j])
            X_perm = hstack([sparse_block, csr_matrix(permuted)]).tocsr()
            drops.append(base_auc - roc_auc_score(y_true, model.predict_proba(X_perm)[:, 1]))
        rows.append({"feature": name, "auc_base": base_auc,
                     "caida_auc_media": float(np.mean(drops)),
                     "caida_auc_de": float(np.std(drops, ddof=1)),
                     "n_repeticiones": PERM_REPEATS})
        log.info("  permutation importance %s: caida media %.5f", name, np.mean(drops))
    return pd.DataFrame(rows).sort_values("caida_auc_media", ascending=False)


def load_reference_model():
    # El joblib se guardo con sklearn_pipeline.py corriendo como __main__, asi que
    # pickle busca FittedPipeline en __main__; se inyecta antes de cargar.
    import __main__
    from sklearn_pipeline import FittedPipeline
    __main__.FittedPipeline = FittedPipeline
    ref_path = ARTIFACTS_DIR / "sklearn" / "models" / "mlp_reference.joblib"
    if not ref_path.exists():
        raise FileNotFoundError(f"No existe {ref_path}. Corre sklearn_pipeline.py primero.")
    ref = joblib.load(ref_path)
    return ref["pipeline"], ref["model"]


def main() -> None:
    pipe, model = load_reference_model()

    con = duckdb.connect()
    con.execute("SET memory_limit='10GB'")
    con.execute("SET threads TO 1")  # muestreo determinista: mismas instancias en LIME y en la sensibilidad
    # En DuckDB, USING SAMPLE se aplica ANTES del WHERE; la subconsulta filtra
    # primero el dia de prueba y despues muestrea, para obtener filas reales del dia 30.
    test_df = con.execute(
        f"SELECT * FROM (SELECT * FROM read_parquet('{PARQUET_GLOB}') WHERE day = {TEST_DAY}) "
        f"USING SAMPLE {TEST_SAMPLE_ROWS} ROWS (reservoir, {SEEDS[0]})"
    ).df()
    y_test = test_df[LABEL_COL].values

    X_test, dense_test = pipe.full_features(test_df, fit=False)
    sparse_block = X_test[:, : X_test.shape[1] - dense_test.shape[1]].tocsr()
    p_test = model.predict_proba(X_test)[:, 1]
    feature_names = [f"{c}_freq" for c in pipe.freq.columns] + [f"{c}_te" for c in pipe.target_enc.columns]

    # Umbral congelado del modelo de referencia (tier 2M, semilla 13) en sklearn.
    runs = pd.read_csv(ARTIFACTS_DIR / "sklearn" / "runs.csv")
    ref_run = runs[(runs["model"] == "mlp") & (runs["tier"] == runs["tier"].max()) & (runs["seed"] == SEEDS[0])]
    threshold = float(ref_run["threshold"].iloc[0])

    preds_df = pd.DataFrame({"p_pred": p_test, "y_true": y_test})
    cases = select_confident_errors(preds_df, threshold)
    cases.to_csv(INTERP_DIR / "instancias_explicadas.csv")
    log.info("Umbral=%.4f; %d FP y %d FN seleccionados", threshold,
             (cases["tipo_error"] == "FP").sum(), (cases["tipo_error"] == "FN").sum())

    if "--discretizacion" in sys.argv:
        log.info("Solo diagnostico de la discretizacion de LIME...")
        q_df, e_df = discretization_check(model, dense_test, sparse_block, cases, feature_names)
        q_df.to_csv(INTERP_DIR / "lime_discretizacion_cuartiles.csv", index=False)
        e_df.to_csv(INTERP_DIR / "lime_discretizacion_efecto.csv", index=False)
        log.info("Diagnostico de discretizacion listo")
        return

    if "--sensibilidad" in sys.argv:
        log.info("Solo sensibilidad de LIME (ancho de kernel x muestras perturbadas)...")
        lime_sensitivity(model, dense_test, sparse_block, cases, feature_names).to_csv(
            INTERP_DIR / "lime_sensibilidad.csv", index=False)
        log.info("Sensibilidad de LIME lista")
        return

    log.info("LIME (%d instancias x %d semillas)...", len(cases), N_STABILITY_SEEDS)
    lime_df = run_lime(model, dense_test, sparse_block, cases, feature_names)
    lime_df.to_csv(INTERP_DIR / "lime_explicaciones.csv", index=False)
    lime_stability(lime_df).to_csv(INTERP_DIR / "lime_estabilidad_jaccard.csv", index=False)

    log.info("SHAP (PermutationExplainer, background=%d)...", N_SHAP_BACKGROUND)
    shap_df = run_shap(model, dense_test, sparse_block, cases, feature_names)
    shap_df.to_csv(INTERP_DIR / "shap_valores.csv", index=False)
    lime_shap_agreement(lime_df, shap_df, feature_names).to_csv(
        INTERP_DIR / "concordancia_lime_shap.csv", index=False
    )

    log.info("Permutation importance (%d filas, %d repeticiones)...", PERM_SAMPLE_ROWS, PERM_REPEATS)
    rng = np.random.default_rng(SEEDS[0])
    idx = rng.choice(len(y_test), size=PERM_SAMPLE_ROWS, replace=False)
    perm_imp = permutation_importance_dense(
        model, sparse_block[idx], dense_test[idx], y_test[idx], feature_names
    )
    perm_imp.to_csv(INTERP_DIR / "permutation_importance.csv", index=False)

    log.info("Interpretabilidad completa. Artifacts en %s", INTERP_DIR)


if __name__ == "__main__":
    main()
