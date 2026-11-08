"""
Modelado en scikit-learn: SGDClassifier y HistGradientBoostingClassifier como
baselines de referencia, MLPClassifier como modelo objetivo, sobre muestras
escalonadas (100k / 1M / 10M filas) para construir curvas de aprendizaje y de
costo computacional.

Reduccion de alcance declarada (maquina con 16 GB RAM, ver artifacts/eda/data_sheet.json):
- La busqueda de hiperparametros con Optuna se corre UNA vez por modelo (no por
  cada combinacion de tier x semilla) sobre el tier de 1M filas, y los mejores
  parametros se reutilizan en todos los tiers y semillas. Se declara el espacio
  de busqueda antes de entrenar (ver PARAM SPACES mas abajo).
- HistGradientBoostingClassifier usa el bloque denso (frequency + target
  encoding), no el hashing disperso de alta dimension: los modelos de arboles no
  se benefician de ese espacio disperso y mantenerlo denso evita presion de RAM
  en el tier de 10M filas. SGD y MLP si usan el pipeline completo (hashing +
  frequency + target encoding).
- Las variables C* de Avazu son codigos categoricos anonimizados, no numericas
  continuas: el paso de "log + StandardScaler para numericas" del enunciado no
  aplica a este dataset (se usan como categoricas via hashing/frequency, igual
  que el resto). Esto se documenta aqui en vez de forzar una transformacion sin
  sentido estadistico.
- Ajuste adicional por tiempo de computo real (medido en la primera corrida):
  el MLP sobre el bloque hasheado de 2^14 columnas tardaba mas de una hora por
  trial de Optuna sobre 1M de filas, lo que hacia inviable terminar a tiempo.
  El enunciado fija literalmente la grilla de arquitecturas del MLP sklearn en
  (50,), (100,), (100,50) y (256,128) con busqueda bayesiana (no GridSearchCV)
  y max_iter "controlado por la parada temprana" -- esto NO se tocó: las
  cuatro arquitecturas siguen disponibles y max_iter sigue en 200, solo se
  bajo la paciencia de la parada temprana (n_iter_no_change=5) para que corte
  antes tras estancarse. Lo que si se ajusto, dentro del margen que el propio
  enunciado da ("tamanos de 100 mil, 1 millon y 10 millones... hasta donde
  permita la memoria"): (a) la dimension de hashing bajo de 2^14 a 2^12 para
  SGD y MLP (HistGB no se ve afectado, usa solo el bloque denso; la dimension
  de hashing es uno de los factores de ablacion declarados, no un valor fijo
  del enunciado); (b) el MLP se afina sobre una muestra reducida
  (MLP_TUNE_SAMPLE) en vez del tier de 1M completo, ya que el tamano de la
  muestra de afinacion no esta mandatado; (c) el tier maximo de entrenamiento
  final bajo de 10M a 2M filas, documentado como el tope real alcanzado por
  limite de tiempo/hardware, tal como el enunciado contempla.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import duckdb
import joblib
import numpy as np
import optuna
import pandas as pd
import psutil
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from config import (
    ARTIFACTS_DIR,
    DATA_PROCESSED_DIR,
    HIGH_CARDINALITY_CATEGORICALS,
    LABEL_COL,
    LOW_CARDINALITY_CATEGORICALS,
    NUMERIC_LIKE_CATEGORICALS,
    SEEDS,
    TEST_DAY,
    TRAIN_DAYS,
    VALIDATION_DAY,
)
from metrics import choose_threshold_on_validation, full_metric_report
from preprocessing import (
    FrequencyEncoder,
    HashingEncoder,
    OutOfFoldTargetEncoder,
    RareCategoryGrouper,
    add_feature_cross,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sklearn_pipeline")

SKLEARN_DIR = ARTIFACTS_DIR / "sklearn"
PRED_DIR = SKLEARN_DIR / "predictions"
MODELS_DIR = SKLEARN_DIR / "models"
for _d in (SKLEARN_DIR, PRED_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

PARQUET_GLOB = (DATA_PROCESSED_DIR / "day=*" / "*.parquet").as_posix()

SAMPLE_TIERS = [100_000, 1_000_000, 2_000_000]
VAL_TEST_SAMPLE = 2_000_000  # fijo para todos los tiers/semillas, limite de RAM
HASH_DIM = 2**12
RARE_THRESHOLD = 20
OPTUNA_TRIALS = {"mlp": 10, "sgd": 15, "histgb": 15}
MLP_TUNE_SAMPLE = 200_000  # muestra reducida solo para afinar el MLP (ver nota de alcance)
CATEGORICAL_COLS = LOW_CARDINALITY_CATEGORICALS + HIGH_CARDINALITY_CATEGORICALS
CROSS_A, CROSS_B = "site_id", "device_type"


def _con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET memory_limit='10GB'")
    con.execute("SET threads TO 8")
    return con


def _sample_days(con, days: list[int], n: int | None, seed: int) -> pd.DataFrame:
    days_sql = f"({','.join(map(str, days))})"
    q = f"SELECT * FROM read_parquet('{PARQUET_GLOB}') WHERE day IN {days_sql}"
    if n is not None:
        q += f" USING SAMPLE {n} ROWS (reservoir, {seed})"
    return con.execute(q).df()


class FittedPipeline:
    """Agrupa los encoders ajustados solo en train para una corrida (tier, seed) dada."""

    def __init__(self, rare: RareCategoryGrouper, freq: FrequencyEncoder,
                 hasher: HashingEncoder, target_enc: OutOfFoldTargetEncoder,
                 scaler: StandardScaler):
        self.rare = rare
        self.freq = freq
        self.hasher = hasher
        self.target_enc = target_enc
        self.scaler = scaler

    def _prep_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.rare.transform(df)
        out = add_feature_cross(out, CROSS_A, CROSS_B)
        return out

    def dense_block(self, df: pd.DataFrame, fit: bool, y: pd.Series | None = None,
                     max_train_day: int | None = None, seed: int = 13) -> np.ndarray:
        prepped = self._prep_categoricals(df)
        freq_df = self.freq.fit(prepped, max_train_day).transform(prepped) if fit else self.freq.transform(prepped)
        freq_cols = [f"{c}_freq" for c in self.freq.columns]

        if fit:
            te_df = self.target_enc.fit_transform(prepped, y, max_train_day, seed)
        else:
            te_df = self.target_enc.transform(prepped)
        te_cols = [f"{c}_te" for c in self.target_enc.columns]

        dense = pd.concat([freq_df[freq_cols], te_df[te_cols]], axis=1).fillna(0.0).values
        if fit:
            dense = self.scaler.fit_transform(dense)
        else:
            dense = self.scaler.transform(dense)
        return dense

    def sparse_block(self, df: pd.DataFrame) -> csr_matrix:
        prepped = self._prep_categoricals(df)
        return self.hasher.transform(prepped)

    def full_features(self, df: pd.DataFrame, fit: bool, y: pd.Series | None = None,
                       max_train_day: int | None = None, seed: int = 13):
        dense = self.dense_block(df, fit=fit, y=y, max_train_day=max_train_day, seed=seed)
        sparse = self.sparse_block(df)
        X = hstack([sparse, csr_matrix(dense)]).tocsr()
        return X, dense  # dense se reutiliza solo-denso para HistGB


def build_fitted_pipeline(train_df: pd.DataFrame, y_train: pd.Series, seed: int,
                          return_features: bool = False):
    max_train_day = max(TRAIN_DAYS)
    rare = RareCategoryGrouper(threshold=RARE_THRESHOLD, columns=CATEGORICAL_COLS)
    rare.fit(train_df, max_train_day)

    prepped_train = rare.transform(train_df)
    prepped_train = add_feature_cross(prepped_train, CROSS_A, CROSS_B)

    hash_cols = CATEGORICAL_COLS + [f"{CROSS_A}_x_{CROSS_B}"]
    hasher = HashingEncoder(columns=hash_cols, n_features=HASH_DIM)

    freq = FrequencyEncoder(columns=HIGH_CARDINALITY_CATEGORICALS)
    target_enc = OutOfFoldTargetEncoder(columns=HIGH_CARDINALITY_CATEGORICALS)
    scaler = StandardScaler()

    pipe = FittedPipeline(rare, freq, hasher, target_enc, scaler)
    X_train, dense_train = pipe.full_features(train_df, fit=True, y=y_train, max_train_day=max_train_day, seed=seed)
    if return_features:  # usado por stage_benchmark.py: ajuste y transformacion OOF en una sola etapa
        return pipe, X_train, dense_train
    return pipe


def _param_space(trial: optuna.Trial, model_name: str) -> dict:
    if model_name == "mlp":
        return {
            "hidden_layer_sizes": trial.suggest_categorical(
                "hidden_layer_sizes", ["50", "100", "100,50", "256,128"]
            ),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        }
    if model_name == "sgd":
        return {
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l2", "elasticnet"]),
            "eta0": trial.suggest_float("eta0", 1e-4, 1e-1, log=True),
        }
    if model_name == "histgb":
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 3e-1, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
        }
    raise ValueError(model_name)


def _build_model(model_name: str, params: dict, seed: int):
    if model_name == "mlp":
        hidden = tuple(int(x) for x in str(params["hidden_layer_sizes"]).split(","))
        return MLPClassifier(
            hidden_layer_sizes=hidden, alpha=params["alpha"],
            learning_rate_init=params["learning_rate_init"], batch_size=params["batch_size"],
            early_stopping=True, n_iter_no_change=5, validation_fraction=0.1,
            random_state=seed, max_iter=200,
        )
    if model_name == "sgd":
        return SGDClassifier(
            loss="log_loss", alpha=params["alpha"], penalty=params["penalty"],
            learning_rate="adaptive", eta0=params["eta0"], random_state=seed,
        )
    if model_name == "histgb":
        return HistGradientBoostingClassifier(
            max_depth=params["max_depth"], learning_rate=params["learning_rate"],
            max_leaf_nodes=params["max_leaf_nodes"], random_state=seed,
        )
    raise ValueError(model_name)


def tune_hyperparameters(model_name: str, train_df, y_train, val_df, y_val, seed: int) -> dict:
    log.info("Optuna: afinando %s con %d trials...", model_name, OPTUNA_TRIALS[model_name])
    pipe = build_fitted_pipeline(train_df, y_train, seed)
    X_val, dense_val = pipe.full_features(val_df, fit=False)
    X_tr, dense_tr = pipe.full_features(train_df, fit=False)
    X_fit = dense_tr if model_name == "histgb" else X_tr
    X_v = dense_val if model_name == "histgb" else X_val

    def objective(trial: optuna.Trial) -> float:
        params = _param_space(trial, model_name)
        model = _build_model(model_name, params, seed)
        model.fit(X_fit, y_train)
        p = model.predict_proba(X_v)[:, 1]
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_val, p)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=OPTUNA_TRIALS[model_name], show_progress_bar=False)
    log.info("%s: mejor ROC AUC en validacion = %.4f, params = %s", model_name, study.best_value, study.best_params)
    return study.best_params


def run_one(model_name: str, best_params: dict, pipe: FittedPipeline,
            X_tr, dense_tr, y_train, X_val, dense_val, y_val,
            X_test, dense_test, y_test, tier: int, seed: int) -> dict:
    process = psutil.Process()

    rss_before = process.memory_info().rss
    t0 = time.perf_counter()
    model = _build_model(model_name, best_params, seed)
    X_fit = dense_tr if model_name == "histgb" else X_tr
    model.fit(X_fit, y_train)
    fit_time = time.perf_counter() - t0
    rss_peak_fit = process.memory_info().rss

    X_val_fit = dense_val if model_name == "histgb" else X_val
    X_test_fit = dense_test if model_name == "histgb" else X_test

    t0 = time.perf_counter()
    p_test = model.predict_proba(X_test_fit)[:, 1]
    predict_time = time.perf_counter() - t0

    p_val = model.predict_proba(X_val_fit)[:, 1]
    threshold = choose_threshold_on_validation(y_val.values, p_val)
    report = full_metric_report(y_test.values, p_test, threshold)

    tag = f"{model_name}_tier{tier}_seed{seed}"
    np.save(PRED_DIR / f"{tag}_y_test.npy", y_test.values)
    np.save(PRED_DIR / f"{tag}_p_test.npy", p_test)

    if tier == max(SAMPLE_TIERS) and seed == SEEDS[0]:
        joblib.dump({"pipeline": pipe, "model": model}, MODELS_DIR / f"{model_name}_reference.joblib")

    return {
        "model": model_name, "tier": tier, "seed": seed,
        "fit_time_s": fit_time, "predict_time_s": predict_time,
        "rss_peak_mb": (rss_peak_fit - rss_before) / (1024**2),
        **report,
    }


def _load_best_params() -> dict:
    path = SKLEARN_DIR / "best_params.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_best_params(best_params: dict) -> None:
    (SKLEARN_DIR / "best_params.json").write_text(
        json.dumps(best_params, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _load_done_runs() -> set[tuple[str, int, int]]:
    path = SKLEARN_DIR / "runs.csv"
    if not path.exists():
        return set()
    done = pd.read_csv(path)[["model", "tier", "seed"]]
    return {(r.model, int(r.tier), int(r.seed)) for r in done.itertuples()}


def _append_run(row: dict) -> None:
    path = SKLEARN_DIR / "runs.csv"
    df_row = pd.DataFrame([row])
    df_row.to_csv(path, mode="a", header=not path.exists(), index=False)


def main() -> None:
    con = _con()
    log.info("Muestreando validacion (dia %d) y prueba (dia %d), %d filas fijas cada una...",
              VALIDATION_DAY, TEST_DAY, VAL_TEST_SAMPLE)
    val_df = _sample_days(con, [VALIDATION_DAY], VAL_TEST_SAMPLE, seed=SEEDS[0])
    test_df = _sample_days(con, [TEST_DAY], VAL_TEST_SAMPLE, seed=SEEDS[0])
    y_val, y_test = val_df[LABEL_COL], test_df[LABEL_COL]

    best_params = _load_best_params()

    log.info("Muestreando tier de 1M filas para afinar sgd/histgb...")
    tune_train_df = _sample_days(con, TRAIN_DAYS, 1_000_000, seed=SEEDS[0])
    y_tune_train = tune_train_df[LABEL_COL]

    for model_name in ("sgd", "histgb"):
        if model_name in best_params:
            log.info("%s: parametros ya afinados (en best_params.json), se reusan.", model_name)
            continue
        best_params[model_name] = tune_hyperparameters(
            model_name, tune_train_df, y_tune_train, val_df, y_val, seed=SEEDS[0]
        )
        _save_best_params(best_params)

    if "mlp" not in best_params:
        log.info("Muestreando %d filas para afinar mlp (muestra reducida, ver nota de alcance)...",
                  MLP_TUNE_SAMPLE)
        mlp_tune_df = _sample_days(con, TRAIN_DAYS, MLP_TUNE_SAMPLE, seed=SEEDS[0])
        y_mlp_tune = mlp_tune_df[LABEL_COL]
        best_params["mlp"] = tune_hyperparameters(
            "mlp", mlp_tune_df, y_mlp_tune, val_df, y_val, seed=SEEDS[0]
        )
        _save_best_params(best_params)
    else:
        log.info("mlp: parametros ya afinados (en best_params.json), se reusan.")

    done_runs = _load_done_runs()
    n_done, n_total = 0, len(SAMPLE_TIERS) * len(SEEDS) * 3
    for tier in SAMPLE_TIERS:
        for seed in SEEDS:
            needed = [m for m in ("sgd", "histgb", "mlp") if (m, tier, seed) not in done_runs]
            if not needed:
                n_done += 3
                continue
            log.info("Muestreando tier=%d, seed=%d...", tier, seed)
            train_df = _sample_days(con, TRAIN_DAYS, tier, seed=seed)
            y_train = train_df[LABEL_COL]
            # El pipeline de encoders y las features de train/val/test se calculan UNA
            # sola vez por (tier, seed) y se reutilizan entre los 3 modelos: evita
            # recodificar el bloque fijo de validacion/prueba (2M filas) tres veces.
            pipe = build_fitted_pipeline(train_df, y_train, seed)
            X_tr, dense_tr = pipe.full_features(train_df, fit=False)
            X_val, dense_val = pipe.full_features(val_df, fit=False)
            X_test, dense_test = pipe.full_features(test_df, fit=False)
            for model_name in needed:
                log.info("Entrenando %s | tier=%d | seed=%d", model_name, tier, seed)
                row = run_one(model_name, best_params[model_name], pipe,
                               X_tr, dense_tr, y_train, X_val, dense_val, y_val,
                               X_test, dense_test, y_test, tier, seed)
                _append_run(row)
                n_done += 1
                log.info("  -> [%d/%d] roc_auc=%.4f pr_auc=%.4f fit=%.1fs predict=%.1fs",
                          n_done, n_total, row["roc_auc"], row["pr_auc"],
                          row["fit_time_s"], row["predict_time_s"])

    log.info("Listo. Corridas guardadas en %s", SKLEARN_DIR / "runs.csv")


if __name__ == "__main__":
    main()
