"""
Benchmarking computacional: tiempos por etapa, scaling fuerte/debil en modo
local de Spark (sin cluster real, se varia local[1,2,4,8,12]), ajuste de la ley
de Amdahl, y punto de cruce sklearn-vs-Spark-local por volumen de datos.
"""
from __future__ import annotations

import json
import logging
import sys
import time

import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from config import ARTIFACTS_DIR, DATA_PROCESSED_DIR, LABEL_COL, TRAIN_DAYS
from plotting import finalize_and_save, new_figure, series_styles
from spark_pipeline import build_spark, build_feature_pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import col

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("benchmarking")

BENCH_DIR = ARTIFACTS_DIR / "benchmarking"
BENCH_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_GLOB = (DATA_PROCESSED_DIR / "day=*" / "*.parquet").as_posix()
CORE_LEVELS = [1, 2, 4, 8, 12]  # nucleos logicos reales de la maquina (ver data_sheet.json)
STRONG_SCALING_ROWS = 2_000_000  # tamano fijo, acotado por tiempo total de la bateria de pruebas


def _timed_spark_fit_lr(master: str, df) -> float:
    spark = build_spark(master=master, driver_memory="8g")
    spark.sparkContext.setLogLevel("WARN")
    sdf = spark.createDataFrame(df)
    pipe = build_feature_pipeline()
    t0 = time.perf_counter()
    fitted = pipe.fit(sdf)
    lr = LogisticRegression(featuresCol="features", labelCol=LABEL_COL, maxIter=30)
    lr.fit(fitted.transform(sdf))
    elapsed = time.perf_counter() - t0
    spark.stop()
    return elapsed


def strong_scaling(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    log.info("Strong scaling: tamano fijo de %d filas, variando local[1,2,4,8,12]...", STRONG_SCALING_ROWS)
    days_sql = f"({','.join(map(str, TRAIN_DAYS))})"
    fixed_df = con.execute(
        f"SELECT * FROM read_parquet('{PARQUET_GLOB}') WHERE day IN {days_sql} "
        f"USING SAMPLE {STRONG_SCALING_ROWS} ROWS (reservoir, 13)"
    ).df()

    rows = []
    for p in CORE_LEVELS:
        elapsed = _timed_spark_fit_lr(f"local[{p}]", fixed_df)
        rows.append({"p_nucleos": p, "tiempo_s": elapsed})
        log.info("  local[%d] -> %.1f s", p, elapsed)

    df = pd.DataFrame(rows)
    t1 = df.loc[df["p_nucleos"] == 1, "tiempo_s"].iloc[0]
    df["speedup"] = t1 / df["tiempo_s"]
    df["eficiencia"] = df["speedup"] / df["p_nucleos"]
    df.to_csv(BENCH_DIR / "strong_scaling.csv", index=False)
    return df


def weak_scaling(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    log.info("Weak scaling: tamano proporcional a nucleos (1 nucleo/1M filas)...")
    days_sql = f"({','.join(map(str, TRAIN_DAYS))})"
    rows = []
    for p in CORE_LEVELS:
        n = p * 1_000_000
        df_p = con.execute(
            f"SELECT * FROM read_parquet('{PARQUET_GLOB}') WHERE day IN {days_sql} "
            f"USING SAMPLE {n} ROWS (reservoir, 13)"
        ).df()
        elapsed = _timed_spark_fit_lr(f"local[{p}]", df_p)
        rows.append({"p_nucleos": p, "n_filas": n, "tiempo_s": elapsed})
        log.info("  local[%d] con %d filas -> %.1f s", p, n, elapsed)

    df = pd.DataFrame(rows)
    df.to_csv(BENCH_DIR / "weak_scaling.csv", index=False)
    return df


def amdahl_law(p: np.ndarray, f: float) -> np.ndarray:
    return 1.0 / (f + (1.0 - f) / p)


def fit_amdahl(strong_df: pd.DataFrame) -> dict:
    p = strong_df["p_nucleos"].values.astype(float)
    speedup = strong_df["speedup"].values.astype(float)
    popt, pcov = curve_fit(amdahl_law, p, speedup, p0=[0.1], bounds=(0.0, 1.0))
    f_hat = float(popt[0])
    se = float(np.sqrt(np.diag(pcov))[0])
    result = {"fraccion_secuencial_f": f_hat, "error_estandar_f": se}
    (BENCH_DIR / "amdahl_fit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def plot_scaling(strong_df: pd.DataFrame, amdahl_params: dict) -> None:
    fig, ax = new_figure()
    styles = series_styles(2)
    p_grid = np.linspace(1, strong_df["p_nucleos"].max(), 100)
    ax.plot(strong_df["p_nucleos"], strong_df["speedup"], label="speedup observado",
            color=styles[0]["color"], linestyle=styles[0]["linestyle"], marker=styles[0]["marker"])
    ax.plot(p_grid, amdahl_law(p_grid, amdahl_params["fraccion_secuencial_f"]),
            label=f"ley de Amdahl (f={amdahl_params['fraccion_secuencial_f']:.2f})",
            color=styles[1]["color"], linestyle=styles[1]["linestyle"])
    ax.plot(p_grid, p_grid, color="#bfbfbf", linestyle=":", linewidth=1, label="speedup ideal")
    ax.set_xlabel("nucleos locales (p)")
    ax.set_ylabel("speedup S(p) = T1 / Tp")
    finalize_and_save(fig, ax, BENCH_DIR / "strong_scaling.png", title="Strong scaling y ajuste de la ley de Amdahl")


def train_fraction(con: duckdb.DuckDBPyConnection) -> float:
    days_sql = f"({','.join(map(str, TRAIN_DAYS))})"
    n_train = con.execute(f"SELECT count(*) FROM read_parquet('{PARQUET_GLOB}') WHERE day IN {days_sql}").fetchone()[0]
    n_total = con.execute(f"SELECT count(*) FROM read_parquet('{PARQUET_GLOB}')").fetchone()[0]
    return n_train / n_total


def effective_sizes(con: duckdb.DuckDBPyConnection) -> dict:
    """En DuckDB, USING SAMPLE se aplica ANTES del WHERE: las consultas de
    sklearn_pipeline.py y de strong/weak scaling muestrean n filas del dataset
    completo y despues filtran por dia, asi que el tamano efectivo es n por la
    fraccion de filas de esos dias (no hay fuga: el filtro temporal se respeta)."""
    frac = train_fraction(con)
    test_n = con.execute(f"SELECT count(*) FROM read_parquet('{PARQUET_GLOB}') WHERE day = 30").fetchone()[0]
    val_n = con.execute(f"SELECT count(*) FROM read_parquet('{PARQUET_GLOB}') WHERE day = 29").fetchone()[0]
    total = con.execute(f"SELECT count(*) FROM read_parquet('{PARQUET_GLOB}')").fetchone()[0]
    sizes = {
        "fraccion_train": frac,
        "sklearn_tiers_nominal_a_efectivo": {str(t): round(t * frac) for t in (100_000, 1_000_000, 2_000_000)},
        "sklearn_validacion_efectiva_aprox": round(2_000_000 * val_n / total),
        "sklearn_prueba_efectiva_exacta": 202_511,
        "strong_scaling_efectivo_aprox": round(STRONG_SCALING_ROWS * frac),
        "weak_scaling_efectivo_aprox": {str(p): round(p * 1_000_000 * frac) for p in CORE_LEVELS},
    }
    (BENCH_DIR / "tamanos_efectivos.json").write_text(json.dumps(sizes, indent=2), encoding="utf-8")
    return sizes


CROSSOVER_SIZES = [80_085, 800_847, 1_601_694, 3_203_388]  # efectivos: los 3 tiers de sklearn y el doble del mayor
SPARK_REPEATS = 3
SKLEARN_BYTES_PER_ROW = 2.6e3  # pico del preprocesamiento medido en stage_benchmark.py (1.972 MB / 800.847 filas)


def _timed_sklearn_sgd(df: pd.DataFrame) -> dict:
    """Preprocesamiento una vez y ajuste de SGDClassifier (log loss, hiperparametros
    afinados) con las 5 semillas, sobre la misma muestra que usa Spark."""
    import gc

    from config import SEEDS
    from sklearn_pipeline import _build_model, build_fitted_pipeline
    best = json.loads((ARTIFACTS_DIR / "sklearn" / "best_params.json").read_text(encoding="utf-8"))["sgd"]
    y = df[LABEL_COL]
    t0 = time.perf_counter()
    _, X, _ = build_fitted_pipeline(df, y, SEEDS[0], return_features=True)
    prep = time.perf_counter() - t0
    times, iters = [], []
    for seed in SEEDS:
        m = _build_model("sgd", best, seed)
        t0 = time.perf_counter()
        m.fit(X, y)
        times.append(time.perf_counter() - t0)
        iters.append(m.n_iter_)
    del X
    gc.collect()
    return {"preproc_sklearn_s": prep, "tiempo_sklearn_sgd_s": float(np.mean(times)),
            "de_sklearn_sgd_s": float(np.std(times, ddof=1)), "iteraciones_sklearn": float(np.mean(iters))}


def _timed_spark_lr_fit_only(tmp) -> dict:
    """Mide SOLO el ajuste de LogisticRegression sobre variables ya calculadas y
    cacheadas, igual que en sklearn (que excluye el preprocesamiento). Los datos
    entran a Spark como Parquet, igual que en stage_benchmark.py. El ajuste se
    repite SPARK_REPEATS veces en la misma sesion; L-BFGS es determinista, asi que
    las repeticiones solo miden la variacion del tiempo."""
    spark = build_spark(master="local[*]", driver_memory="8g")
    spark.sparkContext.setLogLevel("WARN")
    sdf = spark.read.parquet(str(tmp)).cache()
    sdf.count()
    t0 = time.perf_counter()
    fitted = build_feature_pipeline().fit(sdf)
    feats = fitted.transform(sdf).select("features", LABEL_COL).cache()
    feats.count()
    prep = time.perf_counter() - t0
    lr = LogisticRegression(featuresCol="features", labelCol=LABEL_COL, regParam=0.01, maxIter=50)
    times = []
    for _ in range(SPARK_REPEATS):
        t0 = time.perf_counter()
        model = lr.fit(feats)
        times.append(time.perf_counter() - t0)
    out = {"preproc_spark_s": prep, "tiempo_spark_lr_s": float(np.median(times)),
           "de_spark_lr_s": float(np.std(times, ddof=1)), "iteraciones_spark": model.summary.totalIterations,
           "particiones_spark": feats.rdd.getNumPartitions()}
    spark.stop()
    return out


def crossover_point(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Punto de cruce entre entornos con el MISMO tipo de modelo: regresion
    logistica (SGDClassifier con log loss en sklearn, LogisticRegression en Spark,
    con sus hiperparametros afinados). Ambos entornos se miden en la misma sesion,
    sobre la MISMA muestra de cada tamano y solo en el ajuste del modelo. Primero
    se mide sklearn en todos los tamanos, sin ninguna JVM activa (spark.stop() no
    termina el proceso Java, que conserva su memoria), y despues Spark sobre los
    mismos archivos Parquet. Antes de correr sklearn se verifica que la memoria
    disponible alcance para su pico de preprocesamiento; si no alcanza, ese tamano
    solo se mide en Spark."""
    import gc

    import psutil
    import pyarrow as pa
    import pyarrow.parquet as pq

    log.info("Estimando punto de cruce sklearn vs. Spark-local (regresion logistica)...")
    days_sql = f"({','.join(map(str, TRAIN_DAYS))})"
    tmp_dir = BENCH_DIR / "stage_inputs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in CROSSOVER_SIZES:  # pasada 1: muestra, Parquet y sklearn
        df_n = con.execute(
            f"SELECT * FROM (SELECT * FROM read_parquet('{PARQUET_GLOB}') WHERE day IN {days_sql}) "
            f"USING SAMPLE {n} ROWS (reservoir, 13)"
        ).df()
        tmp = tmp_dir / f"cruce_{n}.parquet"
        # Grupos de 122.880 filas, como los que escribe DuckDB en stage_benchmark.py:
        # Spark solo puede repartir un Parquet entre tareas por grupos de filas, y con
        # el tamano por defecto de pyarrow (1 millon) leeria todo en una particion.
        pq.write_table(pa.Table.from_pandas(df_n, preserve_index=False), tmp, row_group_size=122_880)
        need_mb = 1.3 * SKLEARN_BYTES_PER_ROW * n / 1024**2
        avail_mb = psutil.virtual_memory().available / 1024**2
        row = {"n_filas_efectivas": len(df_n), "memoria_necesaria_sklearn_mb": need_mb,
               "memoria_disponible_mb": avail_mb, "_tmp": tmp}
        if avail_mb > need_mb:
            row.update(_timed_sklearn_sgd(df_n))
            log.info("  n=%d sklearn=%.1f s", n, row["tiempo_sklearn_sgd_s"])
        else:
            log.info("  n=%d: sklearn omitido (necesita ~%.0f MB, disponibles %.0f MB)", n, need_mb, avail_mb)
        rows.append(row)
        del df_n
        gc.collect()
    for row in rows:  # pasada 2: Spark sobre los mismos archivos
        row.update(_timed_spark_lr_fit_only(row["_tmp"]))
        log.info("  n=%d -> sklearn=%s s | spark=%.1f s (%d iteraciones, %d particiones)", row["n_filas_efectivas"],
                 f"{row['tiempo_sklearn_sgd_s']:.1f}" if "tiempo_sklearn_sgd_s" in row else "n/d",
                 row["tiempo_spark_lr_s"], row["iteraciones_spark"], row["particiones_spark"])
    for row in rows:
        row.pop("_tmp").unlink(missing_ok=True)

    merged = pd.DataFrame(rows)
    merged["spark_mas_rapido"] = merged["tiempo_spark_lr_s"] < merged["tiempo_sklearn_sgd_s"]
    merged["razon_spark_sobre_sklearn"] = merged["tiempo_spark_lr_s"] / merged["tiempo_sklearn_sgd_s"]
    merged.to_csv(BENCH_DIR / "crossover_point.csv", index=False)

    fig, ax = new_figure()
    styles = series_styles(2)
    ax.plot(merged["n_filas_efectivas"], merged["tiempo_sklearn_sgd_s"], label="scikit-learn (SGD, log loss)",
            color=styles[0]["color"], linestyle=styles[0]["linestyle"], marker=styles[0]["marker"])
    ax.plot(merged["n_filas_efectivas"], merged["tiempo_spark_lr_s"], label="Spark local[*] (LogisticRegression)",
            color=styles[1]["color"], linestyle=styles[1]["linestyle"], marker=styles[1]["marker"])
    ax.set_xscale("log")
    ax.set_xlabel("filas de entrenamiento efectivas (escala log)")
    ax.set_ylabel("tiempo de ajuste del modelo (s)")
    finalize_and_save(fig, ax, BENCH_DIR / "crossover_point.png",
                       title="Punto de cruce: regresion logistica en scikit-learn vs. Spark local")
    return merged


def main() -> None:
    con = duckdb.connect()
    con.execute("SET memory_limit='10GB'")
    if "--solo-cruce" in sys.argv:
        # Repeticion del punto de cruce con muestreo determinista (un hilo).
        con.execute("SET threads TO 1")
        crossover_point(con)
        return
    con.execute("SET threads TO 8")

    strong_df = strong_scaling(con)
    weak_scaling(con)
    amdahl_params = fit_amdahl(strong_df)
    plot_scaling(strong_df, amdahl_params)
    effective_sizes(con)
    con.execute("SET threads TO 1")  # muestreo determinista para el punto de cruce
    crossover_point(con)
    log.info("Benchmarking completo. Artifacts en %s", BENCH_DIR)


if __name__ == "__main__":
    main()
