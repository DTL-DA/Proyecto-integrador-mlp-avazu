"""
Benchmarking por etapas (lectura, preprocesamiento, entrenamiento y prediccion)
con tiempo de pared, memoria pico y filas por segundo, en ambos entornos, mas la
evidencia de la Spark UI (shuffle, spill a disco y sesgo entre tareas) que pide
el enunciado.

Diseno: ambos entornos leen EXACTAMENTE el mismo archivo parquet (mismas filas):
TRAIN_ROWS filas de los dias 21-28 (el tamano efectivo del tier medio de
sklearn) y TEST_ROWS del dia 30. El modelo es la regresion logistica en ambos
(SGDClassifier log loss / LogisticRegression, hiperparametros afinados); en
sklearn se mide ademas el MLP. La memoria pico se mide muestreando el RSS cada
0.1 s: del proceso Python en sklearn y del proceso Java (driver = executor en
modo local) en Spark. Las metricas de la UI se leen de su API REST antes de
cerrar la sesion; la UI se habilita solo en este script.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request

import duckdb
import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq

from config import ARTIFACTS_DIR, DATA_PROCESSED_DIR, LABEL_COL, SEEDS, TEST_DAY, TRAIN_DAYS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("stage_benchmark")

OUT_DIR = ARTIFACTS_DIR / "benchmarking"
TMP_DIR = OUT_DIR / "stage_inputs"
TMP_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_GLOB = (DATA_PROCESSED_DIR / "day=*" / "*.parquet").as_posix()
TRAIN_ROWS = 800_847
TEST_ROWS = 200_000
UI_PORT = 4050


class PeakRSS:
    """Muestrea el RSS de un conjunto de procesos cada 0.1 s y guarda el pico."""

    def __init__(self, procs_fn):
        self.procs_fn = procs_fn
        self.peak = 0
        self._stop = threading.Event()

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            total = 0
            for p in self.procs_fn():
                try:
                    total += p.memory_info().rss
                except psutil.Error:
                    pass
            self.peak = max(self.peak, total)
            time.sleep(0.1)

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def _stage(rows: list, entorno: str, etapa: str, n: int, procs_fn, fn):
    with PeakRSS(procs_fn) as mon:
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
    rows.append({"entorno": entorno, "etapa": etapa, "filas": n, "tiempo_s": dt,
                 "filas_por_s": n / dt if dt > 0 else np.nan, "memoria_pico_mb": mon.peak / 1024**2})
    log.info("  %-8s %-26s %.2fs  pico=%.0f MB", entorno, etapa, dt, mon.peak / 1024**2)
    return out


def write_inputs() -> tuple[str, str]:
    con = duckdb.connect()
    con.execute("SET memory_limit='10GB'")
    con.execute("SET threads TO 1")  # muestreo determinista
    days = ",".join(map(str, TRAIN_DAYS))
    train_path, test_path = TMP_DIR / "train_800k.parquet", TMP_DIR / "test_200k.parquet"
    con.execute(f"COPY (SELECT * FROM (SELECT * FROM read_parquet('{PARQUET_GLOB}') WHERE day IN ({days})) "
                f"USING SAMPLE {TRAIN_ROWS} ROWS (reservoir, {SEEDS[0]})) TO '{train_path.as_posix()}' (FORMAT PARQUET)")
    con.execute(f"COPY (SELECT * FROM (SELECT * FROM read_parquet('{PARQUET_GLOB}') WHERE day = {TEST_DAY}) "
                f"USING SAMPLE {TEST_ROWS} ROWS (reservoir, {SEEDS[0]})) TO '{test_path.as_posix()}' (FORMAT PARQUET)")
    return str(train_path), str(test_path)


def bench_sklearn(rows: list, train_path: str, test_path: str) -> None:
    from sklearn_pipeline import _build_model, build_fitted_pipeline
    best = json.loads((ARTIFACTS_DIR / "sklearn" / "best_params.json").read_text(encoding="utf-8"))
    me = psutil.Process()
    procs = lambda: [me]

    train_df = _stage(rows, "sklearn", "lectura_train", TRAIN_ROWS, procs, lambda: pq.read_table(train_path).to_pandas())
    y = train_df[LABEL_COL]
    # El target encoding de train es out-of-fold, asi que ajuste y transformacion
    # de train ocurren en una sola pasada; se reportan como una etapa.
    pipe, X_tr, _ = _stage(rows, "sklearn", "preprocesamiento_ajuste_y_transform", TRAIN_ROWS, procs,
                           lambda: build_fitted_pipeline(train_df, y, SEEDS[0], return_features=True))
    models = {}
    for name in ("sgd", "mlp"):
        models[name] = _stage(rows, "sklearn", f"entrenamiento_{name}", TRAIN_ROWS, procs,
                              lambda n=name: _build_model(n, best[n], SEEDS[0]).fit(X_tr, y))
    test_df = _stage(rows, "sklearn", "lectura_test", TEST_ROWS, procs, lambda: pq.read_table(test_path).to_pandas())
    X_te, _ = _stage(rows, "sklearn", "preprocesamiento_test", TEST_ROWS, procs,
                     lambda: pipe.full_features(test_df, fit=False))
    for name, m in models.items():
        _stage(rows, "sklearn", f"prediccion_{name}", TEST_ROWS, procs, lambda m=m: m.predict_proba(X_te))


def _ui_get(path: str):
    with urllib.request.urlopen(f"http://127.0.0.1:{UI_PORT}/api/v1{path}", timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def spark_ui_metrics() -> pd.DataFrame:
    app = _ui_get("/applications")[0]["id"]
    out = []
    for st in _ui_get(f"/applications/{app}/stages?status=complete"):
        q = _ui_get(f"/applications/{app}/stages/{st['stageId']}/{st['attemptId']}/taskSummary"
                    f"?quantiles=0.0,0.5,1.0")
        rt = q.get("executorRunTime", [0, 0, 0])
        out.append({"stage_id": st["stageId"], "nombre": st["name"][:80], "num_tareas": st["numTasks"],
                    "input_mb": st.get("inputBytes", 0) / 1024**2,
                    "shuffle_lectura_mb": st.get("shuffleReadBytes", 0) / 1024**2,
                    "shuffle_escritura_mb": st.get("shuffleWriteBytes", 0) / 1024**2,
                    "spill_memoria_mb": st.get("memoryBytesSpilled", 0) / 1024**2,
                    "spill_disco_mb": st.get("diskBytesSpilled", 0) / 1024**2,
                    "tiempo_tarea_mediana_ms": rt[1], "tiempo_tarea_max_ms": rt[2],
                    "sesgo_max_sobre_mediana": rt[2] / rt[1] if rt[1] else np.nan,
                    "tiempo_ejecucion_total_ms": st.get("executorRunTime", 0)})
    return pd.DataFrame(out).sort_values("stage_id")


def bench_spark(rows: list, train_path: str, test_path: str) -> None:
    from pyspark.ml.classification import LogisticRegression
    from pyspark.sql import SparkSession
    from spark_pipeline import build_feature_pipeline

    spark = (SparkSession.builder.appName("stage-benchmark").master("local[*]")
             .config("spark.driver.memory", "8g").config("spark.sql.shuffle.partitions", "12")
             .config("spark.driver.host", "127.0.0.1").config("spark.driver.bindAddress", "127.0.0.1")
             .config("spark.ui.enabled", "true").config("spark.ui.port", str(UI_PORT)).getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    me = psutil.Process()
    procs = lambda: [c for c in me.children(recursive=True) if "java" in c.name().lower()]

    train = _stage(rows, "spark", "lectura_train", TRAIN_ROWS, procs,
                   lambda: (lambda d: (d.count(), d)[1])(spark.read.parquet(train_path).cache()))
    fitted = _stage(rows, "spark", "preprocesamiento_ajuste", TRAIN_ROWS, procs,
                    lambda: build_feature_pipeline().fit(train))
    feats = _stage(rows, "spark", "preprocesamiento_transform", TRAIN_ROWS, procs,
                   lambda: (lambda d: (d.count(), d)[1])(fitted.transform(train).select("features", LABEL_COL).cache()))
    lr = _stage(rows, "spark", "entrenamiento_lr", TRAIN_ROWS, procs,
                lambda: LogisticRegression(featuresCol="features", labelCol=LABEL_COL,
                                           regParam=0.01, maxIter=50).fit(feats))
    test = _stage(rows, "spark", "lectura_test", TEST_ROWS, procs,
                  lambda: (lambda d: (d.count(), d)[1])(spark.read.parquet(test_path).cache()))
    test_feats = _stage(rows, "spark", "preprocesamiento_test", TEST_ROWS, procs,
                        lambda: (lambda d: (d.count(), d)[1])(fitted.transform(test).cache()))
    _stage(rows, "spark", "prediccion_lr", TEST_ROWS, procs,
           lambda: lr.transform(test_feats).select("probability").count())

    try:
        ui = spark_ui_metrics()
        ui.to_csv(OUT_DIR / "spark_ui_stages.csv", index=False)
        log.info("Spark UI: %d stages; spill disco total=%.1f MB; shuffle escritura total=%.1f MB; "
                 "sesgo max=%.1f", len(ui), ui["spill_disco_mb"].sum(), ui["shuffle_escritura_mb"].sum(),
                 ui["sesgo_max_sobre_mediana"].max())
    except Exception as e:  # la UI es evidencia complementaria; no debe tumbar los tiempos
        log.warning("No se pudo leer la API REST de la Spark UI: %s", e)
    spark.stop()


def main() -> None:
    log.info("Escribiendo los mismos insumos parquet para ambos entornos...")
    train_path, test_path = write_inputs()
    rows: list = []
    log.info("scikit-learn:")
    bench_sklearn(rows, train_path, test_path)
    log.info("Spark local[*]:")
    bench_spark(rows, train_path, test_path)
    pd.DataFrame(rows).to_csv(OUT_DIR / "tiempos_por_etapa.csv", index=False)
    log.info("Benchmarking por etapas listo en %s", OUT_DIR)


if __name__ == "__main__":
    main()
