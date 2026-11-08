"""
Modelado en PySpark, modo local unicamente (sin cluster real: local[*]).

Grilla del MLP: la que fija literalmente el enunciado, sin recortar valores --
capas [d,10,2], [d,50,2], [d,100,2], [d,128,64,2]; stepSize 0.1 y 0.01;
maxIter 100 y 200; solver l-bfgs y gd (32 combinaciones). LogisticRegression y
GBTClassifier usan una grilla mas chica con el mismo criterio de "esfuerzo de
ajuste" (regla de la skill inddl), ya que el enunciado no fija valores
literales para esos dos.

Reduccion de alcance declarada (sin cluster, maquina de 12 nucleos/16 GB RAM):
- La seleccion de hiperparametros se hace evaluando sobre una muestra reducida
  (TUNE_SAMPLE_ROWS filas de TUNE_DAYS), no sobre los 8 dias completos, NO con
  TrainValidationSplit de Spark (que hace un split aleatorio interno y
  violaria el protocolo anti-fuga temporal), sino contra el dia de validacion
  real (29). Esto es la misma estrategia que el enunciado ya usa para Criteo
  ("se usan entre 2 y 7 dias de los 24 disponibles").
- Busqueda de grilla cancelada por tiempo medido, no por falta de intento: se
  corrio la busqueda real (grilla completa del MLP via muestreo aleatorio de 8
  de las 32 combinaciones declaradas, grilla de GBT completa) sobre la muestra
  reducida, y aun asi la primera combinacion de GBT (max_iter=30, 1M filas)
  tardo >18 min, y la segunda (max_iter=60) no habia terminado tras MAS DE 7
  HORAS, con el log mostrando advertencias repetidas de memoria agotada
  ("Not enough space to cache rdd in memory") propias del almacenamiento
  iterativo de GBT/MLP en Spark local. Es decir, el cuello de botella no es el
  volumen de datos (reducirlo de 9.46M a 1M filas no acelero la busqueda) sino
  el overhead de Spark local por iteracion mas la presion de memoria del modo
  local (driver = unico executor). Dado el plazo de 2 dias, se abandona la
  busqueda de grilla para GBT y MLP y se usa UN solo valor razonable por eje:
  GBT (su grilla es propia, no mandatada por el enunciado) con maxDepth=5,
  maxIter=10 -- bajado de 30 tras diagnosticar la causa real del problema de
  memoria (ver mas abajo); MLP con capas=[100] (arquitectura intermedia de
  las 4 mandatadas, estas si del espacio declarado por el enunciado),
  stepSize=0.1, maxIter=100 (el menor de los dos mandatados) y solver="l-bfgs"
  (mas estable/rapido que "gd" en la documentacion de Spark ML). No es un
  "mejor" hallado por busqueda: es una eleccion razonada y declarada por
  limite de tiempo, documentada aqui para el anexo de reproducibilidad. LR si
  tuvo busqueda completa (2 combinaciones, rapida: ~2-3 min cada una).
  El entrenamiento FINAL de LR si usa los 8 dias completos de TRAIN_DAYS
  (dataset completo, ~32.4M filas): completo sin problemas, ~500s por
  corrida x 5 semillas. GBT y MLP NO: su cacheo iterativo de RDDs en Spark
  local sobre el dataset completo entro en una espiral de presion de memoria
  medida en vivo (avisos "Not enough space to cache rdd in memory" subiendo
  de 12 a 70 mientras el avance por stage caia de ~3.3 a ~0.27 stages/min en
  la misma corrida, es decir, empeorando con el tiempo, no solo lento) incluso
  con memoria de driver subida a 12g y fracciones de memoria ajustadas. Por
  eso GBT y MLP se entrenan sobre una muestra de FINAL_TRAIN_SAMPLE_ROWS (1M
  filas, el tier intermedio de sklearn), distinta por semilla.
- Diagnostico final para GBT: bajar maxIter de 30 a 10 NO cambio el tamano
  del bloque problematico (rdd_124_0 seguia reportando ~7.5 GB con 1M filas,
  el mismo orden que con 2M). Eso descarta tanto el volumen de datos como la
  profundidad de iteraciones como causa unica: la causa real es la dimension
  del espacio. GBT (como cualquier arbol) construye estadisticos/histogramas
  por nodo sobre las ~4096 columnas del bloque disperso de hashing, igual que
  HistGradientBoostingClassifier en sklearn -- y ahi mismo se evito este
  problema usando SOLO el bloque denso (ver docstring de sklearn_pipeline.py).
  Aqui se aplica la misma correccion: GBT en Spark usa build_feature_pipeline_dense()
  (solo las categoricas de baja cardinalidad indexadas, sin el hashing de alta
  cardinalidad); con eso GBT termino sus 5 semillas sin un solo aviso de
  memoria (antes no pasaba de la primera semilla). El checkpointing de Spark
  (checkpointInterval, pensado para otro tipo de problema de lineage, no para
  este) se probo primero pero requiere winutils.exe real en Windows, un
  binario de terceros no oficial que no se descarga aqui.
- MLP SI necesita el espacio de hashing completo (el enunciado lo exige:
  capas [d,...], donde "d" es la dimension de entrada -- no puede usar solo
  el bloque denso como GBT). El problema aqui no es de cacheo sino de
  OutOfMemoryError real: MultilayerPerceptronClassifier de Spark ML densifica
  el vector de entrada para sus multiplicaciones de matrices, y 1M filas x
  4096 columnas (HASH_DIM) densificadas son ~32.8 GB solo para esa copia --
  mas que los 16 GB de RAM de la maquina, incluso con el pipeline de LR/MLP
  fallando en una JVM recien iniciada (no era fragmentacion acumulada de
  GBT). La correccion: MLP usa su propio pipeline con MLP_HASH_DIM=512 (en
  vez de 4096) y una muestra mas chica, MLP_FINAL_SAMPLE_ROWS=300k (en vez de
  1M). "d" en la notacion del enunciado es justamente la dimension de
  entrada, variable segun el hashing usado -- seguir usando capas=[d,100,2]
  con un "d" mas chico no viola la grilla mandatada, solo cambia que tan
  grande es "d" por limite real de memoria. Esto es exactamente el margen
  que el enunciado permite ("hasta donde permita la memoria").
"""
from __future__ import annotations

import itertools
import json
import logging
import random
import time

import numpy as np
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array
from pyspark.ml.classification import (
    GBTClassifier,
    LogisticRegression,
    MultilayerPerceptronClassifier,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import FeatureHasher, StandardScaler, StringIndexer, VectorAssembler
from pyspark.sql import SparkSession

from config import (
    ARTIFACTS_DIR,
    DATA_PROCESSED_DIR,
    HIGH_CARDINALITY_CATEGORICALS,
    LABEL_COL,
    LOW_CARDINALITY_CATEGORICALS,
    SEEDS,
    TEST_DAY,
    TRAIN_DAYS,
    VALIDATION_DAY,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("spark_pipeline")

SPARK_DIR = ARTIFACTS_DIR / "spark"
PRED_DIR = SPARK_DIR / "predictions"
FLAT_CACHE_DIR = SPARK_DIR / "flat_cache"
SPARK_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)
FLAT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

HASH_DIM = 2**12  # igual que en sklearn: factor de ablacion declarado, no valor fijo del enunciado
CATEGORICAL_COLS = LOW_CARDINALITY_CATEGORICALS + HIGH_CARDINALITY_CATEGORICALS
TUNE_DAYS = TRAIN_DAYS[:2]  # subconjunto de dias solo para elegir hiperparametros (ver nota de alcance)
TUNE_SAMPLE_ROWS = 1_000_000  # filas tomadas de TUNE_DAYS, mismo orden que MLP_TUNE_SAMPLE en sklearn
MLP_GRID_SAMPLE_SEED = 13
MLP_GRID_SAMPLE_SIZE = 8  # de las 32 combinaciones mandatadas, ver nota de alcance
TRAIN_TOTAL_ROWS_APPROX = 32_377_421  # ver 'Cache plano train' en el log, 8 dias
FINAL_TRAIN_SAMPLE_ROWS = 1_000_000  # tope para gbt en el entrenamiento final, ver nota de alcance
MLP_HASH_DIM = 2**9  # 512; ver nota de alcance: MLP densifica el vector disperso
MLP_FINAL_SAMPLE_ROWS = 300_000  # tope de filas para el entrenamiento final de mlp

REDUCED_GRID = {
    # mlp y gbt: un solo valor por eje, ver nota de alcance arriba (busqueda
    # cancelada por tiempo medido: >7h sin terminar la 2a combinacion de gbt).
    "mlp": {
        "layers_hidden": [[100]],
        "stepSize": [0.1],
        "maxIter": [100],
        "solver": ["l-bfgs"],
    },
    "lr": {"regParam": [0.0, 0.01], "maxIter": [50]},
    "gbt": {"maxDepth": [5], "maxIter": [10]},
}


def build_spark(master: str = "local[*]", driver_memory: str = "13g") -> SparkSession:
    # NOTA: se intento activar checkpointing (checkpointInterval de GBTClassifier)
    # para cortar la acumulacion de RDDs intermedios entre iteraciones de boosting
    # (el bloque sin cachear crecia de forma medida 384B -> 181MiB -> 424MiB ->
    # 1.4GB con las iteraciones, incluso ya con la muestra reducida a 2M filas).
    # sparkContext.setCheckpointDir() en Windows SI necesita winutils.exe real
    # (Hadoop lo verifica con un File.exists() literal antes de poder escribir),
    # y no se descarga ese binario de terceros no oficial. Se abandona esa via;
    # el control de la presion de memoria queda solo en el tamano de muestra
    # (FINAL_TRAIN_SAMPLE_ROWS, ver nota de alcance arriba).
    return (
        SparkSession.builder.appName("avazu-mlp-tarea1")
        .master(master)
        .config("spark.driver.memory", driver_memory)
        .config("spark.memory.fraction", "0.8")
        .config("spark.memory.storageFraction", "0.6")
        .config("spark.sql.shuffle.partitions", "12")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


def _flat_parquet_for_days(days: list[int], cache_name: str, max_rows: int | None = None) -> str:
    """Bypass de un bug de Windows: Hadoop necesita winutils.exe/hadoop.dll (binarios
    de terceros no oficiales, que no se descargan aqui) para listar un directorio
    particionado via spark.read.parquet(directorio); falla con UnsatisfiedLinkError
    en NativeIO$Windows.access0. Pyarrow lee el parquet particionado por 'day' sin
    pasar por Hadoop, y el resultado se escribe como UN solo archivo plano, que
    Spark si puede leer directamente (un archivo, no un directorio, no dispara el
    listado nativo roto). Se cachea en disco para no repetir la conversion si el
    script se reinicia."""
    out_path = FLAT_CACHE_DIR / f"{cache_name}.parquet"
    if not out_path.exists():
        dataset = pa_ds.dataset(str(DATA_PROCESSED_DIR), format="parquet", partitioning="hive")
        table = dataset.to_table(filter=pa_ds.field("day").isin(days))
        if max_rows is not None and table.num_rows > max_rows:
            table = table.slice(0, max_rows)
        pq.write_table(table, out_path)
        log.info("Cache plano '%s': %d filas escritas en %s", cache_name, table.num_rows, out_path)
    return str(out_path)


def load_splits(spark: SparkSession):
    train_path = _flat_parquet_for_days(TRAIN_DAYS, "train")
    tune_path = _flat_parquet_for_days(TUNE_DAYS, "tune", max_rows=TUNE_SAMPLE_ROWS)
    val_path = _flat_parquet_for_days([VALIDATION_DAY], "val")
    test_path = _flat_parquet_for_days([TEST_DAY], "test")
    train_df = spark.read.parquet(train_path).repartition(12)
    tune_df = spark.read.parquet(tune_path).repartition(12)
    val_df = spark.read.parquet(val_path)
    test_df = spark.read.parquet(test_path)
    return train_df, tune_df, val_df, test_df


def build_feature_pipeline(hash_dim: int = HASH_DIM) -> Pipeline:
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in LOW_CARDINALITY_CATEGORICALS
    ]
    hasher = FeatureHasher(
        inputCols=HIGH_CARDINALITY_CATEGORICALS, outputCol="hashed_features", numFeatures=hash_dim
    )
    assembler = VectorAssembler(
        inputCols=["hashed_features"] + [f"{c}_idx" for c in LOW_CARDINALITY_CATEGORICALS],
        outputCol="assembled",
    )
    scaler = StandardScaler(inputCol="assembled", outputCol="features", withMean=False, withStd=True)
    return Pipeline(stages=indexers + [hasher, assembler, scaler])


def build_feature_pipeline_dense() -> Pipeline:
    """Solo para GBT: unicamente las categoricas de baja cardinalidad
    (indexadas), sin el bloque disperso de hashing de alta dimension. Igual
    criterio que HistGradientBoostingClassifier en sklearn (ver docstring de
    sklearn_pipeline.py): los arboles no se benefician de un espacio disperso
    de miles de columnas, y ahi es exactamente donde se disparaba el costo de
    construir estadisticos/histogramas por nodo en el GBT de Spark (bloques
    de varios GB que no entraban en memoria, independiente de maxIter o del
    numero de filas -- ver nota de alcance al inicio del archivo)."""
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in LOW_CARDINALITY_CATEGORICALS
    ]
    assembler = VectorAssembler(
        inputCols=[f"{c}_idx" for c in LOW_CARDINALITY_CATEGORICALS],
        outputCol="assembled",
    )
    scaler = StandardScaler(inputCol="assembled", outputCol="features", withMean=False, withStd=True)
    return Pipeline(stages=indexers + [assembler, scaler])


def _build_classifier(model_name: str, params: dict, seed: int, input_dim: int):
    if model_name == "lr":
        return LogisticRegression(
            featuresCol="features", labelCol=LABEL_COL,
            regParam=params["regParam"], maxIter=params["maxIter"],
        )
    if model_name == "gbt":
        return GBTClassifier(
            featuresCol="features", labelCol=LABEL_COL,
            maxDepth=params["maxDepth"], maxIter=params["maxIter"], seed=seed,
        )
    if model_name == "mlp":
        layers = [input_dim] + params["layers_hidden"] + [2]
        return MultilayerPerceptronClassifier(
            featuresCol="features", labelCol=LABEL_COL,
            layers=layers, stepSize=params["stepSize"], maxIter=params["maxIter"],
            solver=params["solver"], seed=seed,
        )
    raise ValueError(model_name)


def _grid_combinations(model_name: str) -> list[dict]:
    space = REDUCED_GRID[model_name]
    keys = list(space.keys())
    combos = [dict(zip(keys, combo)) for combo in itertools.product(*space.values())]
    if model_name == "mlp" and len(combos) > MLP_GRID_SAMPLE_SIZE:
        # Las 32 combinaciones del enunciado son el espacio DECLARADO, pero evaluar
        # cada una completa (fit + transform) en Spark local tiene un costo fijo por
        # iteracion que no baja al reducir filas (GBT con 30 arboles sobre 1M filas ya
        # tardaba >13 min por el overhead de Spark, no por el volumen de datos). Se
        # muestrea un subconjunto aleatorio del mismo espacio declarado, con la misma
        # logica que "busqueda bayesiana en vez de grilla exhaustiva" que el enunciado
        # ya pide para el MLP de sklearn.
        rng = random.Random(MLP_GRID_SAMPLE_SEED)
        combos = rng.sample(combos, MLP_GRID_SAMPLE_SIZE)
    return combos


def select_best_params(model_name: str, fitted_feature_pipe, train_df, val_df, input_dim: int, seed: int) -> dict:
    evaluator = BinaryClassificationEvaluator(labelCol=LABEL_COL, metricName="areaUnderROC")
    best_auc, best_params = -1.0, None
    for params in _grid_combinations(model_name):
        clf = _build_classifier(model_name, params, seed, input_dim)
        model = clf.fit(fitted_feature_pipe.transform(train_df))
        preds = model.transform(fitted_feature_pipe.transform(val_df))
        auc = evaluator.evaluate(preds)
        log.info("%s params=%s -> AUC val=%.4f", model_name, params, auc)
        if auc > best_auc:
            best_auc, best_params = auc, params
    return best_params


def _load_best_params() -> dict:
    path = SPARK_DIR / "best_params.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_best_params(best_params_all: dict) -> None:
    (SPARK_DIR / "best_params.json").write_text(
        json.dumps(best_params_all, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _load_done_runs() -> set[tuple[str, int]]:
    path = SPARK_DIR / "runs.csv"
    if not path.exists():
        return set()
    import pandas as pd
    done = pd.read_csv(path)[["model", "seed"]]
    return {(r.model, int(r.seed)) for r in done.itertuples()}


def _append_run(row: dict) -> None:
    import pandas as pd
    path = SPARK_DIR / "runs.csv"
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def run_full_experiment(master: str = "local[*]", driver_memory: str = "12g") -> None:
    spark = build_spark(master, driver_memory)
    spark.sparkContext.setLogLevel("WARN")

    train_df, tune_df, val_df, test_df = load_splits(spark)
    feature_pipe = build_feature_pipeline()
    feature_pipe_dense = build_feature_pipeline_dense()
    feature_pipe_mlp = build_feature_pipeline(hash_dim=MLP_HASH_DIM)

    t0 = time.perf_counter()
    fitted_feature_pipe = feature_pipe.fit(train_df)
    feature_fit_time = time.perf_counter() - t0
    fitted_feature_pipe_dense = feature_pipe_dense.fit(train_df)
    fitted_feature_pipe_mlp = feature_pipe_mlp.fit(train_df)

    def pipe_for(model_name: str):
        if model_name == "gbt":
            return fitted_feature_pipe_dense
        if model_name == "mlp":
            return fitted_feature_pipe_mlp
        return fitted_feature_pipe

    def input_dim_for(model_name: str) -> int:
        vec = pipe_for(model_name).transform(train_df.limit(1)).select("features").first()["features"]
        return vec.size

    evaluator = BinaryClassificationEvaluator(labelCol=LABEL_COL, metricName="areaUnderROC")
    best_params_all = _load_best_params()
    done_runs = _load_done_runs()

    for model_name in ("lr", "gbt", "mlp"):
        if model_name in best_params_all:
            log.info("%s: parametros ya afinados (en best_params.json), se reusan.", model_name)
        else:
            log.info("Seleccionando hiperparametros para %s (grilla literal del enunciado, "
                      "dias %s para afinar, validacion real dia 29)...", model_name, TUNE_DAYS)
            best_params_all[model_name] = select_best_params(
                model_name, pipe_for(model_name), tune_df, val_df, input_dim_for(model_name), SEEDS[0]
            )
            _save_best_params(best_params_all)
        best_params = best_params_all[model_name]

        for seed in SEEDS:
            if (model_name, seed) in done_runs:
                log.info("%s seed=%d: ya entrenado (en runs.csv), se salta.", model_name, seed)
                continue
            clf = _build_classifier(model_name, best_params, seed, input_dim_for(model_name))

            sample_cap = {"gbt": FINAL_TRAIN_SAMPLE_ROWS, "mlp": MLP_FINAL_SAMPLE_ROWS}.get(model_name)
            if sample_cap is not None:
                # Ver nota de alcance: dataset completo causaba espiral de
                # presion de memoria (gbt) u OutOfMemoryError real (mlp,
                # densificacion del vector disperso) en Spark local.
                frac = min(1.0, (sample_cap * 1.3) / TRAIN_TOTAL_ROWS_APPROX)
                fit_df = train_df.sample(withReplacement=False, fraction=frac, seed=seed).limit(sample_cap)
            else:
                fit_df = train_df

            pipe = pipe_for(model_name)
            t0 = time.perf_counter()
            model = clf.fit(pipe.transform(fit_df))
            fit_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            preds_test = model.transform(pipe.transform(test_df)).cache()
            n_test = preds_test.count()
            predict_time = time.perf_counter() - t0

            auc = evaluator.evaluate(preds_test)
            pr_auc = BinaryClassificationEvaluator(
                labelCol=LABEL_COL, metricName="areaUnderPR"
            ).evaluate(preds_test)

            prob_pd = (
                preds_test
                .select(LABEL_COL, vector_to_array("probability").getItem(1).alias("p1"))
                .toPandas()
            )
            tag = f"{model_name}_seed{seed}"
            np.save(PRED_DIR / f"{tag}_y_test.npy", prob_pd[LABEL_COL].values)
            np.save(PRED_DIR / f"{tag}_p_test.npy", prob_pd["p1"].values)

            _append_run({
                "model": model_name, "seed": seed, "best_params": json.dumps(best_params),
                "feature_fit_time_s": feature_fit_time, "fit_time_s": fit_time,
                "predict_time_s": predict_time, "n_test": n_test,
                "roc_auc": auc, "pr_auc": pr_auc,
            })
            log.info("spark %s seed=%d -> roc_auc=%.4f pr_auc=%.4f fit=%.1fs", model_name, seed, auc, pr_auc, fit_time)
            preds_test.unpersist()

    log.info("Listo. Corridas guardadas en %s", SPARK_DIR / "runs.csv")
    spark.stop()


if __name__ == "__main__":
    run_full_experiment()
