"""
Curva de aprendizaje de los modelos de Spark: AUC en prueba segun el numero de
filas de entrenamiento, para LogisticRegression, GBTClassifier y
MultilayerPerceptronClassifier, con la MISMA configuracion del experimento
principal (spark_pipeline.py): encoders ajustados con todo el train, mismos
hiperparametros (artifacts/spark/best_params.json), mismo muestreo por semilla
y evaluacion sobre el dia 30 completo.

Diseno: tamanos de 30.000, 100.000 y 300.000 filas con la semilla 13. El MLP de
300.000 filas no se vuelve a entrenar: es exactamente la corrida de la semilla 13
del experimento principal y se toma de artifacts/stats. Tambien se agregan como
puntos de referencia la regresion logistica con el train completo y el GBT con
1.000.000 de filas del experimento principal. Para el MLP se guarda el historial
de la funcion objetivo de L-BFGS por iteracion.

Salidas: artifacts/spark/curva_aprendizaje.csv y
artifacts/spark/mlp_historial_objetivo.json
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
from pyspark.ml.functions import vector_to_array
from sklearn.metrics import log_loss, roc_auc_score

from config import ARTIFACTS_DIR, LABEL_COL, SEEDS, TRAIN_DAYS
from spark_pipeline import (
    MLP_HASH_DIM, TRAIN_TOTAL_ROWS_APPROX, _build_classifier, _flat_parquet_for_days, build_feature_pipeline,
    build_feature_pipeline_dense, build_spark, load_splits,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("spark_learning_curve")

SPARK_DIR = ARTIFACTS_DIR / "spark"
SIZES = [30_000, 100_000, 300_000]
SEED = SEEDS[0]


def main() -> None:
    best = json.loads((SPARK_DIR / "best_params.json").read_text(encoding="utf-8"))
    spark = build_spark("local[*]", "12g")
    spark.sparkContext.setLogLevel("WARN")
    train_df, _, _, test_df = load_splits(spark)
    # Los encoders se ajustan sobre la lectura directa del train, sin el
    # repartition(12) de load_splits: los conteos de StringIndexer y las
    # estadisticas del escalador no dependen de como se reparten las filas, y se
    # evita redistribuir 32 millones de filas tres veces.
    train_plain = spark.read.parquet(_flat_parquet_for_days(TRAIN_DAYS, "train"))

    log.info("Ajustando los encoders con todo el train (igual que el experimento principal)...")
    pipes = {
        "lr": build_feature_pipeline().fit(train_plain),
        "gbt": build_feature_pipeline_dense().fit(train_plain),
        "mlp": build_feature_pipeline(hash_dim=MLP_HASH_DIM).fit(train_plain),
    }
    dims = {m: p.transform(train_plain.limit(1)).select("features").first()["features"].size for m, p in pipes.items()}
    log.info("Encoders listos; dimensiones de entrada: %s", dims)

    rows, history = [], {}
    for n in SIZES:
        # Misma muestra que el experimento principal (mismo muestreo y semilla),
        # tomada una vez por tamano y compartida por los tres modelos. Se
        # materializa en cache: sin cache, cada iteracion del optimizador vuelve a
        # leer y redistribuir los 32 millones de filas para tomar la misma muestra.
        frac = min(1.0, (n * 1.3) / TRAIN_TOTAL_ROWS_APPROX)
        fit_df = train_df.sample(withReplacement=False, fraction=frac, seed=SEED).limit(n).cache()
        fit_df.count()
        for model_name in ("lr", "gbt", "mlp"):
            if model_name == "mlp" and n == 300_000:
                continue  # es la corrida de la semilla 13 del experimento principal
            feats = pipes[model_name].transform(fit_df).cache()
            feats.count()
            clf = _build_classifier(model_name, best[model_name], SEED, dims[model_name])
            t0 = time.perf_counter()
            model = clf.fit(feats)
            fit_s = time.perf_counter() - t0
            feats.unpersist()
            pred = (model.transform(pipes[model_name].transform(test_df))
                    .select(LABEL_COL, vector_to_array("probability").getItem(1).alias("p1")).toPandas())
            y, p = pred[LABEL_COL].values, np.clip(pred["p1"].values, 1e-15, 1 - 1e-15)
            row = {"modelo": model_name, "filas_entrenamiento": n, "semilla": SEED, "roc_auc": roc_auc_score(y, p),
                   "log_loss": log_loss(y, p), "tiempo_ajuste_s": fit_s, "fuente": "curva de aprendizaje"}
            if model_name == "mlp":
                summ = model.summary() if callable(getattr(model, "summary", None)) else model.summary
                history[str(n)] = [float(v) for v in summ.objectiveHistory]
                row["iteraciones"] = len(history[str(n)])
            rows.append(row)
            log.info("%s n=%d -> AUC=%.4f log loss=%.4f ajuste=%.1fs", model_name, n, row["roc_auc"], row["log_loss"], fit_s)
        fit_df.unpersist()

    # Puntos del experimento principal (semilla 13), con las mismas metricas exactas.
    ref = pd.read_csv(ARTIFACTS_DIR / "stats" / "spark_metricas_por_corrida.csv").query("seed == @SEED")
    for model_name, n in (("mlp", 300_000), ("gbt", 1_000_000), ("lr", 32_377_421)):
        r = ref[ref["model"] == model_name].iloc[0]
        rows.append({"modelo": model_name, "filas_entrenamiento": n, "semilla": SEED, "roc_auc": r["roc_auc"],
                     "log_loss": r["log_loss"], "tiempo_ajuste_s": r["fit_time_s"], "fuente": "experimento principal"})

    pd.DataFrame(rows).sort_values(["modelo", "filas_entrenamiento"]).to_csv(SPARK_DIR / "curva_aprendizaje.csv", index=False)
    (SPARK_DIR / "mlp_historial_objetivo.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    log.info("Curva de aprendizaje de Spark lista")
    spark.stop()


if __name__ == "__main__":
    main()
