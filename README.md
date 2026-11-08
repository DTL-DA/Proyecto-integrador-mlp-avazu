# Proyecto Integrador 2.12 — MLP para predecir clics en anuncios móviles

Clasificador MLP de clic / no clic en anuncios móviles sobre el dataset
**Avazu CTR Prediction**, con una comparación entre scikit-learn y PySpark en
modo local, interpretabilidad con LIME contrastada con SHAP y la importancia por
permutación, y evaluación estadística con 5 semillas. El resultado se entrega
como un Jupyter Book ejecutable.

**Libro publicado:** https://dtl-da.github.io/Proyecto-integrador-mlp-avazu/

## Estructura

```
src/          scripts que ejecutan los experimentos y guardan resultados
artifacts/    resultados de cada script (métricas, predicciones, tablas, logs)
book/         capítulos del Jupyter Book (solo leen artifacts/, no reentrenan)
_config.yml   configuración del libro
_toc.yml      tabla de contenido
environment.yml
```

`data/` (CSV crudo y Parquet por día) no se versiona; se regenera con `data_prep.py`.

## Entorno

```
conda env create -f environment.yml
conda activate dl_tarea1
python -m ipykernel install --user --name dl_tarea1 --display-name "Python (dl_tarea1)"
```

- **Java 17** (se usó Eclipse Temurin 17.0.20.1). `src/config.py` respeta la
  variable `JAVA_HOME` si está definida; si no, busca `~/Java/jdk-17`.
- **Kaggle**: token de la API en `~/.kaggle/access_token` (nunca dentro del
  repositorio) y reglas de la competencia `avazu-ctr-prediction` aceptadas en
  kaggle.com.
- **Windows**: `config.py` fija el intérprete de los workers de PySpark, el host
  `127.0.0.1` y un `HADOOP_HOME` vacío dentro de `artifacts/`; no hace falta
  `winutils.exe`.

## Orden de ejecución (de punta a punta)

```
cd src
python data_prep.py                      # descarga Avazu y escribe Parquet por día
python manifiesto_datos.py               # filas, tamaño y SHA-256 de cada partición
python eda.py                            # EDA -> artifacts/eda/
python sklearn_pipeline.py               # SGD, HistGB y MLP, 3 tamaños x 5 semillas -> artifacts/sklearn/
python spark_pipeline.py                 # LR, GBT y MLP de Spark x 5 semillas -> artifacts/spark/
python spark_learning_curve.py           # curva de aprendizaje de Spark (30k, 100k y 300k filas)
python stats_tests.py                    # DeLong, McNemar, TOST, agregados -> artifacts/stats/
python benchmarking.py                   # escalamiento fuerte y débil, Amdahl, punto de cruce
python stage_benchmark.py                # tiempos por etapa, memoria pico y Spark UI (máquina sin otra carga)
python interpretability.py               # LIME, SHAP e importancia por permutación
python interpretability.py --sensibilidad    # ancho del kernel x muestras perturbadas
python interpretability.py --discretizacion  # diagnóstico de la discretización de LIME
python ablation.py                       # ablaciones del MLP -> artifacts/ablation/
```

Los scripts de modelado guardan cada corrida al terminarla y retoman desde la
última corrida completa si se interrumpen.

## Construir el libro

Desde la raíz del proyecto:

```
jupyter-book build .
```

El HTML queda en `_build/html/index.html`.

## Reproducibilidad

- Semillas: 13, 29, 47, 71 y 97 (`SEEDS` en `src/config.py`).
- Partición temporal fija: train días 21 a 28, validación día 29, prueba día 30.
- Todas las transformaciones se ajustan solo con los días de train; el target
  encoding de train es fuera de pliegue.
- El manifiesto `artifacts/eda/manifiesto_datos.json` registra el hash SHA-256
  de cada partición, incluida la de prueba.
- Hardware usado: Intel Core i5-1235U (12 hilos), 16 GB de RAM, Windows 11, sin
  GPU y sin clúster de Spark. Los detalles, la configuración de Spark y el
  registro de desviaciones están en el capítulo 11 del libro.
