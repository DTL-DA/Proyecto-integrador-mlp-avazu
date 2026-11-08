"""Rutas y constantes compartidas por todos los scripts del proyecto."""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# PySpark en Windows necesita el worker Python explicito y un host/bind local
# fijo (127.0.0.1); sin esto, el driver pierde la conexion por socket con el
# executor en modo local (visto empiricamente: "java.net.SocketException:
# Connection reset" al hacer fit() de cualquier pipeline, no solo con datos reales).
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
# Java 17: se respeta JAVA_HOME si ya esta definido; si no, se prueba la ruta
# ~/Java/jdk-17 del usuario actual (sin rutas personales fijas en el codigo).
_JAVA_FALLBACK = Path.home() / "Java" / "jdk-17"
if "JAVA_HOME" not in os.environ and _JAVA_FALLBACK.exists():
    os.environ["JAVA_HOME"] = str(_JAVA_FALLBACK)

# Hadoop (empaquetado con PySpark) revienta en Windows con
# "FileNotFoundException: HADOOP_HOME and hadoop.home.dir are unset" apenas
# HADOOP_HOME no esta definido, incluso para operaciones que no necesitan
# winutils.exe de verdad (como crear el directorio temporal de SparkSubmit o
# fijar un checkpoint dir). No se descarga winutils.exe/hadoop.dll (binarios
# de terceros no oficiales); basta con apuntar HADOOP_HOME a un directorio
# cualquiera que exista para que esa comprobacion puntual no truene. Las
# operaciones que SI necesitan el binario nativo (listar un directorio
# particionado) se evitan en spark_pipeline.py leyendo con pyarrow en vez de
# Hadoop.
_HADOOP_HOME_FALLBACK = PROJECT_ROOT / "artifacts" / "spark" / "_hadoop_home_stub"
(_HADOOP_HOME_FALLBACK / "bin").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HADOOP_HOME", str(_HADOOP_HOME_FALLBACK))
os.environ.setdefault("hadoop.home.dir", str(_HADOOP_HOME_FALLBACK))

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RAW_CSV_PATH = DATA_RAW_DIR / "train.csv"
RAW_GZ_PATH = DATA_RAW_DIR / "train.gz"

KAGGLE_COMPETITION = "avazu-ctr-prediction"
EXPECTED_ROW_COUNT = 40_428_967

# Partición temporal declarada antes de cualquier ajuste de encoders/escaladores.
# Días del dataset Avazu: 21 a 30 de octubre de 2014 (codificados como yy=14, mm=10).
TRAIN_DAYS = list(range(21, 29))  # 21-28 inclusive
VALIDATION_DAY = 29
TEST_DAY = 30

SEEDS = [13, 29, 47, 71, 97]  # minimo 5 semillas, fijas y documentadas

RANDOM_STATE_DEFAULT = SEEDS[0]

HASHING_DIMENSIONS = [2**12, 2**14, 2**16]

LOW_CARDINALITY_CATEGORICALS = [
    "banner_pos",
    "site_category",
    "app_category",
    "device_type",
    "device_conn_type",
]

HIGH_CARDINALITY_CATEGORICALS = [
    "site_id",
    "site_domain",
    "app_id",
    "app_domain",
    "device_id",
    "device_ip",
    "device_model",
]

NUMERIC_LIKE_CATEGORICALS = [f"C{i}" for i in (1, 14, 15, 16, 17, 18, 19, 20, 21)]

LABEL_COL = "click"

for _d in (DATA_RAW_DIR, DATA_PROCESSED_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
