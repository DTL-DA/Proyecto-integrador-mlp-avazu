"""
Descarga del dataset Avazu CTR (Kaggle) y conversion a Parquet particionado por dia.

Uso:
    python src/data_prep.py

Etapas:
    1. Descarga train.gz via API de Kaggle (requiere ~/.kaggle/kaggle.json).
    2. Descomprime a data/raw/train.csv (streaming, sin cargar todo en RAM).
    3. Valida el conteo de filas esperado.
    4. Convierte a Parquet particionado por dia con DuckDB (fuera de memoria),
       agregando variables temporales derivadas de `hour` (dia, mes, hora, dia de
       semana, franja horaria).
    5. Escribe la ficha de datos en artifacts/eda/data_sheet.json.
"""
from __future__ import annotations

import gzip
import json
import logging
import shutil
import sys
import time
from datetime import date

import duckdb

from config import (
    ARTIFACTS_DIR,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    EXPECTED_ROW_COUNT,
    KAGGLE_COMPETITION,
    RAW_CSV_PATH,
    RAW_GZ_PATH,
    TEST_DAY,
    TRAIN_DAYS,
    VALIDATION_DAY,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("data_prep")

DRIVER_MEMORY_LIMIT = "10GB"  # acotado a los 16 GB de RAM reales de la maquina


def download_avazu() -> None:
    if RAW_GZ_PATH.exists() or RAW_CSV_PATH.exists():
        log.info("train.gz/train.csv ya existen en data/raw, se omite la descarga.")
        return

    log.info("Autenticando con la API de Kaggle...")
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    log.info("Descargando train.gz de la competencia %s...", KAGGLE_COMPETITION)
    api.competition_download_file(
        KAGGLE_COMPETITION, "train.gz", path=str(DATA_RAW_DIR)
    )
    log.info("Descarga completa: %s", RAW_GZ_PATH)


def decompress_gz() -> None:
    if RAW_CSV_PATH.exists():
        log.info("train.csv ya existe, se omite la descompresion.")
        return

    if not RAW_GZ_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro {RAW_GZ_PATH}. Revisa la descarga de Kaggle."
        )

    log.info("Descomprimiendo %s -> %s (streaming)...", RAW_GZ_PATH, RAW_CSV_PATH)
    t0 = time.time()
    with gzip.open(RAW_GZ_PATH, "rb") as f_in, open(RAW_CSV_PATH, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out, length=64 * 1024 * 1024)
    log.info("Descompresion completa en %.1f s.", time.time() - t0)


def validate_raw_row_count(con: duckdb.DuckDBPyConnection) -> int:
    log.info("Validando conteo de filas de train.csv (streaming, sin cargar en RAM)...")
    (n_rows,) = con.execute(
        f"SELECT COUNT(*) FROM read_csv_auto('{RAW_CSV_PATH.as_posix()}', header=True)"
    ).fetchone()
    log.info("Filas encontradas: %d (esperado: %d)", n_rows, EXPECTED_ROW_COUNT)
    if n_rows != EXPECTED_ROW_COUNT:
        log.warning(
            "El conteo de filas no coincide con el esperado. Se continua, pero "
            "esto debe quedar documentado como limitacion en la ficha de datos."
        )
    return n_rows


def _weekday_name(yy: int, mm: int, dd: int) -> str:
    return date(2000 + yy, mm, dd).strftime("%A")


def build_partitioned_parquet(con: duckdb.DuckDBPyConnection) -> None:
    if any(DATA_PROCESSED_DIR.glob("day=*")):
        log.info("Ya existe parquet particionado en data/processed, se omite la conversion.")
        return

    log.info("Convirtiendo CSV -> Parquet particionado por dia (DuckDB, fuera de memoria)...")
    t0 = time.time()
    con.execute(
        f"""
        COPY (
            SELECT
                *,
                (CAST(hour AS BIGINT) // 1000000)          AS yy,
                (CAST(hour AS BIGINT) // 10000) % 100       AS month,
                (CAST(hour AS BIGINT) // 100) % 100         AS day,
                CAST(hour AS BIGINT) % 100                  AS hour_of_day,
                CASE
                    WHEN CAST(hour AS BIGINT) % 100 BETWEEN 0 AND 5   THEN 'madrugada'
                    WHEN CAST(hour AS BIGINT) % 100 BETWEEN 6 AND 11  THEN 'manana'
                    WHEN CAST(hour AS BIGINT) % 100 BETWEEN 12 AND 17 THEN 'tarde'
                    ELSE 'noche'
                END AS franja_horaria
            FROM read_csv_auto('{RAW_CSV_PATH.as_posix()}', header=True)
        ) TO '{DATA_PROCESSED_DIR.as_posix()}' (FORMAT PARQUET, PARTITION_BY (day), OVERWRITE_OR_IGNORE 1)
        """
    )
    log.info("Conversion completa en %.1f s.", time.time() - t0)


def build_data_sheet(con: duckdb.DuckDBPyConnection, n_rows: int) -> dict:
    log.info("Calculando tasa base de click y tamanos para la ficha de datos...")
    parquet_glob = (DATA_PROCESSED_DIR / "day=*" / "*.parquet").as_posix()

    (base_click_rate,) = con.execute(
        f"SELECT AVG(click) FROM read_parquet('{parquet_glob}')"
    ).fetchone()

    days_present = [
        row[0]
        for row in con.execute(
            f"SELECT DISTINCT day FROM read_parquet('{parquet_glob}') ORDER BY day"
        ).fetchall()
    ]

    raw_csv_size_gb = RAW_CSV_PATH.stat().st_size / (1024**3) if RAW_CSV_PATH.exists() else None
    parquet_size_gb = sum(
        p.stat().st_size for p in DATA_PROCESSED_DIR.rglob("*.parquet")
    ) / (1024**3)

    sheet = {
        "dataset": "Avazu CTR Prediction",
        "fuente": "https://www.kaggle.com/competitions/avazu-ctr-prediction",
        "licencia": "Terminos de la competencia de Kaggle (uso academico/no comercial)",
        "periodo": "21 a 30 de octubre de 2014 (10 dias, codificados yyMMddHH)",
        "particion_temporal_declarada": {
            "train_days": TRAIN_DAYS,
            "validation_day": VALIDATION_DAY,
            "test_day": TEST_DAY,
        },
        "dias_presentes_en_parquet": days_present,
        "n_filas": n_rows,
        "tasa_base_click": base_click_rate,
        "tamano_csv_crudo_gb": raw_csv_size_gb,
        "tamano_parquet_gb": parquet_size_gb,
        "hardware_usado": {
            "ram_gb": 16,
            "nucleos_logicos": 12,
            "spark_modo": "local[*] (sin cluster real)",
        },
        "limitaciones": [
            "El enunciado asume 32-64 GB de RAM; esta maquina tiene 16 GB, por lo "
            "que las muestras de scikit-learn y las grillas de PySpark se redujeron "
            "de forma declarada (ver sklearn_pipeline.py y spark_pipeline.py).",
            "No hay cluster Spark real: el benchmarking de escalamiento se simula "
            "variando el numero de nucleos locales (local[1,2,4,8,12]).",
            "device_id y device_ip tienen cardinalidad muy alta; se usan tecnicas "
            "de hashing/frequency encoding en vez de one-hot encoding directo.",
        ],
    }

    out_path = ARTIFACTS_DIR / "eda" / "data_sheet.json"
    out_path.write_text(json.dumps(sheet, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log.info("Ficha de datos escrita en %s", out_path)
    return sheet


def main() -> None:
    download_avazu()
    decompress_gz()

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{DRIVER_MEMORY_LIMIT}'")
    con.execute("SET threads TO 8")

    n_rows = validate_raw_row_count(con)
    build_partitioned_parquet(con)
    sheet = build_data_sheet(con, n_rows)

    log.info("Listo. Tasa base de click: %.4f", sheet["tasa_base_click"])


if __name__ == "__main__":
    sys.exit(main())
