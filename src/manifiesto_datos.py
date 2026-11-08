"""
Manifiesto de datos: por cada particion diaria del Parquet procesado, el numero
de filas, el tamano, la fecha de ultima modificacion y el hash SHA-256 del
archivo. El hash del dia 30 (prueba) permite verificar que el conjunto de prueba
no cambio desde su creacion; la fecha de modificacion muestra que es anterior a
cualquier entrenamiento.

Uso: python manifiesto_datos.py  ->  artifacts/eda/manifiesto_datos.json
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json

import pyarrow.parquet as pq

from config import ARTIFACTS_DIR, DATA_PROCESSED_DIR, TEST_DAY, TRAIN_DAYS, VALIDATION_DAY


def sha256(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    rows = []
    for day_dir in sorted(DATA_PROCESSED_DIR.glob("day=*")):
        day = int(day_dir.name.split("=")[1])
        for f in sorted(day_dir.glob("*.parquet")):
            st = f.stat()
            rows.append({
                "dia": day,
                "rol": "train" if day in TRAIN_DAYS else "validacion" if day == VALIDATION_DAY else "prueba" if day == TEST_DAY else "otro",
                "archivo": f.relative_to(DATA_PROCESSED_DIR.parent).as_posix(),
                "filas": pq.ParquetFile(f).metadata.num_rows,
                "bytes": st.st_size,
                "modificado": dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                "sha256": sha256(f),
            })
    out = {"generado": dt.datetime.now().isoformat(timespec="seconds"),
           "total_filas": sum(r["filas"] for r in rows), "particiones": rows}
    path = ARTIFACTS_DIR / "eda" / "manifiesto_datos.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(rows)} archivos, {out['total_filas']:,} filas -> {path.name}")


if __name__ == "__main__":
    main()
