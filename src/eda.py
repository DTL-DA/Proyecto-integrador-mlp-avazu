"""
Analisis exploratorio de datos (EDA) sobre el parquet particionado de Avazu.

Todo lo pesado (conteos, agregaciones) se hace en DuckDB directamente sobre el
parquet particionado, sin materializar el dataset completo en memoria. Las
medidas de asociacion y el PSI usan una muestra aleatoria explicita (documentada),
porque mutual information / Cramer's V no estan disponibles como agregacion SQL
nativa escalable.

Salidas: artifacts/eda/*.csv (tablas) y artifacts/eda/*.png (graficas en
escala de grises).
"""
from __future__ import annotations

import json
import logging

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OrdinalEncoder

from config import (
    ARTIFACTS_DIR,
    DATA_PROCESSED_DIR,
    HIGH_CARDINALITY_CATEGORICALS,
    LOW_CARDINALITY_CATEGORICALS,
    NUMERIC_LIKE_CATEGORICALS,
    TEST_DAY,
    TRAIN_DAYS,
)
from plotting import finalize_and_save, new_figure, series_styles

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eda")

EDA_DIR = ARTIFACTS_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_GLOB = (DATA_PROCESSED_DIR / "day=*" / "*.parquet").as_posix()
SAMPLE_SIZE = 2_000_000  # para asociacion/PSI; documentado por limite de RAM (16 GB)


def _con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET memory_limit='10GB'")
    con.execute("SET threads TO 8")
    return con


def click_overall_and_by_time(con: duckdb.DuckDBPyConnection) -> None:
    log.info("Distribucion de click (global, por dia, por hora, por franja)...")

    overall = con.execute(
        f"SELECT click, COUNT(*) AS n FROM read_parquet('{PARQUET_GLOB}') GROUP BY click ORDER BY click"
    ).df()
    overall.to_csv(EDA_DIR / "click_overall.csv", index=False)

    fig, ax = new_figure((4.5, 4.2))
    styles = series_styles(1)
    ax.bar(overall["click"].astype(str), overall["n"], color=styles[0]["color"])
    ax.set_xlabel("click")
    ax.set_ylabel("numero de registros")
    finalize_and_save(fig, ax, EDA_DIR / "click_overall.png", legend=False,
                       title="Distribucion global de click (0 = no, 1 = si)")

    by_day = con.execute(
        f"SELECT day, AVG(click) AS ctr, COUNT(*) AS n FROM read_parquet('{PARQUET_GLOB}') "
        "GROUP BY day ORDER BY day"
    ).df()
    by_day.to_csv(EDA_DIR / "ctr_by_day.csv", index=False)

    by_hour = con.execute(
        f"SELECT hour_of_day, AVG(click) AS ctr, COUNT(*) AS n FROM read_parquet('{PARQUET_GLOB}') "
        "GROUP BY hour_of_day ORDER BY hour_of_day"
    ).df()
    by_hour.to_csv(EDA_DIR / "ctr_by_hour.csv", index=False)

    fig, ax = new_figure()
    style = series_styles(1)[0]
    ax.plot(by_hour["hour_of_day"], by_hour["ctr"], color=style["color"],
            linestyle=style["linestyle"], marker=style["marker"], label="CTR observado")
    ax.set_xlabel("hora del dia (0-23)")
    ax.set_ylabel("tasa de click (CTR)")
    finalize_and_save(fig, ax, EDA_DIR / "ctr_by_hour.png", title="CTR por hora del dia")

    by_slot = con.execute(
        f"SELECT franja_horaria, AVG(click) AS ctr, COUNT(*) AS n FROM read_parquet('{PARQUET_GLOB}') "
        "GROUP BY franja_horaria ORDER BY ctr DESC"
    ).df()
    by_slot.to_csv(EDA_DIR / "ctr_by_franja.csv", index=False)


def categorical_frequencies(con: duckdb.DuckDBPyConnection) -> None:
    log.info("Frecuencias de categoricas de baja cardinalidad...")
    for col in LOW_CARDINALITY_CATEGORICALS:
        freq = con.execute(
            f"SELECT {col}, COUNT(*) AS n, AVG(click) AS ctr FROM read_parquet('{PARQUET_GLOB}') "
            f"GROUP BY {col} ORDER BY n DESC LIMIT 20"
        ).df()
        freq.to_csv(EDA_DIR / f"freq_{col}.csv", index=False)

        fig, ax = new_figure((6.5, 4.2))
        style = series_styles(1)[0]
        ax.bar(freq[col].astype(str), freq["n"], color=style["color"])
        ax.set_xlabel(col)
        ax.set_ylabel("numero de registros")
        ax.tick_params(axis="x", rotation=45)
        finalize_and_save(fig, ax, EDA_DIR / f"freq_{col}.png", legend=False,
                           title=f"Frecuencia de categorias: {col}")


def cardinality_long_tail(con: duckdb.DuckDBPyConnection) -> None:
    log.info("Cardinalidad y cola larga de categoricas de alta cardinalidad...")
    summary_rows = []
    curves = {}
    for col in HIGH_CARDINALITY_CATEGORICALS:
        counts = con.execute(
            f"SELECT {col}, COUNT(*) AS n FROM read_parquet('{PARQUET_GLOB}') "
            f"GROUP BY {col} ORDER BY n DESC"
        ).df()
        total = counts["n"].sum()
        n_categories = len(counts)
        n_singletons = int((counts["n"] == 1).sum())
        cum_frac = counts["n"].cumsum() / total

        summary_rows.append(
            {
                "columna": col,
                "n_categorias": n_categories,
                "pct_singletons": n_singletons / n_categories,
                "categorias_para_80pct_masa": int((cum_frac <= 0.80).sum() + 1),
            }
        )
        curves[col] = cum_frac.values[: min(len(cum_frac), 5000)]

    pd.DataFrame(summary_rows).to_csv(EDA_DIR / "cardinalidad_resumen.csv", index=False)

    # 7 variables en una sola grafica de grises serian indistinguibles entre si
    # (ver anti-patrones de la skill dataviz); se separan en dos paneles (small
    # multiples) de hasta 4 series cada uno.
    cols = list(curves.keys())
    groups = [cols[:4], cols[4:]]
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, group in zip(axes, groups):
        styles = series_styles(len(group))
        for style, col in zip(styles, group):
            curve = curves[col]
            x = np.linspace(0, 1, len(curve))
            ax.plot(x, curve, label=col, color=style["color"],
                    linestyle=style["linestyle"], marker=style["marker"], markevery=0.1, markersize=4)
        ax.grid(True, color="#d9d9d9", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_xlabel("fraccion de categorias (ordenadas por frecuencia desc.)")
        ax.set_ylabel("fraccion acumulada de registros")
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Curva de cola larga por variable")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "cola_larga.png", dpi=150)
    plt.close(fig)


def association_measures(con: duckdb.DuckDBPyConnection) -> None:
    log.info("Midiendo asociacion (informacion mutua, V de Cramer, Spearman) sobre muestra de %d filas...",
              SAMPLE_SIZE)
    sample = con.execute(
        f"SELECT * FROM read_parquet('{PARQUET_GLOB}') WHERE day IN {tuple(TRAIN_DAYS)} "
        f"USING SAMPLE {SAMPLE_SIZE} ROWS"
    ).df()

    cat_cols = LOW_CARDINALITY_CATEGORICALS + HIGH_CARDINALITY_CATEGORICALS
    cat_cols = [c for c in cat_cols if c in sample.columns]

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_cat = enc.fit_transform(sample[cat_cols].astype(str))
    mi = mutual_info_classif(X_cat, sample["click"], discrete_features=True, random_state=13)
    mi_df = pd.DataFrame({"columna": cat_cols, "informacion_mutua": mi}).sort_values(
        "informacion_mutua", ascending=False
    )
    mi_df.to_csv(EDA_DIR / "informacion_mutua.csv", index=False)

    fig, ax = new_figure((6.5, 4.5))
    style = series_styles(1)[0]
    ax.barh(mi_df["columna"], mi_df["informacion_mutua"], color=style["color"])
    ax.invert_yaxis()
    ax.set_xlabel("informacion mutua con click")
    finalize_and_save(fig, ax, EDA_DIR / "informacion_mutua.png", legend=False,
                       title="Informacion mutua de cada variable categorica con click")

    cramers_v_rows = []
    low_card_present = [c for c in LOW_CARDINALITY_CATEGORICALS if c in sample.columns]
    for i, col_a in enumerate(low_card_present):
        for col_b in low_card_present[i + 1 :]:
            table = pd.crosstab(sample[col_a], sample[col_b])
            chi2, _, _, _ = chi2_contingency(table)
            n = table.sum().sum()
            r, k = table.shape
            v = np.sqrt((chi2 / n) / (min(r - 1, k - 1) or 1))
            cramers_v_rows.append({"var_a": col_a, "var_b": col_b, "cramers_v": v})
    pd.DataFrame(cramers_v_rows).to_csv(EDA_DIR / "cramers_v.csv", index=False)

    numeric_like = [c for c in NUMERIC_LIKE_CATEGORICALS if c in sample.columns]
    spearman_rows = []
    for col in numeric_like:
        rho, p = spearmanr(sample[col].astype(float), sample["click"])
        spearman_rows.append({"columna": col, "spearman_rho": rho, "p_valor": p})
    pd.DataFrame(spearman_rows).to_csv(EDA_DIR / "spearman_numeric_like.csv", index=False)

    corr = sample[numeric_like].astype(float).corr(method="spearman")
    fig, ax = new_figure((6, 5.5))
    im = ax.imshow(corr.values, cmap="Greys", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric_like)), numeric_like, rotation=45, ha="right")
    ax.set_yticks(range(len(numeric_like)), numeric_like)
    fig.colorbar(im, ax=ax, label="correlacion de Spearman")
    finalize_and_save(fig, ax, EDA_DIR / "correlacion_numeric_like.png", legend=False,
                       title="Correlacion de Spearman entre variables C* (codigos categoricos)")


def _psi(train_pct: pd.Series, test_pct: pd.Series) -> float:
    eps = 1e-6
    train_pct = train_pct.clip(lower=eps)
    test_pct = test_pct.clip(lower=eps)
    return float(((train_pct - test_pct) * np.log(train_pct / test_pct)).sum())


def population_stability_index(con: duckdb.DuckDBPyConnection) -> None:
    log.info("Calculando PSI (train vs. prueba) por variable...")
    cols = LOW_CARDINALITY_CATEGORICALS + ["hour_of_day", "franja_horaria"]
    rows = []
    for col in cols:
        train_dist = con.execute(
            f"SELECT {col} AS cat, COUNT(*)::DOUBLE / SUM(COUNT(*)) OVER () AS pct "
            f"FROM read_parquet('{PARQUET_GLOB}') WHERE day IN {tuple(TRAIN_DAYS)} GROUP BY {col}"
        ).df().set_index("cat")["pct"]
        test_dist = con.execute(
            f"SELECT {col} AS cat, COUNT(*)::DOUBLE / SUM(COUNT(*)) OVER () AS pct "
            f"FROM read_parquet('{PARQUET_GLOB}') WHERE day = {TEST_DAY} GROUP BY {col}"
        ).df().set_index("cat")["pct"]
        joined = pd.concat([train_dist, test_dist], axis=1, keys=["train", "test"]).fillna(0.0)
        rows.append({"columna": col, "psi": _psi(joined["train"], joined["test"])})

    psi_df = pd.DataFrame(rows).sort_values("psi", ascending=False)
    psi_df.to_csv(EDA_DIR / "psi_train_vs_test.csv", index=False)

    fig, ax = new_figure((6, 4.2))
    style = series_styles(1)[0]
    ax.barh(psi_df["columna"], psi_df["psi"], color=style["color"])
    ax.axvline(0.1, color="#808080", linestyle="--", linewidth=1, label="umbral de alerta (0.1)")
    ax.invert_yaxis()
    ax.set_xlabel("Population Stability Index (PSI)")
    finalize_and_save(fig, ax, EDA_DIR / "psi_train_vs_test.png",
                       title="PSI entre train (dias 21-28) y prueba (dia 30)")


def main() -> None:
    con = _con()
    click_overall_and_by_time(con)
    categorical_frequencies(con)
    cardinality_long_tail(con)
    association_measures(con)
    population_stability_index(con)

    manifest = {
        "sample_size_asociacion_psi": SAMPLE_SIZE,
        "nota": (
            "Las medidas de asociacion y PSI usan una muestra aleatoria de "
            f"{SAMPLE_SIZE} filas de train, no el dataset completo, por limite de "
            "RAM (16 GB). Las agregaciones de conteo/CTR si cubren el dataset completo "
            "via DuckDB fuera de memoria."
        ),
    }
    (EDA_DIR / "eda_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("EDA completo. Artifacts en %s", EDA_DIR)


if __name__ == "__main__":
    main()
