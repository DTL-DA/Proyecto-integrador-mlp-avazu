"""
Logica de preprocesamiento compartida conceptualmente entre scikit-learn y PySpark.

Todas las funciones de ajuste (`fit_*`) reciben explicitamente solo datos de
entrenamiento. Nunca reciben el dataframe completo, para que una fuga de datos
sea un error de llamada visible en el codigo, no un bug silencioso.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import KFold

RARE_TOKEN = "__RARE__"


class LeakageError(RuntimeError):
    """Se lanza cuando una funcion de ajuste recibe datos fuera de train."""


def assert_is_train_only(df: pd.DataFrame, max_day: int, day_col: str = "day") -> None:
    if day_col in df.columns and (df[day_col] > max_day).any():
        raise LeakageError(
            f"Se intento ajustar un encoder/escalador con dias > {max_day} "
            "(validacion o prueba). Revisa el pipeline de llamada."
        )


@dataclass
class RareCategoryGrouper:
    """Agrupa categorias poco frecuentes en un token comun, ajustado solo en train."""

    threshold: int
    columns: list[str]
    frequent_values_: dict[str, set] = field(default_factory=dict)

    def fit(self, df_train: pd.DataFrame, max_train_day: int) -> "RareCategoryGrouper":
        assert_is_train_only(df_train, max_train_day)
        for col in self.columns:
            counts = df_train[col].value_counts()
            self.frequent_values_[col] = set(counts[counts >= self.threshold].index)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.columns:
            frequent = self.frequent_values_.get(col, set())
            out[col] = out[col].where(out[col].isin(frequent), RARE_TOKEN)
        return out


@dataclass
class FrequencyEncoder:
    columns: list[str]
    freq_maps_: dict[str, pd.Series] = field(default_factory=dict)

    def fit(self, df_train: pd.DataFrame, max_train_day: int) -> "FrequencyEncoder":
        assert_is_train_only(df_train, max_train_day)
        n = len(df_train)
        for col in self.columns:
            self.freq_maps_[col] = df_train[col].value_counts() / n
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.columns:
            freq_map = self.freq_maps_[col]
            out[f"{col}_freq"] = out[col].map(freq_map).fillna(0.0)
        return out


class HashingEncoder:
    """Hashing trick via sklearn.FeatureHasher, con dimension declarada antes de entrenar."""

    def __init__(self, columns: list[str], n_features: int):
        self.columns = columns
        self.n_features = n_features
        self._hasher = FeatureHasher(n_features=n_features, input_type="string")

    def transform(self, df: pd.DataFrame):
        rows = (
            [f"{col}={val}" for col, val in zip(self.columns, row)]
            for row in df[self.columns].astype(str).itertuples(index=False)
        )
        return self._hasher.transform(rows)


@dataclass
class OutOfFoldTargetEncoder:
    """Target encoding solo para scikit-learn, con validacion cruzada interna (out-of-fold).

    Limitacion documentada: no tiene equivalente directo en el pipeline de PySpark
    de este proyecto (Spark ML no trae un TargetEncoder nativo con esta semantica),
    por lo que la comparacion sklearn-vs-Spark que use esta variante queda marcada
    como no estrictamente equivalente entre entornos.
    """

    columns: list[str]
    n_splits: int = 5
    smoothing: float = 20.0
    global_mean_: float = field(default=0.0)
    full_maps_: dict[str, pd.Series] = field(default_factory=dict)

    def fit_transform(
        self, df_train: pd.DataFrame, y_train: pd.Series, max_train_day: int, seed: int
    ) -> pd.DataFrame:
        assert_is_train_only(df_train, max_train_day)
        out = df_train.copy()
        self.global_mean_ = float(y_train.mean())
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=seed)

        for col in self.columns:
            oof = np.full(len(df_train), np.nan)
            for train_idx, hold_idx in kf.split(df_train):
                fold_df = df_train.iloc[train_idx]
                fold_y = y_train.iloc[train_idx]
                stats = fold_y.groupby(fold_df[col]).agg(["mean", "count"])
                smoothed = (
                    stats["mean"] * stats["count"] + self.global_mean_ * self.smoothing
                ) / (stats["count"] + self.smoothing)
                oof[hold_idx] = (
                    df_train.iloc[hold_idx][col].map(smoothed).fillna(self.global_mean_).values
                )
            out[f"{col}_te"] = oof

            full_stats = y_train.groupby(df_train[col]).agg(["mean", "count"])
            self.full_maps_[col] = (
                full_stats["mean"] * full_stats["count"] + self.global_mean_ * self.smoothing
            ) / (full_stats["count"] + self.smoothing)

        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.columns:
            out[f"{col}_te"] = out[col].map(self.full_maps_[col]).fillna(self.global_mean_)
        return out


def add_feature_cross(df: pd.DataFrame, col_a: str, col_b: str) -> pd.DataFrame:
    out = df.copy()
    out[f"{col_a}_x_{col_b}"] = out[col_a].astype(str) + "|" + out[col_b].astype(str)
    return out
