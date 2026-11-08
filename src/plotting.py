"""
Estilo de graficas compartido: escala de grises unicamente (exigencia del curso).

Las series se distinguen por tono (claro->oscuro en orden fijo), tipo de linea y
marcador, nunca por color. Lineas delgadas, grilla recesiva, leyenda siempre
presente con 2+ series.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

# Tonos de gris en orden fijo (oscuro -> claro), nunca reasignados por rango/filtro.
GRAY_TONES = ["#1a1a1a", "#4d4d4d", "#808080", "#a6a6a6", "#595959", "#bfbfbf"]
LINESTYLES = ["-", "--", "-.", ":", "-", "--"]
MARKERS = ["o", "s", "^", "D", "v", "x"]


def series_styles(n: int) -> list[dict]:
    if n > len(GRAY_TONES):
        raise ValueError(
            f"Se pidieron {n} series pero solo hay {len(GRAY_TONES)} estilos "
            "distinguibles definidos; agrega mas tonos/linestyles antes de seguir."
        )
    return [
        {"color": GRAY_TONES[i], "linestyle": LINESTYLES[i], "marker": MARKERS[i]}
        for i in range(n)
    ]


def new_figure(figsize: tuple[float, float] = (7.0, 4.2)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, color="#d9d9d9", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return fig, ax


def finalize_and_save(fig, ax, out_path, legend: bool = True, title: str | None = None):
    if title:
        ax.set_title(title, fontsize=11)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
