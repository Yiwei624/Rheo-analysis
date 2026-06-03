from __future__ import annotations

import io
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


DEFAULT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    sample_name TEXT,
    geometry_type TEXT,
    operator TEXT,
    notes TEXT,
    source_filename TEXT,
    r1_m REAL,
    r2_m REAL,
    height_m REAL,
    yield0_pa REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS data_points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    row_index INTEGER NOT NULL,
    segment TEXT,
    shear_rate_1_s REAL,
    shear_stress_pa REAL,
    viscosity_pa_s REAL,
    target_shear_rate_1_s REAL,
    percentage_deviation_pct REAL,
    temperature_c REAL,
    time_s REAL,
    thrust_g REAL,
    accumulated_time_s REAL,
    torque_nm REAL,
    angular_velocity_rad_s REAL,
    notes TEXT,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

CREATE INDEX IF NOT EXISTS idx_data_points_experiment_id ON data_points(experiment_id);
"""

TARGET_COLUMNS = {
    "shear_rate_1_s": [
        "shearrate1s", "shearrate", "shearrate_s", "shearratepersec", "gamma_dot", "gammadot",
        "targetshearrate1s", "targetshearrate",
    ],
    "shear_stress_pa": ["shearstresspa", "shearstress", "stresspa", "tau", "taupa"],
    "viscosity_pa_s": ["viscositypas", "viscosity", "etapas", "eta"],
    "target_shear_rate_1_s": ["targetshearrate1s", "targetshearrate", "programmedshearrate1s"],
    "percentage_deviation_pct": ["percentagedeviation", "percentagedeviationpct", "deviationpct"],
    "temperature_c": ["temperaturec", "temperature", "tempc", "temp"],
    "time_s": ["times", "time"],
    "thrust_g": ["thrustg", "thrust"],
    "accumulated_time_s": ["accumulatedtimes", "accumulatedtime", "elapsedtimes", "elapsedtime"],
    "torque_nm": ["torquenm", "torque"],
    "angular_velocity_rad_s": ["angularvelocityrads", "angularvelocity", "omega"],
    "notes": ["notes", "note", "comment", "comments"],
    "segment": ["segment", "sweep", "branch", "direction"],
}

DB_COLUMNS = [
    "segment",
    "shear_rate_1_s",
    "shear_stress_pa",
    "viscosity_pa_s",
    "target_shear_rate_1_s",
    "percentage_deviation_pct",
    "temperature_c",
    "time_s",
    "thrust_g",
    "accumulated_time_s",
    "torque_nm",
    "angular_velocity_rad_s",
    "notes",
]

PLOT_LABELS = {
    "shear_stress_pa": "Shear Stress τ (Pa)",
    "viscosity_pa_s": "Viscosity η (Pa·s)",
}

SEGMENT_MARKERS = {"up": "o", "down": "s", "all": "^"}
SEGMENT_NAMES = {"up": "up", "down": "down", "all": "all"}


@dataclass
class FitResult:
    model: str
    segment: str
    params: dict[str, float]
    r2: float | None
    rmse: float | None
    n_points: int


def init_db(db_path: str | Path, schema_path: str | Path | None = None) -> Path:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    sql = DEFAULT_SCHEMA_SQL
    if schema_path:
        schema_file = Path(schema_path)
        if schema_file.exists():
            content = schema_file.read_text(encoding="utf-8", errors="ignore")
            if "CREATE TABLE" in content.upper():
                sql = content
    with sqlite3.connect(db_path) as conn:
        conn.executescript(sql)
        conn.commit()
    return db_path


def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _normalize_name(name: Any) -> str:
    text = str(name).strip().lower()
    keep = []
    for ch in text:
        if ch.isalnum():
            keep.append(ch)
    return "".join(keep)


def _best_match_column(columns: Iterable[str], aliases: list[str]) -> str | None:
    normalized_map = {_normalize_name(col): col for col in columns}
    for alias in aliases:
        if alias in normalized_map:
            return normalized_map[alias]
    for norm, original in normalized_map.items():
        if any(alias in norm or norm in alias for alias in aliases):
            return original
    return None


def _coerce_numeric(series: pd.Series) -> pd.Series:
    if series.dtype.kind in "biufc":
        return pd.to_numeric(series, errors="coerce")
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(c).strip() for c in cleaned.columns]
    out = pd.DataFrame(index=cleaned.index)
    for target, aliases in TARGET_COLUMNS.items():
        match = _best_match_column(cleaned.columns, aliases)
        if match is None:
            continue
        if target == "notes":
            out[target] = cleaned[match].astype(str)
        elif target == "segment":
            out[target] = cleaned[match].astype(str).str.strip().str.lower()
        else:
            out[target] = _coerce_numeric(cleaned[match])
    if "shear_rate_1_s" not in out.columns and "target_shear_rate_1_s" in out.columns:
        out["shear_rate_1_s"] = out["target_shear_rate_1_s"]
    if "viscosity_pa_s" not in out.columns and {"shear_rate_1_s", "shear_stress_pa"}.issubset(out.columns):
        denom = out["shear_rate_1_s"].replace(0, np.nan)
        out["viscosity_pa_s"] = out["shear_stress_pa"] / denom
    if "notes" not in out.columns:
        out["notes"] = ""
    out = out.reset_index(drop=True)
    out["row_index"] = np.arange(len(out))
    out["segment"] = _resolve_segments(out)
    return out


def _resolve_segments(df: pd.DataFrame) -> pd.Series:
    if "segment" in df.columns:
        s = df["segment"].astype(str).str.strip().str.lower()
        mapped = s.replace({
            "upsweep": "up", "rampup": "up", "increase": "up", "ascending": "up",
            "downsweep": "down", "rampdown": "down", "decrease": "down", "descending": "down",
        })
        valid = mapped.where(mapped.isin(["up", "down", "all"]))
        if valid.notna().sum() >= max(2, len(valid) // 2):
            return valid.fillna("all")
    x = None
    if "shear_rate_1_s" in df.columns:
        x = pd.to_numeric(df["shear_rate_1_s"], errors="coerce")
    elif "target_shear_rate_1_s" in df.columns:
        x = pd.to_numeric(df["target_shear_rate_1_s"], errors="coerce")
    if x is None or x.notna().sum() < 4:
        return pd.Series(["all"] * len(df), index=df.index)
    x = x.to_numpy(dtype=float)
    finite = np.isfinite(x)
    if finite.sum() < 4:
        return pd.Series(["all"] * len(df), index=df.index)
    x_valid = x[finite]
    if np.all(np.diff(x_valid) >= 0) or np.all(np.diff(x_valid) <= 0):
        return pd.Series(["all"] * len(df), index=df.index)
    peak_idx = int(np.nanargmax(x))
    seg = np.array(["up" if i <= peak_idx else "down" for i in range(len(df))], dtype=object)
    seg[~finite] = "all"
    return pd.Series(seg, index=df.index)


def read_uploaded_table(path_or_buffer: str | Path | io.BytesIO) -> pd.DataFrame:
    suffix = ""
    if isinstance(path_or_buffer, (str, Path)):
        suffix = Path(path_or_buffer).suffix.lower()
    excel_engines = [None, "openpyxl"]
    if suffix in {".xlsx", ".xls"}:
        for engine in excel_engines:
            try:
                return pd.read_excel(path_or_buffer, engine=engine)
            except Exception:
                pass
    for encoding in ["utf-8", "utf-8-sig", "latin1"]:
        try:
            return pd.read_csv(path_or_buffer, encoding=encoding)
        except Exception:
            continue
    for engine in excel_engines:
        try:
            return pd.read_excel(path_or_buffer, engine=engine)
        except Exception:
            continue
    raise ValueError("无法读取文件，请确认是 CSV 或 XLSX。")


def ingest_dataframe(
    db_path: str | Path,
    raw_df: pd.DataFrame,
    *,
    name: str,
    source_filename: str = "",
    sample_name: str = "",
    geometry_type: str = "",
    operator: str = "",
    notes: str = "",
    r1_m: float | None = None,
    r2_m: float | None = None,
    height_m: float | None = None,
    yield0_pa: float | None = None,
) -> int:
    norm = normalize_dataframe(raw_df)
    if "shear_rate_1_s" not in norm.columns and "shear_stress_pa" not in norm.columns:
        raise ValueError("未识别到关键列，至少需要 shear rate 或 shear stress。")
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO experiments
                (name, sample_name, geometry_type, operator, notes, source_filename, r1_m, r2_m, height_m, yield0_pa)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (name, sample_name, geometry_type, operator, notes, source_filename, r1_m, r2_m, height_m, yield0_pa),
        )
        exp_id = int(cur.lastrowid)
        insert_rows = []
        for _, row in norm.iterrows():
            insert_rows.append(
                (
                    exp_id,
                    int(row["row_index"]),
                    str(row.get("segment", "all")),
                    _safe_float(row.get("shear_rate_1_s")),
                    _safe_float(row.get("shear_stress_pa")),
                    _safe_float(row.get("viscosity_pa_s")),
                    _safe_float(row.get("target_shear_rate_1_s")),
                    _safe_float(row.get("percentage_deviation_pct")),
                    _safe_float(row.get("temperature_c")),
                    _safe_float(row.get("time_s")),
                    _safe_float(row.get("thrust_g")),
                    _safe_float(row.get("accumulated_time_s")),
                    _safe_float(row.get("torque_nm")),
                    _safe_float(row.get("angular_velocity_rad_s")),
                    str(row.get("notes", "")),
                )
            )
        conn.executemany(
            """
            INSERT INTO data_points (
                experiment_id, row_index, segment, shear_rate_1_s, shear_stress_pa, viscosity_pa_s,
                target_shear_rate_1_s, percentage_deviation_pct, temperature_c, time_s, thrust_g,
                accumulated_time_s, torque_nm, angular_velocity_rad_s, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            insert_rows,
        )
        conn.commit()
    return exp_id


def list_experiments(db_path: str | Path) -> pd.DataFrame:
    init_db(db_path)
    with _connect(db_path) as conn:
        df = pd.read_sql_query(
            """
            SELECT e.*, COUNT(d.id) AS n_points
            FROM experiments e
            LEFT JOIN data_points d ON d.experiment_id = e.id
            GROUP BY e.id
            ORDER BY e.created_at DESC, e.id DESC
            """,
            conn,
        )
    return df


def get_experiment(db_path: str | Path, experiment_id: int) -> dict[str, Any]:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
    if row is None:
        raise KeyError(f"experiment_id={experiment_id} 不存在")
    return dict(row)


def load_experiment_data(db_path: str | Path, experiment_id: int) -> pd.DataFrame:
    with _connect(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM data_points WHERE experiment_id = ? ORDER BY row_index ASC",
            conn,
            params=(experiment_id,),
        )
    return df


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        v = float(value)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def _segment_data(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if df.empty:
        return {"all": df.copy()}
    out: dict[str, pd.DataFrame] = {}
    for segment, part in df.groupby(df["segment"].fillna("all").astype(str)):
        out[segment] = part.sort_values("row_index").reset_index(drop=True)
    if "up" in out or "down" in out:
        return out
    return {"all": df.sort_values("row_index").reset_index(drop=True)}


def _fit_power_law(part: pd.DataFrame) -> FitResult | None:
    cols = {"shear_rate_1_s", "shear_stress_pa"}
    if not cols.issubset(part.columns):
        return None
    data = part[list(cols)].dropna()
    data = data[(data["shear_rate_1_s"] > 0) & (data["shear_stress_pa"] > 0)]
    if len(data) < 3:
        return None
    x = np.log10(data["shear_rate_1_s"].to_numpy(dtype=float))
    y = np.log10(data["shear_stress_pa"].to_numpy(dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    pred = intercept + slope * x
    y_lin = 10 ** y
    pred_lin = 10 ** pred
    return FitResult(
        model="power_law",
        segment=str(part["segment"].iloc[0]),
        params={"K": float(10 ** intercept), "n": float(slope)},
        r2=_r2_score(y_lin, pred_lin),
        rmse=_rmse(y_lin, pred_lin),
        n_points=len(data),
    )


def _fit_bingham(part: pd.DataFrame) -> FitResult | None:
    cols = {"shear_rate_1_s", "shear_stress_pa"}
    if not cols.issubset(part.columns):
        return None
    data = part[list(cols)].dropna()
    data = data[data["shear_rate_1_s"] >= 0]
    if len(data) < 3:
        return None
    x = data["shear_rate_1_s"].to_numpy(dtype=float)
    y = data["shear_stress_pa"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    pred = intercept + slope * x
    return FitResult(
        model="bingham",
        segment=str(part["segment"].iloc[0]),
        params={"tau0": float(intercept), "mu_p": float(slope)},
        r2=_r2_score(y, pred),
        rmse=_rmse(y, pred),
        n_points=len(data),
    )


def _fit_herschel_bulkley(part: pd.DataFrame, yield0: float | None = None) -> FitResult | None:
    cols = {"shear_rate_1_s", "shear_stress_pa"}
    if not cols.issubset(part.columns):
        return None
    data = part[list(cols)].dropna()
    data = data[(data["shear_rate_1_s"] > 0) & (data["shear_stress_pa"] > 0)]
    if len(data) < 4:
        return None
    x = data["shear_rate_1_s"].to_numpy(dtype=float)
    y = data["shear_stress_pa"].to_numpy(dtype=float)

    def hb_model(xv: np.ndarray, tau0: float, k: float, n: float) -> np.ndarray:
        return tau0 + k * np.power(xv, n)

    tau_guess = max(0.0, min(y) * 0.5 if yield0 is None else float(yield0))
    k_guess = max(1e-6, np.median(y / np.maximum(x, 1e-6) ** 0.5))
    n_guess = 0.5
    try:
        popt, _ = curve_fit(
            hb_model,
            x,
            y,
            p0=[tau_guess, k_guess, n_guess],
            bounds=([0.0, 1e-9, 0.0], [max(y) * 2.0, max(y) * 10.0, 2.0]),
            maxfev=20000,
        )
    except Exception:
        return None
    pred = hb_model(x, *popt)
    return FitResult(
        model="herschel_bulkley",
        segment=str(part["segment"].iloc[0]),
        params={"tau0": float(popt[0]), "K": float(popt[1]), "n": float(popt[2])},
        r2=_r2_score(y, pred),
        rmse=_rmse(y, pred),
        n_points=len(data),
    )


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if len(y_true) < 2:
        return None
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0:
        return None
    return 1.0 - ss_res / ss_tot


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if len(y_true) < 1:
        return None
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_hysteresis_area(df: pd.DataFrame) -> dict[str, float | str] | None:
    segs = _segment_data(df)
    if not {"up", "down"}.issubset(segs.keys()):
        return None
    up = segs["up"][["shear_rate_1_s", "shear_stress_pa"]].dropna()
    down = segs["down"][["shear_rate_1_s", "shear_stress_pa"]].dropna()
    up = up[(up["shear_rate_1_s"] > 0) & (up["shear_stress_pa"] >= 0)].sort_values("shear_rate_1_s")
    down = down[(down["shear_rate_1_s"] > 0) & (down["shear_stress_pa"] >= 0)].sort_values("shear_rate_1_s")
    if len(up) < 3 or len(down) < 3:
        return None
    xmin = max(float(up["shear_rate_1_s"].min()), float(down["shear_rate_1_s"].min()))
    xmax = min(float(up["shear_rate_1_s"].max()), float(down["shear_rate_1_s"].max()))
    if not (xmax > xmin > 0):
        return None
    xs = np.geomspace(xmin, xmax, 200)
    yu = np.interp(xs, up["shear_rate_1_s"], up["shear_stress_pa"])
    yd = np.interp(xs, down["shear_rate_1_s"], down["shear_stress_pa"])
    area = float(np.trapezoid(np.abs(yu - yd), xs))
    signed = float(np.trapezoid(yu - yd, xs))
    return {"method": "logspace_matched_trapezoid", "area": area, "signed_area": signed}


def compute_partial_yield(df: pd.DataFrame, r1_m: float | None, r2_m: float | None, tau_y: float | None) -> pd.DataFrame:
    out = df.copy()
    if r1_m is None or r2_m is None or tau_y is None or tau_y <= 0 or r1_m <= 0 or r2_m <= r1_m:
        out["tau_crit_pa"] = np.nan
        out["lambda_mobilization"] = np.nan
        out["yield_radius_m"] = np.nan
        out["yield_radius_over_r2"] = np.nan
        out["regime"] = "unknown"
        return out
    tau_crit = float(tau_y) * (float(r2_m) / float(r1_m)) ** 2
    tau1 = pd.to_numeric(out.get("shear_stress_pa"), errors="coerce")
    out["tau_crit_pa"] = tau_crit
    out["lambda_mobilization"] = tau1 / tau_crit
    y_radius = float(r1_m) * np.sqrt(np.maximum(tau1, 0.0) / float(tau_y))
    y_radius = np.minimum(y_radius, float(r2_m))
    out["yield_radius_m"] = y_radius
    out["yield_radius_over_r2"] = y_radius / float(r2_m)
    out["regime"] = np.where(out["lambda_mobilization"] >= 1.0, "fully_yielded", "partially_yielded")
    return out


def _fit_table(fits: list[FitResult | None]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fit in fits:
        if fit is None:
            continue
        row = {"model": fit.model, "segment": fit.segment, "r2": fit.r2, "rmse": fit.rmse, "n_points": fit.n_points}
        row.update(fit.params)
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_experiment(db_path: str | Path, experiment_id: int) -> dict[str, Any]:
    meta = get_experiment(db_path, experiment_id)
    df = load_experiment_data(db_path, experiment_id)
    df = compute_partial_yield(df, meta.get("r1_m"), meta.get("r2_m"), meta.get("yield0_pa"))
    segs = _segment_data(df)
    fits: list[FitResult | None] = []
    for seg_name, part in segs.items():
        if seg_name not in {"up", "down", "all"}:
            continue
        fits.append(_fit_power_law(part))
        fits.append(_fit_bingham(part))
        fits.append(_fit_herschel_bulkley(part, meta.get("yield0_pa")))
    fits_df = _fit_table(fits)
    hysteresis = compute_hysteresis_area(df)
    figures = make_single_experiment_figures(df, meta)
    return {
        "metadata": meta,
        "data": df,
        "fits": fits_df,
        "hysteresis": hysteresis,
        "figures": figures,
    }


def _maybe_add_log_fit(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: Any, label: str) -> None:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = np.asarray(x)[mask]
    y = np.asarray(y)[mask]
    if len(x) < 3:
        return
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    try:
        slope, intercept = np.polyfit(np.log10(x), np.log10(y), 1)
    except Exception:
        return
    xs = np.geomspace(float(np.min(x)), float(np.max(x)), 100)
    ys = 10 ** (intercept + slope * np.log10(xs))
    ax.plot(xs, ys, linestyle="--", linewidth=1.0, color=color, alpha=0.9, label=label)


def _set_log_axes(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.25)


def make_single_experiment_figures(df: pd.DataFrame, meta: dict[str, Any]) -> dict[str, plt.Figure]:
    title = f"exp={meta['id']} — {meta.get('name') or ''}".strip()
    segs = _segment_data(df)
    colors = {"up": "tab:blue", "down": "tab:orange", "all": "tab:green"}

    fig1, ax1 = plt.subplots(figsize=(7.2, 4.6))
    for seg_name, part in segs.items():
        if not {"shear_rate_1_s", "shear_stress_pa"}.issubset(part.columns):
            continue
        pdata = part[["shear_rate_1_s", "shear_stress_pa"]].dropna()
        if pdata.empty:
            continue
        ax1.scatter(
            pdata["shear_rate_1_s"],
            pdata["shear_stress_pa"],
            s=18,
            marker=SEGMENT_MARKERS.get(seg_name, "o"),
            label=SEGMENT_NAMES.get(seg_name, seg_name),
            alpha=0.9,
            color=colors.get(seg_name, None),
        )
        _maybe_add_log_fit(
            ax1,
            pdata["shear_rate_1_s"].to_numpy(dtype=float),
            pdata["shear_stress_pa"].to_numpy(dtype=float),
            colors.get(seg_name, None),
            f"{seg_name} fit",
        )
    _set_log_axes(ax1, "Shear rate γ̇ (1/s)", "Shear stress τ (Pa)")
    ax1.set_title(f"Fig1 τ vs γ̇ — {title}")
    ax1.legend()

    fig2, ax2 = plt.subplots(figsize=(7.2, 4.6))
    for seg_name, part in segs.items():
        pdata = part[["shear_rate_1_s", "viscosity_pa_s"]].dropna()
        pdata = pdata[(pdata["shear_rate_1_s"] > 0) & (pdata["viscosity_pa_s"] > 0)]
        if pdata.empty:
            continue
        ax2.scatter(
            pdata["shear_rate_1_s"],
            pdata["viscosity_pa_s"],
            s=18,
            marker=SEGMENT_MARKERS.get(seg_name, "o"),
            label=SEGMENT_NAMES.get(seg_name, seg_name),
            alpha=0.9,
            color=colors.get(seg_name, None),
        )
        _maybe_add_log_fit(
            ax2,
            pdata["shear_rate_1_s"].to_numpy(dtype=float),
            pdata["viscosity_pa_s"].to_numpy(dtype=float),
            colors.get(seg_name, None),
            f"{seg_name} fit",
        )
    _set_log_axes(ax2, "Shear rate γ̇ (1/s)", "Viscosity η (Pa·s)")
    ax2.set_title(f"Fig2 η vs γ̇ — {title}")
    ax2.legend()

    fig3, ax3 = plt.subplots(figsize=(7.2, 4.6))
    if {"shear_rate_1_s", "shear_stress_pa", "lambda_mobilization"}.issubset(df.columns):
        for regime, part in df.groupby("regime"):
            pdata = part[["shear_rate_1_s", "shear_stress_pa"]].dropna()
            pdata = pdata[(pdata["shear_rate_1_s"] > 0) & (pdata["shear_stress_pa"] > 0)]
            if pdata.empty:
                continue
            marker = "o" if regime == "fully_yielded" else "^"
            ax3.scatter(pdata["shear_rate_1_s"], pdata["shear_stress_pa"], s=18, marker=marker, alpha=0.85, label=regime)
        tau_crit = pd.to_numeric(df.get("tau_crit_pa"), errors="coerce").dropna()
        if not tau_crit.empty:
            ax3.axhline(float(tau_crit.iloc[0]), linestyle="--", linewidth=1.0, color="black", label="τ_crit")
    _set_log_axes(ax3, "Shear rate γ̇ (1/s)", "Shear stress τ (Pa)")
    ax3.set_title(f"Fig3 τ vs γ̇ + partial-yield — {title}")
    ax3.legend()

    fig4, ax4 = plt.subplots(figsize=(7.2, 4.6))
    pdata = df[["shear_rate_1_s", "yield_radius_over_r2"]].dropna()
    pdata = pdata[(pdata["shear_rate_1_s"] > 0) & (pdata["yield_radius_over_r2"] > 0)]
    if not pdata.empty:
        ax4.scatter(pdata["shear_rate_1_s"], pdata["yield_radius_over_r2"], s=18, alpha=0.85)
        ax4.axhline(1.0, linestyle="--", linewidth=1.0, color="black", label="full gap")
    ax4.set_xscale("log")
    ax4.set_xlabel("Shear rate γ̇ (1/s)")
    ax4.set_ylabel("r_y / r2")
    ax4.grid(True, which="both", alpha=0.25)
    ax4.set_title(f"Fig4 r_y/r2 vs γ̇ — {title}")
    ax4.legend(loc="best") if ax4.lines else None

    return {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4}


def _interpolate_on_common_grid(df_a: pd.DataFrame, df_b: pd.DataFrame, metric: str, x_min: float | None, x_max: float | None) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    a = df_a[["shear_rate_1_s", metric]].dropna()
    b = df_b[["shear_rate_1_s", metric]].dropna()
    a = a[(a["shear_rate_1_s"] > 0) & np.isfinite(a[metric])]
    b = b[(b["shear_rate_1_s"] > 0) & np.isfinite(b[metric])]
    if len(a) < 3 or len(b) < 3:
        return None
    a = a.sort_values("shear_rate_1_s")
    b = b.sort_values("shear_rate_1_s")
    xmin = max(float(a["shear_rate_1_s"].min()), float(b["shear_rate_1_s"].min()))
    xmax = min(float(a["shear_rate_1_s"].max()), float(b["shear_rate_1_s"].max()))
    if x_min is not None:
        xmin = max(xmin, float(x_min))
    if x_max is not None:
        xmax = min(xmax, float(x_max))
    if not (xmax > xmin > 0):
        return None
    xs = np.geomspace(xmin, xmax, 150)
    ya = np.interp(xs, a["shear_rate_1_s"], a[metric])
    yb = np.interp(xs, b["shear_rate_1_s"], b[metric])
    return xs, ya, yb


def compare_experiments(
    db_path: str | Path,
    experiment_ids: list[int],
    *,
    reference_id: int,
    metric: str = "shear_stress_pa",
    x_min: float | None = None,
    x_max: float | None = None,
    overlay_y_scale: str = "log",
    overlay_y_ticks: list[float] | None = None,
    overlay_legend_ncol: int = 2,
) -> dict[str, Any]:
    if reference_id not in experiment_ids:
        raise ValueError("reference_id 必须包含在 experiment_ids 中。")
    metas = {eid: get_experiment(db_path, eid) for eid in experiment_ids}
    frames = {eid: load_experiment_data(db_path, eid) for eid in experiment_ids}
    ref = frames[reference_id]
    rows: list[dict[str, Any]] = []
    for eid in experiment_ids:
        if eid == reference_id:
            continue
        for segment in ["up", "down", "all"]:
            ref_seg = ref[ref["segment"] == segment] if segment in ref["segment"].unique() else (ref if segment == "all" else pd.DataFrame())
            cur = frames[eid]
            cur_seg = cur[cur["segment"] == segment] if segment in cur["segment"].unique() else (cur if segment == "all" else pd.DataFrame())
            if ref_seg.empty or cur_seg.empty:
                continue
            interp = _interpolate_on_common_grid(cur_seg, ref_seg, metric, x_min, x_max)
            if interp is None:
                continue
            xs, y, yref = interp
            ape = np.abs((y - yref) / np.where(np.abs(yref) < 1e-12, np.nan, yref)) * 100.0
            rmse = np.sqrt(np.mean((y - yref) ** 2))
            rows.append(
                {
                    "experiment_id": eid,
                    "experiment_name": metas[eid].get("name"),
                    "reference_id": reference_id,
                    "segment": segment,
                    "metric": metric,
                    "x_min": float(xs.min()),
                    "x_max": float(xs.max()),
                    "MAPE_pct": float(np.nanmean(ape)),
                    "Max_APE_pct": float(np.nanmax(ape)),
                    "RMSE": float(rmse),
                    "n_grid": int(len(xs)),
                }
            )
    fig = make_overlay_figure(
        frames,
        metas,
        metric=metric,
        x_min=x_min,
        x_max=x_max,
        y_scale=overlay_y_scale,
        y_ticks=overlay_y_ticks,
        legend_ncol=overlay_legend_ncol,
    )
    return {"summary": pd.DataFrame(rows), "figure": fig, "metas": metas}


def make_overlay_figure(
    frames: dict[int, pd.DataFrame],
    metas: dict[int, dict[str, Any]],
    metric: str = "shear_stress_pa",
    *,
    x_min: float | None = None,
    x_max: float | None = None,
    y_scale: str = "log",
    y_ticks: list[float] | None = None,
    legend_ncol: int = 2,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    cmap = plt.cm.get_cmap("tab20")
    marker_map = {"up": "o", "down": "s", "all": "^"}

    for idx, (eid, df) in enumerate(frames.items()):
        color = cmap(idx % 20)
        name = str(metas[eid].get("name") or eid)
        grouped = _segment_data(df)
        for seg_name in ["up", "down"] if ("up" in grouped or "down" in grouped) else ["all"]:
            if seg_name not in grouped:
                continue
            part = grouped[seg_name]
            if not {"shear_rate_1_s", metric}.issubset(part.columns):
                continue
            pdata = part[["shear_rate_1_s", metric]].copy()
            pdata["shear_rate_1_s"] = pd.to_numeric(pdata["shear_rate_1_s"], errors="coerce")
            pdata[metric] = pd.to_numeric(pdata[metric], errors="coerce")
            pdata = pdata.dropna()
            pdata = pdata[(pdata["shear_rate_1_s"] > 0) & np.isfinite(pdata[metric])]
            if y_scale == "log":
                pdata = pdata[pdata[metric] > 0]
            if x_min is not None:
                pdata = pdata[pdata["shear_rate_1_s"] >= float(x_min)]
            if x_max is not None:
                pdata = pdata[pdata["shear_rate_1_s"] <= float(x_max)]
            if pdata.empty:
                continue
            pdata = pdata.sort_values("shear_rate_1_s")
            ax.scatter(
                pdata["shear_rate_1_s"],
                pdata[metric],
                s=48,
                marker=marker_map.get(seg_name, "o"),
                label=f"{name} — {seg_name}",
                color=color,
                alpha=0.95,
                linewidths=0.35,
                edgecolors=color,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Shear rate γ̇ (1/s)")
    ax.set_ylabel(PLOT_LABELS.get(metric, metric))
    if y_scale == "log":
        ax.set_yscale("log")
    else:
        ax.set_yscale("linear")
        if y_ticks:
            clean_ticks = [float(v) for v in y_ticks if pd.notna(v)]
            if clean_ticks:
                ax.set_yticks(clean_ticks)
                ax.set_yticklabels([f"{v:g}" for v in clean_ticks])
    ax.grid(True, which="both", alpha=0.25)
    ax.set_title("Selected experiments overlay")
    ax.legend(fontsize=9, ncol=max(1, int(legend_ncol)), loc="upper left", frameon=True)
    fig.tight_layout()
    return fig


def dataframe_to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            sheet_name = name[:31] or "Sheet1"
            frame.to_excel(writer, index=False, sheet_name=sheet_name)
    buf.seek(0)
    return buf.getvalue()


def figure_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def write_template_files(base_dir: str | Path) -> dict[str, Path]:
    base = Path(base_dir)
    assets = base / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    template = pd.DataFrame(
        columns=[
            "Shear Rate 1/s",
            "Shear Stress Pa",
            "Viscosity Pas",
            "Target Shear Rate 1/s",
            "Percentage Deviation %",
            "Temperature ℃",
            "Time s",
            "Thrust g",
            "Accumulated Time s",
            "Torque Nm",
            "Angular Velocity rad/s",
            "Notes",
        ]
    )
    sample = _make_example_data()
    csv_path = assets / "template_export.csv"
    xlsx_path = assets / "template_export.xlsx"
    example_path = assets / "example_data.csv"
    template.to_csv(csv_path, index=False)
    template.to_excel(xlsx_path, index=False)
    sample.to_csv(example_path, index=False)
    return {"template_csv": csv_path, "template_xlsx": xlsx_path, "example_csv": example_path}


def _make_example_data() -> pd.DataFrame:
    up_rates = np.geomspace(1e-3, 2e2, 60)
    down_rates = up_rates[::-1]
    rng = np.random.default_rng(42)

    def stress_model(x: np.ndarray, plateau: float, rise: float, n: float, offset: float) -> np.ndarray:
        return offset + plateau + rise * np.power(x, n) / (1 + np.power(x / 6.0, -1.2))

    tau_up = stress_model(up_rates, plateau=3.2, rise=2.8, n=0.42, offset=0.3)
    tau_up += np.where(up_rates < 0.01, 6.0 * np.exp(-((np.log10(up_rates) + 2.0) ** 2) / 0.18), 0.0)
    tau_up *= (1 + rng.normal(0, 0.04, len(tau_up)))

    tau_down = stress_model(down_rates, plateau=2.8, rise=2.2, n=0.36, offset=0.25)
    tau_down *= (1 + rng.normal(0, 0.03, len(tau_down)))

    rates = np.concatenate([up_rates, down_rates])
    stress = np.concatenate([tau_up, tau_down])
    seg = np.array(["up"] * len(up_rates) + ["down"] * len(down_rates))
    df = pd.DataFrame(
        {
            "Shear Rate 1/s": rates,
            "Shear Stress Pa": stress,
            "Viscosity Pas": stress / rates,
            "Target Shear Rate 1/s": rates,
            "Percentage Deviation %": 0.0,
            "Temperature ℃": 25.0,
            "Time s": np.arange(len(rates), dtype=float),
            "Thrust g": np.nan,
            "Accumulated Time s": np.arange(len(rates), dtype=float),
            "Torque Nm": np.nan,
            "Angular Velocity rad/s": np.nan,
            "Notes": seg,
            "Segment": seg,
        }
    )
    return df
