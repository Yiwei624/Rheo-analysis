from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from rheo import db as rdb

APP_TITLE = "RheoDB — 流变实验数据库 & 自动分析"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
SCHEMA_PATH = BASE_DIR / "schema.sql"
DOCS_PATH = BASE_DIR / "docs" / "why_12_blade.md"

for folder in [DATA_DIR, UPLOAD_DIR, OUTPUT_DIR, BASE_DIR / "assets", BASE_DIR / "docs"]:
    folder.mkdir(parents=True, exist_ok=True)

rdb.write_template_files(BASE_DIR)

st.set_page_config(page_title=APP_TITLE, layout="wide")


def _mm_to_m(value_mm: float | None) -> float | None:
    if value_mm in (None, 0):
        return None
    return float(value_mm) / 1000.0


def _m_to_mm(value_m: float | None) -> float | None:
    if value_m is None:
        return None
    return float(value_m) * 1000.0


def _get_db_path() -> Path:
    name = st.session_state.get("db_name", "rheo.sqlite")
    if not str(name).lower().endswith((".sqlite", ".db")):
        name = f"{name}.sqlite"
    return DATA_DIR / str(name)


def _init_db() -> Path:
    db_path = _get_db_path()
    rdb.init_db(db_path, SCHEMA_PATH)
    return db_path


def _save_upload(uploaded) -> Path:
    dest = UPLOAD_DIR / uploaded.name
    dest.write_bytes(uploaded.getbuffer())
    return dest


def _show_sidebar() -> Path:
    with st.sidebar:
        st.header("数据库")
        st.text_input("数据库文件名", value="rheo.sqlite", key="db_name")
        db_path = _get_db_path()
        if st.button("初始化 / 修复数据库", use_container_width=True):
            rdb.init_db(db_path, SCHEMA_PATH)
            st.success(f"已准备数据库：{db_path.name}")
        st.caption(f"当前数据库：{db_path}")
        st.divider()
        st.header("模板")
        assets = BASE_DIR / "assets"
        for filename in ["template_export.csv", "template_export.xlsx", "example_data.csv"]:
            path = assets / filename
            if path.exists():
                st.download_button(
                    f"下载 {filename}",
                    data=path.read_bytes(),
                    file_name=filename,
                    use_container_width=True,
                )
    return db_path


def _page_import(db_path: Path) -> None:
    st.subheader("① 导入实验")
    uploaded = st.file_uploader("上传 rheometer 导出的 CSV / XLSX", type=["csv", "xlsx", "xls"])
    col1, col2, col3 = st.columns(3)
    with col1:
        exp_name = st.text_input("实验名称", value="")
        sample_name = st.text_input("样品名称", value="")
        geometry_type = st.selectbox("几何类型", ["", "cone_plate", "cup_bob", "vane_cup"])
    with col2:
        operator = st.text_input("操作人", value="")
        r1_mm = st.number_input("r1 (mm)", min_value=0.0, value=0.0, step=0.1)
        r2_mm = st.number_input("r2 (mm)", min_value=0.0, value=0.0, step=0.1)
    with col3:
        height_mm = st.number_input("高度 H (mm)", min_value=0.0, value=0.0, step=0.1)
        tau_y0 = st.number_input("τy 初值 (Pa)", min_value=0.0, value=0.0, step=0.1)
    notes = st.text_area("备注", value="")

    if uploaded is None:
        st.info("先上传文件。")
        return

    saved = _save_upload(uploaded)
    try:
        raw_df = rdb.read_uploaded_table(saved)
        norm_df = rdb.normalize_dataframe(raw_df)
    except Exception as exc:
        st.error(f"读取失败：{exc}")
        return

    with st.expander("查看原始数据预览", expanded=False):
        st.dataframe(raw_df.head(50), use_container_width=True)
    with st.expander("查看标准化后数据", expanded=True):
        st.dataframe(norm_df.head(50), use_container_width=True)
        st.caption(f"识别到列：{', '.join(norm_df.columns)}")

    if st.button("导入到数据库", type="primary", use_container_width=True):
        name = exp_name.strip() or Path(uploaded.name).stem
        try:
            exp_id = rdb.ingest_dataframe(
                db_path,
                raw_df,
                name=name,
                source_filename=uploaded.name,
                sample_name=sample_name,
                geometry_type=geometry_type,
                operator=operator,
                notes=notes,
                r1_m=_mm_to_m(r1_mm),
                r2_m=_mm_to_m(r2_mm),
                height_m=_mm_to_m(height_mm),
                yield0_pa=float(tau_y0) if tau_y0 > 0 else None,
            )
        except Exception as exc:
            st.error(f"导入失败：{exc}")
            return
        st.success(f"导入成功，experiment_id = {exp_id}")


def _metadata_table(meta: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "字段": ["id", "name", "sample_name", "geometry_type", "operator", "r1 (mm)", "r2 (mm)", "H (mm)", "yield0 (Pa)", "source_filename", "created_at"],
            "值": [
                meta.get("id"),
                meta.get("name"),
                meta.get("sample_name"),
                meta.get("geometry_type"),
                meta.get("operator"),
                _m_to_mm(meta.get("r1_m")),
                _m_to_mm(meta.get("r2_m")),
                _m_to_mm(meta.get("height_m")),
                meta.get("yield0_pa"),
                meta.get("source_filename"),
                meta.get("created_at"),
            ],
        }
    )


def _page_single_analysis(db_path: Path) -> None:
    st.subheader("② 实验浏览 & 分析")
    exp_df = rdb.list_experiments(db_path)
    if exp_df.empty:
        st.info("数据库里还没有实验。")
        return
    show_df = exp_df[["id", "name", "sample_name", "geometry_type", "n_points", "created_at"]]
    st.dataframe(show_df, use_container_width=True)
    exp_id = st.selectbox("选择 experiment_id", options=exp_df["id"].tolist(), format_func=lambda x: f"{x} — {exp_df.loc[exp_df['id']==x, 'name'].iloc[0]}")
    result = rdb.analyze_experiment(db_path, int(exp_id))
    meta = result["metadata"]
    data = result["data"]
    fits = result["fits"]
    hysteresis = result["hysteresis"]

    col1, col2 = st.columns([1.1, 1.4])
    with col1:
        st.markdown("**实验元数据**")
        st.dataframe(_metadata_table(meta), hide_index=True, use_container_width=True)
        st.markdown("**拟合结果**")
        st.dataframe(fits, use_container_width=True)
        if hysteresis:
            st.markdown("**Hysteresis Area**")
            st.json(hysteresis)
        else:
            st.caption("当前实验没有同时识别到 up / down，未计算 hysteresis area。")
        st.download_button(
            "下载拟合结果 Excel",
            data=rdb.dataframe_to_excel_bytes({"fits": fits, "data": data}),
            file_name=f"experiment_{exp_id}_analysis.xlsx",
            use_container_width=True,
        )

    with col2:
        for key, fig in result["figures"].items():
            st.pyplot(fig, clear_figure=False, use_container_width=True)
            st.download_button(
                f"下载 {key}.png",
                data=rdb.figure_to_png_bytes(fig),
                file_name=f"experiment_{exp_id}_{key}.png",
                use_container_width=True,
                key=f"dl_{exp_id}_{key}",
            )
    with st.expander("查看清洗后的数据表", expanded=False):
        st.dataframe(data, use_container_width=True)


def _page_overlay(db_path: Path) -> None:
    st.subheader("③ 多实验对比分析")
    exp_df = rdb.list_experiments(db_path)
    if exp_df.empty:
        st.info("数据库里还没有实验。")
        return

    options = exp_df["id"].tolist()
    selected = st.multiselect(
        "选择要对比的实验",
        options=options,
        default=options[:2],
        format_func=lambda x: f"{x} — {exp_df.loc[exp_df['id']==x, 'name'].iloc[0]}",
    )
    if len(selected) < 2:
        st.info("至少选择 2 个实验。")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        reference_id = st.selectbox("Reference 实验", options=selected, format_func=lambda x: f"{x} — {exp_df.loc[exp_df['id']==x, 'name'].iloc[0]}")
    with c2:
        metric = st.selectbox("比较指标", options=["shear_stress_pa", "viscosity_pa_s"], format_func=lambda x: {"shear_stress_pa": "Shear Stress", "viscosity_pa_s": "Viscosity"}[x])
    with c3:
        x_min = st.number_input("x 最小值 (1/s)", min_value=0.0, value=0.0, step=0.1)
    with c4:
        x_max = st.number_input("x 最大值 (1/s)", min_value=0.0, value=0.0, step=0.1)

    c5, c6, c7 = st.columns([1, 1, 1.6])
    with c5:
        y_scale_mode = st.selectbox(
            "Y 轴显示",
            options=["log", "linear"],
            format_func=lambda x: "10 的幂次（log）" if x == "log" else "普通数值（linear）",
        )
    with c6:
        legend_columns = st.selectbox("图例列数", options=[1, 2, 3], index=1)
    with c7:
        custom_y_ticks_text = st.text_input(
            "Linear Y ticks（留空=自动）",
            value="1,10,15,20",
            disabled=(y_scale_mode != "linear"),
            help="只有在线性坐标时生效，例如 1,10,15,20",
        )

    custom_y_ticks = None
    if y_scale_mode == "linear":
        text = (custom_y_ticks_text or "").strip()
        if text:
            try:
                custom_y_ticks = [float(part.strip()) for part in text.split(",") if part.strip()]
            except ValueError:
                st.warning("Linear Y ticks 请输入逗号分隔的数字，例如 1,10,15,20")
                custom_y_ticks = None

    result = rdb.compare_experiments(
        db_path,
        [int(x) for x in selected],
        reference_id=int(reference_id),
        metric=metric,
        x_min=float(x_min) if x_min > 0 else None,
        x_max=float(x_max) if x_max > 0 else None,
        overlay_y_scale=y_scale_mode,
        overlay_y_ticks=custom_y_ticks,
        overlay_legend_ncol=int(legend_columns),
    )
    st.caption("多实验对比图已改为：不画 fit 直线；up/down 分开展示；同一实验共用颜色，实验名完整显示在图例中。")
    st.pyplot(result["figure"], clear_figure=False, use_container_width=True)
    st.download_button(
        "下载 overlay.png",
        data=rdb.figure_to_png_bytes(result["figure"]),
        file_name="overlay.png",
        use_container_width=True,
    )
    if result["summary"].empty:
        st.warning("没有得到可比较的重叠区间，可能是 x 范围不重合，或某个实验缺少对应 segment。")
    else:
        st.dataframe(result["summary"], use_container_width=True)
        st.download_button(
            "下载误差统计 Excel",
            data=rdb.dataframe_to_excel_bytes({"comparison": result["summary"]}),
            file_name="comparison_summary.xlsx",
            use_container_width=True,
        )


def _page_why_12() -> None:
    st.subheader("④ 为什么 12-blade 最好")
    if DOCS_PATH.exists():
        st.markdown(DOCS_PATH.read_text(encoding="utf-8"))
    else:
        st.info("未找到 why_12_blade.md")


def main() -> None:
    st.title(APP_TITLE)
    st.caption("导入、存库、自动分析、partial-yield 判据、overlay 对比、以及 12-blade 解释页面。")
    db_path = _show_sidebar()
    _init_db()

    tab1, tab2, tab3, tab4 = st.tabs([
        "① 导入实验",
        "② 单实验分析",
        "③ 多实验对比",
        "④ 为什么 12-blade 最好",
    ])
    with tab1:
        _page_import(db_path)
    with tab2:
        _page_single_analysis(db_path)
    with tab3:
        _page_overlay(db_path)
    with tab4:
        _page_why_12()


if __name__ == "__main__":
    main()
