
import json
import os
import math
import platform
import sys
from importlib import import_module
from importlib.metadata import distributions
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Plotly is a nice fit for stacked, shared-axis chromatograms
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# -----------------------------
# Application version metadata
# -----------------------------
# Manually update APP_SEMANTIC_VERSION when releasing a new version.
APP_SEMANTIC_VERSION = "0.43.0"
APP_DISPLAY_FILENAME = os.path.basename(__file__) if "__file__" in globals() else "quad_viewer_app_v43_persistent_uploads.py"

# Libraries shown in the sidebar Environment / Library Info panel.
# Add or remove entries here if the application dependency set changes.
CORE_LIBRARY_MODULES = {
    "Streamlit": "streamlit",
    "Plotly": "plotly",
    "Pandas": "pandas",
    "NumPy": "numpy",
}


def safe_text(value: Any, fallback: str = "UNKNOWN") -> str:
    """Return a compact display string for sidebar metadata."""
    try:
        if value is None:
            return fallback
        text = str(value).strip()
        return text if text else fallback
    except Exception:
        return fallback


def get_module_version(module_name: str) -> str:
    """Return a package/module version without failing if unavailable."""
    try:
        module = import_module(module_name)
        version = getattr(module, "__version__", None)
        return safe_text(version, "Unknown")
    except Exception:
        return "Not installed"


@st.cache_data(show_spinner=False)
def get_installed_package_versions() -> List[Tuple[str, str]]:
    """Return installed package versions for optional deep-debug display."""
    packages: List[Tuple[str, str]] = []
    try:
        for dist in distributions():
            try:
                name = dist.metadata.get("Name", "")
                version = dist.version
                if name:
                    packages.append((str(name), str(version)))
            except Exception:
                continue
    except Exception:
        return []
    return sorted(packages, key=lambda item: item[0].lower())


def render_application_environment_sidebar() -> None:
    """Render combined application version, environment, and library information in the sidebar."""
    version_text = safe_text(APP_SEMANTIC_VERSION, "UNKNOWN VERSION")
    file_text = safe_text(APP_DISPLAY_FILENAME, "UNKNOWN FILE")

    with st.sidebar.expander("App. & Environment", expanded=False):
        st.markdown("**Application version**")
        st.caption(f"Version: {version_text}")
        st.caption(f"File: {file_text}")

        st.markdown("**Runtime**")
        st.caption(f"Python: {safe_text(sys.version.split()[0])}")
        st.caption(f"OS: {safe_text(platform.system())} {safe_text(platform.release(), '')}".strip())

        st.markdown("**Core libraries**")
        for label, module_name in CORE_LIBRARY_MODULES.items():
            st.caption(f"{label}: {get_module_version(module_name)}")

        show_full_environment = st.checkbox(
            "Show package list",
            value=False,
            key="show_full_package_list",
            help="Shows all installed Python packages and versions for debugging and reproducibility checks.",
        )
        if show_full_environment:
            packages = get_installed_package_versions()
            if packages:
                st.code("\n".join(f"{name}=={version}" for name, version in packages), language="text")
            else:
                st.caption("No package information available.")


# -----------------------------
# Helpers
# -----------------------------
def safe_upper_status(val: Any) -> str:
    if val is None:
        return "UNKNOWN"
    return str(val).upper()


def status_color(status_upper: str) -> str:
    # Simple, consistent mapping (can be refined later)
    mapping = {
        "SUCCEEDED": "#2e7d32",
        "RECALLED": "#ef6c00",
        "ABORTED": "#c62828",
        "UNKNOWN": "#616161",
    }
    return mapping.get(status_upper, "#616161")



def normalize_opinion(val: Any) -> str:
    """Normalize raw software_opinion values to UI vocabulary."""
    if val is None:
        return "UNKNOWN"
    s = str(val).strip().lower()
    mapping = {
        "Accept": "ACCEPTED",
        "accept": "ACCEPTED",
        "accepted": "ACCEPTED",
        "reject": "REJECTED",
        "rejected": "REJECTED",
        "inconclusive": "INCONCLUSIVE",
        "error": "ERROR",
        "err": "ERROR",
    }
    return mapping.get(s, str(val).upper() if str(val).strip() else "UNKNOWN")

def normalize_human_opinion(val: Any) -> str:
    """Normalize human_opinion values for UI display."""
    if val is None:
        return "NULL"
    s = str(val).strip().lower()
    if not s or s == "null":
        return "NULL"
    mapping = {
        "accept": "ACCEPTED",
        "accepted": "ACCEPTED",
        "reject": "REJECTED",
        "rejected": "REJECTED",
    }
    return mapping.get(s, str(val).upper())


def opinion_color(opinion_upper: str) -> str:
    mapping = {
        "ACCEPTED": "#2e7d32",
        "REJECTED": "#c62828",
        "INCONCLUSIVE": "#ef6c00",
        "ERROR": "#1565c0",
        "UNKNOWN": "#616161",
    }
    return mapping.get(opinion_upper, "#616161")

def human_opinion_color(opinion_upper: str) -> str:
    mapping = {
        "ACCEPTED": "#2e7d32",
        "REJECTED": "#c62828",
        "NULL": "#616161",
        "UNKNOWN": "#616161",
    }
    return mapping.get(opinion_upper, "#616161")


def format_max_3dp(val: Any) -> str:
    """Format numbers to a maximum of 3 decimal places; otherwise return as string."""
    if val is None:
        return ""
    try:
        # Accept numeric strings too
        if isinstance(val, str):
            s = val.strip()
            if s == "":
                return ""
            try:
                num = float(s)
                # Treat leading-zero digit strings as IDs unless they contain '.' or 'e'
                if s.isdigit() and len(s) > 1 and s.startswith("0"):
                    return s
                val = num
            except Exception:
                return s
        if isinstance(val, (int, float)):
            if not math.isfinite(float(val)):
                return str(val)
            out = f"{float(val):.3f}"
            out = out.rstrip("0").rstrip(".")
            return out
        return str(val)
    except Exception:
        return str(val)


def normalize_concern_category(val: Any) -> str:
    if val is None:
        return "UNKNOWN"
    s = str(val).strip().lower()
    mapping = {
        "Accept": "ACCEPT",
        "accept": "ACCEPT",
        "accepted": "ACCEPT",
        "reject": "REJECT",
        "rejected": "REJECT",
        "inconclusive": "INCONCLUSIVE",
        "error": "ERROR",
    }
    return mapping.get(s, str(val).upper() if str(val).strip() else "UNKNOWN")


def concern_color(cat_upper: str) -> str:
    mapping = {
        "REJECT": "#c62828",
        "ERROR": "#1565c0",
        "INCONCLUSIVE": "#ef6c00",
        "ACCEPT": "#2e7d32",
        "UNKNOWN": "#616161",
    }
    return mapping.get(cat_upper, "#616161")
def list_json_files(folder: str) -> List[str]:
    try:
        files = [f for f in os.listdir(folder) if f.lower().endswith(".json")]
        files.sort()
        return files
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def load_json(path: str) -> Tuple[Optional[dict], Optional[str]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e:
        return None, str(e)


@st.cache_data(show_spinner=False)
def load_dataset(folder: str, files: List[str]) -> Dict[str, Any]:
    """
    Returns:
      {
        "readable": {filename: json_dict},
        "unreadable": {filename: error_str},
        "all_analytes": sorted list
      }
    """
    readable: Dict[str, dict] = {}
    unreadable: Dict[str, str] = {}
    analytes = set()

    for fn in files:
        p = os.path.join(folder, fn)
        data, err = load_json(p)
        if data is None:
            unreadable[fn] = err or "Unknown error"
            continue
        readable[fn] = data
        aqm = data.get("analyte_quad_map", {})
        if isinstance(aqm, dict):
            analytes.update(aqm.keys())

    return {
        "readable": readable,
        "unreadable": unreadable,
        "all_analytes": sorted(analytes, key=lambda s: str(s).lower()),
    }


def parse_uploaded_json_files(uploaded_files: Any) -> Dict[str, Any]:
    """Parse Streamlit UploadedFile objects into the same dataset structure used for folders."""
    readable: Dict[str, dict] = {}
    unreadable: Dict[str, str] = {}
    analytes = set()
    file_names: List[str] = []

    if not uploaded_files:
        return {"readable": readable, "unreadable": unreadable, "all_analytes": [], "files": []}

    for uploaded in uploaded_files[:100]:
        fn = safe_text(getattr(uploaded, "name", None), "uploaded_file.json")
        original_fn = fn
        duplicate_counter = 2
        while fn in readable or fn in unreadable or fn in file_names:
            stem, ext = os.path.splitext(original_fn)
            fn = f"{stem} ({duplicate_counter}){ext or '.json'}"
            duplicate_counter += 1
        file_names.append(fn)

        try:
            raw = uploaded.getvalue()
            text = raw.decode("utf-8-sig")
            data = json.loads(text)
        except Exception as e:
            unreadable[fn] = str(e)
            continue

        if not isinstance(data, dict):
            unreadable[fn] = "Top-level JSON value is not an object"
            continue

        readable[fn] = data
        aqm = data.get("analyte_quad_map", {})
        if isinstance(aqm, dict):
            analytes.update(aqm.keys())

    return {
        "readable": readable,
        "unreadable": unreadable,
        "all_analytes": sorted(analytes, key=lambda s: str(s).lower()),
        "files": file_names,
    }


def get_active_dataset() -> Tuple[str, List[str], Dict[str, Any], List[str]]:
    """Return active dataset source label, file list, dataset dict, and readable file list."""
    source = st.session_state.get("dataset_source", "Upload JSON files")

    if source == "Upload JSON files":
        ds = st.session_state.get("uploaded_dataset", {"readable": {}, "unreadable": {}, "all_analytes": [], "files": []})
        files = list(ds.get("files", []))
        unreadable = ds.get("unreadable", {}) if isinstance(ds, dict) else {}
        readable_files = [f for f in files if f not in unreadable]
        return source, files, ds, readable_files

    folder = st.session_state.get("selected_folder", "")
    files = st.session_state.file_list or (list_json_files(folder) if folder and os.path.isdir(folder) else [])
    ds = load_dataset(folder, files) if folder and os.path.isdir(folder) else {"readable": {}, "unreadable": {}, "all_analytes": []}
    unreadable = ds.get("unreadable", {})
    readable_files = [f for f in files if f not in unreadable]
    return source, files, ds, readable_files


def active_dataset_ready() -> bool:
    source, files, ds, readable_files = get_active_dataset()
    if source == "Upload JSON files":
        return bool(files)
    folder = st.session_state.get("selected_folder", "")
    return bool(folder and os.path.isdir(folder))


def active_dataset_caption() -> str:
    source = st.session_state.get("dataset_source", "Upload JSON files")
    if source == "Upload JSON files":
        files = st.session_state.get("uploaded_dataset", {}).get("files", [])
        return f"Dataset source: Uploaded files ({len(files)} file{'s' if len(files) != 1 else ''})"
    folder = st.session_state.get("selected_folder", "")
    return f"Dataset source: Local folder — {folder}" if folder else "Dataset source: Local folder"


def uploaded_dataset_has_files() -> bool:
    """Return True when an uploaded dataset has already been parsed and cached in session state."""
    ds = st.session_state.get("uploaded_dataset", {})
    return bool(isinstance(ds, dict) and ds.get("files"))


def get_json_from_active_dataset(filename: str, folder: Optional[str] = None, data_by_name: Optional[Dict[str, dict]] = None) -> Optional[dict]:
    """Load JSON by filename from uploaded data or from a local folder."""
    if isinstance(data_by_name, dict) and filename in data_by_name:
        return data_by_name.get(filename)
    if folder:
        data, _ = load_json(os.path.join(folder, filename))
        return data
    return None


def pick_first_qualifier(obj: dict) -> Optional[dict]:
    quals = obj.get("qualifiers", None)
    if isinstance(quals, list) and len(quals) > 0 and isinstance(quals[0], dict):
        return quals[0]
    return None


def channel_from_quad(quad: dict, which: str, ion: str) -> Optional[dict]:
    """
    which: "analyte" or "internal_standard"
    ion: "quantifier" or "qualifier"
    Returns a channel dict with {name, raw, peak} etc or None.
    """
    parent = quad.get(which, None)
    if not isinstance(parent, dict):
        return None

    if ion == "quantifier":
        return parent.get("quantifier", None) if isinstance(parent.get("quantifier", None), dict) else None

    # qualifier: take first qualifier entry
    q = pick_first_qualifier(parent)
    return q


def normalize_to_percent(y: List[float]) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    mx = np.nanmax(arr) if arr.size else np.nan
    if not np.isfinite(mx) or mx <= 0:
        return np.zeros_like(arr)
    return (arr / mx) * 100.0


def extract_xy(trace_obj: Optional[dict]) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(trace_obj, dict):
        return np.array([]), np.array([])
    rt = trace_obj.get("retention_times", [])
    inten = trace_obj.get("intensities", [])
    try:
        x = np.asarray(rt, dtype=float)
        y = np.asarray(inten, dtype=float)
        return x, y
    except Exception:
        return np.array([]), np.array([])


def trace_obj_from_channel(ch: Optional[dict], trace_mode: str) -> Optional[dict]:
    """Return the selected display trace object for a channel."""
    if not isinstance(ch, dict):
        return None

    mode = str(trace_mode).strip().lower()
    key = "smoothed" if mode == "smoothed" else "raw"
    obj = ch.get(key, None)
    return obj if isinstance(obj, dict) else None


def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    try:
        if a is None or b is None:
            return None
        a = float(a)
        b = float(b)
        if b == 0.0 or a == 0.0:
            # Requirement: if either response is missing/null/zero -> None
            return None
        return a / b
    except Exception:
        return None


def get_peak_values(ch: Optional[dict]) -> Tuple[Optional[float], Optional[float]]:
    if not isinstance(ch, dict):
        return None, None
    peak = ch.get("peak", None)
    if not isinstance(peak, dict):
        return None, None
    return peak.get("response", None), peak.get("rt", None)


def canonicalize_transition_key(val: Any) -> Optional[str]:
    """Normalize transition representations like '143 > 70.9' and '143.0>70.90'."""
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    s = s.replace("→", ">")
    s = " ".join(s.split())

    parts = [p.strip() for p in s.split(">")]
    if len(parts) != 2:
        compact = s.replace(" ", "")
        parts = [p.strip() for p in compact.split(">")]
        if len(parts) != 2:
            return compact or None

    def norm_num(token: str) -> str:
        try:
            num = float(token)
            if not math.isfinite(num):
                return token.replace(" ", "")
            return format(num, ".15g")
        except Exception:
            return token.replace(" ", "")

    return f"{norm_num(parts[0])}>{norm_num(parts[1])}"



def transition_candidates_from_channel(ch: Optional[dict]) -> List[str]:
    """Build candidate transition keys from name and mz values for tolerant matching."""
    out: List[str] = []
    if not isinstance(ch, dict):
        return out

    trans = ch.get("transition", {})
    if not isinstance(trans, dict):
        return out

    name = trans.get("name", None)
    key = canonicalize_transition_key(name)
    if key:
        out.append(key)

    parent = trans.get("parent_mz", None)
    product = trans.get("product_mz", None)
    if parent is not None and product is not None:
        key = canonicalize_transition_key(f"{parent}>{product}")
        if key:
            out.append(key)

    # de-duplicate while preserving order
    deduped: List[str] = []
    seen = set()
    for item in out:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped



def extract_expected_ion_ratio_from_efficiencies(quad: Optional[dict], which: str) -> Optional[float]:
    """
    Calculate expected ion ratio as qualifier transition efficiency divided by
    quantifier transition efficiency for the requested species.
    which: "analyte" or "internal_standard"

    Transition efficiencies are sourced only from:
    quad["calibrator"]["transition_efficiencies"]
    """
    if not isinstance(quad, dict):
        return None

    calibrator = quad.get("calibrator", {})
    if not isinstance(calibrator, dict):
        return None

    te = calibrator.get("transition_efficiencies", {})
    if not isinstance(te, dict) or not te:
        return None

    te_species = te.get(which, {})
    if not isinstance(te_species, dict) or not te_species:
        return None

    normalized_efficiencies: Dict[str, float] = {}
    for raw_key, raw_val in te_species.items():
        norm_key = canonicalize_transition_key(raw_key)
        if not norm_key:
            continue
        try:
            val = float(raw_val)
            if math.isfinite(val):
                normalized_efficiencies[norm_key] = val
        except Exception:
            continue

    if not normalized_efficiencies:
        return None

    q_ch = channel_from_quad(quad, which, "quantifier")
    l_ch = channel_from_quad(quad, which, "qualifier")

    q_candidates = transition_candidates_from_channel(q_ch)
    l_candidates = transition_candidates_from_channel(l_ch)

    q_eff = None
    for key in q_candidates:
        if key in normalized_efficiencies:
            q_eff = normalized_efficiencies[key]
            break

    l_eff = None
    for key in l_candidates:
        if key in normalized_efficiencies:
            l_eff = normalized_efficiencies[key]
            break

    if q_eff is None or l_eff is None:
        return None

    if not math.isfinite(q_eff) or not math.isfinite(l_eff) or q_eff == 0.0:
        return None

    return l_eff / q_eff

def channel_display_title(which: str, ion: str, ch: Optional[dict]) -> str:
    base_title = CHANNEL_STYLES[(which, ion)]["title"]
    if isinstance(ch, dict):
        trans = ch.get("transition", {})
        if isinstance(trans, dict):
            nm = trans.get("name", None)
            if nm:
                return f"{base_title}: {nm}"
    return base_title


def normalize_method_value(val: Any) -> Any:
    """Recursively format numbers for display while preserving nested structure."""
    if isinstance(val, dict):
        return {k: normalize_method_value(v) for k, v in val.items()}
    if isinstance(val, list):
        return [normalize_method_value(v) for v in val]
    if isinstance(val, (int, float)):
        try:
            return float(format_max_3dp(val))
        except Exception:
            return format_max_3dp(val)
    return val


def format_method_value(val: Any) -> str:
    """Convert method values to compact display strings."""
    if val is None:
        return "None"
    if isinstance(val, (int, float)):
        return format_max_3dp(val)
    if isinstance(val, str):
        return val
    try:
        compact = json.dumps(normalize_method_value(val), ensure_ascii=False, separators=(", ", ": "))
        return compact
    except Exception:
        return str(val)


def flatten_method_section(section: Any, prefix: str = "") -> Dict[str, str]:
    """
    Flatten a nested method subsection into dotted parameter paths.
    Lists are kept as compact JSON strings to avoid exploding the table.
    """
    out: Dict[str, str] = {}

    if isinstance(section, dict):
        for key, value in section.items():
            new_prefix = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                out.update(flatten_method_section(value, new_prefix))
            else:
                out[new_prefix] = format_method_value(value)
        return out

    if prefix:
        out[prefix] = format_method_value(section)
    else:
        out["value"] = format_method_value(section)
    return out


def build_method_comparison_tables(methods_obj: Dict[str, Any]) -> Tuple[List[Tuple[str, pd.DataFrame]], Dict[str, int]]:
    analyte_methods = methods_obj.get("analyte", {}) if isinstance(methods_obj, dict) else {}
    istd_methods = methods_obj.get("internal_standard", {}) if isinstance(methods_obj, dict) else {}

    section_names = sorted(set(analyte_methods.keys()) | set(istd_methods.keys()), key=str)
    tables: List[Tuple[str, pd.DataFrame]] = []

    identical = 0
    different = 0
    missing = 0
    total = 0

    for section in section_names:
        a_flat = flatten_method_section(analyte_methods.get(section, None), "")
        i_flat = flatten_method_section(istd_methods.get(section, None), "")

        param_names = sorted(set(a_flat.keys()) | set(i_flat.keys()), key=str)
        rows = []
        for param in param_names:
            a_val = a_flat.get(param, "None")
            i_val = i_flat.get(param, "None")
            is_missing = ("None" in (a_val, i_val)) and (a_val != i_val)
            is_diff = a_val != i_val

            total += 1
            if is_missing:
                missing += 1
            elif is_diff:
                different += 1
            else:
                identical += 1

            rows.append(
                {
                    "Parameter": param,
                    "Analyte": a_val,
                    "Internal Standard": i_val,
                    "Match": "Different" if is_diff else "Same",
                }
            )

        tables.append((section, pd.DataFrame(rows)))

    summary = {
        "total": total,
        "identical": identical,
        "different": different,
        "missing": missing,
    }
    return tables, summary


def style_method_comparison(df: pd.DataFrame):
    def row_style(row):
        if row["Match"] == "Different":
            return ["background-color: rgba(255, 236, 179, 0.45)"] * len(row)
        return [""] * len(row)

    return (
        df.style
        .hide(axis="index")
        .apply(row_style, axis=1)
    )



BASELINE_COLOR = "#FF00FF"
RAW_TRACE_COLOR = "#B8BDC7"

CHANNEL_STYLES = {
    ("analyte", "quantifier"): {
        "light": "#9FC9D7",
        "dark": "#0F6D8C",
        "title": "Analyte quantifier",
    },
    ("analyte", "qualifier"): {
        "light": "#C9D7A3",
        "dark": "#5E7F18",
        "title": "Analyte qualifier",
    },
    ("internal_standard", "quantifier"): {
        "light": "#E3AAEC",
        "dark": "#7C2A8A",
        "title": "Internal standard quantifier",
    },
    ("internal_standard", "qualifier"): {
        "light": "#F4C9A6",
        "dark": "#B35B1E",
        "title": "Internal standard qualifier",
    },
}


def extract_summary_value(quad: Optional[dict], which: str, ion: str, parameter: str) -> Optional[float]:
    """Extract response or retention time from a selected channel in a quad."""
    if not isinstance(quad, dict):
        return None

    ch = channel_from_quad(quad, which, ion)
    if not isinstance(ch, dict):
        return None

    peak = ch.get("peak", None)
    if not isinstance(peak, dict):
        return None

    field_map = {
        "Response": "response",
        "Retention time": "rt",
    }
    field = field_map.get(parameter)
    if field is None:
        return None

    val = peak.get(field, None)
    try:
        if val is None:
            return None
        val = float(val)
        if not math.isfinite(val):
            return None
        return val
    except Exception:
        return None


def summary_channel_specs() -> List[Tuple[str, str]]:
    return [
        ("analyte", "quantifier"),
        ("analyte", "qualifier"),
        ("internal_standard", "quantifier"),
        ("internal_standard", "qualifier"),
    ]


def summary_channel_label(which: str, ion: str) -> str:
    return CHANNEL_STYLES[(which, ion)]["title"]


def build_summary_records(
    readable_files: List[str],
    folder: Optional[str],
    analyte: Optional[str],
    parameter: str,
    data_by_name: Optional[Dict[str, dict]] = None,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for seq, fn in enumerate(readable_files, start=1):
        data = get_json_from_active_dataset(fn, folder=folder, data_by_name=data_by_name)
        aqm = data.get("analyte_quad_map", {}) if isinstance(data, dict) else {}
        quad = aqm.get(analyte, None) if isinstance(aqm, dict) and analyte else None

        for which, ion in summary_channel_specs():
            records.append(
                {
                    "Sequence number": seq,
                    "File": fn,
                    "Channel": summary_channel_label(which, ion),
                    "which": which,
                    "ion": ion,
                    "Value": extract_summary_value(quad, which, ion, parameter),
                }
            )
    return records


def build_summary_stats_table(records_df: pd.DataFrame, selected_channels: List[str]) -> pd.DataFrame:
    rows = []
    for channel in selected_channels:
        channel_df = records_df[
            (records_df["Channel"] == channel) & (records_df["Value"].notna())
        ].copy()

        if channel_df.empty:
            rows.append(
                {
                    "Channel": channel,
                    "Mean": "None",
                    "Standard deviation": "None",
                }
            )
            continue

        mean_val = float(channel_df["Value"].mean())
        std_val = float(channel_df["Value"].std(ddof=1)) if len(channel_df) > 1 else 0.0
        rows.append(
            {
                "Channel": channel,
                "Mean": format_max_3dp(mean_val),
                "Standard deviation": format_max_3dp(std_val),
            }
        )

    return pd.DataFrame(rows)


def build_summary_chart(
    records_df: pd.DataFrame,
    selected_channels: List[str],
    parameter: str,
    show_stats_lines: bool,
    show_stats_labels: bool,
) -> go.Figure:
    fig = go.Figure()

    for channel in selected_channels:
        channel_df = records_df[
            (records_df["Channel"] == channel) & (records_df["Value"].notna())
        ].copy()

        if channel_df.empty:
            continue

        first = channel_df.iloc[0]
        color = CHANNEL_STYLES[(first["which"], first["ion"])]["dark"]

        fig.add_trace(
            go.Scatter(
                x=channel_df["Sequence number"],
                y=channel_df["Value"],
                mode="lines+markers",
                name=channel,
                line=dict(color=color, width=2),
                marker=dict(color=color, size=7, symbol="circle"),
                hovertemplate=(
                    "Sequence number=%{x}<br>"
                    + "Channel="
                    + channel
                    + "<br>"
                    + parameter
                    + "=%{y:.3f}<extra></extra>"
                ),
            )
        )

        if show_stats_lines:
            mean_val = float(channel_df["Value"].mean())
            std_val = float(channel_df["Value"].std(ddof=1)) if len(channel_df) > 1 else 0.0

            line_specs = [
                (mean_val, "solid", "Mean"),
                (mean_val + std_val, "dot", "Mean + SD"),
                (mean_val - std_val, "dot", "Mean - SD"),
            ]
            for yv, dash, label in line_specs:
                fig.add_hline(
                    y=yv,
                    line=dict(color=color, width=1.5, dash=dash),
                    opacity=0.7,
                    annotation_text=(f"{channel} {label}" if show_stats_labels else None),
                    annotation_position="top left",
                )

    fig.update_layout(
        height=500,
        margin=dict(l=40, r=20, t=50, b=40),
        legend_title_text="Channel",
        hovermode="closest",
    )
    fig.update_xaxes(title_text="Sequence number", dtick=1)
    fig.update_yaxes(title_text=parameter)
    return fig


def build_ion_ratio_records(
    readable_files: List[str],
    folder: Optional[str],
    analyte: Optional[str],
    expected_ratio_analyte: Optional[float],
    expected_ratio_istd: Optional[float],
    data_by_name: Optional[Dict[str, dict]] = None,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for seq, fn in enumerate(readable_files, start=1):
        data = get_json_from_active_dataset(fn, folder=folder, data_by_name=data_by_name)
        aqm = data.get("analyte_quad_map", {}) if isinstance(data, dict) else {}
        quad = aqm.get(analyte, None) if isinstance(aqm, dict) and analyte else None

        for which, label in [("analyte", "Analyte"), ("internal_standard", "Internal Standard")]:
            q = channel_from_quad(quad, which, "quantifier") if isinstance(quad, dict) else None
            l = channel_from_quad(quad, which, "qualifier") if isinstance(quad, dict) else None
            q_resp, _ = get_peak_values(q)
            l_resp, _ = get_peak_values(l)

            try:
                q_resp = float(q_resp) if q_resp is not None else None
                if q_resp is not None and not math.isfinite(q_resp):
                    q_resp = None
            except Exception:
                q_resp = None

            try:
                l_resp = float(l_resp) if l_resp is not None else None
                if l_resp is not None and not math.isfinite(l_resp):
                    l_resp = None
            except Exception:
                l_resp = None

            derived_expected_ratio = extract_expected_ion_ratio_from_efficiencies(quad, which)
            manual_expected_ratio = expected_ratio_analyte if which == "analyte" else expected_ratio_istd

            records.append(
                {
                    "Sequence number": seq,
                    "File": fn,
                    "Species": label,
                    "Quantifier response": q_resp,
                    "Qualifier response": l_resp,
                    "Observed ion ratio": safe_div(l_resp, q_resp),
                    "Expected ion ratio": manual_expected_ratio if manual_expected_ratio is not None else derived_expected_ratio,
                    "Derived expected ion ratio": derived_expected_ratio,
                }
            )
    return records


def linear_fit_equation(x: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, float]]:
    if len(x) < 2 or len(y) < 2:
        return None
    try:
        m, b = np.polyfit(x, y, 1)
        if not (math.isfinite(m) and math.isfinite(b)):
            return None
        return float(m), float(b)
    except Exception:
        return None


def build_ion_ratio_scatter(
    df_species: pd.DataFrame,
    species_label: str,
    marker_color: str,
    show_trendline: bool,
) -> Tuple[go.Figure, Optional[str]]:
    fig = go.Figure()
    plot_df = df_species[
        df_species["Quantifier response"].notna() &
        df_species["Qualifier response"].notna()
    ].copy()

    if not plot_df.empty:
        fig.add_trace(
            go.Scatter(
                x=plot_df["Quantifier response"],
                y=plot_df["Qualifier response"],
                mode="markers",
                name=species_label,
                marker=dict(color=marker_color, size=8, symbol="circle"),
                hovertemplate=(
                    "Sequence number=%{customdata[0]}<br>"
                    "Quantifier response=%{x:.3f}<br>"
                    "Qualifier response=%{y:.3f}<br>"
                    "Observed ion ratio=%{customdata[1]:.3f}<extra></extra>"
                ),
                customdata=plot_df[["Sequence number", "Observed ion ratio"]].to_numpy(),
            )
        )

    eqn_text = None

    expected_vals = df_species["Expected ion ratio"].dropna()
    if not plot_df.empty and not expected_vals.empty:
        expected_ratio = float(expected_vals.iloc[0])
        x_max = float(plot_df["Quantifier response"].max())
        x_line = np.array([0.0, x_max])
        y_line = expected_ratio * x_line
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="Expected ion ratio",
                line=dict(color=marker_color, width=2, dash="dot"),
                hoverinfo="skip",
            )
        )

    if show_trendline and len(plot_df) >= 2:
        x = plot_df["Quantifier response"].to_numpy(dtype=float)
        y = plot_df["Qualifier response"].to_numpy(dtype=float)
        fit = linear_fit_equation(x, y)
        if fit is not None:
            m, b = fit
            x_line = np.linspace(float(x.min()), float(x.max()), 100)
            y_line = m * x_line + b
            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name="Trendline",
                    line=dict(color="rgba(0,0,0,0.65)", width=2),
                    hoverinfo="skip",
                )
            )
            eqn_text = f"y = {format_max_3dp(m)}x + {format_max_3dp(b)}"

    fig.update_layout(
        height=420,
        margin=dict(l=40, r=20, t=50, b=40),
        legend_title_text="Series",
        hovermode="closest",
        title=species_label,
    )
    fig.update_xaxes(title_text="Quantifier Response")
    fig.update_yaxes(title_text="Qualifier Response")
    return fig, eqn_text



def interpolate_crossing(x1: float, y1: float, x2: float, y2: float, target: float) -> Optional[float]:
    try:
        x1 = float(x1); y1 = float(y1); x2 = float(x2); y2 = float(y2); target = float(target)
        if not all(math.isfinite(v) for v in [x1, y1, x2, y2, target]):
            return None
        if y1 == y2:
            return None
        frac = (target - y1) / (y2 - y1)
        return x1 + frac * (x2 - x1)
    except Exception:
        return None


def extract_peak_shape_metrics(quad: Optional[dict], which: str, ion: str) -> Dict[str, Optional[float]]:
    """Return RT, FWHM and asymmetry for a selected channel."""
    out = {"RT": None, "FWHM": None, "Asymmetry": None}
    if not isinstance(quad, dict):
        return out

    ch = channel_from_quad(quad, which, ion)
    if not isinstance(ch, dict):
        return out

    peak = ch.get("peak", None)
    if not isinstance(peak, dict):
        return out

    rt = peak.get("rt", None)
    try:
        if rt is not None:
            rt = float(rt)
            if math.isfinite(rt):
                out["RT"] = rt
    except Exception:
        out["RT"] = None

    apex = peak.get("apex", None)
    if not isinstance(apex, dict):
        return out

    xs = apex.get("retention_times", [])
    ys = apex.get("intensities", [])
    try:
        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)
    except Exception:
        return out

    if x.size < 3 or y.size < 3 or x.size != y.size:
        return out
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return out

    peak_max = float(np.max(y))
    if not math.isfinite(peak_max) or peak_max <= 0:
        return out

    base = peak.get("baseline", None)
    if not isinstance(base, dict):
        return out

    base_intensities = base.get("intensities", [])
    try:
        base_y = np.asarray(base_intensities, dtype=float)
    except Exception:
        return out

    if base_y.size < 2 or not np.all(np.isfinite(base_y)):
        return out

    baseline_start_intensity = float(base_y[0])
    baseline_end_intensity = float(base_y[-1])
    baseline_mean_intensity = (baseline_start_intensity + baseline_end_intensity) / 2.0
    if not math.isfinite(baseline_mean_intensity):
        return out

    max_idx = int(np.argmax(y))

    def find_crossings(target_level: float) -> Tuple[Optional[float], Optional[float]]:
        t_left_local = None
        for i in range(max_idx - 1, -1, -1):
            y1 = float(y[i])
            y2 = float(y[i + 1])
            if (y1 <= target_level <= y2) or (y1 >= target_level >= y2):
                t_left_local = interpolate_crossing(float(x[i]), y1, float(x[i + 1]), y2, target_level)
                if t_left_local is not None:
                    break

        t_right_local = None
        for i in range(max_idx, len(y) - 1):
            y1 = float(y[i])
            y2 = float(y[i + 1])
            if (y1 >= target_level >= y2) or (y1 <= target_level <= y2):
                t_right_local = interpolate_crossing(float(x[i]), y1, float(x[i + 1]), y2, target_level)
                if t_right_local is not None:
                    break

        return t_left_local, t_right_local

    half_height_level = baseline_mean_intensity + ((peak_max - baseline_mean_intensity) / 2.0)
    t_left_half, t_right_half = find_crossings(half_height_level)

    if t_left_half is not None and t_right_half is not None:
        fwhm = t_right_half - t_left_half
        if math.isfinite(fwhm) and fwhm > 0:
            out["FWHM"] = float(fwhm)

    five_percent_height_level = baseline_mean_intensity + ((peak_max - baseline_mean_intensity) / 20.0)
    t_left_5, t_right_5 = find_crossings(five_percent_height_level)

    rt_for_asym = out["RT"]
    if rt_for_asym is not None and t_left_5 is not None and t_right_5 is not None:
        left_width = rt_for_asym - t_left_5
        right_width = t_right_5 - rt_for_asym
        if all(math.isfinite(v) for v in [left_width, right_width]) and left_width > 0 and right_width >= 0:
            out["Asymmetry"] = float(right_width / left_width)

    return out


def build_peak_shape_records(readable_files: List[str], folder: Optional[str], analyte: Optional[str], parameter: str, data_by_name: Optional[Dict[str, dict]] = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for seq, fn in enumerate(readable_files, start=1):
        data = get_json_from_active_dataset(fn, folder=folder, data_by_name=data_by_name)
        aqm = data.get("analyte_quad_map", {}) if isinstance(data, dict) else {}
        quad = aqm.get(analyte, None) if isinstance(aqm, dict) and analyte else None

        concentration = None
        if isinstance(quad, dict):
            try:
                concentration = float(quad.get("concentration", None))
                if not math.isfinite(concentration):
                    concentration = None
            except Exception:
                concentration = None

        for which, ion in [
            ("analyte", "quantifier"),
            ("analyte", "qualifier"),
            ("internal_standard", "quantifier"),
            ("internal_standard", "qualifier"),
        ]:
            metrics = extract_peak_shape_metrics(quad, which, ion)
            records.append({
                "Sequence number": seq,
                "File": fn,
                "Channel": summary_channel_label(which, ion),
                "which": which,
                "ion": ion,
                "Concentration": concentration,
                "Value": metrics.get(parameter),
            })
    return records


def build_peak_shape_chart(records_df: pd.DataFrame, selected_channels: List[str], parameter: str) -> go.Figure:
    fig = go.Figure()
    for channel in selected_channels:
        channel_df = records_df[(records_df["Channel"] == channel) & records_df["Concentration"].notna() & records_df["Value"].notna()].copy()
        if channel_df.empty:
            continue
        first = channel_df.iloc[0]
        color = CHANNEL_STYLES[(first["which"], first["ion"])]['dark']
        hover_fmt = ".5f" if parameter in ("FWHM", "RT") else ".3f"
        fig.add_trace(go.Scatter(
            x=channel_df["Concentration"],
            y=channel_df["Value"],
            mode="markers",
            name=channel,
            line=dict(color=color, width=2),
            marker=dict(color=color, size=7, symbol="circle"),
            hovertemplate=(
                "Concentration=%{x:.3f}<br>" +
                f"Channel={channel}<br>" +
                parameter + "=%{y" + hover_fmt + "}<extra></extra>"
            ),
        ))
    fig.update_layout(height=500, margin=dict(l=40, r=20, t=50, b=40), legend_title_text="Channel", hovermode="closest")
    fig.update_xaxes(title_text="Concentration")
    fig.update_yaxes(title_text=(f"{parameter} (mins)" if parameter in ("FWHM", "RT") else parameter))
    return fig



def format_calibration_equation(fit_type: Any, coefficients: Any) -> str:
    """Return a display string for the calibration equation."""
    if not isinstance(coefficients, (list, tuple)):
        return "Not available"
    try:
        vals = [float(v) for v in coefficients]
    except Exception:
        return "Not available"

    fit = str(fit_type).strip().lower()
    if fit == "linear" and len(vals) >= 2:
        c, m = vals[0], vals[1]
        return f"y = {m:.6g}x + {c:.6g}"
    if fit == "quadratic" and len(vals) >= 3:
        a, b, c = vals[0], vals[1], vals[2]
        return f"y = {a:.6g}x² + {b:.6g}x + {c:.6g}"
    return "Not available"


def evaluate_calibration_curve(fit_type: Any, coefficients: Any, x_values: np.ndarray) -> Optional[np.ndarray]:
    """Evaluate supported calibration models for plotting the fitted line."""
    if not isinstance(coefficients, (list, tuple)):
        return None
    try:
        vals = [float(v) for v in coefficients]
    except Exception:
        return None

    fit = str(fit_type).strip().lower()
    try:
        x = np.asarray(x_values, dtype=float)
        if fit == "linear" and len(vals) >= 2:
            c, m = vals[0], vals[1]
            y = m * x + c
            return y if np.all(np.isfinite(y)) else None
        if fit == "quadratic" and len(vals) >= 3:
            a, b, c = vals[0], vals[1], vals[2]
            y = a * x * x + b * x + c
            return y if np.all(np.isfinite(y)) else None
    except Exception:
        return None
    return None


def build_calibration_plot(calibrator: Optional[dict]) -> Tuple[Optional[go.Figure], Dict[str, str]]:
    """Build the calibration scatter/fit plot and metadata for display."""
    meta = {
        "equation": "Not available",
        "rsquared": "None",
        "fit_type": "None",
        "weight_function": "None",
    }
    if not isinstance(calibrator, dict):
        return None, meta

    fit_type = None
    settings = calibrator.get("settings", {})
    if isinstance(settings, dict):
        fit_type = settings.get("fit_type", None)
        meta["fit_type"] = str(fit_type) if fit_type is not None and str(fit_type).strip() else "None"
        weight = settings.get("weight_function", None)
        meta["weight_function"] = str(weight) if weight is not None and str(weight).strip() else "None"

    rsq = calibrator.get("rsquared", None)
    try:
        if rsq is not None:
            rsq = float(rsq)
            meta["rsquared"] = f"{rsq:.5f}"
    except Exception:
        pass

    coeffs = calibrator.get("coefficients", None)
    meta["equation"] = format_calibration_equation(fit_type, coeffs)

    conc_raw = calibrator.get("concentration", [])
    rr_raw = calibrator.get("response_ratio", [])
    try:
        conc = np.asarray(conc_raw, dtype=float)
        rr = np.asarray(rr_raw, dtype=float)
    except Exception:
        return None, meta

    if conc.size == 0 or rr.size == 0 or conc.size != rr.size:
        return None, meta

    mask = np.isfinite(conc) & np.isfinite(rr)
    conc = conc[mask]
    rr = rr[mask]
    if conc.size == 0 or rr.size == 0:
        return None, meta

    x_max = float(np.max(conc)) if conc.size else 0.0
    y_max = float(np.max(rr)) if rr.size else 0.0
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=conc,
            y=rr,
            mode="markers",
            name="Calibration points",
            marker=dict(color=CHANNEL_STYLES[("analyte", "quantifier")]["dark"], size=9, symbol="x"),
            hovertemplate="Concentration=%{x:.3f}<br>Response Ratio=%{y:.3f}<extra></extra>",
        )
    )

    if conc.size >= 2:
        x_line = np.linspace(0.0, max(x_max * 1.1, 1e-12), 200)
        y_line = evaluate_calibration_curve(fit_type, coeffs, x_line)
        if y_line is not None:
            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name="Fitted line",
                    line=dict(color=CHANNEL_STYLES[("analyte", "qualifier")]["dark"], width=2),
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        height=360,
        margin=dict(l=40, r=20, t=40, b=40),
        legend_title_text="Series",
        hovermode="closest",
    )
    fig.update_xaxes(title_text="Concentration", range=[0.0, max(x_max * 1.1, 1.0)])
    fig.update_yaxes(title_text="Response Ratio", range=[0.0, max(y_max * 1.1, 1.0)])
    return fig, meta


def render_concern_cards(concerns: Any, empty_message: str = "No concerns to display.") -> None:
    """Render concern cards using the shared display logic."""
    non_accept = []
    if isinstance(concerns, list):
        for c in concerns:
            if isinstance(c, dict) and normalize_concern_category(c.get("category", None)) != "ACCEPT":
                non_accept.append(c)

    if not non_accept:
        st.write(empty_message)
        return

    for c in non_accept:
        raw_cat = c.get("category", None)
        cat = normalize_concern_category(raw_cat)
        col = concern_color(cat)

        stage = format_max_3dp(c.get("stage", ""))
        cid = format_max_3dp(c.get("id", ""))

        en = ""
        tr = c.get("translations", {})
        if isinstance(tr, dict):
            en = tr.get("en", "") or ""

        meta_parts = []
        if stage:
            meta_parts.append(f"Stage: <b>{stage}</b>")
        if cid:
            meta_parts.append(f"ID: <b>{cid}</b>")
        meta_html = " &nbsp;&nbsp; ".join(meta_parts)

        st.markdown(
            f"""
<div style="
  border:1px solid rgba(0,0,0,0.08);
  border-left:6px solid {col};
  border-radius:12px;
  padding:10px 12px;
  margin:8px 0;
  background: rgba(255,255,255,0.6);
">
  <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
    <span style="
      display:inline-block;
      padding:3px 10px;
      border-radius:999px;
      border:1px solid {col};
      color:{col};
      font-weight:800;
      letter-spacing:0.4px;
      font-size:12px;
    ">{cat}</span>
    <span style="color:rgba(0,0,0,0.65); font-size:12px;">{meta_html}</span>
  </div>
  <div style="margin-top:6px; font-size:13px; line-height:1.35;">
    {en if en else "<span style='color:rgba(0,0,0,0.45);'>No description provided.</span>"}
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

def extract_consensus_rt(quad: Optional[dict], which: str) -> Optional[float]:
    """Return the consensus RT for analyte or internal standard from the calibrator section."""
    if not isinstance(quad, dict):
        return None

    calibrator = quad.get("calibrator", {})
    if not isinstance(calibrator, dict):
        return None

    consensus_rts = calibrator.get("consensus_rts", {})
    if not isinstance(consensus_rts, dict):
        return None

    value = consensus_rts.get(which, None)
    try:
        if value is None:
            return None
        value = float(value)
        if not math.isfinite(value):
            return None
        return value
    except Exception:
        return None


def basis_norm(half_support: float, order: int = 6) -> Optional[float]:
    """Integral of (1 - x^2)^order over [-half_support, +half_support]."""
    try:
        half_support = float(half_support)
        order = int(order)
        if not math.isfinite(half_support) or half_support <= 0 or order < 0:
            return None

        integral_unit = sum(
            ((-1) ** n) * math.comb(order, n) * 2.0 / (2 * n + 1)
            for n in range(order + 1)
        )
        norm = half_support * integral_unit
        return float(norm) if math.isfinite(norm) and norm > 0 else None
    except Exception:
        return None


def compact_basis(x: Any, half_support: float, order: int = 6, normalize_area: bool = True) -> np.ndarray:
    """
    C++ equivalent of Basis::Value.

    x is offset from coefficient centre:
        x = rt - coeff_time
    """
    try:
        half_support = float(half_support)
        order = int(order)
        x_arr = np.asarray(x, dtype=float)
    except Exception:
        return np.array([])

    y = np.zeros_like(x_arr, dtype=float)
    if not math.isfinite(half_support) or half_support <= 0 or order < 0:
        return y

    mask = np.abs(x_arr) <= half_support
    if np.any(mask):
        u = x_arr[mask] / half_support
        y[mask] = (1.0 - u**2) ** order

    if normalize_area:
        norm = basis_norm(half_support, order)
        if norm is None:
            return np.zeros_like(x_arr, dtype=float)
        y = y / norm

    return y


def hwhm_to_half_support(hwhm: float, order: int = 6) -> Optional[float]:
    """Convert basis HWHM to compact-support half-width for (1 - u^2)^order."""
    try:
        hwhm = float(hwhm)
        order = int(order)
        if not math.isfinite(hwhm) or hwhm <= 0 or order <= 0:
            return None
        denominator = math.sqrt(1.0 - math.pow(0.5, 1.0 / order))
        if denominator <= 0 or not math.isfinite(denominator):
            return None
        return hwhm / denominator
    except Exception:
        return None


def peak_model_half_support(peak_model: dict, order: int = 6) -> Optional[float]:
    """
    Return compact-support half-width for the expected peak model.

    The model field basis_half_support is interpreted as HWHM and converted to
    the compact-support half-width used by compact_basis. If a future model
    explicitly provides basis_half_width or compact_half_support, those values
    are used directly.
    """
    if not isinstance(peak_model, dict):
        return None

    for key in ("compact_half_support", "basis_half_width"):
        try:
            if key in peak_model:
                val = float(peak_model.get(key))
                if math.isfinite(val) and val > 0:
                    return val
        except Exception:
            pass

    try:
        hwhm = float(peak_model.get("basis_half_support", None))
    except Exception:
        return None
    return hwhm_to_half_support(hwhm, order=order)


def reconstruct_expected_peak(
    peak_model: dict,
    n_points: int = 1000,
    normalize_peak: bool = True,
    order: int = 6,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Reconstruct an expected chromatographic peak from an area-normalized
    compact-support basis model.

    Each mini-peak is calculated as:
        coeff_value * compact_basis(rt - coeff_time, half_support)

    basis_half_support from the model is interpreted as HWHM and converted to
    the compact-support half-width before evaluating the basis. The summed peak
    and mini-peaks are peak-normalized to 100% for plotting on the normalized
    chromatogram y-axis. Invalid models return (None, None, None).
    """
    if not isinstance(peak_model, dict):
        return None, None, None

    try:
        order = int(order)
        support = peak_model_half_support(peak_model, order=order)
        if support is None or not math.isfinite(support) or support <= 0:
            return None, None, None

        coeff_times = np.asarray(peak_model.get("coeff_times", []), dtype=float)
        coeff_values = np.asarray(peak_model.get("coeff_values", []), dtype=float)
    except Exception:
        return None, None, None

    if coeff_times.size == 0 or coeff_values.size == 0:
        return None, None, None
    if coeff_times.size != coeff_values.size:
        return None, None, None
    if not np.all(np.isfinite(coeff_times)) or not np.all(np.isfinite(coeff_values)):
        return None, None, None

    try:
        rt_min = float(peak_model.get("left_rt_limit", np.min(coeff_times) - support))
        rt_max = float(peak_model.get("right_rt_limit", np.max(coeff_times) + support))
    except Exception:
        rt_min = float(np.min(coeff_times) - support)
        rt_max = float(np.max(coeff_times) + support)

    if not (math.isfinite(rt_min) and math.isfinite(rt_max)) or rt_max <= rt_min:
        rt_min = float(np.min(coeff_times) - support)
        rt_max = float(np.max(coeff_times) + support)

    try:
        rt = np.linspace(rt_min, rt_max, int(n_points))
        mini_peaks = np.array([
            c * compact_basis(rt - t, support, order=order, normalize_area=True)
            for t, c in zip(coeff_times, coeff_values)
        ])

        y_total = mini_peaks.sum(axis=0)
        y_max = float(np.nanmax(y_total)) if y_total.size else np.nan
        if not math.isfinite(y_max) or y_max <= 0:
            return None, None, None

        if normalize_peak:
            y_total = (y_total / y_max) * 100.0
            mini_peaks = (mini_peaks / y_max) * 100.0

        return rt, y_total, mini_peaks
    except Exception:
        return None, None, None


def expected_peak_model_center(rt_model: np.ndarray, y_model: np.ndarray, peak_model: Optional[dict] = None) -> Optional[float]:
    """Return the native centre RT of the reconstructed expected peak model.

    Prefer peak_model["rt_max"] when available because it records the model
    apex/centre in the model coordinate system. Fall back to the reconstructed
    maximum y-position if rt_max is absent or invalid.
    """
    try:
        if isinstance(peak_model, dict) and peak_model.get("rt_max", None) is not None:
            centre = float(peak_model.get("rt_max"))
            if math.isfinite(centre):
                return centre
    except Exception:
        pass

    try:
        if rt_model is None or y_model is None or len(rt_model) == 0 or len(y_model) == 0:
            return None
        rt_arr = np.asarray(rt_model, dtype=float)
        y_arr = np.asarray(y_model, dtype=float)
        if rt_arr.size != y_arr.size or rt_arr.size == 0:
            return None
        if not np.any(np.isfinite(y_arr)):
            return None
        idx = int(np.nanargmax(y_arr))
        centre = float(rt_arr[idx])
        return centre if math.isfinite(centre) else None
    except Exception:
        return None


def shift_expected_peak_to_peak_rt(
    rt_model: np.ndarray,
    y_model: np.ndarray,
    peak_model: Optional[dict],
    observed_peak: Optional[dict],
) -> Optional[np.ndarray]:
    """Shift expected model RTs so the model centre coincides with peak.rt."""
    if rt_model is None or y_model is None or not isinstance(observed_peak, dict):
        return None

    try:
        target_rt = float(observed_peak.get("rt", None))
        if not math.isfinite(target_rt):
            return None
    except Exception:
        return None

    centre_rt = expected_peak_model_center(rt_model, y_model, peak_model)
    if centre_rt is None or not math.isfinite(centre_rt):
        return None

    try:
        return np.asarray(rt_model, dtype=float) + (target_rt - centre_rt)
    except Exception:
        return None

def extract_expected_peak_model(quad: Optional[dict], which: str) -> Optional[dict]:
    """Return calibrator.peak_models.analyte or .internal_standard for expected peak rendering."""
    if not isinstance(quad, dict):
        return None
    calibrator = quad.get("calibrator", {})
    if not isinstance(calibrator, dict):
        return None
    peak_models = calibrator.get("peak_models", {})
    if not isinstance(peak_models, dict):
        return None
    model = peak_models.get(which, None)
    return model if isinstance(model, dict) else None


def build_quad_plot(quad: dict, trace_mode: str = "Smoothed", show_apex: bool = False, show_consensus_rt: bool = True, show_expected_peak: bool = True) -> go.Figure:
    """
    Stacked 4-row chromatogram plot:
      1 analyte quantifier
      2 analyte qualifier
      3 istd quantifier
      4 istd qualifier
    Missing channels show an empty row + annotation.
    """
    rows = 4
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(" ", " ", " ", " "),
    )

    specs = [
        ("analyte", "quantifier"),
        ("analyte", "qualifier"),
        ("internal_standard", "quantifier"),
        ("internal_standard", "qualifier"),
    ]
    row_titles = []

    analyte_consensus_rt = extract_consensus_rt(quad, "analyte")
    istd_consensus_rt = extract_consensus_rt(quad, "internal_standard")
    analyte_expected_peak_model = extract_expected_peak_model(quad, "analyte")
    istd_expected_peak_model = extract_expected_peak_model(quad, "internal_standard")

    for i, (which, ion) in enumerate(specs, start=1):
        style = CHANNEL_STYLES[(which, ion)]
        raw_color = style["light"]
        overlay_color = style["dark"]
        ch = channel_from_quad(quad, which, ion)
        row_titles.append(channel_display_title(which, ion, ch))

        # Selected display trace (smoothed by default; raw optional)
        trace_obj = trace_obj_from_channel(ch, trace_mode)
        x_trace, y_trace = extract_xy(trace_obj)
        y_trace_n = normalize_to_percent(y_trace.tolist()) if y_trace.size else np.array([])

        if x_trace.size:
            fig.add_trace(
                go.Scatter(
                    x=x_trace,
                    y=y_trace_n,
                    mode="lines",
                    name=f"{which}:{ion}:{str(trace_mode).lower()}",
                    #line=dict(width=1.2, color=RAW_TRACE_COLOR),
                    line=dict(width=3, color=overlay_color),
                    opacity=0.7,
                    showlegend=False,
                    hovertemplate="Time=%{x:.3f} min<br>Intensity=%{y:.1f}%<extra></extra>",
                ),
                row=i,
                col=1,
            )
        else:
            fig.add_annotation(
                text="(missing channel data)",
                xref="paper",
                yref="paper",
                x=0.02,
                y=1 - (i - 0.5) / rows,
                showarrow=False,
                font=dict(size=12, color="#666"),
            )


        # Peak overlay (fill -> apex -> baseline/markers)
        peak = ch.get("peak", None) if isinstance(ch, dict) else None
        if isinstance(peak, dict) and x_trace.size:
            apex = peak.get("apex", None)
            base = peak.get("baseline", None)

            x_apex, y_apex = extract_xy(apex)
            x_base, y_base = extract_xy(base)

            # Normalize apex/base on SAME max as raw (since we normalized raw already)
            mx = np.nanmax(y_trace) if y_trace.size else np.nan

            def norm_like_raw(arr: np.ndarray) -> np.ndarray:
                if not arr.size or not np.isfinite(mx) or mx <= 0:
                    return np.zeros_like(arr)
                return (arr / mx) * 100.0

            y_apex_n = norm_like_raw(y_apex)
            y_base_n = norm_like_raw(y_base)

            # 1) Shaded integrated region first (above raw, below apex/baseline)
            if x_base.size >= 2:
                t_min = float(np.nanmin(x_base))
                t_max = float(np.nanmax(x_base))

                mask = (x_trace >= t_min) & (x_trace <= t_max)
                if mask.any():
                    xw = x_trace[mask]
                    yw = y_trace_n[mask]

                    try:
                        yb = np.interp(xw, x_base, y_base_n)
                        fig.add_trace(
                            go.Scatter(
                                x=xw,
                                y=yb,
                                mode="lines",
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo="skip",
                            ),
                            row=i,
                            col=1,
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=xw,
                                y=yw,
                                mode="lines",
                                fill="tonexty",
                                fillcolor=raw_color,
                                line=dict(width=0),
                                opacity=0.22,
                                showlegend=False,
                                hoverinfo="skip",
                            ),
                            row=i,
                            col=1,
                        )
                    except Exception:
                        pass

            # 2) Redraw displayed signal trace on top so it stays full width through filled region
            fig.add_trace(
                go.Scatter(
                    x=x_trace,
                    y=y_trace_n,
                    mode="lines",
                    name=f"{which}:{ion}:{str(trace_mode).lower()}:top",
                    line=dict(width=3, color=overlay_color),
                    opacity=0.7,
                    showlegend=False,
                    hovertemplate="Time=%{x:.3f} min<br>Intensity=%{y:.1f}%<extra></extra>",
                ),
                row=i,
                col=1,
            )

            # 3) Apex line next
            if show_apex and x_apex.size:
                fig.add_trace(
                    go.Scatter(
                        x=x_apex,
                        y=y_apex_n,
                        mode="lines",
                        name=f"{which}:{ion}:apex",
                        line=dict(width=3, color="#FF00FF"),
                        showlegend=False,
                        hovertemplate="Apex<br>Time=%{x:.3f} min<br>Intensity=%{y:.1f}%<extra></extra>",
                    ),
                    row=i,
                    col=1,
                )

            # 4) Baseline line + start/end markers last
            if x_base.size:
                fig.add_trace(
                    go.Scatter(
                        x=x_base,
                        y=y_base_n,
                        mode="lines",
                        name=f"{which}:{ion}:baseline",
                        line=dict(width=3, color=BASELINE_COLOR),
                        showlegend=False,
                        hovertemplate="Baseline<br>Time=%{x:.3f} min<br>Intensity=%{y:.1f}%<extra></extra>",
                    ),
                    row=i,
                    col=1,
                )

                try:
                    t_start = float(np.nanmin(x_base))
                    t_end = float(np.nanmax(x_base))
                    marker_height = 10.0  # normalized channels => 10% of channel height

                    # Marker ticks should sit on the baseline, not start from zero.
                    # Use the normalized baseline intensity at each endpoint and extend upward
                    # by 10 percentage points of the normalized channel height.
                    start_idx = int(np.nanargmin(x_base))
                    end_idx = int(np.nanargmax(x_base))
                    y_start_base = float(y_base_n[start_idx])
                    y_end_base = float(y_base_n[end_idx])

                    fig.add_trace(
                        go.Scatter(
                            x=[t_start, t_start],
                            y=[y_start_base, y_start_base + marker_height],
                            mode="lines",
                            line=dict(color=BASELINE_COLOR, width=3),
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=i,
                        col=1,
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=[t_end, t_end],
                            y=[y_end_base, y_end_base + marker_height],
                            mode="lines",
                            line=dict(color=BASELINE_COLOR, width=3),
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=i,
                        col=1,
                    )
                except Exception:
                    pass

            # 5) Expected peak model overlay (above observed peak overlays, below consensus RT marker)
            expected_peak_model = analyte_expected_peak_model if which == "analyte" else istd_expected_peak_model
            if show_expected_peak and expected_peak_model is not None:
                rt_model, y_model, mini_peaks = reconstruct_expected_peak(expected_peak_model)
                rt_model_shifted = shift_expected_peak_to_peak_rt(rt_model, y_model, expected_peak_model, peak)
                if rt_model_shifted is not None and y_model is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=rt_model_shifted,
                            y=y_model,
                            mode="lines",
                            name=f"{which}:expected_peak",
                            line=dict(color="#444444", width=2.5, dash="dot"),
                            opacity=0.95,
                            showlegend=False,
                            hovertemplate="Expected peak<br>Time=%{x:.3f} min<br>Intensity=%{y:.1f}%<extra></extra>",
                        ),
                        row=i,
                        col=1,
                    )

            # Consensus RT marker last so it remains visible above all other elements
            consensus_rt = analyte_consensus_rt if which == "analyte" else istd_consensus_rt
            if show_consensus_rt and x_trace.size and consensus_rt is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[consensus_rt, consensus_rt],
                        y=[0, 100],
                        mode="lines",
                        name=f"{which}:consensus_rt",
                        line=dict(color="#808080", width=2, dash="dash"),
                        showlegend=False,
                        hovertemplate="Consensus RT<br>Time=%{x:.3f} min<extra></extra>",
                    ),
                    row=i,
                    col=1,
                )

            # Peak label: response (nearest int) + rt (3 dp)
            resp = peak.get("response", None)
            rt = peak.get("rt", None)
            if resp is not None and rt is not None:
                try:
                    #label = f"resp={int(round(float(resp)))}  rt={float(rt):.3f}"
                    label = f"{int(round(float(resp)))}<br>{float(rt):.3f}"
                    # Place label near the apex time at y=90% (keeps it off apex most of the time)
                    fig.add_annotation(
                        x=float(rt) + 0.012,
                        y=96,
                        xref=f"x{i}" if i > 1 else "x",
                        yref=f"y{i}" if i > 1 else "y",
                        text=label,
                        showarrow=False,
                        xanchor="left",
                        yanchor="middle",
                        font=dict(size=11, color=overlay_color),
                    )
                except Exception:
                    pass

    # Axes labels (only once)
    fig.update_xaxes(title_text="Time (mins)", row=rows, col=1)
    fig.update_yaxes(title_text="Intensity (%)", row=2, col=1)  # centered-ish

    fig.update_layout(
        height=760,
        margin=dict(l=40, r=20, t=60, b=40),
    )

    # Update channel titles with transition names and align them to the right
    for ann, key, title in zip(
        fig.layout.annotations[:4],
        [
            ("analyte", "quantifier"),
            ("analyte", "qualifier"),
            ("internal_standard", "quantifier"),
            ("internal_standard", "qualifier"),
        ],
        row_titles,
    ):
        ann.text = title
        ann.font.color = CHANNEL_STYLES[key]["dark"]
        ann.font.size = 16
        ann.x = 1.0
        ann.xanchor = "right"

    return fig


# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="Quad Viewer", layout="wide")

# Initialize session state defaults
defaults = {
    "dataset_source": "Upload JSON files",
    "selected_folder": "",
    "file_list": [],
    "uploaded_dataset": {"readable": {}, "unreadable": {}, "all_analytes": [], "files": []},
    "uploaded_signature": None,
    "selected_file": None,
    "selected_index": 0,
    "selected_analyte": None,
    "page": "Dataset",
    "show_methods": False,
    "page_request": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Handle programmatic navigation requests *before* the page widget is instantiated
if "page_request" not in st.session_state:
    st.session_state.page_request = ""
if st.session_state.page_request:
    st.session_state.page = st.session_state.page_request
    st.session_state.page_request = ""

st.sidebar.title("Quad Viewer")

st.sidebar.radio(
    "Page",
    ["Dataset", "Injection / Quad View", "Summary / Trend", "Ion Ratio Analysis", "Peak Shape Analysis"],
    key="page",
)

st.sidebar.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
render_application_environment_sidebar()


# -----------------------------
# Dataset page
# -----------------------------
if st.session_state.page == "Dataset":
    st.title("Dataset")

    previous_source = st.session_state.get("dataset_source", "Upload JSON files")
    dataset_source = st.radio(
        "Dataset source",
        ["Upload JSON files", "Local folder"],
        key="dataset_source",
        horizontal=True,
        help="Use uploaded files for Streamlit Cloud deployment; use Local folder when running on your own machine.",
    )

    if dataset_source != previous_source:
        st.session_state.selected_file = None
        st.session_state.selected_index = 0
        st.session_state.selected_analyte = None
        st.session_state.show_methods = False

    if dataset_source == "Upload JSON files":
        uploaded_files = st.file_uploader(
            "Upload LC-MS injection JSON files",
            type=["json"],
            accept_multiple_files=True,
            key="json_file_uploader",
            help="Upload up to 100 processed injection JSON files. Uploaded data are cached for this Streamlit session so navigation between pages does not clear the active dataset.",
        )

        upload_signature = tuple(
            (getattr(f, "name", ""), getattr(f, "size", None))
            for f in (uploaded_files or [])
        )

        # Important for Streamlit Cloud: the file_uploader widget itself is not a durable
        # dataset store, especially when navigating between pages. Parse and cache uploaded
        # JSON into session_state, and only replace that cached dataset when the uploader
        # contains a non-empty selection that differs from the previous upload signature.
        if upload_signature and upload_signature != st.session_state.get("uploaded_signature"):
            st.session_state.uploaded_signature = upload_signature
            st.session_state.uploaded_dataset = parse_uploaded_json_files(uploaded_files or [])
            st.session_state.file_list = list(st.session_state.uploaded_dataset.get("files", []))
            st.session_state.selected_file = None
            st.session_state.selected_index = 0
            st.session_state.selected_analyte = None

        ds = st.session_state.uploaded_dataset
        files = list(ds.get("files", []))
        st.caption(active_dataset_caption())

        if uploaded_dataset_has_files():
            st.success("Uploaded dataset is cached for this session. You can move between pages without re-uploading the files.")
            if st.button("Clear uploaded dataset"):
                st.session_state.uploaded_dataset = {"readable": {}, "unreadable": {}, "all_analytes": [], "files": []}
                st.session_state.uploaded_signature = None
                st.session_state.file_list = []
                st.session_state.selected_file = None
                st.session_state.selected_index = 0
                st.session_state.selected_analyte = None
                st.rerun()
        elif not uploaded_files:
            st.info("Upload one or more JSON files to begin. The parsed dataset will be retained while this Streamlit session remains active.")

    else:
        colA, colB = st.columns([3, 1])
        with colA:
            folder = st.text_input(
                "Folder containing JSON injections",
                value=st.session_state.selected_folder,
                placeholder=r"e.g. C:\data\injections  or  /home/user/injections",
            )
        with colB:
            browse_clicked = st.button("Browse (local)")
            if browse_clicked:
                try:
                    import tkinter as tk
                    from tkinter import filedialog
                    root = tk.Tk()
                    root.withdraw()
                    chosen = filedialog.askdirectory()
                    root.destroy()
                    if chosen:
                        folder = chosen
                except Exception:
                    st.warning("Local browse not available in this environment. Paste the folder path instead.")

        folder = folder.strip()
        if folder != st.session_state.selected_folder:
            st.session_state.selected_folder = folder
            st.session_state.file_list = []
            st.session_state.selected_file = None
            st.session_state.selected_index = 0
            st.session_state.selected_analyte = None

        if st.session_state.selected_folder:
            st.caption(active_dataset_caption())

        if st.session_state.selected_folder and os.path.isdir(st.session_state.selected_folder):
            files = list_json_files(st.session_state.selected_folder)
            st.session_state.file_list = files
            ds = load_dataset(st.session_state.selected_folder, files)
        else:
            files = []
            ds = {"readable": {}, "unreadable": {}, "all_analytes": []}

    unreadable = ds.get("unreadable", {}) if isinstance(ds, dict) else {}
    analytes = ds.get("all_analytes", []) if isinstance(ds, dict) else []

    if not files:
        if dataset_source == "Upload JSON files":
            st.info("Upload one or more JSON files to begin.")
        else:
            st.info("Select a folder to begin. (Tip: you can paste a folder path.)")
    else:
        st.subheader("Analytes found")
        if analytes:
            st.write(", ".join(analytes))
            st.session_state.selected_analyte = st.selectbox(
                "Select analyte to inspect",
                options=analytes,
                index=analytes.index(st.session_state.selected_analyte) if st.session_state.selected_analyte in analytes else 0,
            )
        else:
            st.info("No analytes found in readable files.")

        st.divider()
        st.subheader("Injections")

        st.markdown(
            """
            <style>
            div[data-testid="stButton"] > button[kind="tertiary"] {
                justify-content: flex-start;
                text-align: left;
                padding-left: 0.25rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        readable_files = [f for f in files if f not in unreadable]

        with st.container(border=True):
            header_cols = st.columns([0.7, 2.8, 2.2, 1.4])
            header_cols[0].markdown("**#**")
            header_cols[1].markdown("**Injection ID**")
            header_cols[2].markdown("**Sample ID**")
            header_cols[3].markdown("**Injection Status**")

        for seq, fn in enumerate(files, start=1):
            with st.container(border=True):
                row_cols = st.columns([0.7, 2.8, 2.2, 1.4])
                row_cols[0].markdown(f"<div style='padding-top:0.35rem; font-weight:600;'>{seq}</div>", unsafe_allow_html=True)

                if fn in unreadable:
                    row_cols[1].markdown(f"<div style='padding-top:0.35rem; color:#c62828; font-weight:600;'>{fn}</div>", unsafe_allow_html=True)
                    row_cols[2].markdown("<div style='padding-top:0.35rem; color:#616161;'>Unreadable</div>", unsafe_allow_html=True)
                    row_cols[3].markdown("<div style='padding-top:0.35rem; color:#616161;'>UNKNOWN</div>", unsafe_allow_html=True)
                    continue

                data = ds.get("readable", {}).get(fn, {})
                aqm = data.get("analyte_quad_map", {}) if isinstance(data, dict) else {}
                first_quad = next((v for v in aqm.values() if isinstance(v, dict)), None) if isinstance(aqm, dict) else None

                injection_id = os.path.splitext(fn)[0]
                sample_id = ""
                status_txt = "UNKNOWN"

                if isinstance(first_quad, dict):
                    inj_block = first_quad.get("injection", {})
                    if isinstance(inj_block, dict):
                        raw_status = inj_block.get("status", None)
                        status_txt = safe_upper_status(raw_status)

                        result = inj_block.get("result", {})
                        if isinstance(result, dict):
                            injection_id = result.get("id", injection_id) or injection_id
                            sample_info = result.get("sample_info", {})
                            if isinstance(sample_info, dict):
                                sample_id = sample_info.get("sample_id", "") or ""

                if row_cols[1].button(str(injection_id), key=f"open_injection_{dataset_source}_{fn}", type="tertiary", use_container_width=True):
                    st.session_state.selected_file = fn
                    if fn in readable_files:
                        st.session_state.selected_index = readable_files.index(fn)
                    st.session_state.page_request = "Injection / Quad View"
                    st.rerun()

                row_cols[2].markdown(
                    f"<div style='padding-top:0.35rem;'>{sample_id or '&nbsp;'}</div>",
                    unsafe_allow_html=True,
                )
                row_cols[3].markdown(
                    f"<div style='padding-top:0.35rem; color:{status_color(status_txt)}; font-weight:700;'>{status_txt}</div>",
                    unsafe_allow_html=True,
                )

        if unreadable:
            with st.expander("Unreadable files (details)", expanded=False):
                for fn, err in unreadable.items():
                    st.write(f"**{fn}** — {err}")

# -----------------------------
# Ion Ratio Analysis page
# -----------------------------
elif st.session_state.page == "Ion Ratio Analysis":
    st.title("Ion Ratio Analysis")

    source, files, ds, readable_files = get_active_dataset()
    folder = st.session_state.selected_folder if source == "Local folder" else None
    data_by_name = ds.get("readable", {}) if source == "Upload JSON files" else None

    if not active_dataset_ready():
        st.warning("No valid dataset selected. Go to Dataset page first.")
        st.stop()

    if not readable_files:
        st.warning("No readable files available in the active dataset.")
        st.stop()

    analyte = st.session_state.selected_analyte
    if not analyte:
        st.warning("No analyte selected. Go to Dataset page first.")
        st.stop()

    st.caption(active_dataset_caption())
    st.caption(f"Selected analyte: {analyte}")
    st.caption("Expected ion ratios are derived from transition efficiencies and can be adjusted.")

    expected_analyte_default = None
    expected_istd_default = None
    for fn in readable_files:
        data = get_json_from_active_dataset(fn, folder=folder, data_by_name=data_by_name)
        aqm = data.get("analyte_quad_map", {}) if isinstance(data, dict) else {}
        quad = aqm.get(analyte, None) if isinstance(aqm, dict) and analyte else None

        if expected_analyte_default is None:
            expected_analyte_default = extract_expected_ion_ratio_from_efficiencies(quad, "analyte")
        if expected_istd_default is None:
            expected_istd_default = extract_expected_ion_ratio_from_efficiencies(quad, "internal_standard")

        if expected_analyte_default is not None and expected_istd_default is not None:
            break

    analyte_default_source = (
        st.session_state.selected_folder,
        st.session_state.selected_analyte,
        "analyte",
    )
    istd_default_source = (
        st.session_state.selected_folder,
        st.session_state.selected_analyte,
        "internal_standard",
    )

    if "expected_ratio_analyte_input" not in st.session_state:
        st.session_state.expected_ratio_analyte_input = (
            float(expected_analyte_default) if expected_analyte_default is not None else 0.0
        )
        st.session_state.expected_ratio_analyte_source = analyte_default_source
    elif st.session_state.get("expected_ratio_analyte_source") != analyte_default_source:
        st.session_state.expected_ratio_analyte_input = (
            float(expected_analyte_default) if expected_analyte_default is not None else 0.0
        )
        st.session_state.expected_ratio_analyte_source = analyte_default_source

    if "expected_ratio_istd_input" not in st.session_state:
        st.session_state.expected_ratio_istd_input = (
            float(expected_istd_default) if expected_istd_default is not None else 0.0
        )
        st.session_state.expected_ratio_istd_source = istd_default_source
    elif st.session_state.get("expected_ratio_istd_source") != istd_default_source:
        st.session_state.expected_ratio_istd_input = (
            float(expected_istd_default) if expected_istd_default is not None else 0.0
        )
        st.session_state.expected_ratio_istd_source = istd_default_source

    show_trendline = st.checkbox("Show trendline", value=True, key="ionratio_show_trendline")

    c1, c2 = st.columns(2, gap="large")

    with c1:
        expected_ratio_analyte = st.number_input(
            "Expected analyte ion ratio",
            min_value=0.0,
            step=0.00001,
            format="%.5f",
            key="expected_ratio_analyte_input",
        )

    with c2:
        expected_ratio_istd = st.number_input(
            "Expected internal standard ion ratio",
            min_value=0.0,
            step=0.00001,
            format="%.5f",
            key="expected_ratio_istd_input",
        )

    ion_records = build_ion_ratio_records(
        readable_files,
        folder,
        analyte,
        expected_ratio_analyte if expected_ratio_analyte > 0 else None,
        expected_ratio_istd if expected_ratio_istd > 0 else None,
        data_by_name=data_by_name,
    )
    ion_df = pd.DataFrame(ion_records)

    with c1:
        analyte_df = ion_df[ion_df["Species"] == "Analyte"].copy()
        analyte_fig, analyte_eqn = build_ion_ratio_scatter(
            analyte_df,
            "Analyte ion ratio",
            CHANNEL_STYLES[("analyte", "quantifier")]["dark"],
            show_trendline,
        )
        st.plotly_chart(analyte_fig, use_container_width=True)
        if show_trendline:
            st.caption(f"Trendline equation: {analyte_eqn or 'not available'}")
        if analyte_df[
            analyte_df["Quantifier response"].notna() & analyte_df["Qualifier response"].notna()
        ].empty:
            st.warning("No data available for analyte ion ratio plot.")

    with c2:
        istd_df = ion_df[ion_df["Species"] == "Internal Standard"].copy()
        istd_fig, istd_eqn = build_ion_ratio_scatter(
            istd_df,
            "Internal standard ion ratio",
            CHANNEL_STYLES[("internal_standard", "quantifier")]["dark"],
            show_trendline,
        )
        st.plotly_chart(istd_fig, use_container_width=True)
        if show_trendline:
            st.caption(f"Trendline equation: {istd_eqn or 'not available'}")
        if istd_df[
            istd_df["Quantifier response"].notna() & istd_df["Qualifier response"].notna()
        ].empty:
            st.warning("No data available for internal standard ion ratio plot.")

# -----------------------------
# Summary / Trend page
# -----------------------------
elif st.session_state.page == "Summary / Trend":
    st.title("Summary / Trend")

    source, files, ds, readable_files = get_active_dataset()
    folder = st.session_state.selected_folder if source == "Local folder" else None
    data_by_name = ds.get("readable", {}) if source == "Upload JSON files" else None

    if not active_dataset_ready():
        st.warning("No valid dataset selected. Go to Dataset page first.")
        st.stop()

    if not readable_files:
        st.warning("No readable files available in the active dataset.")
        st.stop()

    analyte = st.session_state.selected_analyte
    if not analyte:
        st.warning("No analyte selected. Go to Dataset page first.")
        st.stop()

    st.caption(active_dataset_caption())
    st.caption(f"Selected analyte: {analyte}")
    st.caption("View response or retention time trends across the selected dataset.")

    ctrl1, ctrl2, ctrl3 = st.columns([1.2, 2.6, 1.6])

    with ctrl1:
        parameter = st.radio(
            "Parameter",
            ["Response", "Retention time"],
            key="summary_parameter",
        )

    channel_options = [summary_channel_label(which, ion) for which, ion in summary_channel_specs()]
    default_channels = [channel_options[0], channel_options[2]]

    with ctrl2:
        selected_channels = st.multiselect(
            "Channels",
            options=channel_options,
            default=st.session_state.get("summary_channels", default_channels),
            key="summary_channels",
        )

    with ctrl3:
        show_stats_lines = st.checkbox(
            "Show mean ± SD lines",
            value=True,
            key="summary_show_stats",
        )
        show_stats_labels = st.checkbox(
            "Show line labels",
            value=True,
            key="summary_show_stats_labels",
        )

    records = build_summary_records(readable_files, folder, analyte, parameter, data_by_name=data_by_name)
    records_df = pd.DataFrame(records)

    if not selected_channels:
        st.info("Select one or more channels to display.")
    else:
        fig = build_summary_chart(
            records_df,
            selected_channels,
            parameter,
            show_stats_lines,
            show_stats_labels,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Summary statistics")
        stats_df = build_summary_stats_table(records_df, selected_channels)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

        no_data_channels = []
        for ch in selected_channels:
            valid = records_df[(records_df["Channel"] == ch) & (records_df["Value"].notna())]
            if valid.empty:
                no_data_channels.append(ch)

        if no_data_channels:
            st.warning("No data available for: " + ", ".join(no_data_channels))

# -----------------------------
# Peak Shape Analysis page
# -----------------------------
elif st.session_state.page == "Peak Shape Analysis":
    st.title("Peak Shape Analysis")

    source, files, ds, readable_files = get_active_dataset()
    folder = st.session_state.selected_folder if source == "Local folder" else None
    data_by_name = ds.get("readable", {}) if source == "Upload JSON files" else None

    if not active_dataset_ready():
        st.warning("No valid dataset selected. Go to Dataset page first.")
        st.stop()

    if not readable_files:
        st.warning("No readable files available in the active dataset.")
        st.stop()

    analyte = st.session_state.selected_analyte
    if not analyte:
        st.warning("No analyte selected. Go to Dataset page first.")
        st.stop()

    st.caption(active_dataset_caption())
    st.caption(f"Selected analyte: {analyte}")
    st.caption("View analyte and internal standard peak shape metrics against analyte concentration across the selected dataset.")

    ctrl1, ctrl2 = st.columns([1.3, 2.4])
    with ctrl1:
        parameter = st.radio("Parameter", ["FWHM", "RT", "Asymmetry"], key="peakshape_parameter", help="""
        FWHM: Calculated from Apex values, not smoothed data                     
        RT: Pulled from peak.rt                             
        Asymmetry: Calculated from Apex values, not smoothed data
        """)

    peakshape_channel_options = [
        summary_channel_label("analyte", "quantifier"),
        summary_channel_label("analyte", "qualifier"),
        summary_channel_label("internal_standard", "quantifier"),
        summary_channel_label("internal_standard", "qualifier"),
    ]
    default_peakshape_channels = st.session_state.get(
        "peakshape_channels",
        [
            summary_channel_label("analyte", "quantifier"),
            summary_channel_label("analyte", "qualifier"),
        ],
    )
    with ctrl2:
        selected_channels = st.multiselect(
            "Channels",
            options=peakshape_channel_options,
            default=default_peakshape_channels,
            key="peakshape_channels",
        )

    records = build_peak_shape_records(readable_files, folder, analyte, parameter, data_by_name=data_by_name)
    records_df = pd.DataFrame(records)

    if not selected_channels:
        st.info("Select one or more channels to display.")
    else:
        fig = build_peak_shape_chart(records_df, selected_channels, parameter)
        st.plotly_chart(fig, use_container_width=True)

        no_data_channels = []
        for ch in selected_channels:
            valid = records_df[(records_df["Channel"] == ch) & records_df["Concentration"].notna() & records_df["Value"].notna()]
            if valid.empty:
                no_data_channels.append(ch)

        if no_data_channels:
            st.warning("No data available for: " + ", ".join(no_data_channels))

# -----------------------------
# Quad view page
# -----------------------------
else:
    st.title("Injection / Quad View")

    source, files, ds, readable_files = get_active_dataset()
    folder = st.session_state.selected_folder if source == "Local folder" else None
    data_by_name = ds.get("readable", {}) if source == "Upload JSON files" else None

    if not active_dataset_ready():
        st.warning("No valid dataset selected. Go to Dataset page first.")
        st.stop()

    if not readable_files:
        st.warning("No readable files available in the active dataset.")
        st.stop()


    # Clamp selected_index
    idx = st.session_state.selected_index
    idx = max(0, min(idx, len(readable_files) - 1))
    st.session_state.selected_index = idx
    st.session_state.selected_file = readable_files[idx]

    # Navigation controls
    nav1, nav2, nav3, nav4 = st.columns([1, 1, 3, 2])

    with nav1:
        prev_disabled = (idx == 0)
        if st.button("⬅ Previous", disabled=prev_disabled):
            st.session_state.selected_index = idx - 1
            st.session_state.selected_file = readable_files[idx - 1]
            st.rerun()

    with nav2:
        next_disabled = (idx == len(readable_files) - 1)
        if st.button("Next ➡", disabled=next_disabled):
            st.session_state.selected_index = idx + 1
            st.session_state.selected_file = readable_files[idx + 1]
            st.rerun()

    with nav3:
        st.caption(f"Injection {idx + 1} of {len(readable_files)} — {st.session_state.selected_file}")

    with nav4:
        # Jump selector
        chosen = st.selectbox(
            "Jump to injection",
            options=readable_files,
            index=idx,
        )
        if chosen != st.session_state.selected_file:
            st.session_state.selected_file = chosen
            st.session_state.selected_index = readable_files.index(chosen)
            st.rerun()

    st.caption(active_dataset_caption())

    # Load current injection JSON
    inj = get_json_from_active_dataset(st.session_state.selected_file, folder=folder, data_by_name=data_by_name)
    if inj is None:
        st.error("Unable to load selected JSON from the active dataset.")
        st.stop()

    analyte = st.session_state.selected_analyte
    aqm = inj.get("analyte_quad_map", {})
    quad = aqm.get(analyte, None) if isinstance(aqm, dict) else None

    # Header
    header_left, header_mid, header_right = st.columns([2.2, 2.2, 1.6])

    # Derive injection identifier (prefer quad.injection.result.id; fall back to filename)
    inj_id = None
    if isinstance(quad, dict):
        inj_block = quad.get("injection", {})
        if isinstance(inj_block, dict):
            res = inj_block.get("result", {})
            if isinstance(res, dict):
                inj_id = res.get("id", None)
    if not inj_id:
        inj_id = os.path.splitext(st.session_state.selected_file)[0]

    with header_left:
        st.subheader(analyte or "(no analyte selected)")

    with header_mid:
        st.markdown(f"**Injection ID**: `{inj_id}`")

    with header_right:
        # Human Opinion + Software Opinion badges (labeled)
        raw_human_opinion = None
        raw_opinion = None

        if isinstance(quad, dict):
            raw_human_opinion = quad.get("human_opinion", None)
            raw_opinion = quad.get("software_opinion", None)

        human_opinion_txt = normalize_human_opinion(raw_human_opinion)
        opinion_txt = normalize_opinion(raw_opinion)

        b1, b2 = st.columns(2, gap="small")

        with b1:
            st.markdown("**Human Opinion**")
            st.markdown(
                f"<div style='display:inline-block; padding:6px 10px; border-radius:8px; "
                f"border:2px solid {human_opinion_color(human_opinion_txt)}; color:{human_opinion_color(human_opinion_txt)}; "
                f"font-weight:800; letter-spacing:0.5px;'>{human_opinion_txt}</div>",
                unsafe_allow_html=True,
            )

        with b2:
            st.markdown("**Software Opinion**")
            st.markdown(
                f"<div style='display:inline-block; padding:6px 10px; border-radius:8px; "
                f"border:2px solid {opinion_color(opinion_txt)}; color:{opinion_color(opinion_txt)}; "
                f"font-weight:800; letter-spacing:0.5px;'>{opinion_txt}</div>",
                unsafe_allow_html=True,
            )


    if analyte and (quad is None):
        st.warning("Selected analyte is not present in this injection; showing missing-data layout.")

    # Main layout: plot + calculated values
    left, right = st.columns([3.2, 1.2], gap="large")

    with left:
        ctrl_trace, ctrl_apex, ctrl_consensus, ctrl_expected = st.columns([1.45, 1.05, 1.10, 1.20])
        with ctrl_trace:
            trace_mode = st.radio(
                "Trace display",
                ["Smoothed", "Raw"],
                horizontal=True,
                index=0,
            )
        with ctrl_apex:
            show_apex = st.checkbox(
                "Show apex line",
                value=False,
                help="Derived from Apex (unsmoothed).",
            )
        with ctrl_consensus:
            show_consensus_rt = st.checkbox(
                "Show consensus RT",
                value=True,
                help="Shows consensus retention time markers from the calibrator consensus RT values.",
            )
        with ctrl_expected:
            show_expected_peak = st.checkbox(
                "Show expected peak",
                value=True,
                help="Reconstructed using coefficients for ϕ(x)=(1−x²)^6 for |x|≤1, and centred on Observed RT on baseline = 0; !!REFERENCE ONLY!!",
            )

        if isinstance(quad, dict):
            fig = build_quad_plot(quad, trace_mode=trace_mode, show_apex=show_apex, show_consensus_rt=show_consensus_rt, show_expected_peak=show_expected_peak)
        else:
            # Build an empty plot with missing annotations by passing a dummy quad
            fig = build_quad_plot({"analyte": {}, "internal_standard": {}}, trace_mode=trace_mode, show_apex=show_apex, show_consensus_rt=show_consensus_rt, show_expected_peak=show_expected_peak)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Results")
        if isinstance(quad, dict):
            conc = quad.get("concentration", None)
            rr = quad.get("response_ratio", None)
        else:
            conc = None
            rr = None

        st.metric("Concentration", "None" if conc is None else f"{float(conc):.2f}")
        st.metric("Response ratio", "None" if rr is None else f"{float(rr):.3f}")

        # Ion ratios (computed)
        if isinstance(quad, dict):
            aq = channel_from_quad(quad, "analyte", "quantifier")
            al = channel_from_quad(quad, "analyte", "qualifier")
            iq = channel_from_quad(quad, "internal_standard", "quantifier")
            il = channel_from_quad(quad, "internal_standard", "qualifier")

            a_q_resp, _ = get_peak_values(aq)
            a_l_resp, _ = get_peak_values(al)
            i_q_resp, _ = get_peak_values(iq)
            i_l_resp, _ = get_peak_values(il)

            analyte_ir = safe_div(a_l_resp, a_q_resp)
            istd_ir = safe_div(i_l_resp, i_q_resp)
        else:
            analyte_ir = None
            istd_ir = None

        expected_analyte_ir = extract_expected_ion_ratio_from_efficiencies(quad, "analyte") if isinstance(quad, dict) else None
        expected_istd_ir = extract_expected_ion_ratio_from_efficiencies(quad, "internal_standard") if isinstance(quad, dict) else None

        ion_col1, ion_col2 = st.columns(2)
        with ion_col1:
            st.metric("Analyte ion ratio (Qua/Quan)", "None" if analyte_ir is None else f"{analyte_ir:.5f}")
            st.metric("ISTD ion ratio (Qua/Quan)", "None" if istd_ir is None else f"{istd_ir:.5f}")

        with ion_col2:
            st.metric("Expected analyte ion ratio (Qua/Quan)", "None" if expected_analyte_ir is None else f"{expected_analyte_ir:.5f}")
            st.metric("Expected ISTD ion ratio (Qua/Quan)", "None" if expected_istd_ir is None else f"{expected_istd_ir:.5f}")


        st.divider()
        st.subheader("Concerns")
        concerns = quad.get("concerns", []) if isinstance(quad, dict) else []
        render_concern_cards(concerns)


    # Optional analyte_istd_methods
    st.divider()
    st.session_state.show_methods = st.checkbox(
        "Show analyte_istd_methods (optional)",
        value=bool(st.session_state.show_methods),
    )

    if st.session_state.show_methods:
        methods_map = inj.get("analyte_istd_methods", {})
        methods_obj = methods_map.get(analyte, None) if isinstance(methods_map, dict) else None

        st.subheader("Method settings")

        with st.expander("Method comparison", expanded=False):
            if not methods_obj:
                st.info("No analyte_istd_methods found for this analyte.")
            else:
                tables, summary = build_method_comparison_tables(methods_obj)
                st.caption(
                    f'{summary["total"]} parameters compared · '
                    f'{summary["identical"]} same · '
                    f'{summary["different"]} different · '
                    f'{summary["missing"]} missing on one side'
                )

                for section_name, df_section in tables:
                    st.markdown(f"**{section_name}**")
                    styled = style_method_comparison(df_section)
                    st.dataframe(styled, use_container_width=True)

        with st.expander("Calibration", expanded=False):
            calibrator = quad.get("calibrator", None) if isinstance(quad, dict) else None
            cal_fig, cal_meta = build_calibration_plot(calibrator)

            cal_left, cal_right = st.columns([2.3, 1.2], gap="large")
            with cal_left:
                if cal_fig is None:
                    st.info("Calibration is not available for this analyte.")
                else:
                    st.plotly_chart(cal_fig, use_container_width=True)
            with cal_right:
                st.markdown(f"**Fit type**  \n{cal_meta['fit_type']}")
                st.markdown(f"**Weight function**  \n{cal_meta['weight_function']}")
                st.markdown(f"**R²**  \n{cal_meta['rsquared']}")
                st.markdown(f"**Equation**  \n`{cal_meta['equation']}`")

            st.markdown("**Calibration concerns**")
            cal_concerns = calibrator.get("concerns", []) if isinstance(calibrator, dict) else []
            render_concern_cards(cal_concerns)
