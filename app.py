import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
import textwrap
import json
import uuid
import re

# --- CONFIGURATION & STYLING ---
st.set_page_config(layout="wide", page_title="Custom CA Studio")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Quicksand', sans-serif; }
        h1, h2, h3 { font-family: 'Quicksand', sans-serif; font-weight: 700; color: #1e1e1e;}
       
        .metric-box { background-color: #f8f9fa; border: 1px solid #e0e0e0; padding: 15px; border-radius: 8px; text-align: center; margin-bottom: 20px;}
        .metric-title { font-size: 0.9em; font-weight: 600; color: #555; text-transform: uppercase;}
        .metric-value { font-size: 1.8em; font-weight: 800; color: #2e7d32; margin: 5px 0;}
       
        .sidebar-header { margin-top: 15px; padding-bottom: 5px; border-bottom: 2px solid #eaeaea; font-size: 1.1em; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'processed' not in st.session_state: st.session_state.processed = False
if 'df_b_master' not in st.session_state: st.session_state.df_b_master = pd.DataFrame()
if 'df_a_master' not in st.session_state: st.session_state.df_a_master = pd.DataFrame()
if 'passive_layers' not in st.session_state: st.session_state.passive_layers = []  # Stores list of dicts: {id, name, df, shape, visible}
if 'max_dim' not in st.session_state: st.session_state.max_dim = 2
if 's_vals' not in st.session_state: st.session_state.s_vals = []
if 'hidden_items' not in st.session_state: st.session_state.hidden_items = []
if 'map_rot' not in st.session_state: st.session_state.map_rot = 0
if 'show_base_cols' not in st.session_state: st.session_state.show_base_cols = True
if 'show_base_rows' not in st.session_state: st.session_state.show_base_rows = True

# --- PROJECT SAVE/LOAD SERIALIZATION ---
def serialize_project():
    """Converts the entire session state into a robust JSON string using record format."""
    project_data = {
        "version": "1.3",
        "processed": st.session_state.processed,
        "max_dim": st.session_state.max_dim,
        "s_vals": list(st.session_state.s_vals) if isinstance(st.session_state.s_vals, np.ndarray) else st.session_state.s_vals,
        "hidden_items": st.session_state.hidden_items,
        "map_rot": st.session_state.map_rot,
        "show_base_cols": st.session_state.show_base_cols,
        "show_base_rows": st.session_state.show_base_rows,
        "df_b_master": st.session_state.df_b_master.to_dict(orient='records') if not st.session_state.df_b_master.empty else None,
        "df_a_master": st.session_state.df_a_master.to_dict(orient='records') if not st.session_state.df_a_master.empty else None,
        "passive_layers": []
    }
    for layer in st.session_state.passive_layers:
        project_data["passive_layers"].append({
            "id": layer["id"],
            "name": layer["name"],
            "shape": layer["shape"],
            "visible": layer["visible"],
            "df": layer["df"].to_dict(orient='records')
        })
    return json.dumps(project_data, indent=2)

def deserialize_project(json_str):
    """Restores the session state gracefully, patching older versions dynamically."""
    try:
        data = json.loads(json_str)
        st.session_state.processed = data.get("processed", False)
        st.session_state.max_dim = data.get("max_dim", 2)
        st.session_state.s_vals = np.array(data.get("s_vals", []))
        st.session_state.hidden_items = data.get("hidden_items", [])
        st.session_state.map_rot = data.get("map_rot", 0)
        st.session_state.show_base_cols = data.get("show_base_cols", True)
        st.session_state.show_base_rows = data.get("show_base_rows", True)
       
        # Helper to rebuild DataFrames safely regardless of version format
        def load_df(df_data):
            if not df_data:
                return pd.DataFrame()
            # If the user uploads an older V1.2 "split" format, adapt it dynamically
            if isinstance(df_data, dict) and "columns" in df_data and "data" in df_data:
                return pd.DataFrame(**df_data)
            # If it is the new, clean V1.3 "records" list format
            return pd.DataFrame(df_data)
           
        st.session_state.df_b_master = load_df(data.get("df_b_master"))
        st.session_state.df_a_master = load_df(data.get("df_a_master"))
       
        st.session_state.passive_layers = []
        for layer_data in data.get("passive_layers", []):
            st.session_state.passive_layers.append({
                "id": layer_data.get("id", str(uuid.uuid4())[:8]),
                "name": layer_data["name"],
                "shape": layer_data["shape"],
                "visible": layer_data["visible"],
                "df": load_df(layer_data["df"])
            })
        return True
    except Exception as e:
        st.error(f"Failed to restore project file: {e}")
        return False

# --- ENTERPRISE HEADER DETECTOR & SANITIZER ---
def clean_header_formatting(s):
    """Strips trailing carriage returns, hyphens, or underscores commonly exported by survey platforms."""
    if pd.isna(s):
         return ""
    # Split by newline (like "Simply\n-----------") and take the first clean part
    s_str = str(s).split('\n')[0].strip()
    # Strip trailing punctuation artifacts
    return s_str.rstrip('-_= ')

def detect_header_row(file_buffer, file_is_csv, expected_labels=None):
    """Heuristically scans the first 15 rows of a dirty spreadsheet to find where the table headers actually start."""
    file_buffer.seek(0)
    try:
        if file_is_csv:
            df_raw = pd.read_csv(file_buffer, header=None, nrows=15)
        else:
            df_raw = pd.read_excel(file_buffer, header=None, nrows=15)
    except Exception:
        file_buffer.seek(0)
        return 0
    finally:
        file_buffer.seek(0)

    best_row_idx = 0
    max_score = -1000

    labels_set = set()
    if expected_labels is not None:
        for lbl in expected_labels:
            clean = re.sub(r'[^\w\s]', '', str(lbl)).lower().replace(' ', '').strip()
            labels_set.add(clean)

    for idx, row in df_raw.iterrows():
        matches = 0
        non_empty = 0
        strings_count = 0
        numbers_count = 0
       
        for cell in row:
            if pd.isna(cell) or str(cell).strip() == "":
                continue
            non_empty += 1
           
            # Remove formatting punctuation to verify if numeric
            cell_clean = str(cell).strip().replace(',', '').replace('%', '').replace('$', '')
            try:
                float(cell_clean)
                numbers_count += 1
            except ValueError:
                strings_count += 1
                cell_norm = re.sub(r'[^\w\s]', '', cell_clean).lower().replace(' ', '').strip()
                if expected_labels is not None and cell_norm in labels_set:
                    matches += 10
                elif any(k in cell_norm for k in ['total', 'brand', 'attribute', 'statement', 'demographic', 'segment', 'universe', 'respondent']):
                    matches += 2

        # SCORING ALGORITHM: Header rows have high string counts, near-zero numbers, and many non-empty columns
        score = matches + strings_count - (2.0 * numbers_count) + (0.1 * non_empty)
        if score > max_score and non_empty > 2:
            max_score = score
            best_row_idx = idx

    return best_row_idx

# --- CORE MATH FUNCTIONS ---
def normalize_str(s_series):
    # Aggressively strips ALL punctuation and ALL whitespace so column mapping never fails
    return s_series.astype(str).str.lower().str.replace(r'[^\w\s]', '', regex=True).str.replace(r'\s+', '', regex=True).str.strip()

def rotate_coords(df, angle_deg):
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    coords = df[['x', 'y']].values
    rotated = coords @ R.T
    df_new = df.copy()
    df_new['x'], df_new['y'] = rotated[:, 0], rotated[:, 1]
    return df_new

def process_ca(uploaded_file):
    try:
        uploaded_file.seek(0)
        file_is_csv = uploaded_file.name.endswith('.csv')
       
        # Detect the true header row
        header_row_idx = detect_header_row(uploaded_file, file_is_csv)
       
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, header=header_row_idx) if file_is_csv else pd.read_excel(uploaded_file, header=header_row_idx)
       
        # Sanitize formatting artifacts from the headers
        df.columns = [clean_header_formatting(c) for c in df.columns]
        df.iloc[:, 0] = df.iloc[:, 0].apply(clean_header_formatting)
       
        df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.strip()
        df = df.set_index(df.columns[0])
       
        for col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[,$%]', '', regex=True), errors='coerce')
        df = df.fillna(0)
       
        # AGGRESSIVE SCRUBBER: Destroy completely blank rows/cols and ghost "Unnamed/NaN" tags
        df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
        clean_cols = [c for c in df.columns if "unnamed" not in str(c).lower() and str(c).strip() != ""]
        df = df[clean_cols]
        clean_rows = [r for r in df.index if "unnamed" not in str(r).lower() and str(r).lower() != "nan" and str(r).strip() != ""]
        df = df.loc[clean_rows]
       
        # Purge any "Total" or "Universe" columns and rows completely
        u_idx_row = df.index.astype(str).str.strip().str.contains(r"^(?:Study Universe|Total Population|Grand Total|Total Market|Total)$", case=False, regex=True)
        u_idx_col = df.columns.astype(str).str.strip().str.contains(r"^(?:Study Universe|Total Population|Grand Total|Total Market|Total)$", case=False, regex=True)
       
        df_math = df.loc[~u_idx_row, ~u_idx_col].copy()
        df_math = df_math.loc[(df_math != 0).any(axis=1)]
       
        N = df_math.values
        matrix_sum = N.sum()
        if matrix_sum == 0: return False
       
        P = N / matrix_sum
        r = P.sum(axis=1)
        c = P.sum(axis=0)
        E = np.outer(r, c)
        E[E < 1e-9] = 1e-9
        R = (P - E) / np.sqrt(E)
        U, s, Vh = np.linalg.svd(R, full_matrices=False)
       
        max_dim = min(5, len(s))
        st.session_state.max_dim = max_dim
        st.session_state.s_vals = s[:max_dim]
       
        # Safe Slicing to prevent Array Broadcast Shape Errors
        U = U[:, :max_dim]
        s_sliced = s[:max_dim]
        Vh = Vh[:max_dim, :]
       
        row_coords = (U * s_sliced) / np.sqrt(r[:, np.newaxis])
        col_coords = (Vh.T * s_sliced) / np.sqrt(c[:, np.newaxis])
       
        df_b_master = pd.DataFrame(col_coords, columns=[f'Dim{i+1}' for i in range(max_dim)])
        df_b_master['Label'] = df_math.columns.values
        st.session_state.df_b_master = df_b_master
       
        df_a_master = pd.DataFrame(row_coords, columns=[f'Dim{i+1}' for i in range(max_dim)])
        df_a_master['Label'] = df_math.index.values
        st.session_state.df_a_master = df_a_master
       
        st.session_state.processed = True
        return True
    except Exception as e:
        st.error(f"Failed to process file: {e}")
        return False

def process_passive(file, name, mode):
    try:
        file.seek(0)
        file_is_csv = file.name.endswith('.csv')
       
        # Match against active base coordinates for 100% precision header matching
        expected_labels = []
        if not st.session_state.df_b_master.empty:
            expected_labels.extend(st.session_state.df_b_master['Label'].tolist())
        if not st.session_state.df_a_master.empty:
            expected_labels.extend(st.session_state.df_a_master['Label'].tolist())
           
        # Detect the true header row
        header_row_idx = detect_header_row(file, file_is_csv, expected_labels)
       
        file.seek(0)
        df = pd.read_csv(file, header=header_row_idx) if file_is_csv else pd.read_excel(file, header=header_row_idx)
       
        # Sanitize formatting artifacts from the headers
        df.columns = [clean_header_formatting(c) for c in df.columns]
        df.iloc[:, 0] = df.iloc[:, 0].apply(clean_header_formatting)
       
        df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.strip()
        df = df.set_index(df.columns[0])
        for col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[,$%]', '', regex=True), errors='coerce')
        df = df.fillna(0)
       
        # AGGRESSIVE SCRUBBER: Destroy completely blank rows/cols and ghost "Unnamed/NaN" tags
        df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
        clean_cols = [c for c in df.columns if "unnamed" not in str(c).lower() and str(c).strip() != ""]
        df = df[clean_cols]
        clean_rows = [r for r in df.index if "unnamed" not in str(r).lower() and str(r).lower() != "nan" and str(r).strip() != ""]
        df = df.loc[clean_rows]
       
        # Purge "Total" rows/cols from passive layers so they don't plot as ghost dots!
        u_idx_row = df.index.astype(str).str.strip().str.contains(r"^(?:Study Universe|Total Population|Grand Total|Total Market|Total)$", case=False, regex=True)
        u_idx_col = df.columns.astype(str).str.strip().str.contains(r"^(?:Study Universe|Total Population|Grand Total|Total Market|Total)$", case=False, regex=True)
        df = df.loc[~u_idx_row, ~u_idx_col].copy()
       
        base_cols_norm = normalize_str(st.session_state.df_b_master['Label'])
        base_idx_norm = normalize_str(st.session_state.df_a_master['Label'])
        col_mapper = {n: i for i, n in enumerate(base_cols_norm)}
        row_mapper = {n: i for i, n in enumerate(base_idx_norm)}
       
        max_d = st.session_state.max_dim
        s = st.session_state.s_vals[:max_d]
       
        proj = np.array([])
        shape = 'star'
       
        if mode == "Rows (Match by Columns)":
            p_cols_norm = normalize_str(pd.Series(df.columns))
            if sum(1 for x in p_cols_norm if x in col_mapper) > 0:
                p_aligned = pd.DataFrame(0.0, index=df.index, columns=st.session_state.df_b_master['Label'])
                for orig, norm in zip(df.columns, p_cols_norm):
                    if norm in col_mapper: p_aligned.iloc[:, col_mapper[norm]] = df[orig].values
               
                # Robust bulletproof matrix multiplication
                row_sums = p_aligned.sum(axis=1).values
                row_sums = np.where(row_sums == 0, 1, row_sums) # Prevent divide by zero
                base_coords = st.session_state.df_b_master[[f'Dim{i+1}' for i in range(max_d)]].values
               
                proj = (p_aligned.values / row_sums[:, None]) @ base_coords / s
                shape = 'star'
        else:
            p_idx_norm = normalize_str(pd.Series(df.index))
            if sum(1 for x in p_idx_norm if x in row_mapper) > 0:
                p_aligned = pd.DataFrame(0.0, index=st.session_state.df_a_master['Label'], columns=df.columns)
                for orig, norm in zip(df.index, p_idx_norm):
                    if norm in row_mapper: p_aligned.iloc[row_mapper[norm], :] = df.loc[orig].values
               
                # Robust bulletproof matrix multiplication
                col_sums = p_aligned.sum(axis=0).values
                col_sums = np.where(col_sums == 0, 1, col_sums)
                base_coords = st.session_state.df_a_master[[f'Dim{i+1}' for i in range(max_d)]].values
               
                proj = (p_aligned.values / col_sums[None, :]).T @ base_coords / s
                shape = 'diamond'
               
        if proj.size > 0:
            res = pd.DataFrame(proj, columns=[f'Dim{k+1}' for k in range(max_d)])
            res['Label'] = df.index if mode == "Rows (Match by Columns)" else df.columns
           
            # Return stable structural layer object
            return {
                "id": str(uuid.uuid4())[:8],
                "name": name,
                "df": res,
                "shape": shape,
                "visible": True
            }
        return None
    except Exception as e:
        st.error(f"Passive Error on {name}: {e}")
        return None

# --- UI LAYOUT ---
st.title("🗺️ CA Presentation Studio")
st.markdown("Upload any raw crosstab grid (Columns = Brands/Groups, Rows = Attributes/Statements). The engine will automatically map the mathematical relationships and prepare them for PowerPoint export.")

# --- SIDEBAR: STATE & LAYER MANAGERS ---
with st.sidebar:
    st.markdown('<div class="sidebar-header">💾 Save / Load Studio Project</div>', unsafe_allow_html=True)
   
    # Exporter
    if st.session_state.processed:
        proj_json = serialize_project()
        st.download_button(
            label="💾 Download Project File (.castudio)",
            data=proj_json,
            file_name="presentation_studio_workspace.castudio",
            mime="application/json",
            use_container_width=True
        )
    else:
        st.caption("Perform an analysis or load an existing project to begin.")
       
    # Importer
    load_file = st.file_uploader("📂 Load Project File", type=['castudio', 'json'], help="Upload a previously saved .castudio project to restore your map instantly!")
    if load_file is not None:
        if st.button("🔄 Import Workspace", use_container_width=True):
            loaded_json = load_file.read().decode("utf-8")
            if deserialize_project(loaded_json):
                st.success("Project restored!")
                st.rerun()

    st.markdown('<div class="sidebar-header">📂 1. Core Map Data</div>', unsafe_allow_html=True)
    core_file = st.file_uploader("Upload Base Crosstab", type=['csv', 'xlsx'])
    if core_file:
        if st.button("🚀 Run Analysis", use_container_width=True):
            st.session_state.passive_layers = []
            st.session_state.hidden_items = []
            process_ca(core_file)

    if st.session_state.processed:
        st.markdown('<div class="sidebar-header">👁️ 2. Map Layer Manager</div>', unsafe_allow_html=True)
       
        # Unified Base Toggles
        st.session_state.show_base_cols = st.checkbox("👁️ Base Columns (Brands)", value=st.session_state.show_base_cols)
        st.session_state.show_base_rows = st.checkbox("👁️ Base Rows (Attributes)", value=st.session_state.show_base_rows)
       
        # Stable Passive Layer Interface
        if st.session_state.passive_layers:
            st.markdown("**Overlay Layers:**")
            for i, layer in enumerate(st.session_state.passive_layers):
                col_tog, col_del = st.columns([4, 1])
                with col_tog:
                    # Using permanent ID so list deletions never cause sliding toggle glitch
                    is_vis = st.checkbox(f"👁️ {layer['name']}", value=layer['visible'], key=f"vis_layer_{layer['id']}")
                    st.session_state.passive_layers[i]['visible'] = is_vis
                with col_del:
                    if st.button("🗑️", key=f"del_l_{layer['id']}", help="Remove Layer"):
                        st.session_state.passive_layers.pop(i)
                        st.rerun()

        st.markdown('<div class="sidebar-header">⚙️ 3. Map Dimensions</div>', unsafe_allow_html=True)
        col_x, col_y = st.columns(2)
        with col_x: x_ax = st.selectbox("X-Axis", range(1, st.session_state.max_dim + 1), index=0)
        with col_y: y_ax = st.selectbox("Y-Axis", range(1, st.session_state.max_dim + 1), index=1 if st.session_state.max_dim > 1 else 0)
       
        st.session_state.map_rot = st.slider("Rotate Map (Degrees)", 0, 360, 0, step=90)
       
        st.markdown('<div class="sidebar-header">➕ 4. Add Passive Layers</div>', unsafe_allow_html=True)
        st.caption("Upload supplementary grids to overlay onto the base map.")
        p_file = st.file_uploader("Upload Passive File", type=['csv', 'xlsx'])
        p_name = st.text_input("Layer Name", "New Layer")
        p_mode = st.radio("Align By:", ["Rows (Match by Columns)", "Columns (Match by Rows)"])
        if p_file and st.button("Overlay Layer"):
            res_dict = process_passive(p_file, p_name, p_mode)
            if res_dict is not None:
                st.session_state.passive_layers.append(res_dict)
                st.success(f"Added {p_name}!")
                st.rerun()
            else:
                st.error("Could not align layer. Check your column/row names.")

# --- MAIN CANVAS ---
if st.session_state.processed:
    df_b = st.session_state.df_b_master.copy()
    df_b['x'], df_b['y'] = df_b[f'Dim{x_ax}'], df_b[f'Dim{y_ax}']
   
    df_a = st.session_state.df_a_master.copy()
    df_a['x'], df_a['y'] = df_a[f'Dim{x_ax}'], df_a[f'Dim{y_ax}']
   
    # --- VISUAL SETTINGS TOOLBAR ---
    with st.expander("🎨 Visual & Export Settings", expanded=True):
        t_col1, t_col2, t_col3, t_col4 = st.columns(4)
        with t_col1:
            col_color = st.color_picker("Column Color", "#1f77b4")
            col_shape = st.selectbox("Column Shape", ['circle', 'square', 'diamond', 'star'], index=0)
            col_size = st.slider("Column Dot Size", 5, 30, 16)
        with t_col2:
            row_color = st.color_picker("Row Color", "#d62728")
            row_shape = st.selectbox("Row Shape", ['circle', 'square', 'diamond', 'star'], index=1)
            row_size = st.slider("Row Dot Size", 5, 30, 10)
        with t_col3:
            lbl_pos = st.selectbox("Label Anchor", ["Radial (Auto-Spread)", "Top", "Bottom", "Right", "Left"])
            tail_len = st.slider("Connector Line Length", 10, 100, 30)
            lbl_size = st.slider("Font Size", 8, 24, 12)
        with t_col4:
            wrap_len = st.slider("Max Chars Per Line", 15, 100, 35)
            map_height = st.slider("Canvas Height", 500, 1200, 750, step=50)
            passive_boost = st.slider("Passive Dot Spread", 1.0, 10.0, 1.5, step=0.5, help="Demographics naturally clump in the center. Use this to violently stretch them outward to make them readable!")

    st.button("🔄 Unhide All Labels", on_click=lambda: st.session_state.update({'hidden_items': []}))

    df_p_list = []
    for l in st.session_state.passive_layers:
        p_df = l['df'].copy()
        # Apply the passive spread boost multiplier!
        p_df['x'] = p_df[f'Dim{x_ax}'] * passive_boost
        p_df['y'] = p_df[f'Dim{y_ax}'] * passive_boost
        p_df['Visible'] = l['visible']
        p_df['Shape'] = l['shape']
        p_df['LayerName'] = l['name']
        df_p_list.append(p_df)

    if st.session_state.map_rot != 0:
        df_b = rotate_coords(df_b, st.session_state.map_rot)
        df_a = rotate_coords(df_a, st.session_state.map_rot)
        df_p_list = [rotate_coords(p, st.session_state.map_rot) for p in df_p_list]

    eig = np.array(st.session_state.s_vals)**2
    tot_var = np.sum(eig)
    v_x = (eig[x_ax-1] / tot_var) * 100
    v_y = (eig[y_ax-1] / tot_var) * 100
    stability = v_x + v_y

    st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">CURRENT VIEW STABILITY (AXIS {x_ax} + AXIS {y_ax})</div>
            <div class="metric-value" style="color: {'#2e7d32' if stability >= 60 else '#c62828'};">{stability:.1f}%</div>
            <div style="font-size:0.85em; color:#666;">(Axis {x_ax}: {v_x:.1f}% | Axis {y_ax}: {v_y:.1f}%)</div>
        </div>
    """, unsafe_allow_html=True)

    fig = go.Figure()
    annotations = []

    def add_layer_to_fig(df_layer, color, shape, size, name):
        if df_layer.empty: return
       
        cx = float(df_layer['x'].mean())
        cy = float(df_layer['y'].mean())
       
        for _, row in df_layer.iterrows():
            is_visible = row['Label'] not in st.session_state.hidden_items
            wrapped_label = "<br>".join(textwrap.wrap(str(row['Label']), width=wrap_len))
           
            if lbl_pos == "Top": ax, ay = 0, -tail_len
            elif lbl_pos == "Bottom": ax, ay = 0, tail_len
            elif lbl_pos == "Left": ax, ay = -tail_len, 0
            elif lbl_pos == "Right": ax, ay = tail_len, 0
            else:
                try:
                    dx = float(row['x']) - cx
                    dy = float(row['y']) - cy
                    dist = (dx**2 + dy**2)**0.5
                    if dist > 1e-5:
                        ax, ay = (dx/dist)*tail_len, -(dy/dist)*tail_len
                    else:
                        ax, ay = 0, -tail_len
                except Exception:
                    ax, ay = 0, -tail_len

            fig.add_trace(go.Scatter(
                x=[row['x']], y=[row['y']], mode='markers',
                marker=dict(size=size, symbol=shape, color=color, line=dict(width=1, color='white')),
                customdata=[row['Label']], hovertemplate="<b>%{customdata}</b><extra></extra>",
                name=name, showlegend=False, visible=True if is_visible else False
            ))
           
            font_dict = dict(size=lbl_size, color=color, family="Quicksand")
            annotations.append(dict(
                x=row['x'], y=row['y'], xref="x", yref="y",
                text=wrapped_label,
                showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor=color,
                ax=ax, ay=ay, font=font_dict,
                visible=True if is_visible else False
            ))

    # Sidebar unified toggles determine rendering behavior
    if st.session_state.show_base_cols:
        add_layer_to_fig(df_b, col_color, col_shape, col_size, "Columns")
           
    if st.session_state.show_base_rows:
        add_layer_to_fig(df_a, row_color, row_shape, row_size, "Rows")
   
    p_colors = ['#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    for i, p_df in enumerate(df_p_list):
        if p_df.empty or not p_df['Visible'].iloc[0]: continue
        c = p_colors[i % len(p_colors)]
        s = p_df['Shape'].iloc[0]
        n = p_df['LayerName'].iloc[0]
        add_layer_to_fig(p_df, c, s, col_size - 2, n)

    # Calculate Axis Bounds
    all_x = []
    all_y = []
    if st.session_state.show_base_cols:
        all_x.extend(df_b['x'].tolist())
        all_y.extend(df_b['y'].tolist())
    if st.session_state.show_base_rows:
        all_x.extend(df_a['x'].tolist())
        all_y.extend(df_a['y'].tolist())
    for p in df_p_list:
        if p['Visible'].iloc[0]:
            all_x.extend(p['x'].tolist())
            all_y.extend(p['y'].tolist())
   
    max_val = max(np.max(np.abs(all_x)), np.max(np.abs(all_y))) * 1.15 if all_x else 1

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', dragmode='pan',
        margin=dict(l=0, r=0, t=0, b=0), height=map_height,
        xaxis=dict(range=[-max_val, max_val], fixedrange=False, zeroline=True, zerolinecolor='#eee', showgrid=False, showticklabels=False),
        yaxis=dict(range=[-max_val, max_val], fixedrange=False, zeroline=True, zerolinecolor='#eee', showgrid=False, showticklabels=False, scaleanchor="x", scaleratio=1),
        annotations=annotations
    )

    exp_config = {
        'displayModeBar': True,
        'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
        'scrollZoom': True,
        'edits': {'annotationTail': True, 'annotationText': True, 'annotationPosition': False},
        'toImageButtonOptions': {'format': 'png', 'filename': "CA_Export", 'height': 720, 'width': 1280, 'scale': 3}
    }

    st.info("💡 **Instructions:** You can now zoom and pan around the map! Click and drag any text label to un-clutter the map. Click directly on a dot to hide it entirely. Hover over the top right to download a high-res 16:9 PNG for PowerPoint. (Tip: Use the 'Reset Axes' house icon before exporting so your images stack perfectly in PPT!)")

    map_event = st.plotly_chart(
        fig, use_container_width=True, config=exp_config,
        on_select="rerun", selection_mode="points", key="main_studio_map"
    )

    if map_event and map_event.selection.get("points"):
        clicked_pts = [pt["customdata"] for pt in map_event.selection["points"] if "customdata" in pt]
        if clicked_pts:
            changed = False
            for cp in clicked_pts:
                if cp not in st.session_state.hidden_items:
                    st.session_state.hidden_items.append(cp)
                    changed = True
            if changed: st.rerun()

else:
    st.info("👈 Upload a Core Data crosstab or restore a `.castudio` project in the sidebar to begin building your map.")
