# -*- coding: utf-8 -*-
"""OpenBeam-ML: Web-based Diagnostic Benchmark Tool for Predicting the Shear Strength
of Reinforced Concrete Deep Beams with Web Openings.

Modeled strictly after OpenPunch-RF (Figure 10 of rev2 paper and Streamlit Cloud app):
https://flatslabswithopening-3cv4z4eknhg6afxk27ee5z.streamlit.app/

Run via command:
    streamlit run app.py
"""
import os, json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
MOD = ROOT
RES = os.path.join(ROOT, 'results')
ASSETS = ROOT
BEAM_IMG = os.path.join(ASSETS, 'beam_opening.png')

st.set_page_config(
    page_title="OpenBeam-ML: Shear Strength of RC Deep Beams with Openings",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for OpenPunch-RF style matching Figure 10
st.markdown("""
<style>
    /* Metric headline cards */
    .metric-container {
        background-color: #ffffff;
        border-radius: 6px;
        padding: 14px 18px 10px 18px;
        margin-bottom: 15px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-label {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        color: #2c3e50;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1a252f;
        line-height: 1.1;
    }
    .metric-unit {
        font-size: 1.05rem;
        font-weight: 500;
        color: #7f8c8d;
    }
    .metric-bar {
        height: 6px;
        background-color: #3498db;
        border-radius: 3px;
        margin-top: 10px;
    }
    
    /* Research alert box */
    .research-alert {
        background-color: #fef9e7;
        border: 1px solid #f9e79f;
        border-radius: 6px;
        padding: 14px 18px;
        color: #7d6608;
        font-size: 0.92rem;
        line-height: 1.45;
        margin-bottom: 20px;
    }
    .warning-alert {
        background-color: #fdedec;
        border: 1px solid #f5b7b1;
        border-radius: 6px;
        padding: 12px 16px;
        color: #78281f;
        font-size: 0.90rem;
        line-height: 1.4;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    
    /* Sidebar styling */
    .sidebar-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1a252f;
        margin-bottom: 12px;
    }
    .derived-box {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 0.82rem;
        line-height: 1.5;
        color: #333333;
        margin-top: 10px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_resources():
    model_path = os.path.join(MOD, 'best_model.joblib')
    meta_path = os.path.join(MOD, 'app_meta.json')
    model = joblib.load(model_path)
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    return model, meta


model, meta = load_resources()
FEATURES = meta['features']
DOM = meta['domain']

PARAM_INFO = {
    'H': ('Beam height', 'H', 'mm'),
    'b': ('Beam width', 'b', 'mm'),
    "f'c": ('Concrete compressive strength', "f'c", 'MPa'),
    'a/H': ('Shear span-to-height ratio', 'a/H', '-'),
    'ax/a': ('Opening width ratio', 'ax/a', '-'),
    'ay/H': ('Opening height ratio', 'ay/H', '-'),
    'Xo/a': ('Opening horizontal position ratio', 'Xo/a', '-'),
    'Yo/H': ('Opening vertical position ratio', 'Yo/H', '-'),
    'ps': ('Main longitudinal reinforcement ratio', 'ρs', '%'),
    'psw': ('Vertical web reinforcement ratio', 'ρsw', '%'),
    'psh': ('Horizontal web reinforcement ratio', 'ρsh', '%'),
}

# =========================================================================
# SIDEBAR: Input Case & Derived Variables (OpenPunch-RF Layout)
# =========================================================================
with st.sidebar:
    st.markdown('<div class="sidebar-header">Input case</div>', unsafe_allow_html=True)
    
    # 1. Geometry & Material
    c_h, c_fc = st.columns(2)
    H = c_h.number_input('h — beam height (mm)', min_value=150.0, max_value=2000.0, value=500.0, step=10.0)
    fc = c_fc.number_input("f'c (MPa)", min_value=10.0, max_value=150.0, value=40.0, step=1.0)
    
    c_b, c_a = st.columns(2)
    b = c_b.number_input('b — beam width (mm)', min_value=50.0, max_value=1000.0, value=120.0, step=5.0)
    a = c_a.number_input('a — shear span (mm)', min_value=100.0, max_value=3000.0, value=450.0, step=10.0)
    
    # Reinforcement ratios (%)
    st.markdown('**Reinforcement ratios (%)**')
    c_ps, c_psw = st.columns(2)
    ps = c_ps.number_input('ρs (%)', min_value=0.1, max_value=10.0, value=3.2, step=0.1, help="Main longitudinal flexural tensile reinforcement")
    psw = c_psw.number_input('ρsw (%)', min_value=0.0, max_value=6.0, value=0.65, step=0.05, help="Vertical web stirrup reinforcement ratio")
    psh = st.number_input('ρsh (%)', min_value=0.0, max_value=4.0, value=0.30, step=0.05, help="Horizontal web reinforcement ratio")
    
    # Opening configuration
    st.markdown('---')
    st.markdown('<div style="font-size:0.75rem; font-weight:700; color:#555; text-transform:uppercase;">OPENING CONFIGURATION</div>', unsafe_allow_html=True)
    opening_case = st.selectbox(
        'Case',
        options=['Solid beam / no opening', 'Beam with opening in shear span'],
        index=1
    )
    
    has_open = (opening_case == 'Beam with opening in shear span')
    if has_open:
        c_ax, c_ay = st.columns(2)
        ax_val = c_ax.number_input('ax — opening width (mm)', min_value=0.0, max_value=1500.0, value=150.0, step=5.0)
        ay_val = c_ay.number_input('ay — opening height (mm)', min_value=0.0, max_value=1500.0, value=120.0, step=5.0)
        
        c_xo, c_yo = st.columns(2)
        Xo_val = c_xo.number_input('Xo — position (mm)', min_value=0.0, max_value=3000.0, value=150.0, step=5.0, help="Horizontal distance from support to opening edge")
        Yo_val = c_yo.number_input('Yo — position (mm)', min_value=0.0, max_value=2000.0, value=180.0, step=5.0, help="Vertical distance from beam soffit to opening edge")
    else:
        ax_val = 0.0
        ay_val = 0.0
        Xo_val = 0.0
        Yo_val = 0.0
        
    # Dimensionless ratios calculation
    ratio_a_H = a / H
    ratio_ax_a = ax_val / a if a > 0 else 0.0
    ratio_ay_H = ay_val / H if H > 0 else 0.0
    ratio_Xo_a = Xo_val / a if a > 0 else 0.0
    ratio_Yo_H = Yo_val / H if H > 0 else 0.0
    
    # Sidebar box: DERIVED MODEL VARIABLES (exact match of OpenPunch-RF)
    st.markdown('<div style="font-size:0.75rem; font-weight:700; color:#555; text-transform:uppercase; margin-top:10px;">DERIVED MODEL VARIABLES</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="derived-box">
        <b>a/H</b> = {ratio_a_H:.3f}<br>
        <b>ax/a</b> = {ratio_ax_a:.3f}<br>
        <b>ay/H</b> = {ratio_ay_H:.3f}<br>
        <b>Xo/a</b> = {ratio_Xo_a:.3f}<br>
        <b>Yo/H</b> = {ratio_Yo_H:.3f}<br>
        <b>f'c</b> = {fc:.1f} MPa<br>
        <b>ρs</b> = {ps:.2f} %<br>
        <b>ρsw</b> = {psw:.2f} %<br>
        <b>ρsh</b> = {psh:.2f} %
    </div>
    """, unsafe_allow_html=True)
    
    # Optional image toggle in sidebar
    if os.path.exists(BEAM_IMG):
        with st.expander("📷 Xem sơ đồ dầm (Sketch)", expanded=False):
            st.image(BEAM_IMG, caption="Sơ đồ ký hiệu hình học", use_container_width=True)
            
    btn_predict = st.button("Run ML diagnostic", type="primary", use_container_width=True)

# Build feature dictionary
input_dict = {
    'H': float(H),
    'b': float(b),
    "f'c": float(fc),
    'a/H': float(ratio_a_H),
    'ax/a': float(ratio_ax_a),
    'ay/H': float(ratio_ay_H),
    'Xo/a': float(ratio_Xo_a),
    'Yo/H': float(ratio_Yo_H),
    'ps': float(ps),
    'psw': float(psw),
    'psh': float(psh)
}
X_input = pd.DataFrame([input_dict])[FEATURES]

# Predict with model
predicted_vu = float(model.predict(X_input)[0])
nominal_stress = (predicted_vu * 1000.0) / (b * H) if (b * H) > 0 else 0.0

# =========================================================================
# MAIN APP BODY: Matches OpenPunch-RF Structure (Header, Cards, Table, Notes)
# =========================================================================

# App Header
st.title("OpenBeam-ML")
st.markdown(
    "A diagnostic benchmark tool for shear strength prediction of RC deep beams with web openings. "
    f"The interface keeps practical inputs **h, b, a, opening dimensions ($a_x, a_y, X_o, Y_o$), and reinforcement ratios**, "
    f"while internally deriving the variables required by the optimized **{meta['best_model']}** model."
)

# Research diagnostic alert box (Figure 10 banner)
st.markdown("""
<div class="research-alert">
    <b>Research diagnostic.</b> Predictions are data-driven and should be interpreted within the experimental domain used to train the model. 
    The application is intended for research and preliminary assessment, not as a replacement for code-based design verification.
</div>
""", unsafe_allow_html=True)

# Navigation tabs for multiple modes
tab_single, tab_batch, tab_about = st.tabs([
    "Headline diagnostic — single case",
    "Batch processing (CSV)",
    "Model architecture & training data"
])

# -------------------------------------------------------------------------
# TAB 1: Single Case Diagnostic (Exact replica of Figure 10)
# -------------------------------------------------------------------------
with tab_single:
    # Embedded Geometric Schematic Image
    if os.path.exists(BEAM_IMG):
        with st.expander("📐 Sơ đồ hình học dầm cao BTCT có lỗ mở bụng dầm (Geometric notation & layout)", expanded=True):
            col_img, col_txt = st.columns([1.5, 1])
            with col_img:
                st.image(BEAM_IMG, caption="Sơ đồ ký hiệu kích thước hình học dầm và lỗ khoét chịu tải trọng tập trung P", use_container_width=True)
            with col_txt:
                st.markdown("""
                **Ký hiệu hình học chính:**
                * **$H$**: Chiều cao tổng thể của dầm (mm)
                * **$L$**: Chiều dài nhịp dầm giữa hai gối tựa (mm)
                * **$a$**: Chiều dài nhịp cắt từ gối tựa đến điểm đặt tải trọng $P$ (mm)
                * **$P$**: Tải trọng tập trung tác dụng ở mặt trên dầm
                * **$a_x$**: Bề rộng lỗ khoét xuyên bụng dầm (mm)
                * **$a_y$**: Chiều cao lỗ khoét xuyên bụng dầm (mm)
                * **$X_o$**: Khoảng cách ngang từ gối tựa đến mép lỗ mở (mm)
                * **$Y_o$**: Khoảng cách đứng từ mép đáy dầm đến mép dưới lỗ mở (mm)
                * **$b$**: Bề rộng tiết diện dầm (mm)
                """)
                
    st.markdown("### Headline diagnostic — single case")
    
    # 3 Metric Headline Cards (Figure 10 top row)
    col_card1, col_card2, col_card3 = st.columns(3)
    
    with col_card1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">OPTIMIZED {meta['best_model'].upper()} (PREDICTED VU)</div>
            <div class="metric-value">{predicted_vu:,.1f} <span class="metric-unit">kN</span></div>
            <div class="metric-bar"></div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_card2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">NOMINAL SHEAR STRESS, VU / (B · H)</div>
            <div class="metric-value">{nominal_stress:.2f} <span class="metric-unit">MPa</span></div>
            <div class="metric-bar"></div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_card3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">DERIVED SHEAR-SPAN RATIO, A/H</div>
            <div class="metric-value">{ratio_a_H:.3f}</div>
            <div class="metric-bar"></div>
        </div>
        """, unsafe_allow_html=True)

    # Applicability bracket — model-input diagnostics (Figure 10 Table)
    st.markdown("### Applicability bracket — model-input diagnostics")
    
    table_rows = []
    outside_features = []
    for f in FEATURES:
        var_name, var_sym, var_unit = PARAM_INFO[f]
        val = input_dict[f]
        t_min = DOM[f]['min']
        t_max = DOM[f]['max']
        is_in = (t_min <= val <= t_max)
        if not is_in:
            outside_features.append(var_sym)
        status = "Within range" if is_in else "Outside range"
        table_rows.append({
            "Variable": var_name,
            "Symbol": var_sym,
            "Current value": f"{val:.3f}",
            "Unit": var_unit,
            "Training min": f"{t_min:.3f}",
            "Training max": f"{t_max:.3f}",
            "Status": status
        })
        
    df_diag = pd.DataFrame(table_rows)
    
    def highlight_status(val):
        if val == 'Outside range':
            return 'background-color: #fadbd8; color: #78281f; font-weight: bold;'
        return 'background-color: #eafaf1; color: #145a32;'

    st.dataframe(
        df_diag.style.map(highlight_status, subset=['Status']),
        use_container_width=True,
        hide_index=True
    )

    # Geometry-region interpretation (Figure 10 Warning)
    st.markdown("### Geometry-region interpretation")
    
    physical_warnings = []
    if has_open:
        if (Yo_val + ay_val) > H:
            physical_warnings.append(f"Opening extends above beam top surface ($Y_o + a_y = {Yo_val + ay_val:.1f} > H = {H:.1f}$ mm).")
        if (Xo_val + ax_val) > (a * 1.5):
            physical_warnings.append(f"Opening position extends far beyond shear span ($X_o + a_x > 1.5a$).")
            
    for pw in physical_warnings:
        st.error(f"⚠️ Geometry Error: {pw}")
        
    if outside_features:
        out_str = ", ".join(outside_features)
        st.markdown(f"""
        <div class="warning-alert">
            <b>Applicability warning.</b> The following predictor(s) fall outside the reported training ranges: <b>{out_str}</b>. 
            The {meta['best_model']} prediction should therefore be interpreted with additional caution.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success("✅ All predictor variables lie securely within the experimental database range.")

    # Expanders at bottom (Figure 10 exact expanders)
    with st.expander(f"▶ {meta['best_model']} model information"):
        st.markdown(f"""
        * **Algorithm:** {meta['best_model']} (Trained and optimized using Bayesian Optimization over 100 trials).
        * **Objective function:** 5-fold cross-validation Mean Absolute Error (MAE) exclusively on 80% training data (157 specimens).
        * **Testing Performance (20% unseen data):**
          * Coefficient of Determination ($R^2$): **{meta['test_metrics']['R2']:.4f}**
          * Root Mean Squared Error (RMSE): **{meta['test_metrics']['RMSE']:.2f} kN**
          * Mean Absolute Error (MAE): **{meta['test_metrics']['MAE']:.2f} kN**
          * Mean Absolute Percentage Error (MAPE): **{meta['test_metrics']['MAPE']:.2f} %**
        * **Database scale:** 197 high-quality experimental deep beams with web openings ($H = 300 - 800$ mm, $V_u = 51 - 1248$ kN).
        """)

    with st.expander("▶ Predictor transformation used by the app"):
        st.markdown("""
        The web interface accepts structural dimensions ($H, b, a, a_x, a_y, X_o, Y_o$) and automatically translates them into 
        mechanically meaningful dimensionless parameters required by the machine learning engine:
        * **Shear span-to-height ratio:** $a/H = a / H$ (controls direct arch inclination $\\theta$)
        * **Opening width ratio:** $a_x/a = a_x / a$ (quantifies horizontal interruption length along the diagonal compression strut)
        * **Opening height ratio:** $a_y/H = a_y / H$ (quantifies vertical loss of web depth)
        * **Opening coordinates:** $X_o/a$ (relative horizontal position from support), $Y_o/H$ (relative vertical position from soffit)
        * **Reinforcement ratios:** $\\rho_s$ (longitudinal tension tie), $\\rho_{sw}$ (vertical shear stirrups), $\\rho_{sh}$ (horizontal web bars).
        """)

    with st.expander("▶ Important modelling note & structural mechanics"):
        st.markdown("""
        * **Load transfer mechanism:** Deep beams with $a/H \\le 2.0$ transmit shear primarily through **direct strut action** (compression strut connecting load point to support).
        * **Opening interruption:** A web opening in the shear span interrupts this direct path, forcing the compressive stress field to split into upper and lower chords, 
          leading to early diagonal cracking at opening corners and significant reduction in ultimate shear capacity $V_u$.
        * **Scope:** This tool is intended for rapid preliminary design, parametric investigations, and checking existing beams. 
          It does not replace comprehensive structural engineering calculations or compliance with relevant building codes (ACI 318, Eurocode 2).
        """)

# -------------------------------------------------------------------------
# TAB 2: Batch Processing (CSV)
# -------------------------------------------------------------------------
with tab_batch:
    st.markdown("### Batch prediction (CSV upload)")
    st.write("Upload a CSV file containing multiple deep beam configurations to compute predictions simultaneously.")
    
    sample_data = {
        'H': [500.0, 600.0, 400.0],
        'b': [120.0, 150.0, 100.0],
        "f'c": [40.0, 50.0, 35.0],
        'a/H': [0.90, 0.75, 1.10],
        'ax/a': [0.30, 0.25, 0.40],
        'ay/H': [0.20, 0.18, 0.25],
        'Xo/a': [0.50, 0.45, 0.55],
        'Yo/H': [0.50, 0.50, 0.50],
        'ps': [3.20, 4.00, 2.80],
        'psw': [0.65, 0.80, 0.50],
        'psh': [0.30, 0.40, 0.20]
    }
    df_template = pd.DataFrame(sample_data)
    csv_bytes = df_template.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV Template", data=csv_bytes, file_name="deep_beam_template.csv", mime="text/csv")
    
    uploaded_file = st.file_uploader("Upload CSV file for batch processing", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            batch_df.columns = [c.strip() for c in batch_df.columns]
            
            missing_cols = [f for f in FEATURES if f not in batch_df.columns]
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            else:
                preds = model.predict(batch_df[FEATURES].astype(float))
                batch_df['Predicted Vu (kN)'] = np.round(preds, 2)
                batch_df['Nominal vu (MPa)'] = np.round((batch_df['Predicted Vu (kN)'] * 1000.0) / (batch_df['b'] * batch_df['H']), 2)
                
                def check_row_domain(row):
                    out = [f for f in FEATURES if not (DOM[f]['min'] <= row[f] <= DOM[f]['max'])]
                    return "Within range" if len(out) == 0 else f"Out of bounds: {', '.join(out)}"
                    
                batch_df['Applicability Status'] = batch_df[FEATURES].apply(check_row_domain, axis=1)
                
                st.markdown("#### Batch Results:")
                st.dataframe(batch_df, use_container_width=True)
                
                res_csv = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button("💾 Download Results (CSV)", data=res_csv, file_name="batch_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error processing CSV: {e}")

# -------------------------------------------------------------------------
# TAB 3: Model & Database Information
# -------------------------------------------------------------------------
with tab_about:
    st.markdown("### Model Architecture & Experimental Database")
    st.markdown(f"""
    #### 1. Machine Learning Engine
    * **Algorithm:** {meta['best_model']}
    * **Hyperparameter optimization:** Optuna Bayesian Optimization (100 trials, Tree-structured Parzen Estimator).
    * **Cross-Validation:** 5-fold cross-validation on 80% training data to eliminate data leakage and overfitting.
    * **Evaluation metrics:**
      * $R^2$: {meta['test_metrics']['R2']:.4f}
      * RMSE: {meta['test_metrics']['RMSE']:.2f} kN
      * MAE: {meta['test_metrics']['MAE']:.2f} kN
      * MAPE: {meta['test_metrics']['MAPE']:.2f} %

    #### 2. Experimental Database
    The model was trained on an experimental database of **197 RC deep beams with web openings** compiled from major published experimental studies worldwide.
    Four extreme specimens with height $H = 2000$ mm were systematically excluded based on Bažant's Size Effect Law to prevent leverage distortion.
    
    #### 3. Range of Experimental Database
    """)
    dom_summary = []
    for f in FEATURES:
        dom_summary.append({
            "Feature": f,
            "Description": PARAM_INFO[f][0],
            "Unit": PARAM_INFO[f][2],
            "Min": DOM[f]['min'],
            "Median": DOM[f]['median'],
            "Max": DOM[f]['max']
        })
    st.table(pd.DataFrame(dom_summary))
