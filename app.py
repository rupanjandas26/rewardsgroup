import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.formula.api as smf
import numpy as np

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Wipro Rewards Strategy Engine",
    page_icon="wx",
    layout="wide"
)

st.title("Wipro Rewards Analytics: Strategic Decision Engine")
st.markdown("""
**System Status:** Deep Research Mode (Mincer Equation Enabled).
**Objective:** Move from 'Volume-based' (Tenure) to 'Value-based' (Performance) compensation models.
""")

# -----------------------------------------------------------------------------
# 2. ROBUST DATA LOADER (The "No-Crash" Engine)
# -----------------------------------------------------------------------------
@st.cache_data
def load_raw_data(file):
    """Loads the raw file without crashing."""
    try:
        df = pd.read_excel(file, engine='pyxlsb')
    except:
        df = pd.read_excel(file)
    # Clean whitespace from headers
    df.columns = df.columns.str.strip()
    return df

def clean_data(df, pay_col, band_col, exp_col, rating_col, mkt_col):
    """
    Applies logic to clean data based on user-selected columns.
    """
    clean_df = df.copy()
    
    # 1. Handle Currency & Pay (PPP Adjustment)
    # We assume the input is nominal; we convert to PPP USD.
    ppp_factors = {'USD': 1.0, 'INR': 1/22.54, 'PHP': 1/19.16, 'AUD': 1/1.4}
    
    if 'Currency' in clean_df.columns:
        clean_df['Currency'] = clean_df['Currency'].astype(str).str.strip().str.upper()
        clean_df['PPP_Factor'] = clean_df['Currency'].map(ppp_factors).fillna(1.0)
    else:
        clean_df['PPP_Factor'] = 1.0 # Default
        
    # Create the Target Variable
    clean_df[pay_col] = pd.to_numeric(clean_df[pay_col], errors='coerce')
    clean_df['Annual_TCC_PPP'] = clean_df[pay_col] * clean_df['PPP_Factor']
    
    # 2. Handle Band (Categorical Order)
    hierarchy = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "D1", "D2", "E", "F"]
    clean_df[band_col] = clean_df[band_col].astype(str).str.strip()
    # Only keep bands that exist in hierarchy to avoid errors
    valid_bands = [b for b in hierarchy if b in clean_df[band_col].unique()]
    clean_df['Band_Clean'] = pd.Categorical(clean_df[band_col], categories=valid_bands, ordered=True)
    
    # 3. Handle Experience (Numeric)
    clean_df['Experience_Clean'] = pd.to_numeric(clean_df[exp_col], errors='coerce').fillna(0)
    
    # 4. Handle Rating (Numeric)
    clean_df['Rating_Clean'] = pd.to_numeric(clean_df[rating_col], errors='coerce').fillna(3.0)
    
    # 5. Handle Compa-Ratio
    if mkt_col and mkt_col in clean_df.columns:
        clean_df[mkt_col] = pd.to_numeric(clean_df[mkt_col], errors='coerce')
        # Market data also needs PPP adjustment if it's in local currency
        clean_df['P50_PPP'] = clean_df[mkt_col] * clean_df['PPP_Factor']
        clean_df['Compa_Ratio'] = clean_df['Annual_TCC_PPP'] / clean_df['P50_PPP']
    else:
        clean_df['Compa_Ratio'] = 0.0
        
    return clean_df

# -----------------------------------------------------------------------------
# 3. SIDEBAR: DATA INGESTION & MAPPING
# -----------------------------------------------------------------------------
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Wipro Dataset (.xlsb/.xlsx)", type=['xlsb', 'xlsx'])

if uploaded_file:
    # Load Raw
    raw_df = load_raw_data(uploaded_file)
    st.sidebar.success("File Loaded.")
    
    st.sidebar.header("2. Map Columns")
    st.sidebar.info("Select the columns from your file that match these variables.")
    
    # --- INTELLIGENT AUTO-SELECTION ---
    # We try to guess the index of the right column to make it easier for the user
    def get_index(options, search_terms):
        for i, opt in enumerate(options):
            if any(term in opt.lower() for term in search_terms):
                return i
        return 0

    cols = raw_df.columns.tolist()
    
    # 1. Pay
    pay_idx = get_index(cols, ['tcc', 'total', 'pay', 'salary'])
    pay_col = st.sidebar.selectbox("Total Pay (Annual TCC)", cols, index=pay_idx)
    
    # 2. Band
    band_idx = get_index(cols, ['band', 'grade', 'level'])
    band_col = st.sidebar.selectbox("Job Band", cols, index=band_idx)
    
    # 3. Experience
    exp_idx = get_index(cols, ['exp', 'tenure', 'years'])
    exp_col = st.sidebar.selectbox("Experience / Tenure", cols, index=exp_idx)
    
    # 4. Performance
    rate_idx = get_index(cols, ['rating', 'perf', 'score'])
    rating_col = st.sidebar.selectbox("Performance Rating", cols, index=rate_idx)
    
    # 5. Market P50 (Optional)
    mkt_idx = get_index(cols, ['p50', 'median', 'market'])
    mkt_col = st.sidebar.selectbox("Market Median (P50)", ['None'] + cols, index=mkt_idx+1 if mkt_idx else 0)
    if mkt_col == 'None': mkt_col = None

    # --- PROCESS DATA ---
    df = clean_data(raw_df, pay_col, band_col, exp_col, rating_col, mkt_col)
    
    st.sidebar.markdown("---")
    st.sidebar.write(f"**Active Records:** {len(df):,}")

    # -------------------------------------------------------------------------
    # MAIN TABS
    # -------------------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["📊 Diagnostic Dashboard", "Ez Statistical Analysis", "🛠️ Decision Tools"])

    # --- TAB 1: DIAGNOSTICS ---
    with tab1:
        st.subheader("Current State Overview")
        
        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg Pay (PPP USD)", f"${df['Annual_TCC_PPP'].mean():,.0f}")
        m2.metric("Avg Experience", f"{df['Experience_Clean'].mean():.1f} Yrs")
        m3.metric("Avg Rating", f"{df['Rating_Clean'].mean():.2f}")
        m4.metric("Avg Compa-Ratio", f"{df['Compa_Ratio'].mean():.2f}")
        
        # Plots
        st.markdown("### 🔎 Pay vs. Experience Analysis")
        fig = px.scatter(
            df, 
            x='Experience_Clean', 
            y='Annual_TCC_PPP', 
            color='Band_Clean',
            title="Scatter: Does Tenure drive Pay? (Colored by Band)",
            hover_data=['Rating_Clean'],
            opacity=0.6,
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- TAB 2: REGRESSION (MINCER) ---
    with tab2:
        st.header("Deep Research: The Mincer Earnings Function")
        st.markdown(r"""
        We use **Multiple Linear Regression (OLS)** to isolate the specific dollar value of **Experience** while controlling for **Band** and **Performance**.
        
        $$ Pay = \alpha + \beta_1(Experience) + \beta_2(Rating) + \beta_3(Band) + \epsilon $$
        """)
        
        # Prepare Data
        reg_data = df[['Annual_TCC_PPP', 'Experience_Clean', 'Rating_Clean', 'Band_Clean']].dropna()
        
        # Run Model
        formula = "Annual_TCC_PPP ~ Experience_Clean + Rating_Clean + C(Band_Clean)"
        model = smf.ols(formula=formula, data=reg_data).fit()
        
        # Display Key Insights
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.subheader("Key Drivers Identified")
            st.metric("Model Accuracy (R²)", f"{model.rsquared:.1%}", help="How much of the pay variation is explained by these factors.")
            
            # Extract Coefficients
            base_pay = model.params['Intercept']
            exp_value = model.params['Experience_Clean']
            perf_value = model.params['Rating_Clean']
            
            st.write("---")
            st.write(f"**Base Intercept:** ${base_pay:,.0f}")
            st.write(f"**Value of 1 Year Exp:** ${exp_value:,.0f}")
            st.write(f"**Value of 1 Rating Pt:** ${perf_value:,.0f}")
            
            if perf_value < exp_value:
                st.warning("⚠️ Insight: Tenure is currently rewarded more than Performance.")
            else:
                st.success("✅ Insight: Performance is highly valued.")

        with c2:
            st.subheader("Statistical Summary (For Verification)")
            st.code(model.summary().as_text())
            
        # Save params for Tool B
        st.session_state['model_params'] = model.params

    # --- TAB 3: LOGIC TOOLS ---
    with tab3:
        st.header("Managerial Decision Tools")
        
        # TOOL A: FLIGHT RISK
        st.subheader("🚨 Tool A: 'Critical Talent' Flight Risk Detector")
        st.info("Logic: High Performer (4+) + Experienced (>3y) + Underpaid (Compa < 0.85)")
        
        risk_mask = (
            (df['Rating_Clean'] >= 4.0) & 
            (df['Experience_Clean'] > 3.0) & 
            (df['Compa_Ratio'] > 0.01) & # Ensure market data exists
            (df['Compa_Ratio'] < 0.85)
        )
        risk_df = df[risk_mask].copy()
        
        if not risk_df.empty:
            # Calculate cost to fix
            risk_df['Retention_Cost'] = risk_df['P50_PPP'] - risk_df['Annual_TCC_PPP']
            total_cost = risk_df['Retention_Cost'].sum()
            
            k1, k2 = st.columns(2)
            k1.error(f"{len(risk_df)} Employees at Risk")
            k2.metric("Budget Needed to Retain", f"${total_cost:,.0f}")
            
            st.dataframe(risk_df[[band_col, exp_col, rating_col, 'Compa_Ratio', 'Retention_Cost']])
        else:
            st.success("No Critical Risks identified with current logic.")

        st.markdown("---")

        # TOOL B: OFFER CALCULATOR
        st.subheader("🧮 Tool B: AI-Driven Offer Calculator")
        st.info("Predicts 'Fair Market Pay' based on the Regression Model trained in Tab 2.")
        
        if 'model_params' in st.session_state:
            params = st.session_state['model_params']
            
            # Inputs
            oc1, oc2, oc3 = st.columns(3)
            in_exp = oc1.number_input("Candidate Experience (Yrs)", 0, 30, 5)
            in_rating = oc2.number_input("Assumed Rating (Default 3)", 1, 5, 3)
            
            # Get bands from categorical
            avail_bands = df['Band_Clean'].unique().tolist()
            in_band = oc3.selectbox("Target Band", sorted([str(b) for b in avail_bands]))
            
            # Calculate
            # 1. Base + Exp + Rating
            prediction = params['Intercept'] + (params['Experience_Clean'] * in_exp) + (params['Rating_Clean'] * in_rating)
            
            # 2. Add Band Premium
            # Statsmodels creates keys like "C(Band_Clean)[T.B1]"
            # The reference band (first one) has no key (value 0)
            band_key = f"C(Band_Clean)[T.{in_band}]"
            band_premium = params.get(band_key, 0.0)
            
            final_offer = prediction + band_premium
            
            st.success(f"Recommended Offer: ${final_offer:,.0f}")
            st.caption(f"Range (+/- 10%): ${final_offer*0.9:,.0f} to ${final_offer*1.1:,.0f}")
            
        else:
            st.warning("Please visit 'Statistical Analysis' tab first to train the model.")

else:
    st.info("👋 Welcome! Please upload your Wipro Excel file in the sidebar to begin.")
