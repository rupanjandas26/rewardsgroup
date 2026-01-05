import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Wipro Rewards Analytics System", layout="wide")

st.title("Wipro Rewards Analytics: Strategic Decision Engine")
st.markdown("""
**System Status:** Re-Engineered for Statistical Rigor (Mincer Equation).
**Objective:** Optimize Compensation Strategy using Multiple Regression & Logic Engines.
**Strategic Context:** Transition from 'Volume' (Tenure-based) to 'Value' (Performance-based) models.
""")

# -----------------------------------------------------------------------------
# 2. DATA PROCESSING ENGINE (The ETL Layer)
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_clean_data(file):
    """
    Basic loading and cleaning. Complex mapping happens in the main app flow
    to allow for user interaction if auto-detection fails.
    """
    try:
        df = pd.read_excel(file, engine='pyxlsb')
    except:
        df = pd.read_excel(file)

    # Clean headers
    df.columns = df.columns.str.strip()
    
    # 1. Handle Currency
    # We try to auto-detect, otherwise default to USD
    if 'Currency' in df.columns:
        df['Currency'] = df['Currency'].astype(str).str.strip().str.upper()
    else:
        df['Currency'] = 'USD'

    ppp_factors = {'USD': 1.0, 'INR': 1/22.54, 'PHP': 1/19.16}
    df['PPP_Factor'] = df['Currency'].map(ppp_factors).fillna(1.0)

    # 2. Handle Band Hierarchy
    hierarchy = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "D1", "D2", "E"]
    if 'Band' in df.columns:
        df['Band'] = df['Band'].astype(str).str.strip()
        # Only keep categories that exist in the data to avoid empty category errors
        actual_bands = [b for b in hierarchy if b in df['Band'].unique()]
        df['Band'] = pd.Categorical(df['Band'], categories=actual_bands, ordered=True)

    # 3. Handle Ratings (Auto-average if multiple columns found)
    rating_cols = ['Rating_2022', 'Rating_2023', 'Rating_2024', 'Performance_Rating', 'Rating', 'Perf_Rating']
    found_rating_cols = [c for c in rating_cols if c in df.columns]
    
    if found_rating_cols:
        for col in found_rating_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['Clean_Rating'] = df[found_rating_cols].mean(axis=1)
    else:
        df['Clean_Rating'] = 3.0 # Default neutral

    # 4. Handle Experience
    # Try common names
    if 'Experience' in df.columns:
        df['Clean_Experience'] = pd.to_numeric(df['Experience'], errors='coerce')
    elif 'Tenure' in df.columns:
        df['Clean_Experience'] = pd.to_numeric(df['Tenure'], errors='coerce')
    elif 'Years_Exp' in df.columns:
        df['Clean_Experience'] = pd.to_numeric(df['Years_Exp'], errors='coerce')
    else:
        df['Clean_Experience'] = 0.0

    return df

# -----------------------------------------------------------------------------
# 3. SIDEBAR & INITIALIZATION
# -----------------------------------------------------------------------------
st.sidebar.header("Data Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload Wipro Database", type=['xlsb', 'xlsx'])

if uploaded_file:
    df = load_and_clean_data(uploaded_file)
    
    # --- INTELLIGENT COLUMN MAPPING ---
    # We define the target column name we WANT
    target_col = 'Annual TCC (PPP USD)'
    
    # 1. Try to find the Pay column automatically
    possible_pay_names = ['Annual TCC', 'Annual_TCC', 'Annual Base Pay', 'Total Pay', 'CTC', 'Fixed Pay']
    found_pay_col = None
    
    # Check exact matches first
    for candidate in possible_pay_names:
        if candidate in df.columns:
            found_pay_col = candidate
            break
            
    # 2. If not found, ASK THE USER
    if found_pay_col is None:
        st.sidebar.error("⚠️ Could not auto-detect Pay Column.")
        st.sidebar.markdown("Please select the column that represents **Annual Pay**:")
        found_pay_col = st.sidebar.selectbox("Select Pay Column", df.columns)
    else:
        st.sidebar.success(f"Mapped Pay Column: {found_pay_col}")

    # 3. Create the Standardized Target Column
    # Convert to numeric and apply PPP factor
    df[found_pay_col] = pd.to_numeric(df[found_pay_col], errors='coerce')
    df[target_col] = df[found_pay_col] * df['PPP_Factor']

    # 4. Calculate Compa-Ratio (Optional but recommended)
    if 'P50 (PPP USD)' not in df.columns:
        # Try to find a P50/Market column to map
        possible_mkt_names = ['P50', 'Market Median', 'Market P50']
        found_mkt_col = None
        for candidate in possible_mkt_names:
            if candidate in df.columns:
                found_mkt_col = candidate
                break
        
        if found_mkt_col:
            df['P50 (PPP USD)'] = pd.to_numeric(df[found_mkt_col], errors='coerce') * df['PPP_Factor']
            df['Compa_Ratio'] = df[target_col] / df['P50 (PPP USD)']
        else:
            df['Compa_Ratio'] = 1.0 # Default if market data missing
            
    st.sidebar.markdown(f"**Records Loaded:** {len(df):,}")
    
    # -------------------------------------------------------------------------
    # MAIN APP TABS
    # -------------------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["1. Descriptive Diagnostics", "2. Statistical Rigor (OLS)", "3. Logic Engines"])

    # --- TAB 1: DESCRIPTIVE ---
    with tab1:
        st.subheader("Diagnostic Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Pay (PPP)", f"${df[target_col].mean():,.0f}")
        col2.metric("Avg Experience", f"{df['Clean_Experience'].mean():.1f} Years")
        col3.metric("Avg Compa-Ratio", f"{df['Compa_Ratio'].mean():.2f}")
        
        st.markdown("### Pay vs Experience (Raw Data)")
        if 'Band' in df.columns:
            fig = px.scatter(df, x='Clean_Experience', y=target_col, color='Band', 
                             opacity=0.5, height=500, title="Scatter: Tenure vs Pay (by Band)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Band column missing. Please ensure your file has a 'Band' column.")

    # --- TAB 2: REGRESSION (MINCER EQUATION) ---
    with tab2:
        st.header("The 'Mincer Equation' Analysis")
        st.markdown("**Equation:** $$Pay = \\beta_0 + \\beta_1(Exp) + \\beta_2(Perf) + \\beta_3(Band)$$")
        
        # Prepare Data
        reg_cols = [target_col, 'Clean_Experience', 'Clean_Rating', 'Band']
        
        if all(c in df.columns for c in reg_cols):
            df_reg = df[reg_cols].dropna().copy()
            
            # Run OLS
            formula = f"Q('{target_col}') ~ Clean_Experience + Clean_Rating + C(Band)"
            model = smf.ols(formula=formula, data=df_reg).fit()

            # Display
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.subheader("Key Drivers")
                st.metric("R-Squared", f"{model.rsquared:.3f}")
                
                params = model.params
                intercept = params['Intercept']
                exp_coef = params['Clean_Experience']
                perf_coef = params['Clean_Rating']
                
                st.write(f"**Base Pay:** ${intercept:,.2f}")
                st.write(f"**Value of 1 Year Exp:** ${exp_coef:,.2f}")
                st.write(f"**Value of 1 Rating Point:** ${perf_coef:,.2f}")

            with col_res2:
                st.text(model.summary().as_text())

            st.session_state['reg_params'] = model.params
        else:
            st.error(f"Missing columns for regression. Ensure {reg_cols} exist.")

    # --- TAB 3: LOGIC ENGINES ---
    with tab3:
        st.header("Algorithmic Decision Engines")
        
        # Logic A: Flight Risk
        st.subheader("Tool A: Flight Risk Predictor")
        
        if 'Compa_Ratio' in df.columns:
            risk_mask = (
                (df['Compa_Ratio'] < 0.85) & 
                (df['Clean_Rating'] >= 4.0) & 
                (df['Clean_Experience'] > 3.0)
            )
            flight_risk_df = df[risk_mask].copy()
            
            if not flight_risk_df.empty:
                st.error(f"High Risk Employees: {len(flight_risk_df)}")
                st.dataframe(flight_risk_df)
            else:
                st.success("No critical flight risks identified.")
        
        st.markdown("---")
        
        # Logic B: Offer Calculator
        st.subheader("Tool B: Offer Calculator")
        
        if 'reg_params' in st.session_state:
            params = st.session_state['reg_params']
            
            # Input
            c1, c2 = st.columns(2)
            in_exp = c1.number_input("Experience (Years)", 0.0, 40.0, 5.0)
            
            # Get valid bands from data
            valid_bands = df['Band'].unique().tolist()
            in_band = c2.selectbox("Target Band", sorted([str(b) for b in valid_bands]))
            
            # Calculate
            base = params['Intercept']
            exp_val = params['Clean_Experience'] * in_exp
            band_key = f"C(Band)[T.{in_band}]"
            band_val = params.get(band_key, 0.0)
            perf_val = params['Clean_Rating'] * 3.0 # Assume Avg rating
            
            total = base + exp_val + band_val + perf_val
            
            st.metric("Recommended Offer", f"${total:,.0f}")
            st.caption(f"Range: ${total*0.9:,.0f} - ${total*1.1:,.0f}")

else:
    st.info("Awaiting Data Upload. Please use the sidebar to upload the Wipro Database.")
