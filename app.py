import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import plotly.express as px

# -----------------------------------------------------------------------------
# 1. APP CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Wipro Rewards Strategy Engine",
    page_icon="📊",
    layout="wide"
)

st.title("Wipro Rewards Strategy Engine")
st.markdown("""
**System Focus:** Predictive Analytics & Logic-Based Decision Support.
**Methodology:** Mincer Earnings Function (OLS Regression).
""")

# -----------------------------------------------------------------------------
# 2. DATA PIPELINE (ETL)
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_clean_data(file):
    """
    Ingests raw Wipro HR data, normalizes currency to PPP, 
    and prepares regression variables.
    """
    # Load Data (Support for both.xlsb and.xlsx)
    try:
        df = pd.read_excel(file, engine='pyxlsb')
    except:
        df = pd.read_excel(file)
        
    # Standardize Column Names (Strip whitespace)
    df.columns = df.columns.str.strip()

    # --- A. CURRENCY NORMALIZATION (PPP Adjustment) ---
    # Rates derived from internal project context
    ppp_rates = {'USD': 1.0, 'INR': 1/22.54, 'PHP': 1/19.16}
    
    if 'Currency' in df.columns:
        df['PPP_Factor'] = df['Currency'].map(ppp_rates).fillna(1.0)
    else:
        df['PPP_Factor'] = 1.0
        
    # Create Target Variable: Annual TCC (PPP USD)
    if 'Annual TCC' in df.columns:
        df = pd.to_numeric(df, errors='coerce') * df['PPP_Factor']
    
    # Create Market Benchmark: P50 (PPP USD)
    if 'P50' in df.columns:
        df['P50_PPP'] = pd.to_numeric(df['P50'], errors='coerce') * df['PPP_Factor']
        # Calculate Compa-Ratio
        df = df / df['P50_PPP']

    # --- B. HIERARCHY ENFORCEMENT ---
    # Define strict band order for regression reference levels
    band_order =
    
    if 'Band' in df.columns:
        # Filter only bands that exist in the specific dataset to prevent errors
        valid_bands =.unique()]
        df = pd.Categorical(df, categories=valid_bands, ordered=True)

    # --- C. PERFORMANCE METRIC AGGREGATION ---
    # Average current and historical ratings to smooth out anomalies
    rating_cols =
    valid_rating_cols = [c for c in rating_cols if c in df.columns]
    
    if valid_rating_cols:
        for col in valid_rating_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df[valid_rating_cols].mean(axis=1)
    else:
        df = 3.0 # Default fallback

    # --- D. EXPERIENCE CLEANING ---
    if 'Experience' in df.columns:
        df['Clean_Experience'] = pd.to_numeric(df['Experience'], errors='coerce')

    # --- E. COLUMN SELECTION ---
    # Keep only relevant columns for analysis to optimize memory
    keep_cols =
    # Filter for columns that actually exist
    final_cols = [c for c in keep_cols if c in df.columns]
    
    return df[final_cols].copy()

# -----------------------------------------------------------------------------
# 3. SIDEBAR & EXECUTION
# -----------------------------------------------------------------------------
st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload Wipro Data (.xlsb)", type=['xlsb', 'xlsx'])

if uploaded_file:
    df = load_and_clean_data(uploaded_file)
    st.sidebar.success(f"Loaded {len(df):,} records")
    
    # Create Tabs
    tab_reg, tab_tools = st.tabs()

    # =========================================================================
    # TAB 1: REGRESSION ANALYSIS (The Mincer Equation)
    # =========================================================================
    with tab_reg:
        st.subheader("Statistical Validation: The Mincer Earnings Function")
        st.markdown("Isolating the value of **Experience** while controlling for **Job Band** and **Performance**.")
        
        # 1. Run Regression
        # Drop missing values only for the regression subset
        reg_df = df.dropna(subset=)
        
        formula = "Annual_TCC_PPP ~ Clean_Experience + Avg_Rating + C(Band)"
        model = smf.ols(formula=formula, data=reg_df).fit()
        
        # 2. Extract Key Metrics
        r2 = model.rsquared
        params = model.params
        
        # 3. Display Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Model Fit (R²)", f"{r2:.1%}", help="Explains X% of pay variation")
        col2.metric("Base Pay (Intercept)", f"${params['Intercept']:,.0f}")
        col3.metric("Exp Premium (per Year)", f"${params['Clean_Experience']:,.0f}")

        # 4. Detailed Summary
        st.markdown("### Regression Coefficients (The 'Price' of Attributes)")
        
        # Format coefficients into a clean dataframe for display
        coef_data =
        for name, value in params.items():
            if "Band" in name:
                readable_name = name.replace("C(Band)", "")
                coef_data.append({"Factor": readable_name, "Premium ($)": value})
            elif "Experience" in name:
                coef_data.append({"Factor": "Years of Experience", "Premium ($)": value})
            elif "Rating" in name:
                coef_data.append({"Factor": "Performance Rating (1 pt)", "Premium ($)": value})
        
        coef_df = pd.DataFrame(coef_data).set_index("Factor")
        st.dataframe(coef_df.style.format("${:,.0f}"), use_container_width=True)
        
        with st.expander("View Full Statistical Output (OLS Summary)"):
            st.code(model.summary().as_text())
            
        # Store params for the Calculator
        st.session_state['reg_params'] = params

    # =========================================================================
    # TAB 2: DECISION TOOLS (Logic Engines)
    # =========================================================================
    with tab_tools:
        st.header("Logic-Based Decision Engines")
        
        col_tool_a, col_tool_b = st.columns(2)
        
        # --- TOOL A: FLIGHT RISK PREDICTOR ---
        with col_tool_a:
            st.subheader("🚨 Critical Talent Flight Risk")
            st.info("Logic: High Performer (4+) & Experienced (>3yr) & Underpaid (<85% Market).")
            
            if 'Compa_Ratio' in df.columns:
                # Logic Engine A
                risk_mask = (
                    (df < 0.85) &
                    (df >= 4.0) &
                    (df['Clean_Experience'] > 3.0)
                )
                
                risk_df = df[risk_mask].copy()
                
                # Remediation Cost Logic
                risk_df = risk_df['P50_PPP'] - risk_df
                total_cost = risk_df.sum()
                
                st.metric("Total Retention Budget Needed", f"${total_cost:,.0f}")
                st.warning(f"{len(risk_df)} High-Risk Employees Identified")
                
                st.dataframe(
                    risk_df]
                   .style.format({'Compa_Ratio': '{:.2f}', 'Cost_to_Retain': '${:,.0f}'}),
                    height=300
                )
            else:
                st.error("Missing Market Data (P50) for Risk Analysis")

        # --- TOOL B: NEW HIRE OFFER CALCULATOR ---
        with col_tool_b:
            st.subheader("💰 Algorithmic Offer Calculator")
            st.info("Logic: Uses Regression Coefficients to ensure Internal Equity.")
            
            if 'reg_params' in st.session_state:
                p = st.session_state['reg_params']
                
                # Inputs
                bands =.unique()]
                input_band = st.selectbox("Select Band", bands, index=2) # Default B1/B2
                input_exp = st.number_input("Years of Experience", 0.0, 30.0, 5.0)
                
                # Logic Engine B: Math
                # Pay = Intercept + (Exp * Coef) + (Band * Coef)
                base = p['Intercept']
                exp_val = p['Clean_Experience'] * input_exp
                
                # Handle Categorical Lookup safely
                band_key = f"C(Band)"
                band_val = p.get(band_key, 0.0) # 0.0 if reference category
                
                predicted_offer = base + exp_val + band_val
                
                # Output Visuals
                st.markdown("### Recommended Offer Corridor")
                c1, c2, c3 = st.columns(3)
                c1.metric("Min (-10%)", f"${predicted_offer*0.9:,.0f}")
                c2.metric("Target (Equity)", f"${predicted_offer:,.0f}")
                c3.metric("Max (+10%)", f"${predicted_offer*1.1:,.0f}")
                
                # Logic Explanation
                st.caption(f"""
                **Math Breakdown:**
                Base (${base:,.0f}) + 
                Exp Premium (${exp_val:,.0f}) + 
                Band Premium (${band_val:,.0f})
                """)
                
            else:
                st.warning("Please run the Regression analysis in the first tab to calibrate the model.")

else:
    st.info("👆 Please upload the Wipro data file to begin.")
