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
def process_data(file):
    """
    Robust Data Cleaning & Feature Engineering.
    Handles Currency Conversion, Band Sorting, and Rating Aggregation.
    """
    # Load Data (Pyxlsb for binary excel support)
    try:
        df = pd.read_excel(file, engine='pyxlsb')
    except:
        df = pd.read_excel(file) # Fallback for xlsx

    df.columns = df.columns.str.strip()

    # --- A. CURRENCY CONVERSION (PPP Adjustment) ---
    # Nominal comparison is invalid. We use PPP factors (Approximate for 2025).
    ppp_factors = {'USD': 1.0, 'INR': 1/22.54, 'PHP': 1/19.16}
    
    if 'Currency' not in df.columns:
        df['Currency'] = 'USD' # Default fallback
    
    # Calculate Factors
    df['PPP_Factor'] = df['Currency'].map(ppp_factors).fillna(1.0)
    
    # Convert Pay Columns to PPP USD
    pay_cols =
    for col in pay_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df[col] * df['PPP_Factor']

    # --- B. FEATURE ENGINEERING: BAND HIERARCHY ---
    # Explicit order is critical for Regression Reference Category and Visuals
    # Bands: A3 (Entry) -> D3 (Leadership)
    hierarchy =
    
    # Create Ordered Categorical Variable
    # We filter hierarchy to only include bands present in data to avoid errors
    present_bands =.unique()]
    band_type = pd.CategoricalDtype(categories=hierarchy, ordered=True)
    df = df.astype(band_type)
    
    # --- C. FEATURE ENGINEERING: RATINGS ---
    # Average multiple years to get a stable "Performance" metric
    rating_cols =
    for col in rating_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if rating_cols:
        df = df[rating_cols].mean(axis=1)
    else:
        df = 3.0 # Neutral default if missing

    # --- D. FEATURE ENGINEERING: EXPERIENCE & COMPA-RATIO ---
    if 'Experience' in df.columns:
        df['Clean_Experience'] = pd.to_numeric(df['Experience'], errors='coerce')
    
    # Calculate Compa-Ratio (Internal Pay vs Market Median)
    if 'P50 (PPP USD)' in df.columns and 'Annual_TCC (PPP USD)' in df.columns:
        df = df / df

    return df

# -----------------------------------------------------------------------------
# 3. SIDEBAR & INITIALIZATION
# -----------------------------------------------------------------------------
st.sidebar.header("Data Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload Wipro Database (.xlsb)", type=['xlsb', 'xlsx'])

if uploaded_file:
    df = process_data(uploaded_file)
    st.sidebar.success("Data Successfully Processed")
    st.sidebar.markdown(f"**Records:** {len(df):,}")
    
    # Tabs for Report Sections
    tab1, tab2, tab3 = st.tabs()

    # -------------------------------------------------------------------------
    # TAB 1: DESCRIPTIVE (Brief Overview)
    # -------------------------------------------------------------------------
    with tab1:
        st.subheader("Diagnostic Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Pay (PPP)", f"${df.mean():,.0f}")
        col2.metric("Avg Experience", f"{df['Clean_Experience'].mean():.1f} Years")
        col3.metric("Avg Compa-Ratio", f"{df.mean():.2f}")
        
        st.markdown("### Pay vs Experience (Raw Data)")
        st.caption("The 'Descriptive' view showing the raw correlation, colored by Job Band.")
        fig = px.scatter(df, x='Clean_Experience', y='Annual_TCC (PPP USD)', color='Band', 
                         opacity=0.5, height=500)
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 2: STATISTICAL RIGOR (The Mincer Equation)
    # -------------------------------------------------------------------------
    with tab2:
        st.header("The 'Mincer Equation' Analysis")
        st.markdown("""
        To isolate the impact of Experience independent of Band, we run a **Multiple Linear Regression**.
        This overcomes the 'Omitted Variable Bias' of the previous draft.
        
        **Equation:** $$Pay = \\beta_0 + \\beta_1(Exp) + \\beta_2(Perf) + \\beta_3(Band)$$
        """)

        # 1. Prepare Data for Regression (Drop NaNs in relevant cols)
        reg_cols =
        df_reg = df[reg_cols].dropna().copy()

        # 2. Run OLS using Statsmodels (Formula API)
        # Q() handles spaces in column names
        formula = "Q('Annual_TCC (PPP USD)') ~ Clean_Experience + Performance_Rating + C(Band)"
        model = smf.ols(formula=formula, data=df_reg).fit()

        # 3. Display Results
        col_res1, col_res2 = st.columns()
        
        with col_res1:
            st.subheader("Model Key Metrics")
            st.metric("R-Squared", f"{model.rsquared:.3f}", delta_color="normal", 
                     help="Percentage of pay variation explained by the model")
            st.metric("Adj. R-Squared", f"{model.rsquared_adj:.3f}")
            st.metric("AIC (Model Fit)", f"{model.aic:,.0f}")
            
            st.markdown("---")
            st.markdown("**Coefficient Interpretation:**")
            
            # Extract Intercept & Exp Coefficient safely
            params = model.params
            intercept = params['Intercept']
            exp_coef = params['Clean_Experience']
            perf_coef = params
            
            st.write(f"**Base Pay (Intercept):** ${intercept:,.2f}")
            st.write(f"**Value of 1 Year Exp:** ${exp_coef:,.2f}")
            st.write(f"**Value of 1 Rating Point:** ${perf_coef:,.2f}")
            
            if perf_coef < 1000:
                st.error("Warning: Performance Pay Premium is low!")

        with col_res2:
            st.subheader("Detailed Regression Summary")
            st.caption("Full OLS output for Faculty Verification of Significance (P>|t|)")
            # Convert summary to string and display as code for "Academic" look
            st.code(model.summary().as_text(), language='text')

        # 4. Extract Coefficients for Logic B (Cache them in session state)
        st.session_state['reg_params'] = model.params

    # -------------------------------------------------------------------------
    # TAB 3: LOGIC-BASED DECISION TOOLS
    # -------------------------------------------------------------------------
    with tab3:
        st.header("Algorithmic Decision Engines")
        
        # --- LOGIC A: FLIGHT RISK PREDICTOR ---
        st.subheader("Tool A: 'Critical Talent' Flight Risk Predictor")
        st.info("Logic: Flags High Performers (Rating >= 4) with High Experience (> 3 Yrs) who are Underpaid (Compa < 0.85).")
        
        # The Boolean Mask (Logic Engine)
        risk_mask = (
            (df < 0.85) & 
            (df >= 4.0) & 
            (df['Clean_Experience'] > 3.0)
        )
        flight_risk_df = df[risk_mask].copy()
        
        # Calculate Cost to Retain (Bring to P50)
        flight_risk_df = flight_risk_df - flight_risk_df
        total_risk_cost = flight_risk_df.sum()
        
        c1, c2 = st.columns(2)
        c1.error(f"High Risk Employees Identified: {len(flight_risk_df)}")
        c2.metric("Total Budget to Stabilize (Retention Cost)", f"${total_risk_cost:,.0f}")
        
        st.markdown("**Target List for Immediate Intervention:**")
        st.dataframe(flight_risk_df])
        
        st.markdown("---")

        # --- LOGIC B: NEW HIRE OFFER CALCULATOR ---
        st.subheader("Tool B: Algorithmic Offer Calculator")
        st.info("Uses the regression coefficients from Tab 2 to generate an Internal Equity-based offer.")

        if 'reg_params' in st.session_state:
            params = st.session_state['reg_params']
            hierarchy =
            
            # User Inputs
            c_input1, c_input2 = st.columns(2)
            input_exp = c_input1.number_input("Candidate Experience (Years)", 0.0, 40.0, 5.0)
            input_band = c_input2.selectbox("Target Band", hierarchy)
            
            # The Calculator Logic
            # Formula: Intercept + (Exp * Coef) + (Band_Premium)
            # Band Premium is tricky because reference category is 0.0
            
            base_val = params['Intercept']
            exp_val = params['Clean_Experience'] * input_exp
            
            # Look up Band Coefficient
            # Statsmodels names them like "C(Band)"
            band_key = f"C(Band)"
            band_val = params.get(band_key, 0.0) # Returns 0.0 if it's the reference band or not found
            
            predicted_pay = base_val + exp_val + band_val
            
            # Display Output
            st.markdown(f"### Recommended Offer Range for {input_band} with {input_exp} Yrs Exp")
            
            col_off1, col_off2, col_off3 = st.columns(3)
            col_off1.metric("Min Offer (-10%)", f"${predicted_pay * 0.9:,.0f}")
            col_off2.metric("Target (Predicted)", f"${predicted_pay:,.0f}")
            col_off3.metric("Max Offer (+10%)", f"${predicted_pay * 1.1:,.0f}")
            
            st.caption(f"Calculation: Base (${base_val:,.0f}) + Exp Premium (${exp_val:,.0f}) + Band Premium (${band_val:,.0f})")
            
        else:
            st.warning("Please run the Regression in Tab 2 first to train the model coefficients.")

else:
    st.info("Awaiting Data Upload. Please use the sidebar to upload the Wipro Database (.xlsb or.xlsx).")

