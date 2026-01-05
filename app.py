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
    # Load Data (Pyxlsb for binary excel support, Fallback to standard xlsx)
    try:
        df = pd.read_excel(file, engine='pyxlsb')
    except:
        df = pd.read_excel(file)

    df.columns = df.columns.str.strip()

    # --- A. CURRENCY CONVERSION (PPP Adjustment) ---
    # Nominal comparison is invalid. We use PPP factors (Approximate for 2025).
    ppp_factors = {'USD': 1.0, 'INR': 1/22.54, 'PHP': 1/19.16}
    
    if 'Currency' not in df.columns:
        df['Currency'] = 'USD' # Default fallback
    
    # Calculate Factors
    df['PPP_Factor'] = df['Currency'].map(ppp_factors).fillna(1.0)
    
    # Identify Pay Columns to convert
    pay_cols = ['Annual TCC', 'Annual Base Pay', 'Target Incentive', 'P50', 'P25', 'P75']
    
    for col in pay_cols:
        if col in df.columns:
            # Create a new column with (PPP USD) suffix
            new_col_name = f"{col} (PPP USD)"
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[new_col_name] = df[col] * df['PPP_Factor']
            
    # Standardize the main target variable name for the rest of the app
    target_col = 'Annual TCC (PPP USD)'
    if target_col not in df.columns and 'Annual TCC' in df.columns:
         # If the loop above didn't catch it due to exact naming mismatch
         df[target_col] = df['Annual TCC'] * df['PPP_Factor']

    # --- B. FEATURE ENGINEERING: BAND HIERARCHY ---
    # Explicit order is critical for Regression Reference Category
    hierarchy = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "D1", "D2", "E"]
    
    # Create Ordered Categorical Variable
    if 'Band' in df.columns:
        # Filter hierarchy to only include bands actually in the dataset to prevent categoricals errors
        df['Band'] = df['Band'].astype(str).str.strip()
        df['Band'] = pd.Categorical(df['Band'], categories=hierarchy, ordered=True)
    
    # --- C. FEATURE ENGINEERING: RATINGS ---
    # Average multiple years to get a stable "Performance" metric
    rating_cols = ['Rating_2022', 'Rating_2023', 'Rating_2024', 'Performance_Rating']
    
    # Check which columns actually exist in the uploaded file
    existing_rating_cols = [c for c in rating_cols if c in df.columns]
    
    if existing_rating_cols:
        for col in existing_rating_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # Create a clean average rating column
        df['Clean_Rating'] = df[existing_rating_cols].mean(axis=1)
    else:
        df['Clean_Rating'] = 3.0 # Neutral default if missing to prevent crash

    # --- D. FEATURE ENGINEERING: EXPERIENCE & COMPA-RATIO ---
    if 'Experience' in df.columns:
        df['Clean_Experience'] = pd.to_numeric(df['Experience'], errors='coerce')
    else:
        df['Clean_Experience'] = 0.0
    
    # Calculate Compa-Ratio (Internal Pay vs Market Median)
    # Using the PPP Adjusted columns created above
    p50_col = 'P50 (PPP USD)'
    if p50_col in df.columns and target_col in df.columns:
        df['Compa_Ratio'] = df[target_col] / df[p50_col]
    else:
        df['Compa_Ratio'] = 0.0 # Default if market data missing

    return df

# -----------------------------------------------------------------------------
# 3. SIDEBAR & INITIALIZATION
# -----------------------------------------------------------------------------
st.sidebar.header("Data Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload Wipro Database (.xlsb or .xlsx)", type=['xlsb', 'xlsx'])

if uploaded_file:
    df = process_data(uploaded_file)
    
    # Ensure naming consistency for regression
    target_col = 'Annual TCC (PPP USD)'
    
    if target_col not in df.columns:
        st.error(f"Critical Error: Could not generate '{target_col}'. Check input file headers.")
        st.stop()
    
    st.sidebar.success("Data Successfully Processed")
    st.sidebar.markdown(f"**Records:** {len(df):,}")
    
    # Tabs for Report Sections
    tab1, tab2, tab3 = st.tabs(["1. Descriptive Diagnostics", "2. Statistical Rigor (OLS)", "3. Logic Engines"])

    # -------------------------------------------------------------------------
    # TAB 1: DESCRIPTIVE (Brief Overview)
    # -------------------------------------------------------------------------
    with tab1:
        st.subheader("Diagnostic Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Pay (PPP)", f"${df[target_col].mean():,.0f}")
        col2.metric("Avg Experience", f"{df['Clean_Experience'].mean():.1f} Years")
        col3.metric("Avg Compa-Ratio", f"{df['Compa_Ratio'].mean():.2f}")
        
        st.markdown("### Pay vs Experience (Raw Data)")
        st.caption("The 'Descriptive' view showing the raw correlation, colored by Job Band.")
        
        # Safe plot
        if 'Band' in df.columns:
            fig = px.scatter(df, x='Clean_Experience', y=target_col, color='Band', 
                             opacity=0.5, height=500, title="Scatter: Tenure vs Pay (by Band)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Band column missing for visualization.")

    # -------------------------------------------------------------------------
    # TAB 2: STATISTICAL RIGOR (The Mincer Equation)
    # -------------------------------------------------------------------------
    with tab2:
        st.header("The 'Mincer Equation' Analysis")
        st.markdown("""
        To isolate the impact of Experience independent of Band, we run a **Multiple Linear Regression**.
        This overcomes the 'Omitted Variable Bias' of simple correlation.
        
        **Equation:** $$Pay = \\beta_0 + \\beta_1(Exp) + \\beta_2(Perf) + \\beta_3(Band)$$
        """)
        
        

        # 1. Prepare Data for Regression (Drop NaNs in relevant cols)
        reg_cols = [target_col, 'Clean_Experience', 'Clean_Rating', 'Band']
        
        # Verify columns exist
        if all(c in df.columns for c in reg_cols):
            df_reg = df[reg_cols].dropna().copy()

            # 2. Run OLS using Statsmodels (Formula API)
            # Q() handles spaces in column names
            formula = f"Q('{target_col}') ~ Clean_Experience + Clean_Rating + C(Band)"
            model = smf.ols(formula=formula, data=df_reg).fit()

            # 3. Display Results
            col_res1, col_res2 = st.columns(2)
            
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
                perf_coef = params['Clean_Rating']
                
                st.write(f"**Base Pay (Intercept):** ${intercept:,.2f}")
                st.write(f"**Value of 1 Year Exp:** ${exp_coef:,.2f}")
                st.write(f"**Value of 1 Rating Point:** ${perf_coef:,.2f}")
                
                if perf_coef < 1000:
                    st.warning("⚠️ Insight: Performance Pay Premium is low compared to Experience.")

            with col_res2:
                st.subheader("Detailed Regression Summary")
                st.caption("Full OLS output for Faculty Verification of Significance (P>|t|)")
                # Convert summary to string and display as code for "Academic" look
                st.text(model.summary().as_text())

            # 4. Extract Coefficients for Logic B (Cache them in session state)
            st.session_state['reg_params'] = model.params
        else:
            st.error(f"Missing columns for regression. Required: {reg_cols}")

    # -------------------------------------------------------------------------
    # TAB 3: LOGIC-BASED DECISION TOOLS
    # -------------------------------------------------------------------------
    with tab3:
        st.header("Algorithmic Decision Engines")
        
        # --- LOGIC A: FLIGHT RISK PREDICTOR ---
        st.subheader("Tool A: 'Critical Talent' Flight Risk Predictor")
        st.info("Logic: Flags High Performers (Rating >= 4) with High Experience (> 3 Yrs) who are Underpaid (Compa < 0.85).")
        
        

        if 'Compa_Ratio' in df.columns:
            # The Boolean Mask (Logic Engine)
            risk_mask = (
                (df['Compa_Ratio'] < 0.85) & 
                (df['Clean_Rating'] >= 4.0) & 
                (df['Clean_Experience'] > 3.0)
            )
            flight_risk_df = df[risk_mask].copy()
            
            if not flight_risk_df.empty:
                # Calculate Cost to Retain (Bring to P50)
                # Cost = P50 - Current Pay
                if 'P50 (PPP USD)' in df.columns:
                    flight_risk_df['Retention_Cost'] = flight_risk_df['P50 (PPP USD)'] - flight_risk_df[target_col]
                    total_risk_cost = flight_risk_df['Retention_Cost'].sum()
                    
                    c1, c2 = st.columns(2)
                    c1.error(f"High Risk Employees Identified: {len(flight_risk_df)}")
                    c2.metric("Total Budget to Stabilize (Retention Cost)", f"${total_risk_cost:,.0f}")
                    
                    st.markdown("**Target List for Immediate Intervention:**")
                    st.dataframe(flight_risk_df[['ID', 'Band', 'Clean_Experience', 'Clean_Rating', 'Compa_Ratio', 'Retention_Cost']])
                else:
                     st.warning("P50 market data missing, cannot calculate retention cost.")
            else:
                st.success("No critical flight risk employees identified based on current logic criteria.")
        else:
            st.warning("Compa-Ratio could not be calculated.")
        
        st.markdown("---")

        # --- LOGIC B: NEW HIRE OFFER CALCULATOR ---
        st.subheader("Tool B: Algorithmic Offer Calculator")
        st.info("Uses the regression coefficients from Tab 2 to generate an Internal Equity-based offer.")

        if 'reg_params' in st.session_state:
            params = st.session_state['reg_params']
            # Re-define hierarchy for the select box
            hierarchy = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "D1", "D2", "E"]
            
            # User Inputs
            c_input1, c_input2 = st.columns(2)
            input_exp = c_input1.number_input("Candidate Experience (Years)", 0.0, 40.0, 5.0)
            input_band = c_input2.selectbox("Target Band", hierarchy)
            
            # The Calculator Logic
            # Formula: Intercept + (Exp * Coef) + (Band_Premium)
            
            # 1. Base Pay & Exp Premium
            base_val = params['Intercept']
            exp_val = params['Clean_Experience'] * input_exp
            
            # 2. Band Premium
            # Statsmodels formats categorical coefficients as C(Band)[T.BandName]
            # The "Reference Category" (usually A1) will not have a key, so we default to 0
            band_key = f"C(Band)[T.{input_band}]"
            band_val = params.get(band_key, 0.0) 
            
            # 3. Perf Premium (Assume '3' Average rating for new hire offer baseline)
            perf_val = params['Clean_Rating'] * 3.0
            
            predicted_pay = base_val + exp_val + band_val + perf_val
            
            # Display Output
            st.markdown(f"### Recommended Offer Range for {input_band} with {input_exp} Yrs Exp")
            
            col_off1, col_off2, col_off3 = st.columns(3)
            col_off1.metric("Min Offer (-10%)", f"${predicted_pay * 0.9:,.0f}")
            col_off2.metric("Target (Predicted)", f"${predicted_pay:,.0f}")
            col_off3.metric("Max Offer (+10%)", f"${predicted_pay * 1.1:,.0f}")
            
            st.caption(f"Calculation Breakdown: Base (${base_val:,.0f}) + Exp Premium (${exp_val:,.0f}) + Band Premium (${band_val:,.0f}) + Avg Perf Adjustment (${perf_val:,.0f})")
            
        else:
            st.warning("Please run the Regression in Tab 2 first to train the model coefficients.")

else:
    st.info("Awaiting Data Upload. Please use the sidebar to upload the Wipro Database (.xlsb or .xlsx).")
