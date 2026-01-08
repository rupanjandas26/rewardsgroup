import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# 1. APP CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Wipro Rewards Analytics: Dual-Engine AI",
    page_icon="⚖️",
    layout="wide"
)

st.title("Wipro Rewards Analytics: Dual-Engine AI")
st.markdown("""
**System Architecture:**
1.  **Econometric Engine (Mincer OLS):** Establishes the "Fair Price" of talent based on Market Rules.
2.  **AI Engine (K-Means):** Detects "Hidden Segments" and Pay Inequity Anomalies.
""")

# -----------------------------------------------------------------------------
# 2. ROBUST DATA PIPELINE (ETL)
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_process_data(file):
    """
    Ingests Wipro data, handles PPP conversion, computes 3-Year Avg Ratings,
    and prepares variables for Mincer Regression and K-Means.
    """
    # A. Ingestion strategy
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file) 
    except Exception as e:
        return None, f"Error loading file: {e}"

    # Clean headers
    df.columns = df.columns.str.strip()

    # --- LOGIC 1: CURRENCY & PPP NORMALIZATION ---
    target_col = 'Annual_TCC_PPP'
    
    # Check for pre-calculated PPP column first
    if 'Annual_TCC (PPP USD)' in df.columns:
        df[target_col] = pd.to_numeric(df, errors='coerce')
    elif 'Annual_TCC' in df.columns:
        # Fallback PPP Logic
        ppp_factors = {'USD': 1.0, 'INR': 0.044, 'PHP': 0.052} 
        # Create Currency column if missing (default to USD)
        curr_col = df['Currency'] if 'Currency' in df.columns else pd.Series(*len(df))
        # Map currency to factor
        df['PPP_Factor'] = curr_col.map(ppp_factors).fillna(1.0)
        df[target_col] = pd.to_numeric(df, errors='coerce') * df['PPP_Factor']
    else:
        return None, "Critical Error: No Pay Column (Annual_TCC) found. Please check Excel headers."

    # --- LOGIC 2: 3-YEAR PERFORMANCE AVERAGE ---
    # Calculates 'Sustained Performance' rather than just current year
    rating_cols =
    existing_ratings = [c for c in rating_cols if c in df.columns]
    
    if existing_ratings:
        # Convert to numeric first
        for col in existing_ratings:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # Calculate row-wise mean
        df = df[existing_ratings].mean(axis=1)
    elif 'Performance_Rating' in df.columns:
        df = pd.to_numeric(df, errors='coerce')
    else:
        df = 3.0 # Default neutral rating if data is missing

    # --- LOGIC 3: EXPERIENCE VS TENURE ---
    # We use Total Experience for Market Value (Mincer Equation)
    if 'Experience' in df.columns:
        df = pd.to_numeric(df['Experience'], errors='coerce')
    elif 'Tenure' in df.columns:
        df = pd.to_numeric(df, errors='coerce')
    else:
        df = 0.0

    # --- LOGIC 4: BAND HIERARCHY ---
    # Explicit Wipro Hierarchy for correct regression reference
    band_order =
    
    if 'Band' in df.columns:
        df = df.astype(str).str.strip()
        # Filter bands to only those present in the data to avoid empty category errors
        valid_bands_in_data =.unique()]
        
        # If no standard bands found, fall back to sorted unique values
        if not valid_bands_in_data:
            valid_bands_in_data = sorted(df.unique())
            
        df = pd.Categorical(df, categories=valid_bands_in_data, ordered=True)

    # --- LOGIC 5: MARKET BENCHMARK (For Flight Risk) ---
    if 'P50 (PPP USD)' in df.columns:
        df['Market_P50'] = pd.to_numeric(df, errors='coerce')
        df = df[target_col] / df['Market_P50']
    elif 'Compa_Ratio' in df.columns:
        df = pd.to_numeric(df, errors='coerce')
        # Back-calculate P50 for cost estimation
        df['Market_P50'] = df[target_col] / df
    else:
        df = 1.0 
        df['Market_P50'] = df[target_col]

    # Drop rows with missing critical values for regression
    df_clean = df.dropna(subset=).copy()
    
    return df_clean, None

# -----------------------------------------------------------------------------
# 3. SIDEBAR & EXECUTION
# -----------------------------------------------------------------------------
st.sidebar.header("Data Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload 'my_edited_table.xlsx' or CSV", type=['xlsx', 'xlsb', 'csv'])

if uploaded_file:
    df, error_msg = load_and_process_data(uploaded_file)
    
    if error_msg:
        st.error(error_msg)
    elif df is not None and not df.empty:
        st.sidebar.success(f"Successfully Loaded: {len(df):,} Employees")
        
        # TABS
        tab1, tab2, tab3 = st.tabs()

        # =====================================================================
        # TAB 1: MINCER EARNINGS FUNCTION
        # =====================================================================
        with tab1:
            st.subheader("The Mincer Earnings Function (OLS Regression)")
            st.markdown(r"""
            **The Math:** $$Pay = \alpha + \beta_1(Exp) + \beta_2(Perf) + \beta_3(Band)$$
            
            We use **Total Experience** (not just tenure) to capture total human capital accumulation, 
            and **Average Rating** (3-year view) to capture sustained merit.
            """)
            
            # Run OLS
            # Using Q() to quote variable names with spaces if necessary
            formula = "Annual_TCC_PPP ~ Total_Experience + Avg_Rating + C(Band)"
            
            try:
                model = smf.ols(formula=formula, data=df).fit()
                
                # Save for calculators
                st.session_state['reg_params'] = model.params
                
                # Metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("R-Squared", f"{model.rsquared:.2%}", "Model Fit")
                c2.metric("Base Pay (Intercept)", f"${model.params['Intercept']:,.0f}")
                c3.metric("Exp Premium (per Year)", f"${model.params.get('Total_Experience', 0):,.0f}")
                
                st.markdown("### 📊 Coefficient Analysis")
                
                # Extract Band Coefficients cleanly
                coef_data =
                for idx, val in model.params.items():
                    if "Band" in idx:
                        # Clean up statsmodels naming "C(Band)" -> "B1"
                        band_name = idx.replace("C(Band)", "")
                        coef_data.append({"Factor": f"Promotion to {band_name}", "Pay Premium": val})
                    elif "Avg_Rating" in idx:
                        coef_data.append({"Factor": "1 Point Rating Increase", "Pay Premium": val})
                
                if coef_data:
                    coef_df = pd.DataFrame(coef_data)
                    st.dataframe(coef_df.style.format({"Pay Premium": "${:,.0f}"}), use_container_width=True)
                else:
                    st.warning("No significant band coefficients found. Ensure 'Band' column has multiple levels.")
                    
            except Exception as e:
                st.error(f"Regression Failed: {str(e)}")

        # =====================================================================
        # TAB 2: K-MEANS CLUSTERING (AI SEGMENTATION)
        # =====================================================================
        with tab2:
            st.subheader("AI-Driven Workforce Segmentation")
            st.markdown("""
            **Why K-Means?** Regression assumes linear rules for everyone. Clustering finds hidden "Tribes" 
            based on **Pay, Experience, and Performance**.
            """)
            
            # 1. Feature Selection
            cluster_cols =
            
            # Check if cols exist
            if all(c in df.columns for c in cluster_cols):
                cluster_data = df[cluster_cols].copy()
                
                # 2. Scaling
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(cluster_data)
                
                # 3. K-Means
                k = st.slider("Select Number of Clusters", 2, 6, 4)
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                df['Cluster'] = kmeans.fit_predict(X_scaled)
                
                # 4. Visualization
                fig_cluster = px.scatter(
                    df, 
                    x='Total_Experience', 
                    y='Annual_TCC_PPP', 
                    color='Cluster',
                    size='Avg_Rating',
                    title="Workforce Tribes: Pay vs. Experience (Color = Cluster)",
                    template="plotly_white"
                )
                st.plotly_chart(fig_cluster, use_container_width=True)
                
                # 5. Audit Table
                st.markdown("### 🕵️ Cluster Audit: Identify Inequity")
                cluster_summary = df.groupby('Cluster')].mean().reset_index()
                
                st.dataframe(cluster_summary.style.format({
                    'Annual_TCC_PPP': '${:,.0f}', 
                    'Total_Experience': '{:.1f} Yrs', 
                    'Avg_Rating': '{:.2f}',
                    'Compa_Ratio': '{:.2f}'
                }).background_gradient(subset=, cmap='RdYlGn'))
                
            else:
                st.error("Missing columns for clustering.")

        # =====================================================================
        # TAB 3: LOGIC-BASED TOOLS
        # =====================================================================
        with tab3:
            st.header("Strategic HR Tools")
            
            col_a, col_b = st.columns(2)
            
            # TOOL A: New Hire Offer Calculator
            with col_a:
                st.subheader("💰 Smart Offer Calculator")
                st.info("Predicts 'Internal Equity' price using Mincer Coefficients.")
                
                if 'reg_params' in st.session_state:
                    p = st.session_state['reg_params']
                    
                    # Get bands from dataframe
                    bands = sorted(df.unique().astype(str))
                    
                    target_band = st.selectbox("Role Band", bands)
                    candidate_exp = st.number_input("Candidate Experience (Yrs)", 0, 30, 5)
                    
                    # Logic
                    base = p['Intercept']
                    # Construct the correct key for statsmodels coefficient map
                    # It usually looks like "C(Band)"
                    band_key = f"C(Band)"
                    band_premium = p.get(band_key, 0.0) 
                    exp_val = p.get('Total_Experience', 0) * candidate_exp
                    # Assume target rating of 3 (Meets Expectations) for new hire
                    rating_val = p.get('Avg_Rating', 0) * 3.0 
                    
                    fair_pay = base + band_premium + exp_val + rating_val
                    
                    st.metric("Fair Market Offer", f"${fair_pay:,.0f}")
                    st.write(f"**Range (+/- 10%):** \n${fair_pay*0.9:,.0f} - ${fair_pay*1.1:,.0f}")
                else:
                    st.warning("Please run Regression (Tab 1) first.")

            # TOOL B: Flight Risk
            with col_b:
                st.subheader("🚨 Flight Risk Detector")
                st.info("Logic: High Performer (Rating > 3.5) & Underpaid (Compa < 0.85).")
                
                if 'Compa_Ratio' in df.columns:
                    risk_mask = (df >= 3.5) & (df < 0.85)
                    risk_df = df[risk_mask].copy()
                    
                    st.metric("At-Risk Employees", len(risk_df))
                    
                    if not risk_df.empty:
                        # Cost to fix = Move to Market P50
                        risk_df['Cost_to_Fix'] = risk_df['Market_P50'] - risk_df
                        total_fix = risk_df['Cost_to_Fix'].sum()
                        
                        st.metric("Retention Budget", f"${total_fix:,.0f}")
                        st.dataframe(risk_df])
                    else:
                        st.success("No critical flight risks found.")
                else:
                    st.warning("Compa-Ratio not available.")

else:
    st.info("Waiting for data file...")
