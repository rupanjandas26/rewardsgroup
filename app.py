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
    # A. Ingestion strategy for various file types
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.xlsb'):
            df = pd.read_excel(file, engine='pyxlsb')
        else:
            df = pd.read_excel(file) 
    except Exception as e:
        return None, f"Error loading file: {e}"

    # Clean headers
    df.columns = df.columns.str.strip()

    # --- LOGIC 1: CURRENCY & PPP NORMALIZATION ---
    # We prefer the pre-calculated 'Annual_TCC (PPP USD)' if it exists.
    
    target_col = 'Annual_TCC_PPP'
    
    if 'Annual_TCC (PPP USD)' in df.columns:
        df[target_col] = pd.to_numeric(df['Annual_TCC (PPP USD)'], errors='coerce')
    elif 'Annual_TCC' in df.columns:
        # Fallback PPP Logic
        ppp_factors = {'USD': 1.0, 'INR': 1/22.54, 'PHP': 1/19.16} 
        # Ensure Currency column exists, default to USD if missing
        curr_col = df['Currency'] if 'Currency' in df.columns else pd.Series(['USD']*len(df))
        df['PPP_Factor'] = curr_col.map(ppp_factors).fillna(1.0)
        df[target_col] = pd.to_numeric(df['Annual_TCC'], errors='coerce') * df['PPP_Factor']
    else:
        return None, "Critical Error: No Pay Column (Annual_TCC) found."

    # --- LOGIC 2: 3-YEAR PERFORMANCE AVERAGE (FIXED) ---
    # CRITICAL FIX: Explicitly convert to numeric before calculating mean
    rating_cols = ['Current_Rating', 'Previous_Rating', 'Pre_Previous_Rating']
    existing_ratings = [c for c in rating_cols if c in df.columns]
    
    if existing_ratings:
        # Force numeric conversion (turns ' ' or errors into NaN)
        for col in existing_ratings:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Row-wise mean, ignoring NaNs
        df['Avg_Rating'] = df[existing_ratings].mean(axis=1)
    elif 'Performance_Rating' in df.columns:
        df['Avg_Rating'] = pd.to_numeric(df['Performance_Rating'], errors='coerce')
    else:
        df['Avg_Rating'] = 3.0 # Neutral default

    # Fill NaN ratings with a neutral 3.0 to prevent regression errors later
    df['Avg_Rating'] = df['Avg_Rating'].fillna(3.0)

    # --- LOGIC 3: EXPERIENCE VS TENURE ---
    if 'Experience' in df.columns:
        df['Total_Experience'] = pd.to_numeric(df['Experience'], errors='coerce')
    elif 'Tenure' in df.columns:
        df['Total_Experience'] = pd.to_numeric(df['Tenure'], errors='coerce')
    else:
        df['Total_Experience'] = 0.0
    
    # Fill NaN experience with 0
    df['Total_Experience'] = df['Total_Experience'].fillna(0.0)

    # --- LOGIC 4: BAND HIERARCHY ---
    # Critical for OLS to set the "Reference Group" correctly
    band_order = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'D1', 'D2', 'E']
    
    if 'Band' in df.columns:
        # Ensure Band is string first
        df['Band'] = df['Band'].astype(str).str.strip()
        
        # Only keep bands that exist in this dataset to prevent categorical errors
        unique_bands_in_data = df['Band'].unique()
        found_bands = [b for b in band_order if b in unique_bands_in_data]
        
        # Add any bands found in data but not in our list to the end
        extra_bands = [b for b in unique_bands_in_data if b not in found_bands]
        final_bands = found_bands + extra_bands
        
        df['Band'] = pd.Categorical(df['Band'], categories=final_bands, ordered=True)

    # --- LOGIC 5: MARKET BENCHMARK ---
    if 'P50 (PPP USD)' in df.columns:
        df['Market_P50'] = pd.to_numeric(df['P50 (PPP USD)'], errors='coerce')
        df['Compa_Ratio'] = df[target_col] / df['Market_P50']
    elif 'Compa_Ratio' in df.columns:
        df['Compa_Ratio'] = pd.to_numeric(df['Compa_Ratio'], errors='coerce')
    else:
        df['Compa_Ratio'] = 1.0 # Default

    # Final Cleanup: Drop rows where critical modeling data is missing
    modeling_cols = [target_col, 'Total_Experience', 'Avg_Rating', 'Band']
    df_clean = df.dropna(subset=modeling_cols).copy()
    
    return df_clean, None

# -----------------------------------------------------------------------------
# 3. SIDEBAR & EXECUTION
# -----------------------------------------------------------------------------
st.sidebar.header("Data Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload Wipro Data (.xlsx, .xlsb, .csv)", type=['xlsx', 'xlsb', 'csv'])

if uploaded_file:
    df, error_msg = load_and_process_data(uploaded_file)
    
    if error_msg:
        st.error(error_msg)
    else:
        st.sidebar.success(f"Successfully Loaded: {len(df):,} Employees")
        
        # TABS
        tab1, tab2, tab3 = st.tabs(["Regression Analysis", "K-Means Clustering", "Decision Tools"])

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
            # Q() handles spaces in column names automatically
            formula = "Q('Annual_TCC_PPP') ~ Total_Experience + Avg_Rating + C(Band)"
            try:
                model = smf.ols(formula=formula, data=df).fit()
                
                # Save for calculators
                st.session_state['reg_params'] = model.params
                
                # Metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("R-Squared (Explained Variance)", f"{model.rsquared:.2%}", "Higher is better")
                c2.metric("Base Pay (Intercept)", f"${model.params['Intercept']:,.0f}")
                c3.metric("Premium per Year of Exp", f"${model.params['Total_Experience']:,.0f}")
                
                st.markdown("### 📊 Coefficient Analysis")
                st.info("This table shows exactly how much Wipro pays for each attribute, holding everything else constant.")
                
                # Extract Band Coefficients cleanly
                coef_data = []
                for idx, val in model.params.items():
                    if "Band" in idx:
                        # Clean up statsmodels naming 'C(Band)[T.A1]' -> 'A1'
                        band_name = idx.replace("C(Band)[T.", "").replace("]", "")
                        coef_data.append({"Factor": f"Promotion to {band_name}", "Pay Premium": val})
                    elif "Avg_Rating" in idx:
                        coef_data.append({"Factor": "1 Point Increase in Rating", "Pay Premium": val})
                
                coef_df = pd.DataFrame(coef_data)
                st.dataframe(coef_df.style.format({"Pay Premium": "${:,.0f}"}), use_container_width=True)
                
            except Exception as e:
                st.error(f"Regression Failed: {e}")
                st.write("Tip: Ensure your data has enough rows and the 'Band' column is clean.")

        # =====================================================================
        # TAB 2: K-MEANS CLUSTERING (AI SEGMENTATION)
        # =====================================================================
        with tab2:
            st.subheader("AI-Driven Workforce Segmentation")
            st.markdown("""
            **Why K-Means?** Regression assumes everyone follows the same rules. K-Means discovers hidden "Tribes" 
            in your workforce based on **Pay, Experience, and Performance**.
            """)
            
            # 1. Feature Selection
            # We cluster on: Experience (Seniority), Rating (Merit), Pay (Cost)
            cluster_cols = ['Total_Experience', 'Avg_Rating', 'Annual_TCC_PPP']
            
            # Ensure no NaNs in clustering data
            cluster_data = df[cluster_cols].dropna().copy()
            
            if not cluster_data.empty:
                # 2. Scaling (Crucial for K-Means)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(cluster_data)
                
                # 3. K-Means Execution (HARDCODED to 4 CLUSTERS)
                k = 4
                kmeans = KMeans(n_clusters=k, random_state=42)
                
                # Assign clusters back to the main dataframe (using index alignment)
                df.loc[cluster_data.index, 'Cluster'] = kmeans.fit_predict(X_scaled)
                
                # 4. Visualizing the Segments
                st.markdown("### 🧬 Cluster Visualization")
                # Drop rows that weren't clustered (if any)
                plot_df = df.dropna(subset=['Cluster']).copy()
                plot_df['Cluster'] = plot_df['Cluster'].astype(str)
                
                fig_cluster = px.scatter(
                    plot_df, 
                    x='Total_Experience', 
                    y='Annual_TCC_PPP', 
                    color='Cluster',
                    size='Avg_Rating',
                    hover_data=['Band', 'Avg_Rating'],
                    title="Workforce Tribes: Pay vs. Experience (Color = Cluster)",
                    template="plotly_white"
                )
                st.plotly_chart(fig_cluster, use_container_width=True)
                
                # 5. Interpreting the Clusters (The "So What?")
                st.markdown("### 🕵️ Audit: Identifying the 'Risk' Cluster")
                cluster_summary = plot_df.groupby('Cluster').agg({
                    'Annual_TCC_PPP': 'mean',
                    'Total_Experience': 'mean',
                    'Avg_Rating': 'mean',
                    'Band': 'count'
                }).reset_index()
                
                cluster_summary.columns = ['Cluster', 'Avg Pay', 'Avg Exp', 'Avg Rating', 'Count']
                
                # Display Summary (Gradient Removed to fix ImportError)
                st.dataframe(cluster_summary.style.format({
                    'Avg Pay': '${:,.0f}', 
                    'Avg Exp': '{:.1f} Yrs', 
                    'Avg Rating': '{:.2f}'
                }))
                
                st.caption("Look for clusters with **High Rating/Exp** but **Low Pay**. These are your Flight Risks.")
            else:
                st.warning("Not enough clean data for clustering.")

        # =====================================================================
        # TAB 3: LOGIC-BASED TOOLS
        # =====================================================================
        with tab3:
            st.header("Strategic HR Tools")
            
            col_a, col_b = st.columns(2)
            
            # TOOL A: New Hire Offer Calculator (Regression Based)
            with col_a:
                st.subheader("💰 Scientific Offer Calculator")
                st.info("Predicts 'Internal Equity' price using Mincer Coefficients.")
                
                # Inputs
                if 'reg_params' in st.session_state:
                    p = st.session_state['reg_params']
                    
                    # Extract bands from params keys for dropdown
                    bands = [k.replace("C(Band)[T.", "").replace("]", "") for k in p.keys() if "Band" in k]
                    
                    if bands:
                        target_band = st.selectbox("Role Band", sorted(bands))
                        candidate_exp = st.number_input("Candidate Total Experience (Yrs)", 0, 30, 5)
                        target_rating = 3.0 # Assume 'Meets Expectations' for new hire standard
                        
                        # Calculation
                        base = p['Intercept']
                        band_premium = p.get(f"C(Band)[T.{target_band}]", 0)
                        exp_val = p['Total_Experience'] * candidate_exp
                        rating_val = p['Avg_Rating'] * target_rating
                        
                        fair_pay = base + band_premium + exp_val + rating_val
                        
                        st.metric("Fair Market Offer", f"${fair_pay:,.0f}")
                        st.write(f"**Range (+/- 10%):** \n${fair_pay*0.9:,.0f} - ${fair_pay*1.1:,.0f}")
                    else:
                        st.warning("Could not extract bands from regression. Check data.")
                else:
                    st.warning("Model not trained. Go to Tab 1 first.")

            # TOOL B: Flight Risk (Boolean Logic Based)
            with col_b:
                st.subheader("🚨 'Red Zone' Flight Risk")
                st.info("Identifies Underpaid High Performers (The 'Regrettable Loss' group).")
                
                # Logic:
                # 1. High Performance (Avg Rating > 3.5)
                # 2. Experienced (> 5 Years)
                # 3. Underpaid (Compa Ratio < 0.85)
                
                risk_mask = (
                    (df['Avg_Rating'] >= 3.5) & 
                    (df['Total_Experience'] > 5.0) & 
                    (df['Compa_Ratio'] < 0.85)
                )
                
                risk_df = df[risk_mask].copy()
                
                st.metric("At-Risk Employees", len(risk_df))
                
                if not risk_df.empty:
                    # Calculate cost to fix (raise to 1.0 Compa Ratio)
                    if 'Market_P50' in risk_df.columns:
                        risk_df['Cost_to_Fix'] = (risk_df['Market_P50'] - risk_df['Annual_TCC_PPP'])
                        # Clean negative costs (if any)
                        risk_df['Cost_to_Fix'] = risk_df['Cost_to_Fix'].apply(lambda x: max(x, 0))
                        total_fix = risk_df['Cost_to_Fix'].sum()
                        st.metric("Budget to Retain", f"${total_fix:,.0f}")
                    
                    st.dataframe(risk_df[['Band', 'Avg_Rating', 'Compa_Ratio', 'Annual_TCC_PPP']])
                else:
                    st.success("No critical flight risks identified based on current logic.")

else:
    st.info("Waiting for data file... Please upload .xlsb, .xlsx, or .csv")
