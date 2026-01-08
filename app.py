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
    
    if 'Annual_TCC (PPP USD)' in df.columns:
        # Use existing pre-calculated column
        df[target_col] = pd.to_numeric(df, errors='coerce')
    elif 'Annual_TCC' in df.columns:
        # Fallback PPP Logic
        ppp_factors = {'USD': 1.0, 'INR': 0.044, 'PHP': 0.052} 
        curr_col = df['Currency'] if 'Currency' in df.columns else 'USD'
        # Map currency to factor, default to 1.0 if not found
        df['PPP_Factor'] = curr_col.map(ppp_factors).fillna(1.0)
        df[target_col] = pd.to_numeric(df, errors='coerce') * df['PPP_Factor']
    else:
        return None, "Critical Error: No Pay Column (Annual_TCC) found."

    # --- LOGIC 2: 3-YEAR PERFORMANCE AVERAGE ---
    potential_rating_cols =
    existing_ratings = [c for c in potential_rating_cols if c in df.columns]
    
    if existing_ratings:
        # Row-wise mean, ignoring NaNs
        for col in existing_ratings:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df[existing_ratings].mean(axis=1)
    elif 'Performance_Rating' in df.columns:
        df = pd.to_numeric(df, errors='coerce')
    else:
        df = 3.0 # Neutral default

    # --- LOGIC 3: EXPERIENCE VS TENURE ---
    # Create 'Total_Experience' column safely
    if 'Experience' in df.columns:
        df = pd.to_numeric(df['Experience'], errors='coerce')
    elif 'Tenure' in df.columns:
        df = pd.to_numeric(df, errors='coerce')
    else:
        df = 0.0

    # --- LOGIC 4: BAND HIERARCHY ---
    # Define standard Wipro bands order
    band_order =
    
    if 'Band' in df.columns:
        # Clean band data
        df = df.astype(str).str.strip()
        # Find which bands from our standard list actually exist in the data
        valid_bands_in_data =.unique()]
        
        # If no standard bands found, just use the sorted unique values from data
        if not valid_bands_in_data:
            valid_bands_in_data = sorted(df.unique())
            
        df = pd.Categorical(df, categories=valid_bands_in_data, ordered=True)

    # --- LOGIC 5: MARKET BENCHMARK ---
    # Used for Flight Risk Logic
    if 'P50 (PPP USD)' in df.columns:
        df['Market_P50'] = pd.to_numeric(df, errors='coerce')
        df = df[target_col] / df['Market_P50']
    elif 'Compa_Ratio' in df.columns:
        df = pd.to_numeric(df, errors='coerce')
        # Back-calculate P50 if missing (Pay / Ratio = P50)
        df['Market_P50'] = df[target_col] / df
    else:
        df = 1.0 
        df['Market_P50'] = df[target_col]

    # Final Cleanup: Drop rows where critical regression data is missing
    critical_cols =
    if 'Band' in df.columns:
        critical_cols.append('Band')
        
    df_clean = df.dropna(subset=critical_cols).copy()
    
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
    else:
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
            formula = "Q('Annual_TCC_PPP') ~ Total_Experience + Avg_Rating + C(Band)"
            try:
                model = smf.ols(formula=formula, data=df).fit()
                
                # Save for calculators
                st.session_state['reg_params'] = model.params
                
                # Metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("R-Squared (Explained Variance)", f"{model.rsquared:.2%}", "Higher is better")
                c2.metric("Base Pay (Intercept)", f"${model.params['Intercept']:,.0f}")
                c3.metric("Premium per Year of Exp", f"${model.params.get('Total_Experience', 0):,.0f}")
                
                st.markdown("### 📊 Coefficient Analysis")
                st.info("This table shows exactly how much Wipro pays for each attribute, holding everything else constant.")
                
                # Extract Band Coefficients cleanly
                coef_data =
                for idx, val in model.params.items():
                    if "Band" in idx:
                        # Clean up statsmodels naming "C(Band)" -> "B1"
                        band_name = idx.replace("C(Band)", "")
                        coef_data.append({"Factor": f"Promotion to {band_name}", "Pay Premium": val})
                    elif "Avg_Rating" in idx:
                        coef_data.append({"Factor": "1 Point Increase in Rating", "Pay Premium": val})
                
                if coef_data:
                    coef_df = pd.DataFrame(coef_data)
                    st.dataframe(coef_df.style.format({"Pay Premium": "${:,.0f}"}), use_container_width=True)
                else:
                    st.warning("No significant coefficients found for Bands or Rating.")
                    
            except Exception as e:
                st.error(f"Regression Failed: {e}. Ensure data has 'Band', 'Total_Experience', and 'Avg_Rating' columns.")

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
            cluster_cols =
            
            # Ensure columns exist before clustering
            if all(col in df.columns for col in cluster_cols):
                cluster_data = df[cluster_cols].copy()
                
                # 2. Scaling (Crucial for K-Means)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(cluster_data)
                
                # 3. K-Means Execution
                k = st.slider("Select Number of Clusters", 2, 6, 4)
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                df['Cluster'] = kmeans.fit_predict(X_scaled)
                
                # 4. Visualizing the Segments
                st.markdown("### 🧬 Cluster Visualization")
                fig_cluster = px.scatter(
                    df, 
                    x='Total_Experience', 
                    y='Annual_TCC_PPP', 
                    color='Cluster',
                    size='Avg_Rating',
                    hover_data=,
                    title="Workforce Tribes: Pay vs. Experience (Color = Cluster)",
                    template="plotly_white"
                )
                st.plotly_chart(fig_cluster, use_container_width=True)
                
                # 5. Interpreting the Clusters (The "So What?")
                st.markdown("### 🕵️ Audit: Identifying the 'Risk' Cluster")
                cluster_summary = df.groupby('Cluster').agg({
                    'Annual_TCC_PPP': 'mean',
                    'Total_Experience': 'mean',
                    'Avg_Rating': 'mean',
                    'Compa_Ratio': 'mean' 
                }).reset_index()
                
                cluster_summary.columns =
                
                # Highlight Logic
                st.dataframe(cluster_summary.style.format({
                    'Avg Pay': '${:,.0f}', 
                    'Avg Exp': '{:.1f} Yrs', 
                    'Avg Rating': '{:.2f}',
                    'Avg Compa-Ratio': '{:.2f}'
                }).background_gradient(subset=['Avg Pay'], cmap='RdYlGn'))
                
                st.caption("Look for clusters with **Green Rating/Exp** but **Red Pay**. These are your Flight Risks.")
            else:
                st.error("Missing columns for clustering. Need Experience, Rating, and Pay.")

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
                    # Clean up statsmodels naming "C(Band)" -> "B1"
                    bands =", "") for k in p.keys() if "Band" in k]
                    
                    if bands:
                        target_band = st.selectbox("Role Band", sorted(bands))
                        candidate_exp = st.number_input("Candidate Total Experience (Yrs)", 0, 30, 5)
                        target_rating = 3.0 # Assume 'Meets Expectations' for new hire
                        
                        # Calculation
                        base = p['Intercept']
                        
                        # Use exact key matching for band
                        band_key = f"C(Band)"
                        band_premium = p.get(band_key, 0)
                        
                        exp_val = p.get('Total_Experience', 0) * candidate_exp
                        rating_val = p.get('Avg_Rating', 0) * target_rating
                        
                        fair_pay = base + band_premium + exp_val + rating_val
                        
                        st.metric("Fair Market Offer", f"${fair_pay:,.0f}")
                        st.write(f"**Range (+/- 10%):** \n${fair_pay*0.9:,.0f} - ${fair_pay*1.1:,.0f}")
                    else:
                        st.warning("No band data available in regression model.")
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
                
                if 'Compa_Ratio' in df.columns:
                    risk_mask = (
                        (df >= 3.5) & 
                        (df > 5.0) & 
                        (df < 0.85)
                    )
                    
                    risk_df = df[risk_mask].copy()
                    
                    st.metric("At-Risk Employees", len(risk_df))
                    
                    if not risk_df.empty:
                        # Calculate cost to fix (raise to 1.0 Compa Ratio)
                        # Cost = Market_P50 - Current Pay
                        risk_df['Cost_to_Fix'] = (risk_df['Market_P50'] - risk_df)
                        total_fix = risk_df['Cost_to_Fix'].sum()
                        
                        st.metric("Budget to Retain", f"${total_fix:,.0f}")
                        
                        # Show top risks
                        st.dataframe(risk_df])
                    else:
                        st.success("No critical flight risks identified based on current logic.")
                else:
                    st.warning("Compa-Ratio data missing.")

else:
    st.info("Waiting for data file...")
