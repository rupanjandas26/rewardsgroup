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
2.  **AI Engine (K-Means):** Automatically segments workforce into **Strategic Quadrants** (e.g., Flight Risks, Stable Stars).
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
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.xlsb'):
            df = pd.read_excel(file, engine='pyxlsb')
        else:
            df = pd.read_excel(file) 
    except Exception as e:
        return None, f"Error loading file: {e}"

    # 1. Clean Headers
    df.columns = df.columns.str.strip()

    # 2. PPP Conversion Factors
    # We prioritize calculated columns if they exist, else we build them
    ppp_factors = {'USD': 1.0, 'INR': 1/22.54, 'PHP': 1/19.16, 'AUD': 1/1.4, 'EUR': 1/0.9}
    
    # Ensure Currency Column
    if 'Currency' not in df.columns:
        df['Currency'] = 'USD'
    
    df['PPP_Factor'] = df['Currency'].astype(str).str.strip().str.upper().map(ppp_factors).fillna(1.0)
    
    # Target Pay (Annual_TCC_PPP)
    if 'Annual_TCC (PPP USD)' in df.columns:
        df['Annual_TCC_PPP'] = pd.to_numeric(df['Annual_TCC (PPP USD)'], errors='coerce')
    elif 'Annual_TCC' in df.columns:
        df['Annual_TCC_PPP'] = pd.to_numeric(df['Annual_TCC'], errors='coerce') * df['PPP_Factor']
    else:
        return None, "Critical Error: 'Annual_TCC' column missing."

    # 3. Market Benchmark (P50) & Compa-Ratio
    if 'P50 (PPP USD)' in df.columns:
        df['P50_PPP'] = pd.to_numeric(df['P50 (PPP USD)'], errors='coerce')
    elif 'P50' in df.columns:
        df['P50_PPP'] = pd.to_numeric(df['P50'], errors='coerce') * df['PPP_Factor']
    else:
        # Fallback: Create internal P50 if market data missing (not ideal but prevents crash)
        df['P50_PPP'] = df['Annual_TCC_PPP'].median()

    # Calculate Compa-Ratio
    df['Compa_Ratio'] = df['Annual_TCC_PPP'] / df['P50_PPP']

    # 4. Rating Logic (3-Year Average)
    rating_cols = ['Current_Rating', 'Previous_Rating', 'Pre_Previous_Rating']
    existing_ratings = [c for c in rating_cols if c in df.columns]
    
    if existing_ratings:
        for col in existing_ratings:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['Avg_Rating'] = df[existing_ratings].mean(axis=1)
    elif 'Performance_Rating' in df.columns:
        df['Avg_Rating'] = pd.to_numeric(df['Performance_Rating'], errors='coerce')
    else:
        df['Avg_Rating'] = 3.0 
    
    df['Avg_Rating'] = df['Avg_Rating'].fillna(3.0)

    # 5. Experience Logic
    if 'Experience' in df.columns:
        df['Total_Experience'] = pd.to_numeric(df['Experience'], errors='coerce')
    elif 'Tenure' in df.columns:
        df['Total_Experience'] = pd.to_numeric(df['Tenure'], errors='coerce')
    else:
        df['Total_Experience'] = 0.0
    df['Total_Experience'] = df['Total_Experience'].fillna(0.0)

    # 6. Band Hierarchy
    band_order = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'D1', 'D2', 'E']
    if 'Band' in df.columns:
        df['Band'] = df['Band'].astype(str).str.strip()
        unique_bands = df['Band'].unique()
        found_bands = [b for b in band_order if b in unique_bands]
        extra_bands = [b for b in unique_bands if b not in found_bands]
        df['Band'] = pd.Categorical(df['Band'], categories=found_bands + extra_bands, ordered=True)

    # 7. Robustness Filter (As per your code snippet)
    # Remove outliers where Compa Ratio is < 0.4 or > 2.5
    df = df[(df['Compa_Ratio'] > 0.4) & (df['Compa_Ratio'] < 2.5)].copy()

    modeling_cols = ['Annual_TCC_PPP', 'Total_Experience', 'Avg_Rating', 'Band', 'Compa_Ratio']
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
        
        tab1, tab2, tab3 = st.tabs(["Regression Analysis", "K-Means Clustering", "Decision Tools"])

        # =====================================================================
        # TAB 1: MINCER EARNINGS FUNCTION
        # =====================================================================
        with tab1:
            st.subheader("The Mincer Earnings Function (OLS Regression)")
            st.markdown(r"""
            **The Math:** $$Pay = \alpha + \beta_1(Exp) + \beta_2(Perf) + \beta_3(Band)$$
            We use **Total Experience** and **Average Rating** to determine fair pay.
            """)
            
            formula = "Q('Annual_TCC_PPP') ~ Total_Experience + Avg_Rating + C(Band)"
            try:
                model = smf.ols(formula=formula, data=df).fit()
                st.session_state['reg_params'] = model.params
                
                c1, c2, c3 = st.columns(3)
                c1.metric("R-Squared", f"{model.rsquared:.2%}", "Model Fit")
                c2.metric("Base Pay (Intercept)", f"${model.params['Intercept']:,.0f}")
                c3.metric("Exp Premium (per Yr)", f"${model.params['Total_Experience']:,.0f}")
                
                st.markdown("### 📊 Coefficient Analysis")
                coef_data = []
                for idx, val in model.params.items():
                    if "Band" in idx:
                        band_name = idx.replace("C(Band)[T.", "").replace("]", "")
                        coef_data.append({"Factor": f"Promotion to {band_name}", "Pay Premium": val})
                    elif "Avg_Rating" in idx:
                        coef_data.append({"Factor": "1 Point Increase in Rating", "Pay Premium": val})
                
                coef_df = pd.DataFrame(coef_data)
                st.dataframe(coef_df.style.format({"Pay Premium": "${:,.0f}"}), use_container_width=True)
                
            except Exception as e:
                st.error(f"Regression Failed: {e}")

        # =====================================================================
        # TAB 2: K-MEANS SEGMENTATION (Strategic Risk)
        # =====================================================================
        with tab2:
            st.subheader("AI-Driven Workforce Segmentation")
            st.markdown("""
            **Quadrant Analysis:** We cluster employees based on **Performance (Avg Rating)** and **Pay Fairness (Compa-Ratio)**.
            This automatically identifies Flight Risks and Mis-priced talent.
            """)
            
            # Prepare Data for Clustering (Quadrant Logic)
            cluster_cols = ['Avg_Rating', 'Compa_Ratio']
            cluster_data = df[cluster_cols].dropna().copy()
            
            if not cluster_data.empty:
                # 1. Scale
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(cluster_data)
                
                # 2. Run K-Means (Fixed k=4 for Quadrants)
                kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
                cluster_data['Cluster'] = kmeans.fit_predict(X_scaled)
                
                # 3. DYNAMIC LABELING LOGIC (From Snippet)
                # We compare cluster centers to the Global Average
                global_avg_rating = cluster_data['Avg_Rating'].mean()
                global_avg_compa = cluster_data['Compa_Ratio'].mean()
                
                # Compute centroid for each cluster
                cluster_stats = cluster_data.groupby('Cluster')[['Avg_Rating', 'Compa_Ratio']].mean()
                
                labels = {}
                for i, row in cluster_stats.iterrows():
                    # Logic: High Perf / Low Pay = Flight Risk
                    if row['Avg_Rating'] >= global_avg_rating and row['Compa_Ratio'] < global_avg_compa:
                        labels[i] = "🚨 Flight Risk (Underpaid Star)"
                    # Logic: High Perf / High Pay = Stable Star
                    elif row['Avg_Rating'] >= global_avg_rating and row['Compa_Ratio'] >= global_avg_compa:
                        labels[i] = "⭐ Stable Star"
                    # Logic: Low Perf / High Pay = Overpaid
                    elif row['Avg_Rating'] < global_avg_rating and row['Compa_Ratio'] >= global_avg_compa:
                        labels[i] = "🔻 Overpaid / Low Perf"
                    # Logic: Low Perf / Low Pay = Core
                    else:
                        labels[i] = "⚖️ Core Employee"
                
                # Map labels back to dataframe
                df.loc[cluster_data.index, 'Cluster_Label'] = cluster_data['Cluster'].map(labels)
                
                # 4. VISUALIZATION
                st.markdown("### 🧬 The Strategic Talent Map")
                plot_df = df.dropna(subset=['Cluster_Label']).copy()
                
                # Define specific colors if possible, otherwise auto
                color_map = {
                    "🚨 Flight Risk (Underpaid Star)": "red",
                    "⭐ Stable Star": "green",
                    "🔻 Overpaid / Low Perf": "orange",
                    "⚖️ Core Employee": "blue"
                }

                fig_cluster = px.scatter(
                    plot_df, 
                    x='Avg_Rating', 
                    y='Compa_Ratio', 
                    color='Cluster_Label',
                    color_discrete_map=color_map,
                    hover_data=['Band', 'Total_Experience', 'Annual_TCC_PPP'],
                    title=f"Talent Map (Avg Rating vs Compa-Ratio)",
                    template="plotly_white"
                )
                
                # Add reference lines
                fig_cluster.add_hline(y=1.0, line_dash="dash", line_color="black", annotation_text="Market P50")
                fig_cluster.add_vline(x=3.0, line_dash="dash", line_color="black", annotation_text="Expectation")
                
                st.plotly_chart(fig_cluster, use_container_width=True)
                
                # 5. SUMMARY TABLE
                st.markdown("### 📊 Segment Impact Analysis")
                summary = plot_df.groupby('Cluster_Label').agg({
                    'Annual_TCC_PPP': 'mean',
                    'Avg_Rating': 'mean',
                    'Compa_Ratio': 'mean',
                    'Band': 'count'
                }).reset_index()
                
                summary.columns = ['Segment', 'Avg Pay', 'Avg Rating', 'Avg Compa-Ratio', 'Headcount']
                
                st.dataframe(summary.style.format({
                    'Avg Pay': '${:,.0f}',
                    'Avg Rating': '{:.2f}',
                    'Avg Compa-Ratio': '{:.2f}'
                }))

            else:
                st.warning("Not enough clean data for clustering.")

        # =====================================================================
        # TAB 3: STRATEGIC TOOLS (Offer Calculator)
        # =====================================================================
        with tab3:
            st.header("Strategic HR Tools")
            
            # (Tool B Removed as requested)
            
            st.subheader("💰 Scientific Offer Calculator")
            st.info("Predicts 'Internal Equity' price using Mincer Coefficients from Tab 1.")
            
            if 'reg_params' in st.session_state:
                p = st.session_state['reg_params']
                bands = [k.replace("C(Band)[T.", "").replace("]", "") for k in p.keys() if "Band" in k]
                
                if bands:
                    c1, c2, c3 = st.columns(3)
                    target_band = c1.selectbox("Role Band", sorted(bands))
                    candidate_exp = c2.number_input("Total Experience (Yrs)", 0, 30, 5)
                    target_rating = c3.slider("Target Rating Assumption", 1.0, 5.0, 3.0)
                    
                    base = p['Intercept']
                    band_premium = p.get(f"C(Band)[T.{target_band}]", 0)
                    exp_val = p['Total_Experience'] * candidate_exp
                    rating_val = p['Avg_Rating'] * target_rating
                    
                    fair_pay = base + band_premium + exp_val + rating_val
                    
                    st.divider()
                    col_mid = st.columns([1,2,1])
                    col_mid[1].metric("Recommended Offer (Midpoint)", f"${fair_pay:,.0f}")
                    col_mid[1].caption(f"Range: ${fair_pay*0.9:,.0f} - ${fair_pay*1.1:,.0f}")
                else:
                    st.warning("Could not extract bands from regression.")
            else:
                st.warning("Please run the Regression in Tab 1 first to train the model.")

else:
    st.info("Waiting for data file... Please upload .xlsb, .xlsx, or .csv")
