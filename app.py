import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Wipro Rewards Strategy Engine",
    page_icon="📊",
    layout="wide"
)

st.title("Wipro Rewards Analytics: Strategic Decision Engine")
st.markdown("""
**System Status:** Deep Research Mode Active.
**Methodology:** Mincer Earnings Function (Fair Pay) + K-Means Clustering (Flight Risk).
**Input:** Raw HR Data (Auto-processed for PPP & Metrics).
""")

# -----------------------------------------------------------------------------
# 2. AUTOMATED DATA PIPELINE (ETL Layer)
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_process_data(file):
    """
    1. Loads Raw Data (.xlsb/.xlsx)
    2. Converts Currencies to PPP USD
    3. Cleans Experience/Ratings/Bands
    4. Calculates Compa-Ratios
    """
    # A. LOAD
    try:
        df = pd.read_excel(file, engine='pyxlsb')
    except:
        df = pd.read_excel(file)
    
    df.columns = df.columns.str.strip()

    # B. PPP CONVERSION FACTORS (2025 Est)
    ppp_factors = {
        'USD': 1.0,
        'INR': 1 / 22.54,
        'PHP': 1 / 19.16,
        'AUD': 1 / 1.4,
        'EUR': 1 / 0.9
    }
    
    # Handle Currency Column
    if 'Currency' in df.columns:
        df['Currency'] = df['Currency'].astype(str).str.strip().str.upper()
    else:
        df['Currency'] = 'USD'
        
    df['PPP_Factor'] = df['Currency'].map(ppp_factors).fillna(1.0)

    # C. CREATE TARGET COLUMNS (PPP USD)
    # We look for standard raw names and create new columns
    pay_map = {
        'Annual_TCC': 'Annual_TCC (PPP USD)',
        'Annual Base Pay': 'Annual_Base (PPP USD)', 
        'P50': 'P50 (PPP USD)'
    }
    
    for raw, new in pay_map.items():
        if raw in df.columns:
            df[raw] = pd.to_numeric(df[raw], errors='coerce')
            df[new] = df[raw] * df['PPP_Factor']
            
    # D. CLEAN ATTRIBUTES
    # Experience
    if 'Experience' in df.columns:
        df['Experience_Clean'] = pd.to_numeric(df['Experience'], errors='coerce').fillna(0)
    elif 'Tenure' in df.columns:
        df['Experience_Clean'] = pd.to_numeric(df['Tenure'], errors='coerce').fillna(0)
    else:
        df['Experience_Clean'] = 0.0

    # Rating
    if 'Performance_Rating' in df.columns:
        df['Rating_Clean'] = pd.to_numeric(df['Performance_Rating'], errors='coerce').fillna(3.0)
    elif 'Rating' in df.columns:
         df['Rating_Clean'] = pd.to_numeric(df['Rating'], errors='coerce').fillna(3.0)
    else:
        df['Rating_Clean'] = 3.0

    # Band (Ordered Categorical)
    hierarchy = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "D1", "D2", "E", "F", "AA", "TRB"]
    if 'Band' in df.columns:
        # Filter for valid bands
        valid = [b for b in hierarchy if b in df['Band'].unique()]
        extra = [b for b in df['Band'].unique() if b not in hierarchy]
        df['Band_Clean'] = pd.Categorical(df['Band'], categories=valid+extra, ordered=True)
        
    # E. COMPA-RATIO
    if 'Annual_TCC (PPP USD)' in df.columns and 'P50 (PPP USD)' in df.columns:
        df['Compa_Ratio_Clean'] = df['Annual_TCC (PPP USD)'] / df['P50 (PPP USD)']
    else:
        df['Compa_Ratio_Clean'] = 0.0

    # Return clean df
    return df

# -----------------------------------------------------------------------------
# 3. SIDEBAR (Data Upload)
# -----------------------------------------------------------------------------
st.sidebar.header("Data Injection")
uploaded_file = st.sidebar.file_uploader("Upload Raw HR Dataset", type=['xlsb', 'xlsx'])

if uploaded_file:
    with st.spinner("Running Deep Research Pipeline..."):
        df = load_and_process_data(uploaded_file)
        
    st.sidebar.success(f"Processed {len(df):,} Records")
    st.sidebar.info("✅ PPP Conversion Complete\n✅ Mincer Variables Ready")

    # -------------------------------------------------------------------------
    # MAIN ANALYTICS TABS
    # -------------------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["📉 Mincer Regression (Fair Pay)", "🧩 K-Means Segmentation (Risk)", "🧮 Strategic Calculators"])

    # --- TAB 1: MINCER REGRESSION ---
    with tab1:
        st.subheader("Fair Pay Analysis: The Mincer Earnings Function")
        st.markdown(r"$$ Pay (PPP) = \alpha + \beta_1(Exp) + \beta_2(Perf) + \beta_3(Band) $$")
        
        # 1. Prepare Data
        # Drop rows where critical regression vars are missing
        reg_cols = ['Annual_TCC (PPP USD)', 'Experience_Clean', 'Rating_Clean', 'Band_Clean']
        if all(c in df.columns for c in reg_cols):
            df_reg = df.dropna(subset=reg_cols).copy()
            
            # 2. Run OLS Model
            formula = "Q('Annual_TCC (PPP USD)') ~ Experience_Clean + Rating_Clean + C(Band_Clean)"
            model = smf.ols(formula=formula, data=df_reg).fit()
            
            # Save for Calculator
            st.session_state['reg_model'] = model

            # 3. Key Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Model R-Squared", f"{model.rsquared:.2%}", help="How much pay variation is explained by the model.")
            c2.metric("Base Intercept", f"${model.params['Intercept']:,.0f}")
            c3.metric("Value of 1 Yr Exp", f"${model.params['Experience_Clean']:,.0f}")

            # 4. Visual: Price of Attributes
            st.markdown("### The 'Price' of Attributes")
            coef_df = pd.DataFrame({'Factor': model.params.index, 'Value': model.params.values}).iloc[1:]
            coef_df['Factor'] = coef_df['Factor'].str.replace("C(Band_Clean)[T.", "Band: ", regex=False).str.replace("]", "", regex=False)
            
            fig = px.bar(coef_df, x='Value', y='Factor', orientation='h', 
                         title="Dollar Impact of Promotion vs. Experience", color='Value')
            st.plotly_chart(fig, use_container_width=True)
            
            # 5. Visual: Pay Equity Audit (Residuals)
            st.markdown("### Equity Audit: Distribution of Pay Gaps")
            df_reg['Predicted'] = model.predict(df_reg)
            df_reg['Residuals'] = df_reg['Annual_TCC (PPP USD)'] - df_reg['Predicted']
            
            fig_res = px.histogram(df_reg, x='Residuals', nbins=50, title="Distribution of Underpaid vs Overpaid Staff")
            fig_res.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Fair Pay")
            st.plotly_chart(fig_res, use_container_width=True)
            
        else:
            st.error("Missing required columns for Regression (Pay, Exp, Rating, Band).")

    # --- TAB 2: K-MEANS CLUSTERING ---
    with tab2:
        st.subheader("Strategic Segmentation: K-Means Clustering")
        
        # 1. Prepare Data
        # Filter relevant columns and drop NaNs
        cluster_cols = ['Compa_Ratio_Clean', 'Rating_Clean']
        df_cluster = df.dropna(subset=cluster_cols).copy()
        
        # Scale Data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_cluster[cluster_cols])
        
        # 2. Elbow Plot (Pre-Analysis)
        st.markdown("**Step 1: Justification (Elbow Method)**")
        inertia = []
        for k in range(1, 8):
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
            inertia.append(km.inertia_)
            
        fig_elbow = px.line(x=range(1, 8), y=inertia, markers=True, title="Elbow Plot: Why we chose 4 Clusters")
        st.plotly_chart(fig_elbow, use_container_width=True)
        
        # 3. Run K-Means (k=4)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        df_cluster['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # 4. Dynamic Auto-Labeling
        # Compare cluster centers to global averages to label quadrants
        avg_rating = df_cluster['Rating_Clean'].mean()
        avg_compa = df_cluster['Compa_Ratio_Clean'].mean()
        
        cluster_stats = df_cluster.groupby('Cluster')[cluster_cols].mean()
        labels = {}
        risk_label = ""
        
        for i, row in cluster_stats.iterrows():
            if row['Rating_Clean'] >= avg_rating and row['Compa_Ratio_Clean'] < avg_compa:
                labels[i] = "🚨 Flight Risk (Underpaid Star)"
                risk_label = labels[i]
            elif row['Rating_Clean'] >= avg_rating and row['Compa_Ratio_Clean'] >= avg_compa:
                labels[i] = "⭐ Stable Star"
            elif row['Rating_Clean'] < avg_rating and row['Compa_Ratio_Clean'] >= avg_compa:
                labels[i] = "🔻 Overpaid / Low Perf"
            else:
                labels[i] = "⚖️ Core Employee"
                
        df_cluster['Cluster_Label'] = df_cluster['Cluster'].map(labels)
        st.session_state['risk_label'] = risk_label

        # 5. Visual: Talent Map
        st.markdown("**Step 2: The Talent Map**")
        fig_map = px.scatter(df_cluster, x='Rating_Clean', y='Compa_Ratio_Clean', 
                             color='Cluster_Label', symbol='Cluster_Label',
                             title="Employee Segmentation Matrix", hover_data=['Band_Clean'])
        fig_map.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Market P50")
        st.plotly_chart(fig_map, use_container_width=True)
        
        # 6. Visual: Financial Impact
        st.markdown("**Step 3: Financial Impact Analysis**")
        # Calc Retention Cost for everyone
        df_cluster['Retention_Cost'] = df_cluster['P50 (PPP USD)'] - df_cluster['Annual_TCC (PPP USD)']
        df_cluster['Retention_Cost'] = df_cluster['Retention_Cost'].apply(lambda x: max(x, 0))
        
        risk_summary = df_cluster.groupby('Cluster_Label')['Retention_Cost'].sum().reset_index()
        fig_cost = px.bar(risk_summary, x='Retention_Cost', y='Cluster_Label', orientation='h',
                          title="Total Budget Needed to Stabilize Each Segment", color='Retention_Cost')
        st.plotly_chart(fig_cost, use_container_width=True)

    # --- TAB 3: CALCULATORS ---
    with tab3:
        st.header("Decision Support Tools")
        
        col1, col2 = st.columns(2)
        
        # CALCULATOR 1: OFFER GENERATOR
        with col1:
            st.subheader("1. AI Offer Calculator")
            if 'reg_model' in st.session_state:
                model = st.session_state['reg_model']
                
                in_exp = st.number_input("Experience (Yrs)", 0, 40, 5)
                in_rating = st.number_input("Target Rating", 1, 5, 3)
                # Get bands
                bands = df['Band_Clean'].unique().tolist()
                in_band = st.selectbox("Target Band", sorted([str(b) for b in bands]))
                
                # Predict
                base = model.params['Intercept'] + (model.params['Experience_Clean'] * in_exp) + (model.params['Rating_Clean'] * in_rating)
                band_key = f"C(Band_Clean)[T.{in_band}]"
                band_prem = model.params.get(band_key, 0.0)
                
                pred_pay = base + band_prem
                st.metric("Fair Market Offer (PPP)", f"${pred_pay:,.0f}")
                st.caption(f"Range: ${pred_pay*0.9:,.0f} - ${pred_pay*1.1:,.0f}")
            else:
                st.warning("Model not trained. Check Tab 1.")
                
        # CALCULATOR 2: RETENTION BUDGET
        with col2:
            st.subheader("2. Flight Risk Budget")
            if 'risk_label' in st.session_state:
                r_label = st.session_state['risk_label']
                
                # Filter specific risk group
                risk_df = df_cluster[df_cluster['Cluster_Label'] == r_label]
                
                total_risk_cost = risk_df['Retention_Cost'].sum()
                
                st.error(f"Target Segment: {r_label}")
                st.metric("Employees at Risk", len(risk_df))
                st.metric("Immediate Budget Needed", f"${total_risk_cost:,.0f}")
                
                with st.expander("View Employee List"):
                    st.dataframe(risk_df[['Band_Clean', 'Experience_Clean', 'Retention_Cost']])
            else:
                st.warning("Clusters not generated. Check Tab 2.")

else:
    st.info("👋 Waiting for Data Upload...")
