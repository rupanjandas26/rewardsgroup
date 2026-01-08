import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.formula.api as smf
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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
**System Status:** Advanced Analytics Mode.
**Modules Active:** 1. **Mincer Regression:** For Fair Pay Prediction (Internal Equity).
2. **K-Means Clustering:** For Talent Segmentation & Flight Risk Analysis.
""")

# -----------------------------------------------------------------------------
# 2. DATA PROCESSING ENGINE
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(file):
    """
    Robust loader that handles both Excel and CSV formats.
    """
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return pd.DataFrame()

    # Clean Headers
    df.columns = df.columns.str.strip()
    return df

def clean_and_prep_data(df, pay_col, band_col, exp_col, rating_col, mkt_col):
    """
    Prepares the data for ML models.
    """
    # Create working copy
    data = df.copy()
    
    # 1. Target Variable (Pay)
    # Ensure it is numeric
    data[pay_col] = pd.to_numeric(data[pay_col], errors='coerce')
    # If the user selects a nominal column, we might need PPP. 
    # For now, we assume the user maps the "PPP" column if available.
    data['Target_Pay'] = data[pay_col]

    # 2. Features
    data['Experience'] = pd.to_numeric(data[exp_col], errors='coerce').fillna(0)
    data['Rating'] = pd.to_numeric(data[rating_col], errors='coerce').fillna(3.0) # Default to avg
    
    # 3. Band (Categorical)
    # Define standard hierarchy for ordering
    hierarchy = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "D1", "D2", "E", "F", "AA", "TRB"]
    data[band_col] = data[band_col].astype(str).str.strip()
    # Keep only valid bands
    valid_bands = [b for b in hierarchy if b in data[band_col].unique()]
    # Add any bands found in data but not in hierarchy to the end (safety net)
    extra_bands = [b for b in data[band_col].unique() if b not in hierarchy]
    full_order = valid_bands + extra_bands
    
    data['Band_Clean'] = pd.Categorical(data[band_col], categories=full_order, ordered=True)

    # 4. Compa-Ratio (Crucial for Clustering)
    if mkt_col and mkt_col in data.columns:
        data['Market_Ref'] = pd.to_numeric(data[mkt_col], errors='coerce')
        data['Compa_Ratio'] = data['Target_Pay'] / data['Market_Ref']
    else:
        # Fallback if no market data: Use internal median of the Band
        st.warning("Market Reference (P50) not provided. Calculating Internal Compa-Ratio based on Band Median.")
        band_medians = data.groupby('Band_Clean')['Target_Pay'].transform('median')
        data['Compa_Ratio'] = data['Target_Pay'] / band_medians

    # Remove outliers for clean regression/clustering (Optional but recommended)
    # data = data[data['Compa_Ratio'].between(0.5, 2.0)] 

    return data.dropna(subset=['Target_Pay', 'Experience', 'Rating', 'Compa_Ratio'])

# -----------------------------------------------------------------------------
# 3. SIDEBAR: INTELLIGENT MAPPING
# -----------------------------------------------------------------------------
st.sidebar.header("1. Data Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV/Excel)", type=['csv', 'xlsx', 'xlsb'])

if uploaded_file:
    raw_df = load_data(uploaded_file)
    st.sidebar.success(f"Loaded {len(raw_df)} records.")
    
    st.sidebar.header("2. Variable Mapping")
    cols = raw_df.columns.tolist()
    
    # Auto-find likely columns
    def find_col(keywords):
        for c in cols:
            if any(k.lower() in c.lower() for k in keywords):
                return c
        return cols[0]

    # Mappings
    pay_col = st.sidebar.selectbox("Pay Column (PPP USD)", cols, index=cols.index(find_col(['ppp', 'tcc', 'total'])))
    band_col = st.sidebar.selectbox("Band/Grade", cols, index=cols.index(find_col(['band', 'grade'])))
    exp_col = st.sidebar.selectbox("Experience/Tenure", cols, index=cols.index(find_col(['exp', 'tenure', 'years'])))
    rating_col = st.sidebar.selectbox("Performance Rating", cols, index=cols.index(find_col(['rating', 'perf'])))
    mkt_col = st.sidebar.selectbox("Market P50 (Optional)", ['None'] + cols, index=0)
    
    if mkt_col == 'None': mkt_col = None

    # Process Data
    df = clean_and_prep_data(raw_df, pay_col, band_col, exp_col, rating_col, mkt_col)
    
    st.sidebar.markdown("---")
    st.sidebar.write(f"**Analysis Ready:** {len(df)} Employees")

    # -------------------------------------------------------------------------
    # TABS FOR ANALYSIS
    # -------------------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["📉 Mincer Regression", "🧩 K-Means Segmentation", "🧮 Decision Calculators"])

    # --- TAB 1: MINCER REGRESSION ---
    with tab1:
        st.subheader("Deep Research: Mincer Earnings Function")
        st.markdown(r"""
        **Methodology:** OLS Multiple Regression.
        **Goal:** Isolate the impact of *Experience* and *Performance* on Pay, controlling for *Band*.
        $$ Pay = \alpha + \beta_1(Exp) + \beta_2(Perf) + \beta_3(Band) + \epsilon $$
        """)
        
        # Run OLS
        # We use Q() to handle column names with spaces
        formula = "Target_Pay ~ Experience + Rating + C(Band_Clean)"
        model = smf.ols(formula=formula, data=df).fit()
        
        # Save params for calculators
        st.session_state['reg_model'] = model

        # Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("R-Squared (Explained Variance)", f"{model.rsquared:.2%}")
        c2.metric("Base Pay (Intercept)", f"${model.params['Intercept']:,.0f}")
        c3.metric("Pay per Year of Exp", f"${model.params['Experience']:,.0f}")

        # Visualizing Coefficients
        st.markdown("### 📊 Impact of Drivers on Pay")
        coef_df = pd.DataFrame({
            'Factor': model.params.index,
            'Value ($)': model.params.values
        }).iloc[1:] # Skip intercept for plotting
        
        # Clean up labels for plot
        coef_df['Factor'] = coef_df['Factor'].astype(str).str.replace("C(Band_Clean)[T.", "Band: ", regex=False).str.replace("]", "", regex=False)
        
        fig_coef = px.bar(coef_df, x='Value ($)', y='Factor', orientation='h', 
                          title="Marginal Dollar Value of Each Factor",
                          color='Value ($)', color_continuous_scale='Bluered')
        st.plotly_chart(fig_coef, use_container_width=True)
        
        with st.expander("View Detailed Regression Statistics"):
            st.code(model.summary().as_text())

    # --- TAB 2: K-MEANS CLUSTERING ---
    with tab2:
        st.subheader("Advanced Segmentation: K-Means Clustering")
        st.markdown("""
        **Methodology:** Unsupervised Machine Learning.
        **Goal:** Segment employees into 4 distinct clusters based on **Performance** and **Compa-Ratio** to identify strategic groups (e.g., Flight Risks vs. Stars).
        """)

        # Prepare Features for Clustering
        X = df[['Compa_Ratio', 'Rating']].copy()
        
        # 1. Scale Data (Important because Rating is 1-5 and Compa is 0.5-1.5)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 2. Fit K-Means
        k = 4
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # 3. Interpret Clusters (Give them meaningful names automatically)
        # We calculate mean Rating and Compa for each cluster to name them
        cluster_summary = df.groupby('Cluster')[['Rating', 'Compa_Ratio']].mean()
        
        # Logic to auto-label
        labels = {}
        for i, row in cluster_summary.iterrows():
            r = row['Rating']
            c = row['Compa_Ratio']
            if r > 3.5 and c < 0.95:
                labels[i] = "🚨 Flight Risk (High Perf, Low Pay)"
            elif r > 3.5 and c >= 0.95:
                labels[i] = "⭐ Star Retained (High Perf, High Pay)"
            elif r < 3.0 and c > 1.05:
                labels[i] = "🔻 Overpaid / Low Perf"
            else:
                labels[i] = "⚖️ Core Employees"
        
        df['Cluster_Label'] = df['Cluster'].map(labels)
        
        # 4. Visualization
        col_k1, col_k2 = st.columns([3, 1])
        
        with col_k1:
            fig_cluster = px.scatter(
                df, x='Rating', y='Compa_Ratio', 
                color='Cluster_Label', 
                symbol='Cluster_Label',
                title="Employee Segmentation Map",
                hover_data=['Band_Clean', 'Experience', 'Target_Pay'],
                height=500
            )
            # Add reference lines
            fig_cluster.add_hline(y=1.0, line_dash="dash", line_color="grey", annotation_text="Market Median")
            fig_cluster.add_vline(x=3.0, line_dash="dash", line_color="grey", annotation_text="Avg Perf")
            st.plotly_chart(fig_cluster, use_container_width=True)
            
        with col_k2:
            st.markdown("**Cluster Counts:**")
            counts = df['Cluster_Label'].value_counts()
            st.dataframe(counts)
            
            # Save risk cluster for tool
            risk_label = "🚨 Flight Risk (High Perf, Low Pay)"
            # Find the actual label used for risk (in case logic varied)
            risk_cluster_name = next((v for v in labels.values() if "Flight Risk" in v), None)
            st.session_state['risk_cluster_name'] = risk_cluster_name

    # --- TAB 3: CALCULATORS ---
    with tab3:
        st.header("Strategic Calculators")
        
        # CALCULATOR A: OFFER GENERATOR (Regression-Based)
        st.subheader("1. New Hire Offer Calculator")
        st.info("Uses the **Mincer Regression Coefficients** to suggest an equitable offer.")
        
        if 'reg_model' in st.session_state:
            model = st.session_state['reg_model']
            
            # Inputs
            c1, c2, c3 = st.columns(3)
            in_exp = c1.number_input("Years of Experience", 0, 40, 5)
            in_rating = c2.number_input("Target Performance Rating", 1, 5, 3)
            # Band selection
            bands = df['Band_Clean'].unique().tolist()
            in_band = c3.selectbox("Job Band", sorted([str(b) for b in bands]))
            
            # Prediction Logic
            # Base + Exp + Rating
            pred = model.params['Intercept'] + (model.params['Experience'] * in_exp) + (model.params['Rating'] * in_rating)
            
            # Band Premium
            # Statsmodels keys: C(Band_Clean)[T.BandName]
            # Reference category (first band) is 0
            band_key = f"C(Band_Clean)[T.{in_band}]"
            band_val = model.params.get(band_key, 0.0)
            
            final_pred = pred + band_val
            
            # Output
            st.metric("Fair Market Offer (Predicted)", f"${final_pred:,.0f}")
            st.caption(f"Recommended Range: ${final_pred*0.9:,.0f} - ${final_pred*1.1:,.0f}")
        else:
            st.warning("Please run Tab 1 first.")
            
        st.markdown("---")
        
        # CALCULATOR B: RETENTION BUDGET (Cluster-Based)
        st.subheader("2. Retention Budget Estimator")
        st.info("Identifies employees in the **'Flight Risk' Cluster** (ML-identified) and calculates cost to fix.")
        
        if 'risk_cluster_name' in st.session_state and st.session_state['risk_cluster_name']:
            risk_name = st.session_state['risk_cluster_name']
            
            # Filter Data
            risk_df = df[df['Cluster_Label'] == risk_name].copy()
            
            if not risk_df.empty:
                # Calc cost to bring to 1.0 Compa Ratio (Market Median)
                risk_df['Cost_to_Retain'] = (risk_df['Target_Pay'] / risk_df['Compa_Ratio']) - risk_df['Target_Pay']
                
                # Only positive costs
                risk_df['Cost_to_Retain'] = risk_df['Cost_to_Retain'].apply(lambda x: max(x, 0))
                
                total_cost = risk_df['Cost_to_Retain'].sum()
                
                k1, k2 = st.columns(2)
                k1.error(f"High Risk Employees: {len(risk_df)}")
                k2.metric("Total Adjustment Budget Needed", f"${total_cost:,.0f}")
                
                with st.expander("View Employee List"):
                    st.dataframe(risk_df[['Band_Clean', 'Experience', 'Rating', 'Compa_Ratio', 'Cost_to_Retain']])
            else:
                st.success("No employees found in the Flight Risk cluster.")
        else:
            st.warning("Please run Tab 2 first to generate clusters.")

else:
    st.info("👋 Upload your CSV/Excel file to begin.")
