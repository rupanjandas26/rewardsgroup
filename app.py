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
**System Status:** Automated Deep Research Mode.
**Data Source:** Automatically utilizing pre-processed PPP & Metric columns (Post-Column Y).
""")

# -----------------------------------------------------------------------------
# 2. AUTOMATED DATA ENGINE (Strictly using Pre-Calculated Columns)
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_process_data(file):
    """
    Loads data and automatically grabs the specific pre-calculated columns 
    located after the 'Unnamed: 24' (Column Y) separator.
    """
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return pd.DataFrame()

    # 1. Clean Headers
    df.columns = df.columns.str.strip()
    
    # 2. DEFINE EXACT COLUMN MAPPING (Based on your file structure)
    # We prioritize the columns found after Column Y for accuracy
    target_map = {
        'Pay': 'Annual_TCC (PPP USD)',
        'Market': 'P50 (PPP USD)',
        'Rating': 'Performance_Rating',
        'Compa': 'Compa_Ratio',
        'Exp': 'Experience', # From earlier section
        'Band': 'Band'       # From earlier section
    }
    
    # 3. VALIDATE CRITICAL COLUMNS EXIST
    missing = [val for key, val in target_map.items() if val not in df.columns]
    if missing:
        st.error(f"CRITICAL ERROR: The following required columns are missing: {missing}")
        st.info("Ensure you are uploading the 'my_edited_table' file with the calculated PPP columns.")
        st.stop()

    # 4. TYPE CONVERSION & CLEANING
    # Strictly use the values provided in the file, do not re-calculate
    df['Target_Pay'] = pd.to_numeric(df[target_map['Pay']], errors='coerce')
    df['Market_Ref'] = pd.to_numeric(df[target_map['Market']], errors='coerce')
    df['Experience_Clean'] = pd.to_numeric(df[target_map['Exp']], errors='coerce').fillna(0)
    df['Rating_Clean'] = pd.to_numeric(df[target_map['Rating']], errors='coerce').fillna(3.0)
    df['Calculated_Compa_Ratio'] = pd.to_numeric(df[target_map['Compa']], errors='coerce')
    
    # 5. BAND ORDERING
    hierarchy = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "D1", "D2", "E", "F", "AA", "TRB"]
    valid_bands = [b for b in hierarchy if b in df[target_map['Band']].unique()]
    extra_bands = [b for b in df[target_map['Band']].unique() if b not in hierarchy]
    
    df['Band_Clean'] = pd.Categorical(
        df[target_map['Band']], 
        categories=valid_bands + extra_bands, 
        ordered=True
    )

    # Drop rows where critical calculations are missing (NaNs in Pay or Rating)
    return df.dropna(subset=['Target_Pay', 'Experience_Clean', 'Calculated_Compa_Ratio', 'Rating_Clean'])

# -----------------------------------------------------------------------------
# 3. SIDEBAR (File Upload Only)
# -----------------------------------------------------------------------------
st.sidebar.header("Data Injection")
uploaded_file = st.sidebar.file_uploader("Upload 'my_edited_table.xlsx' or .csv", type=['csv', 'xlsx', 'xlsb'])

if uploaded_file:
    df = load_and_process_data(uploaded_file)
    st.sidebar.success(f"Processing Complete: {len(df):,} Rows")
    st.sidebar.success("✅ Successfully locked onto Post-Calculation Columns")

    # -------------------------------------------------------------------------
    # MAIN ANALYTICS TABS
    # -------------------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["📉 Mincer Regression (Fair Pay)", "🧩 K-Means Clustering (Risk)", "🧮 Auto-Calculators"])

    # --- TAB 1: MINCER REGRESSION ---
    with tab1:
        st.subheader("Fair Pay Analysis: Mincer Earnings Function")
        st.markdown(r"$$ Pay (PPP) = \alpha + \beta_1(Exp) + \beta_2(Perf) + \beta_3(Band) $$")
        
        # 1. Run Regression
        # Q() handles spaces in column names
        formula = "Q('Annual_TCC (PPP USD)') ~ Experience_Clean + Rating_Clean + C(Band_Clean)"
        model = smf.ols(formula=formula, data=df).fit()
        
        # Store for Calculator
        st.session_state['reg_model'] = model

        # 2. Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Model Strength (R²)", f"{model.rsquared:.2%}")
        c2.metric("Base Intercept", f"${model.params['Intercept']:,.0f}")
        c3.metric("Value of 1 Yr Exp", f"${model.params['Experience_Clean']:,.0f}")

        # 3. Visualization
        st.write("### Regression Coefficients (Dollar Value of Each Factor)")
        coef_data = pd.DataFrame({
            'Factor': model.params.index,
            'Value': model.params.values
        }).iloc[1:] # Skip intercept
        
        # Clean labels
        coef_data['Factor'] = coef_data['Factor'].str.replace("C(Band_Clean)[T.", "Band: ", regex=False).str.replace("]", "", regex=False)
        
        fig = px.bar(coef_data, x='Value', y='Factor', orientation='h', color='Value')
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("View Full Statistical Summary"):
            st.text(model.summary().as_text())

    # --- TAB 2: K-MEANS CLUSTERING ---
    with tab2:
        st.subheader("Employee Segmentation: K-Means Clustering")
        st.markdown("Segments employees using the pre-calculated **Performance_Rating** and **Compa_Ratio**.")
        
        # 1. Prep Features (Using the robust end-of-file columns)
        X = df[['Calculated_Compa_Ratio', 'Rating_Clean']].copy()
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 2. Train K-Means (k=4)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # 3. Auto-Label Clusters
        cluster_stats = df.groupby('Cluster')[['Rating_Clean', 'Calculated_Compa_Ratio']].mean()
        
        labels = {}
        risk_cluster_id = -1
        
        for i, row in cluster_stats.iterrows():
            r = row['Rating_Clean']
            c = row['Calculated_Compa_Ratio']
            
            # Logic tailored to your cleaned data ranges
            if r >= 3.5 and c <= 0.90:
                labels[i] = "🚨 Flight Risk (Underpaid Star)"
                risk_cluster_id = i
            elif r >= 3.5 and c > 0.90:
                labels[i] = "⭐ Stable Star (Fairly Paid)"
            elif r < 3.0 and c > 1.05:
                labels[i] = "🔻 Overpaid Low Performer"
            else:
                labels[i] = "⚖️ Standard Performer"
                
        df['Cluster_Label'] = df['Cluster'].map(labels)
        
        # Save risk label specifically for the calculator
        st.session_state['risk_label'] = labels.get(risk_cluster_id, "🚨 Flight Risk (Underpaid Star)")

        # 4. Visuals
        col_k1, col_k2 = st.columns([3, 1])
        with col_k1:
            fig_k = px.scatter(
                df, x='Rating_Clean', y='Calculated_Compa_Ratio',
                color='Cluster_Label', symbol='Cluster_Label',
                title="Talent Segmentation Map",
                hover_data=['Band', 'Experience', 'Annual_TCC (PPP USD)']
            )
            fig_k.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Market P50")
            st.plotly_chart(fig_k, use_container_width=True)
            
        with col_k2:
            st.write("**Segment Distribution**")
            st.dataframe(df['Cluster_Label'].value_counts())

    # --- TAB 3: CALCULATORS ---
    with tab3:
        st.header("Decision Support Tools")
        
        # 1. OFFER CALCULATOR
        st.subheader("1. AI Offer Calculator")
        st.info("Predicts Pay using the Mincer Regression Model.")
        
        if 'reg_model' in st.session_state:
            model = st.session_state['reg_model']
            
            c1, c2, c3 = st.columns(3)
            in_exp = c1.number_input("Experience (Yrs)", 0, 40, 5)
            in_rating = c2.number_input("Expected Rating", 1, 5, 3)
            in_band = c3.selectbox("Target Band", df['Band_Clean'].unique().tolist())
            
            # Calculate: Intercept + Exp*B1 + Rating*B2
            base = model.params['Intercept'] + (model.params['Experience_Clean'] * in_exp) + (model.params['Rating_Clean'] * in_rating)
            # Band Premium
            band_key = f"C(Band_Clean)[T.{in_band}]"
            band_prem = model.params.get(band_key, 0.0)
            
            total = base + band_prem
            
            st.metric("Predicted Fair Pay (PPP USD)", f"${total:,.0f}")
            st.caption(f"Range: ${total*0.9:,.0f} - ${total*1.1:,.0f}")
            
        # 2. RETENTION CALCULATOR
        st.markdown("---")
        st.subheader("2. Flight Risk Retention Budget")
        st.info("Calculates cost to fix employees in the 'Flight Risk' Cluster.")
        
        risk_label = st.session_state.get('risk_label')
        if risk_label:
            # Filter specifically for the Risk Cluster identified in Tab 2
            risk_df = df[df['Cluster_Label'] == risk_label].copy()
            
            if not risk_df.empty:
                # Use pre-calculated P50 and TCC
                risk_df['Gap'] = risk_df['P50 (PPP USD)'] - risk_df['Annual_TCC (PPP USD)']
                
                # Only consider positive gaps (where they are underpaid)
                total_gap = risk_df[risk_df['Gap'] > 0]['Gap'].sum()
                
                k1, k2 = st.columns(2)
                k1.error(f"Employees at Risk: {len(risk_df)}")
                k2.metric("Budget to Normalize to P50", f"${total_gap:,.0f}")
                
                with st.expander("See Employee List"):
                    st.dataframe(risk_df[['ID', 'Band', 'Experience', 'Annual_TCC (PPP USD)', 'Gap']])
            else:
                st.success("No critical flight risks detected in this cluster.")
        else:
            st.warning("Please run Tab 2 first to generate clusters.")

else:
    st.info("Waiting for file upload...")
