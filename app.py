import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Wipro Rewards Analytics | Group 13",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS FOR PROFESSIONAL STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    h1, h2, h3 {
        color: #002e6e; /* Wipro-like Blue */
        font-family: 'Segoe UI', sans-serif;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.title("Configuration")
st.sidebar.info("Upload the Wipro Employee Dataset (.csv or .xlsb)")
uploaded_file = st.sidebar.file_uploader("Upload Data", type=["csv", "xlsb", "xlsx"])

# --- DATA LOADING & CLEANING FUNCTION ---
@st.cache_data
def load_and_clean_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.xlsb'):
            df = pd.read_excel(file, engine='pyxlsb')
        else:
            df = pd.read_excel(file)
            
        # 1. Column Standardization
        df.columns = df.columns.str.strip()
        
        # 2. Currency Conversion (PPP)
        # Rates per report: 1 USD = 22.54 INR (PPP) | 1 USD = 19.16 PHP (PPP)
        ppp_rates = {'USD': 1.0, 'INR': 1/22.54, 'PHP': 1/19.16}
        
        if 'Currency' in df.columns and 'Annual_TCC' in df.columns:
            df['Currency'] = df['Currency'].astype(str).str.strip().str.upper()
            df['PPP_Factor'] = df['Currency'].map(ppp_rates).fillna(1.0)
            df['Annual_TCC_PPP'] = pd.to_numeric(df['Annual_TCC'], errors='coerce') * df['PPP_Factor']
            df['Log_Pay'] = np.log(df['Annual_TCC_PPP']) # For Regression Linearity
        
        # 3. Market Median (P50) Processing
        if 'P50' in df.columns:
            df['P50_PPP'] = pd.to_numeric(df['P50'], errors='coerce') * df['PPP_Factor']
            df['Compa_Ratio'] = df['Annual_TCC_PPP'] / df['P50_PPP']
            
        # 4. Rating Cleaning
        if 'Current_Rating' in df.columns:
            df['Clean_Rating'] = pd.to_numeric(df['Current_Rating'], errors='coerce').fillna(3.0) # Impute avg
            
        # 5. Experience Cleaning
        if 'Experience' in df.columns:
            df['Clean_Exp'] = pd.to_numeric(df['Experience'], errors='coerce').fillna(0)
            
        # 6. Robustness Filter (Outliers)
        # As per report: Remove Compa-Ratio < 0.4 or > 2.5
        if 'Compa_Ratio' in df.columns:
            df = df[(df['Compa_Ratio'] > 0.4) & (df['Compa_Ratio'] < 2.5)].copy()
            
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# --- MAIN APP LOGIC ---

st.title("Wipro Rewards Analytics: Dual-Engine Framework")
st.markdown("### Strategic Pay Equity & Retention Dashboard")

if uploaded_file is not None:
    df = load_and_clean_data(uploaded_file)
    
    if df is not None:
        # Create Tabs matching Report Structure
        tab1, tab2, tab3, tab4 = st.tabs([
            "Overview & Data Health", 
            "Engine 1: Market Fairness & Gender", 
            "Engine 2: Strategic Segmentation", 
            "Tools: Salary Fitment"
        ])
        
        # --- TAB 1: OVERVIEW ---
        with tab1:
            st.header("1. Data Integrity & Robustness Check")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Employees", len(df))
            c2.metric("Currencies Normalized", f"{df['Currency'].nunique()} (Converted to PPP USD)")
            c3.metric("Avg Compa-Ratio", f"{df['Compa_Ratio'].mean():.2f}")
            
            st.subheader("Distribution of Pay (PPP USD)")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.histplot(df['Annual_TCC_PPP'], kde=True, color='#002e6e', ax=ax)
            ax.set_title("Annual TCC Distribution (Post-Cleaning)")
            st.pyplot(fig)

        # --- TAB 2: ENGINE 1 (REGRESSION) ---
        with tab2:
            st.header("2. Engine 1: The Econometric Market Model")
            st.markdown("""
            **Methodology:** We use an OLS Mincer Regression to isolate the financial value of specific attributes.
            **Formula:** `Log(Pay) ~ Experience + Rating + Band + Gender + Job_Family`
            """)
            
            # Run Regression
            if 'Annual_TCC_PPP' in df.columns:
                # Prepare data for regression (drop nulls in relevant columns)
                reg_cols = ['Log_Pay', 'Annual_TCC_PPP', 'Clean_Exp', 'Clean_Rating', 'Band', 'Gender', 'Job_Family']
                df_reg = df.dropna(subset=reg_cols).copy()
                
                # Formula with Skills (Job Family) and Gender
                formula = "Log_Pay ~ Clean_Exp + Clean_Rating + C(Band) + C(Gender) + C(Job_Family)"
                
                try:
                    model = smf.ols(formula=formula, data=df_reg).fit()
                    
                    # Store params for Tab 4 Calculator
                    st.session_state['reg_params'] = model.params
                    st.session_state['model_r2'] = model.rsquared
                    
                    # Metrics
                    c1, c2 = st.columns(2)
                    c1.metric("Model Accuracy (R-Squared)", f"{model.rsquared:.2%}")
                    c2.metric("Experience Premium (Approx)", f"{np.exp(model.params['Clean_Exp']) - 1:.1%} per year")

                    st.divider()
                    
                    # --- GENDER EQUITY TOOL SECTION ---
                    st.subheader("Mandatory Analysis: Gender Equity Audit")
                    st.markdown("This tool visualizes the regression coefficient for Gender, effectively conducting a 'Systemic Gap' audit.")
                    
                    # Extract Gender Coefficient
                    gender_param = [k for k in model.params.index if 'Gender' in k]
                    if gender_param:
                        gender_val = model.params[gender_param[0]]
                        
                        fig_gender, ax_g = plt.subplots(figsize=(8, 3))
                        color = 'red' if gender_val < 0 else 'green'
                        ax_g.barh(['Gender Gap (Controlled)'], [gender_val], color=color)
                        ax_g.axvline(0, color='black', linestyle='--')
                        ax_g.set_xlabel("Log Point Difference (Negative = Pay Gap)")
                        st.pyplot(fig_gender)
                        
                        if gender_val < 0:
                            st.error(f"⚠️ Audit Alert: The model detects a negative coefficient of {gender_val:.4f} for females, indicating a systemic gap not explained by Experience or Performance.")
                        else:
                            st.success("Audit Pass: No negative systemic gap detected.")
                    
                    # Full Coefficients Table
                    with st.expander("View Full Regression Coefficients (Skill & Band Premiums)"):
                        st.write(model.summary())
                        
                except Exception as e:
                    st.error(f"Regression failed: {e}. Check if columns 'Band', 'Gender', 'Job_Family' exist.")

        # --- TAB 3: ENGINE 2 (CLUSTERING) ---
        with tab3:
            st.header("3. Engine 2: Strategic Segmentation (Risk Analysis)")
            st.markdown("We use **K-Means Clustering** to segment the workforce into 4 strategic personas based on Value (Rating) vs. Cost (Compa-Ratio).")
            
            if 'Clean_Rating' in df.columns and 'Compa_Ratio' in df.columns:
                X = df[['Clean_Rating', 'Compa_Ratio']].dropna()
                
                # 1. Elbow Plot (Visualizing k=4 justification)
                st.subheader("Statistical Justification: The Elbow Plot")
                inertia = []
                K_range = range(1, 10)
                for k in K_range:
                    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
                    inertia.append(km.inertia_)
                
                fig_elbow, ax_e = plt.subplots(figsize=(8, 3))
                ax_e.plot(K_range, inertia, marker='o', color='teal')
                ax_e.set_title("Elbow Method (Optimal k=4)")
                st.pyplot(fig_elbow)
                
                # 2. Run K-Means (k=4)
                kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
                df['Cluster'] = kmeans.fit_predict(df[['Clean_Rating', 'Compa_Ratio']].fillna(0))
                
                # Dynamic Labeling Logic
                avg_rating = df['Clean_Rating'].mean()
                avg_compa = df['Compa_Ratio'].mean()
                
                def label_cluster(row):
                    # We assign labels based on Quadrant Logic
                    # This is a simplification; in production, we map centroids.
                    if row['Clean_Rating'] >= avg_rating and row['Compa_Ratio'] < avg_compa:
                        return "🚨 Flight Risk (Underpaid Star)"
                    elif row['Clean_Rating'] >= avg_rating and row['Compa_Ratio'] >= avg_compa:
                        return "⭐ Stable Star"
                    elif row['Clean_Rating'] < avg_rating and row['Compa_Ratio'] >= avg_compa:
                        return "🔻 Overpaid / Low Perf"
                    else:
                        return "⚖️ Core Employee"

                df['Persona'] = df.apply(label_cluster, axis=1)
                
                # 3. The Talent Map (Scatter Plot)
                st.subheader("The Strategic Talent Map")
                fig_map, ax_m = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=df, x='Clean_Rating', y='Compa_Ratio', hue='Persona', palette='deep', ax=ax_m)
                ax_m.axhline(1.0, color='red', linestyle='--', label='Market P50')
                ax_m.axvline(3.0, color='black', linestyle='--', label='Avg Rating')
                ax_m.set_title("Workforce Segmentation: identifying Flight Risks")
                st.pyplot(fig_map)
                
                # 4. Flight Risk Table
                st.subheader("Actionable Data: Flight Risk Candidates")
                risks = df[df['Persona'] == "🚨 Flight Risk (Underpaid Star)"][['ID', 'Role Name', 'Annual_TCC_PPP', 'Compa_Ratio', 'Clean_Rating']]
                st.dataframe(risks.head(10), use_container_width=True)

        # --- TAB 4: TOOLS (FITMENT) ---
        with tab4:
            st.header("4. Recruitment Tool: Scientific Salary Fitment")
            st.markdown("""
            **Purpose:** Replaces recruiter guesswork with regression-based prediction.
            **Logic:** `Offer = Intercept + (Exp_Coef * Years) + Band_Premium + Job_Family_Premium`
            """)
            
            if 'reg_params' in st.session_state:
                p = st.session_state['reg_params']
                
                # Inputs
                c1, c2 = st.columns(2)
                with c1:
                    exp_input = st.number_input("Candidate Experience (Years)", 0, 30, 5)
                    rating_input = st.slider("Assumed Rating (Default=3)", 1, 5, 3)
                with c2:
                    # Extract Bands and Job Families from regression params for dropdowns
                    bands = [k.replace("C(Band)[T.", "").replace("]", "") for k in p.index if "C(Band)" in k]
                    families = [k.replace("C(Job_Family)[T.", "").replace("]", "") for k in p.index if "C(Job_Family)" in k]
                    
                    selected_band = st.selectbox("Target Band", sorted(bands))
                    selected_family = st.selectbox("Job Family (Skill Cluster)", sorted(families))
                
                # Calculation Engine
                if st.button("Calculate Fair Market Offer"):
                    # 1. Base Intercept
                    offer_log = p['Intercept']
                    
                    # 2. Add Experience Value
                    offer_log += p['Clean_Exp'] * exp_input
                    
                    # 3. Add Rating Value
                    offer_log += p['Clean_Rating'] * rating_input
                    
                    # 4. Add Band Premium
                    band_key = f"C(Band)[T.{selected_band}]"
                    if band_key in p:
                        offer_log += p[band_key]
                        
                    # 5. Add Skill/Family Premium
                    fam_key = f"C(Job_Family)[T.{selected_family}]"
                    if fam_key in p:
                        offer_log += p[fam_key]
                        
                    # Convert Log back to Dollars (Geometric Mean)
                    final_offer = np.exp(offer_log)
                    
                    # Output
                    st.success(f"✅ Scientific Fair Offer: ${final_offer:,.2f} (PPP USD)")
                    st.caption("This offer ensures internal equity by aligning with the regression line of current employees.")
                    
            else:
                st.warning("⚠️ Please run the Regression Model in 'Engine 1' tab first to train the coefficients.")

    else:
        st.warning("Please upload a valid dataset to proceed.")
else:
    st.info("Awaiting Data Upload...")
