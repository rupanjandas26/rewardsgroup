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

# --- GLOBAL VISUALIZATION SETTINGS ---
# Force a clean, white background for all plots to ensure readability
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['text.color'] = '#333333'
plt.rcParams['axes.labelcolor'] = '#333333'
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'

# --- CSS FOR PROFESSIONAL STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #f4f6f9;
    }
    h1, h2, h3 {
        color: #002e6e; /* Wipro Blue */
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #002e6e;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 4px;
        color: #555;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #002e6e;
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.title("Configuration")
st.sidebar.info("Upload Wipro Employee Dataset")
uploaded_file = st.sidebar.file_uploader("Allowed formats: .csv, .xlsb, .xlsx", type=["csv", "xlsb", "xlsx"])

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
            # Log Pay for Regression Linearity
            df = df[df['Annual_TCC_PPP'] > 0] # Remove zero/negative pay
            df['Log_Pay'] = np.log(df['Annual_TCC_PPP']) 
        
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
st.markdown("### Strategic Pay Equity & Retention Dashboard | Group 13")

if uploaded_file is not None:
    df = load_and_clean_data(uploaded_file)
    
    if df is not None:
        # Create Tabs matching Report Structure
        tab1, tab2, tab3, tab4 = st.tabs([
            "1. Overview & Data Health", 
            "2. Engine 1: Market Fairness & Gender", 
            "3. Engine 2: Strategic Segmentation", 
            "4. Tools: Salary Fitment"
        ])
        
        # --- TAB 1: OVERVIEW ---
        with tab1:
            st.header("1. Data Integrity & Robustness Check")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Employees", f"{len(df):,}")
            c2.metric("Currencies Normalized", f"{df['Currency'].nunique()} (PPP USD)")
            c3.metric("Avg Compa-Ratio", f"{df['Compa_Ratio'].mean():.2f}")
            
            st.subheader("Distribution of Annual Pay (PPP USD)")
            # Forced white background for plot
            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor('white')
            
            sns.histplot(df['Annual_TCC_PPP'], kde=True, color='#002e6e', ax=ax, edgecolor='white')
            ax.set_title("Annual TCC Distribution (Post-Cleaning)", fontweight='bold')
            ax.set_xlabel("Annual TCC (PPP USD)")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        # --- TAB 2: ENGINE 1 (REGRESSION) ---
        with tab2:
            st.header("2. Engine 1: The Econometric Market Model")
            st.markdown("""
            **Methodology:** OLS Mincer Regression to isolate financial value of attributes.
            **Formula:** `Log(Pay) ~ Experience + Rating + Band + Gender + Job_Family`
            """)
            
            if 'Annual_TCC_PPP' in df.columns:
                # Prepare data
                reg_cols = ['Log_Pay', 'Annual_TCC_PPP', 'Clean_Exp', 'Clean_Rating', 'Band', 'Gender', 'Job_Family']
                # Check if columns exist
                missing_cols = [c for c in reg_cols if c not in df.columns and c != 'Log_Pay']
                
                if not missing_cols:
                    df_reg = df.dropna(subset=reg_cols).copy()
                    
                    # Formula
                    formula = "Log_Pay ~ Clean_Exp + Clean_Rating + C(Band) + C(Gender) + C(Job_Family)"
                    
                    try:
                        model = smf.ols(formula=formula, data=df_reg).fit()
                        
                        # Store params
                        st.session_state['reg_params'] = model.params
                        st.session_state['model_r2'] = model.rsquared
                        
                        # Metrics
                        c1, c2 = st.columns(2)
                        c1.metric("Model Accuracy (R-Squared)", f"{model.rsquared:.2%}")
                        c2.metric("Experience Premium", f"{np.exp(model.params['Clean_Exp']) - 1:.1%} per year")

                        st.divider()
                        
                        # --- GENDER EQUITY TOOL ---
                        st.subheader("Mandatory Analysis: Gender Equity Audit")
                        st.markdown("Visualizing the regression coefficient for Gender (Systemic Gap Audit).")
                        
                        gender_param = [k for k in model.params.index if 'Gender' in k]
                        if gender_param:
                            gender_val = model.params[gender_param[0]]
                            
                            fig_gender, ax_g = plt.subplots(figsize=(10, 3))
                            fig_gender.patch.set_facecolor('white')
                            
                            color = '#d62728' if gender_val < 0 else '#2ca02c' # Red if gap, Green if premium
                            ax_g.barh(['Gender Gap (Controlled)'], [gender_val], color=color)
                            ax_g.axvline(0, color='black', linestyle='--')
                            ax_g.set_xlabel("Log Point Difference (Negative = Pay Gap)", fontweight='bold')
                            ax_g.set_title("Systemic Gender Pay Gap Audit", fontweight='bold')
                            
                            # Add data label
                            ax_g.text(gender_val, 0, f" {gender_val:.4f}", va='center', fontweight='bold')
                            
                            st.pyplot(fig_gender)
                            
                            if gender_val < 0:
                                st.error(f"Audit Alert: Negative coefficient ({gender_val:.4f}) detected for females.")
                            else:
                                st.success("Audit Pass: No negative systemic gap detected.")
                        
                        # Coefficients Table
                        with st.expander("View Full Regression Coefficients"):
                            st.write(model.summary())
                            
                    except Exception as e:
                         st.error(f"Regression Error: {e}")
                else:
                    st.warning(f"Missing columns for regression: {missing_cols}")

        # --- TAB 3: ENGINE 2 (CLUSTERING) ---
        with tab3:
            st.header("3. Engine 2: Strategic Segmentation")
            st.markdown("K-Means Clustering: Segmenting by Value (Rating) vs. Cost (Compa-Ratio).")
            
            if 'Clean_Rating' in df.columns and 'Compa_Ratio' in df.columns:
                X = df[['Clean_Rating', 'Compa_Ratio']].dropna()
                
                # 1. Elbow Plot
                st.subheader("Statistical Justification: The Elbow Plot")
                inertia = []
                K_range = range(1, 10)
                for k in K_range:
                    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
                    inertia.append(km.inertia_)
                
                fig_elbow, ax_e = plt.subplots(figsize=(8, 3))
                fig_elbow.patch.set_facecolor('white')
                
                ax_e.plot(K_range, inertia, marker='o', color='#002e6e', linewidth=2)
                ax_e.set_title("Elbow Method (Optimal k=4)", fontweight='bold')
                ax_e.set_xlabel("Number of Clusters (k)")
                ax_e.set_ylabel("Inertia")
                ax_e.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig_elbow)
                
                # 2. Run K-Means (k=4)
                kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
                df['Cluster'] = kmeans.fit_predict(df[['Clean_Rating', 'Compa_Ratio']].fillna(0))
                
                # Labeling
                avg_rating = df['Clean_Rating'].mean()
                avg_compa = df['Compa_Ratio'].mean()
                
                def label_cluster(row):
                    if row['Clean_Rating'] >= avg_rating and row['Compa_Ratio'] < avg_compa:
                        return "Flight Risk (Underpaid Star)"
                    elif row['Clean_Rating'] >= avg_rating and row['Compa_Ratio'] >= avg_compa:
                        return "Stable Star"
                    elif row['Clean_Rating'] < avg_rating and row['Compa_Ratio'] >= avg_compa:
                        return "Overpaid / Low Perf"
                    else:
                        return "Core Employee"

                df['Persona'] = df.apply(label_cluster, axis=1)
                
                # 3. Talent Map
                st.subheader("The Strategic Talent Map")
                fig_map, ax_m = plt.subplots(figsize=(10, 6))
                fig_map.patch.set_facecolor('white')
                
                # Use a high contrast palette
                sns.scatterplot(
                    data=df, 
                    x='Clean_Rating', 
                    y='Compa_Ratio', 
                    hue='Persona', 
                    palette='tab10', 
                    s=60, 
                    alpha=0.7, 
                    ax=ax_m
                )
                
                ax_m.axhline(1.0, color='#d62728', linestyle='--', linewidth=2, label='Market P50')
                ax_m.axvline(3.0, color='black', linestyle='--', linewidth=2, label='Avg Rating')
                
                ax_m.set_title("Workforce Segmentation: Identifying Flight Risks", fontweight='bold')
                ax_m.set_xlabel("Performance Rating")
                ax_m.set_ylabel("Compa-Ratio (Pay Competitiveness)")
                ax_m.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                
                st.pyplot(fig_map)
                
                # 4. Table
                st.subheader("Flight Risk Candidates")
                risks = df[df['Persona'] == "Flight Risk (Underpaid Star)"][['ID', 'Role Name', 'Annual_TCC_PPP', 'Compa_Ratio', 'Clean_Rating']]
                st.dataframe(risks.head(10), use_container_width=True)

        # --- TAB 4: TOOLS (FITMENT) ---
        with tab4:
            st.header("4. Recruitment Tool: Scientific Salary Fitment")
            st.markdown("Predictive Calculator based on Regression Coefficients.")
            
            if 'reg_params' in st.session_state:
                p = st.session_state['reg_params']
                
                c1, c2 = st.columns(2)
                with c1:
                    exp_input = st.number_input("Experience (Years)", 0, 30, 5)
                    rating_input = st.slider("Assumed Rating", 1, 5, 3)
                with c2:
                    # Parse Bands/Families from params
                    bands = [k.replace("C(Band)[T.", "").replace("]", "") for k in p.index if "C(Band)" in k]
                    families = [k.replace("C(Job_Family)[T.", "").replace("]", "") for k in p.index if "C(Job_Family)" in k]
                    
                    selected_band = st.selectbox("Target Band", sorted(bands)) if bands else None
                    selected_family = st.selectbox("Job Family", sorted(families)) if families else None
                
                if st.button("Calculate Fair Market Offer"):
                    # Calc Logic
                    offer_log = p['Intercept']
                    offer_log += p['Clean_Exp'] * exp_input
                    offer_log += p['Clean_Rating'] * rating_input
                    
                    if selected_band:
                        band_key = f"C(Band)[T.{selected_band}]"
                        if band_key in p: offer_log += p[band_key]
                        
                    if selected_family:
                        fam_key = f"C(Job_Family)[T.{selected_family}]"
                        if fam_key in p: offer_log += p[fam_key]
                        
                    final_offer = np.exp(offer_log)
                    
                    st.success(f"Scientific Fair Offer: ${final_offer:,.2f} (PPP USD)")
                    st.info("Offer aligns with internal equity of current workforce.")
                    
            else:
                st.warning("Please run Regression in Tab 2 first.")

    else:
        st.warning("Please upload a valid dataset.")
else:
    st.info("Awaiting Data Upload...")
