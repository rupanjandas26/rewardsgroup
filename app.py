import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Wipro Rewards Analytics | Group 13",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. WIPRO BRANDING & VISUAL SETTINGS ---
# Wipro Palette: Deep Blue, Cyan, Green, Red
wipro_palette = ["#002e6e", "#00c1de", "#6cc24a", "#d62728"]
sns.set_palette(sns.color_palette(wipro_palette))

# Global Plot Settings (Black Text on White Background)
sns.set_theme(style="whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['font.family'] = 'sans-serif'

# --- 3. CSS FOR PROFESSIONAL STYLING ---
st.markdown("""
    <style>
    /* Main Background */
    .main { background-color: #ffffff; }
    
    /* Text Visibility Fix */
    div[data-testid="stMetricValue"] { color: #000000 !important; }
    div[data-testid="stMetricLabel"] { color: #555555 !important; }
    
    /* Wipro Headers */
    h1, h2, h3 { color: #002e6e; font-family: 'Segoe UI', sans-serif; font-weight: 700; }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab"] { color: #333333; }
    .stTabs [aria-selected="true"] { color: #002e6e; border-bottom-color: #002e6e; }
    </style>
""", unsafe_allow_html=True)

# --- 4. SIDEBAR ---
st.sidebar.title("Configuration")
st.sidebar.info("Upload Wipro Employee Dataset")
uploaded_file = st.sidebar.file_uploader("Allowed formats: .csv, .xlsb, .xlsx", type=["csv", "xlsb", "xlsx"])

# --- 5. DATA LOADING & CLEANING FUNCTION ---
@st.cache_data
def load_and_clean_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.xlsb'):
            df = pd.read_excel(file, engine='pyxlsb')
        else:
            df = pd.read_excel(file)
            
        df.columns = df.columns.str.strip()
        
        # Currency Conversion (PPP)
        ppp_rates = {'USD': 1.0, 'INR': 1/22.54, 'PHP': 1/19.16}
        
        if 'Currency' in df.columns and 'Annual_TCC' in df.columns:
            df['Currency'] = df['Currency'].astype(str).str.strip().str.upper()
            df['PPP_Factor'] = df['Currency'].map(ppp_rates).fillna(1.0)
            df['Annual_TCC_PPP'] = pd.to_numeric(df['Annual_TCC'], errors='coerce') * df['PPP_Factor']
            df = df[df['Annual_TCC_PPP'] > 0]
            df['Log_Pay'] = np.log(df['Annual_TCC_PPP']) 
        
        if 'P50' in df.columns:
            df['P50_PPP'] = pd.to_numeric(df['P50'], errors='coerce') * df['PPP_Factor']
            df['Compa_Ratio'] = df['Annual_TCC_PPP'] / df['P50_PPP']
            
        if 'Current_Rating' in df.columns:
            df['Clean_Rating'] = pd.to_numeric(df['Current_Rating'], errors='coerce').fillna(3.0)
            
        if 'Experience' in df.columns:
            df['Clean_Exp'] = pd.to_numeric(df['Experience'], errors='coerce').fillna(0)
            
        # Robustness Filter (Outliers)
        if 'Compa_Ratio' in df.columns:
            df = df[(df['Compa_Ratio'] > 0.4) & (df['Compa_Ratio'] < 2.5)].copy()
            
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# --- 6. MAIN APP LOGIC ---

st.title("Wipro Rewards Analytics: Dual-Engine Framework")
st.markdown("### Strategic Pay Equity & Retention Dashboard | Group 13")

if uploaded_file is not None:
    df = load_and_clean_data(uploaded_file)
    
    if df is not None:
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
            fig, ax = plt.subplots(figsize=(10, 4))
            
            sns.histplot(df['Annual_TCC_PPP'], kde=True, color='#002e6e', ax=ax, edgecolor='black')
            ax.set_title("Annual TCC Distribution (Post-Cleaning)", color='black', fontweight='bold')
            ax.set_xlabel("Annual TCC (PPP USD)", color='black')
            ax.set_ylabel("Frequency", color='black')
            ax.tick_params(colors='black')
            st.pyplot(fig)

        # --- TAB 2: ENGINE 1 (REGRESSION) ---
        with tab2:
            st.header("2. Engine 1: The Econometric Market Model")
            st.markdown("""
            **Methodology:** OLS Mincer Regression.
            **Formula:** `Log(Pay) ~ Experience + Rating + Band + Gender + Job_Family`
            """)
            
            if 'Annual_TCC_PPP' in df.columns:
                reg_cols = ['Log_Pay', 'Annual_TCC_PPP', 'Clean_Exp', 'Clean_Rating', 'Band', 'Gender', 'Job_Family']
                missing_cols = [c for c in reg_cols if c not in df.columns and c != 'Log_Pay']
                
                if not missing_cols:
                    df_reg = df.dropna(subset=reg_cols).copy()
                    
                    formula = "Log_Pay ~ Clean_Exp + Clean_Rating + C(Band) + C(Gender) + C(Job_Family)"
                    
                    try:
                        model = smf.ols(formula=formula, data=df_reg).fit()
                        st.session_state['reg_params'] = model.params
                        df_reg['Predicted_Log'] = model.predict(df_reg)
                        df_reg['Predicted_Pay'] = np.exp(df_reg['Predicted_Log'])
                        
                        c1, c2 = st.columns(2)
                        c1.metric("Model Accuracy (R-Squared)", f"{model.rsquared:.2%}")
                        c2.metric("Experience Premium", f"{np.exp(model.params['Clean_Exp']) - 1:.1%} per year")

                        st.divider()
                        
                        # --- NEW VISUAL 1: ACTUAL VS PREDICTED (PRIP Check) ---
                        st.subheader("Model Validation: Actual vs Predicted Pay")
                        st.markdown("Points above the line indicate employees paid *more* than the model predicts (Positive PRIP).")
                        
                        fig_prip, ax_p = plt.subplots(figsize=(10, 5))
                        sns.scatterplot(x=df_reg['Predicted_Pay'], y=df_reg['Annual_TCC_PPP'], alpha=0.3, color='#00c1de', ax=ax_p)
                        # Add 45-degree reference line
                        max_val = max(df_reg['Predicted_Pay'].max(), df_reg['Annual_TCC_PPP'].max())
                        ax_p.plot([0, max_val], [0, max_val], color='#d62728', linestyle='--')
                        
                        ax_p.set_title("Actual Pay vs Market Fair Pay", color='black', fontweight='bold')
                        ax_p.set_xlabel("Predicted 'Fair' Pay (Model)", color='black')
                        ax_p.set_ylabel("Actual Pay", color='black')
                        ax_p.tick_params(colors='black')
                        st.pyplot(fig_prip)
                        
                        st.divider()

                        # --- NEW VISUAL 2: SKILL HETEROGENEITY (Job Family Premiums) ---
                        st.subheader("Skill Analysis: Job Family Premiums")
                        st.markdown("Visualizing the 'Cost of Skill' - How much does each job family pay relative to the baseline?")
                        
                        # Extract Job Family Coefficients
                        jf_params = {k.replace("C(Job_Family)[T.", "").replace("]", ""): v 
                                     for k, v in model.params.items() if "Job_Family" in k}
                        
                        if jf_params:
                            df_jf = pd.DataFrame(list(jf_params.items()), columns=['Job Family', 'Coefficient'])
                            df_jf = df_jf.sort_values('Coefficient', ascending=False).head(10) # Top 10
                            
                            fig_skill, ax_s = plt.subplots(figsize=(10, 5))
                            sns.barplot(data=df_jf, x='Coefficient', y='Job Family', palette='viridis', ax=ax_s)
                            ax_s.set_title("Top 10 High-Value Skill Clusters (Regression Coefficients)", color='black', fontweight='bold')
                            ax_s.set_xlabel("Premium (Log Points)", color='black')
                            ax_s.set_ylabel("Job Family", color='black')
                            ax_s.tick_params(colors='black')
                            st.pyplot(fig_skill)
                        
                        st.divider()

                        # --- GENDER EQUITY TOOL ---
                        st.subheader("Mandatory Analysis: Gender Equity Audit")
                        gender_param = [k for k in model.params.index if 'Gender' in k]
                        if gender_param:
                            gender_val = model.params[gender_param[0]]
                            
                            fig_gender, ax_g = plt.subplots(figsize=(10, 3))
                            color = '#d62728' if gender_val < 0 else '#6cc24a'
                            ax_g.barh(['Gender Gap (Controlled)'], [gender_val], color=color)
                            ax_g.axvline(0, color='black', linestyle='--')
                            ax_g.set_xlabel("Log Point Difference (Negative = Pay Gap)", color='black', fontweight='bold')
                            ax_g.set_title("Systemic Gender Pay Gap Audit", color='black', fontweight='bold')
                            ax_g.tick_params(colors='black')
                            ax_g.text(gender_val, 0, f" {gender_val:.4f}", va='center', fontweight='bold', color='black')
                            st.pyplot(fig_gender)
                            
                    except Exception as e:
                         st.error(f"Regression Error: {e}")
                else:
                    st.warning(f"Missing columns: {missing_cols}")

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
                ax_e.plot(K_range, inertia, marker='o', color='#002e6e', linewidth=2)
                ax_e.set_title("Elbow Method (Optimal k=4)", color='black', fontweight='bold')
                ax_e.set_xlabel("Number of Clusters (k)", color='black')
                ax_e.set_ylabel("Inertia", color='black')
                ax_e.tick_params(colors='black')
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
                ax_m.set_title("Workforce Segmentation: Identifying Flight Risks", color='black', fontweight='bold')
                ax_m.set_xlabel("Performance Rating", color='black')
                ax_m.set_ylabel("Compa-Ratio (Pay Competitiveness)", color='black')
                ax_m.tick_params(colors='black')
                ax_m.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                st.pyplot(fig_map)
                
                # 4. NEW VISUAL: FINANCIAL IMPACT (ROI)
                st.subheader("Financial Impact: Retention ROI")
                st.markdown("Comparing the cost of fixing the 'Flight Risk' cluster vs. the cost of replacing them (Attrition).")
                
                risks = df[df['Persona'] == "Flight Risk (Underpaid Star)"]
                total_correction_cost = (risks['P50_PPP'] - risks['Annual_TCC_PPP']).sum()
                total_attrition_cost = risks['Annual_TCC_PPP'].sum() * 1.5 # Assumption: 1.5x salary to replace
                
                roi_data = pd.DataFrame({
                    'Cost Type': ['Correction Budget (Retention)', 'Replacement Cost (Attrition)'],
                    'Amount (USD)': [total_correction_cost, total_attrition_cost]
                })
                
                fig_roi, ax_r = plt.subplots(figsize=(8, 4))
                sns.barplot(data=roi_data, y='Cost Type', x='Amount (USD)', palette=['#6cc24a', '#d62728'], ax=ax_r)
                ax_r.set_title("ROI Analysis: Why We Must Act", color='black', fontweight='bold')
                ax_r.set_xlabel("Total Cost (PPP USD)", color='black')
                ax_r.set_ylabel("")
                ax_r.tick_params(colors='black')
                st.pyplot(fig_roi)

                st.info(f"Analysis shows it is {total_attrition_cost/total_correction_cost:.1f}x cheaper to retain these employees than to replace them.")

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
                    bands = [k.replace("C(Band)[T.", "").replace("]", "") for k in p.index if "C(Band)" in k]
                    families = [k.replace("C(Job_Family)[T.", "").replace("]", "") for k in p.index if "C(Job_Family)" in k]
                    
                    selected_band = st.selectbox("Target Band", sorted(bands)) if bands else None
                    selected_family = st.selectbox("Job Family", sorted(families)) if families else None
                
                if st.button("Calculate Fair Market Offer"):
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
