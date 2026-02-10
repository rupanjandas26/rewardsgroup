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
    page_title="RM | Group 13",
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
    .stTabs [data-baseweb="tab"] { color: #333333; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: #002e6e; border-bottom-color: #002e6e; }
    
    /* Expander Styling */
    .streamlit-expanderHeader { color: #002e6e; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

# --- 4. SIDEBAR ---
st.sidebar.title("Configuration")
st.sidebar.info("Upload Wipro Employee Dataset (Note: After Upload, wait for a few minutes for the analysis to complete.)")
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

st.title("Reward Management Group Assignment: Wipro Analysis")
st.markdown("### Group 13")

if uploaded_file is not None:
    df = load_and_clean_data(uploaded_file)
    
    if df is not None:
        tab1, tab2, tab3, tab4 = st.tabs([
            "1. Overview & Data Health", 
            "2. Engine 1: Market Fairness & Gender", 
            "3. Engine 2: Strategic Segmentation", 
            "4. Salary Fitment Tool"
        ])
        
        # --- TAB 1: OVERVIEW ---
        with tab1:
            st.header("1. Data Integrity & Robustness Check")
            
            # Key Metrics Row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Employees", f"{len(df):,}")
            c2.metric("Avg Compa-Ratio", f"{df['Compa_Ratio'].mean():.2f}")
            c3.metric("Avg Tenure", f"{df['Clean_Exp'].mean():.1f} Yrs")
            unique_roles = df['Role Name'].nunique() if 'Role Name' in df.columns else 0
            c4.metric("Unique Roles", unique_roles)
            
            st.divider()
            
            # Row 1 of Visuals: Distributions
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.subheader("A. Pay Distribution (PPP USD)")
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.histplot(df['Annual_TCC_PPP'], kde=True, color='#002e6e', ax=ax, edgecolor='black')
                ax.set_title("Are we paying competitively?", color='black', fontweight='bold')
                ax.set_xlabel("Annual TCC", color='black')
                ax.tick_params(colors='black')
                st.pyplot(fig)
                st.caption("A healthy distribution should be bell-shaped. Skewness indicates pay compression.")

            with col_right:
                st.subheader("B. Headcount by Band")
                if 'Band' in df.columns:
                    fig_band, ax_b = plt.subplots(figsize=(8, 4))
                    band_counts = df['Band'].value_counts().sort_index()
                    sns.barplot(x=band_counts.index, y=band_counts.values, color='#00c1de', ax=ax_b)
                    ax_b.set_title("Organizational Shape (Pyramid Check)", color='black', fontweight='bold')
                    ax_b.tick_params(colors='black')
                    st.pyplot(fig_band)
                    st.caption("Checks the hierarchy. Top-heavy orgs (Inverted Pyramid) have higher costs.")
            
            st.divider()

            # Row 2 of Visuals: Boxplots
            st.subheader("C. Internal Equity Check: Pay Spread by Band")
            if 'Band' in df.columns:
                fig_box, ax_box = plt.subplots(figsize=(10, 5))
                # Sort bands if possible
                bands_sorted = sorted(df['Band'].unique().astype(str))
                sns.boxplot(data=df, x='Band', y='Annual_TCC_PPP', order=bands_sorted, palette='Blues', ax=ax_box)
                ax_box.set_title("Pay Ranges & Overlaps", color='black', fontweight='bold')
                ax_box.set_yscale('log') # Log scale handles wide pay ranges better
                ax_box.set_ylabel("Annual Pay (Log Scale)", color='black')
                ax_box.tick_params(colors='black')
                st.pyplot(fig_box)
                st.caption("This chart reveals if lower bands are overlapping too much with higher bands (Pay Compression).")

        # --- TAB 2: ENGINE 1 (REGRESSION) ---
        with tab2:
            st.header("Engine 1: The Econometric Market Model")
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
                        
                        # --- GENDER EQUITY TOOL ---
                        st.subheader("Mandatory Analysis: Gender Equity Audit")
                        
                        # Explanation Block
                        st.markdown("""
                        **Business Relevance:** Simple comparisons of "Average Male Pay vs. Average Female Pay" are misleading because they don't account for Job Level or Experience. 
                        This tool runs a **Controlled Regression Audit** to isolate the specific impact of Gender on Pay while holding all other factors constant.
                        """)
                        
                        col_g_chart, col_g_text = st.columns([2, 1])
                        
                        gender_param = [k for k in model.params.index if 'Gender' in k]
                        
                        with col_g_chart:
                            if gender_param:
                                gender_val = model.params[gender_param[0]]
                                fig_gender, ax_g = plt.subplots(figsize=(8, 4))
                                color = '#d62728' if gender_val < 0 else '#6cc24a'
                                ax_g.barh(['Gender Gap (Controlled)'], [gender_val], color=color)
                                ax_g.axvline(0, color='black', linestyle='--')
                                ax_g.set_xlabel("Log Point Difference", color='black', fontweight='bold')
                                ax_g.set_title("Coefficient of Gender (Impact on Pay)", color='black', fontweight='bold')
                                ax_g.tick_params(colors='black')
                                ax_g.text(gender_val, 0, f" {gender_val:.4f}", va='center', fontweight='bold', color='black')
                                st.pyplot(fig_gender)

                        with col_g_text:
                            st.markdown("#### How to Read This")
                            if gender_val < 0:
                                st.error(f"**Systemic Gap Detected: {gender_val:.1%}**")
                                st.markdown(f"""
                                **Interpretation:** The bar is RED and to the left. 
                                This indicates that after controlling for Experience, Band, and Role, **Female employees are penalized by {abs(gender_val):.1%}** compared to men.
                                """)
                            else:
                                st.success("**No Systemic Bias**")
                                
                                # Dynamic Calculation of Male Rep in Senior Bands
                                if 'Band' in df.columns:
                                    unique_bands = sorted(df['Band'].unique().astype(str))
                                    # Assume top 3 bands are "Senior" for specific analysis
                                    senior_bands = unique_bands[-3:] if len(unique_bands) >= 3 else unique_bands
                                    senior_df = df[df['Band'].isin(senior_bands)]
                                    male_pct = 0
                                    if len(senior_df) > 0:
                                        male_pct = len(senior_df[senior_df['Gender']=='MALE']) / len(senior_df)
                                    
                                    st.markdown(f"""
                                    The bar is GREEN or near zero. The model shows no statistical penalty for gender. 
                                    Any raw pay gaps are likely due to structural factors (e.g., representation in senior bands).
                                    
                                    **Specific Data:** In the top seniority bands (**{', '.join(senior_bands)}**), Men currently hold **{male_pct:.1%}** of the positions. 
                                    This structural imbalance pulls up the raw average pay for men, even though they are paid fairly for the role itself.
                                    """)

                        st.divider()

                        # --- SKILL HETEROGENEITY ---
                        st.subheader("Skill Analysis: Job Family Premiums")
                        
                        st.markdown("""
                        **Business Relevance:**
                        A "One-Size-Fits-All" pay model fails in a complex organization. A 'Cloud Architect' costs more than an 'Administrator' even if they are in the same Band.
                        This chart visualizes the **Market Premium** associated with specific job families, proving the model prices skills accurately.
                        """)
                        
                        jf_params = {k.replace("C(Job_Family)[T.", "").replace("]", ""): v 
                                     for k, v in model.params.items() if "Job_Family" in k}
                        
                        if jf_params:
                            df_jf = pd.DataFrame(list(jf_params.items()), columns=['Job Family', 'Coefficient'])
                            df_jf = df_jf.sort_values('Coefficient', ascending=False).head(10)
                            
                            fig_skill, ax_s = plt.subplots(figsize=(10, 5))
                            sns.barplot(data=df_jf, x='Coefficient', y='Job Family', palette='viridis', ax=ax_s)
                            ax_s.set_title("Top 10 High-Value Skill Clusters (Market Premium)", color='black', fontweight='bold')
                            ax_s.set_xlabel("Premium above Baseline (Log Points)", color='black')
                            ax_s.tick_params(colors='black')
                            st.pyplot(fig_skill)
                            
                            st.info("""
                            **Glossary: What is a Log Point?**
                            In this regression, Log Points measure the percentage difference from the baseline.
                            * **+0.10** means this Job Family pays roughly **10% more** than the company average.
                            * **-0.05** would mean it pays **5% less**.
                            * Bars extending right indicate "Premium" skills that cost more to hire.
                            """)

                    except Exception as e:
                         st.error(f"Regression Error: {e}")
                else:
                    st.warning(f"Missing columns: {missing_cols}")

        # --- TAB 3: ENGINE 2 (CLUSTERING) ---
        with tab3:
            st.header("Engine 2: Strategic Segmentation")
            st.markdown("K-Means Clustering: Segmenting by Value (Rating) vs. Cost (Compa-Ratio).")
            
            if 'Clean_Rating' in df.columns and 'Compa_Ratio' in df.columns:
                X = df[['Clean_Rating', 'Compa_Ratio']].dropna()
                
                # Run K-Means (k=4)
                kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
                df['Cluster'] = kmeans.fit_predict(df[['Clean_Rating', 'Compa_Ratio']].fillna(0))
                
                # Labeling
                avg_rating = df['Clean_Rating'].mean()
                avg_compa = df['Compa_Ratio'].mean()
                
                def label_cluster(row):
                    if row['Clean_Rating'] >= avg_rating and row['Compa_Ratio'] < avg_compa:
                        return "Flight Risk - Underpaid Star"
                    elif row['Clean_Rating'] >= avg_rating and row['Compa_Ratio'] >= avg_compa:
                        return "Stable Star"
                    elif row['Clean_Rating'] < avg_rating and row['Compa_Ratio'] >= avg_compa:
                        return "Overpaid / Low Performance"
                    else:
                        return "Core Employee"

                df['Persona'] = df.apply(label_cluster, axis=1)
                
                # Talent Map
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
                ax_m.set_title("Workforce Segmentation", color='black', fontweight='bold')
                ax_m.set_xlabel("Performance Rating", color='black')
                ax_m.set_ylabel("Compa-Ratio", color='black')
                ax_m.tick_params(colors='black')
                ax_m.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                st.pyplot(fig_map)
                
                st.divider()

                # --- FINANCIAL IMPACT (ROI) ---
                st.subheader("Financial Impact: Retention ROI Analysis")
                
                col_roi_chart, col_roi_text = st.columns([2, 1])
                
                risks = df[df['Persona'] == "Flight Risk - Underpaid Star"]
                total_correction_cost = (risks['P50_PPP'] - risks['Annual_TCC_PPP']).sum()
                total_attrition_cost = risks['Annual_TCC_PPP'].sum() * 1.5 
                
                with col_roi_chart:
                    roi_data = pd.DataFrame({
                        'Cost Type': ['Correction (Retention)', 'Replacement (Attrition)'],
                        'Amount (USD)': [total_correction_cost, total_attrition_cost]
                    })
                    
                    fig_roi, ax_r = plt.subplots(figsize=(8, 4))
                    sns.barplot(data=roi_data, y='Cost Type', x='Amount (USD)', palette=['#6cc24a', '#d62728'], ax=ax_r)
                    ax_r.set_title("Cost of Action vs Inaction", color='black', fontweight='bold')
                    ax_r.tick_params(colors='black')
                    st.pyplot(fig_roi)
                
                with col_roi_text:
                    st.markdown("#### Business Context")
                    st.markdown(f"""
                    **The Logic:**
                    * **Correction Cost:** The exact budget needed to raise these specific "Underpaid Stars" to the Market Median.
                    * **Replacement Cost:** Industry standard (1.5x Salary). Includes Agency Fees, Onboarding (3-6mo), and Lost Productivity.
                    
                    **Verdict:**
                    It is **{total_attrition_cost/total_correction_cost:.1f}x cheaper** to fix their pay now than to let them leave.
                    """)

        # --- TAB 4: TOOLS (FITMENT) ---
        with tab4:
            st.header("Recruitment Tool: Scientific Salary Fitment")
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
