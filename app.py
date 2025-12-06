import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Page Config ---
st.set_page_config(page_title="Rewards Managemenr: Group 13", layout="wide")
st.title("Total Rewards & Workforce Analytics Dashboard")

# --- 2. File Uploader ---
st.sidebar.header("Upload your files here")
uploaded_file = st.sidebar.file_uploader("I**IMPORTANT:** Upload your file here, make sure the name of the file is **'data set'** and the format is **.xlsb**", type=['xlsb', 'xlsx'])

# --- 3. Data Processing Function ---
@st.cache_data
def process_data(file):
    # Load Data
    df = pd.read_excel(file, engine='pyxlsb')
    
    # Clean Column Names
    df.columns = df.columns.str.strip()
    
    # 1. Currency Conversion Setup
    if 'Currency' in df.columns:
        df['Currency'] = df['Currency'].astype(str).str.strip().str.upper()

    market_rates = {'USD': 1.0, 'INR': 0.0119, 'PHP': 0.0172}
    ppp_rates = {'USD': 1.0, 'INR': 1 / 22.54, 'PHP': 1 / 19.16}

    def convert_currency(row, col_name, rate_dict):
        currency = row['Currency']
        amount = row[col_name]
        if currency in rate_dict and pd.notnull(amount):
            return amount * rate_dict[currency]
        else:
            return None

    target_cols = ['Annual_TCC', 'P10', 'P25', 'P50', 'P75', 'P90']
    # Check if columns exist before processing
    existing_cols = [c for c in target_cols if c in df.columns]
    
    for col in existing_cols:
        df[f'{col} (Nominal USD)'] = df.apply(lambda row: convert_currency(row, col, market_rates), axis=1)
        df[f'{col} (PPP USD)'] = df.apply(lambda row: convert_currency(row, col, ppp_rates), axis=1)

    # 2. Band Sorting
    hierarchy_order = ['AA', 'A3', 'B1', 'B2', 'B3', 'C1', 'C3', 'D1', 'D2', 'D3']
    if 'Band' in df.columns:
        band_type = pd.CategoricalDtype(categories=hierarchy_order, ordered=True)
        df['Band'] = df['Band'].astype(band_type)
        df = df.sort_values(by='Band')

    # 3. Market Positioning (Compa-Ratio)
    market_col = 'P50 (PPP USD)'
    if market_col in df.columns:
        df[market_col] = pd.to_numeric(df[market_col], errors='coerce')
        # Filter out 0 or null market data to avoid division errors
        df = df[df[market_col] > 0]
        df['Compa_Ratio'] = df['Annual_TCC (PPP USD)'] / df[market_col]

        def get_positioning(ratio):
            if pd.isna(ratio): return 'Missing Data'
            elif ratio < 0.8: return 'Significant Lag (<80%)'
            elif 0.8 <= ratio < 0.95: return 'Lag (80-95%)'
            elif 0.95 <= ratio <= 1.05: return 'At Market (95-105%)'
            elif 1.05 < ratio <= 1.2: return 'Lead (105-120%)'
            else: return 'Significant Lead (>120%)'

        df['Positioning_Status'] = df['Compa_Ratio'].apply(get_positioning)

    # 4. Ratings Clean Up
    rating_cols = [col for col in df.columns if 'Rating' in col or 'Perf' in col]
    for col in rating_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if rating_cols:
        df['Performance_Rating'] = df[rating_cols].mean(axis=1)

    # 5. Skill Clean Up
    if 'Skill' in df.columns:
        df['Skill'] = df['Skill'].astype(str).str.upper().str.strip()
        df['Skill'] = df['Skill'].replace(['NAN', 'NONE', '', 'nan'], 'NON PREMIUM')
        skill_order = ['NON PREMIUM', 'PREMIUM', 'SUPER PREMIUM', 'ULTRA PREMIUM']
        df['Clean_Skill'] = pd.Categorical(df['Skill'], categories=skill_order, ordered=True)

    # 6. Additional Cleaning for Visuals
    if 'Tenure' in df.columns:
        df['Clean_Tenure'] = pd.to_numeric(df['Tenure'], errors='coerce')
    if 'Experience' in df.columns:
        df['Clean_Experience'] = pd.to_numeric(df['Experience'], errors='coerce')
    
    # Build vs Buy Logic
    if 'Clean_Experience' in df.columns and 'Clean_Tenure' in df.columns:
        df['Prior_Experience'] = (df['Clean_Experience'] - df['Clean_Tenure']).clip(lower=0)
        df['Hiring_Source'] = np.where(df['Prior_Experience'] < 2.0, 'Home Grown (Build)', 'Outside Hire (Buy)')
        
    return df

# --- 4. Main App Logic ---

if uploaded_file is None:
    st.info("⬅️ Please use the sidebar and upload the appropriate file to begin analysis.")
    st.stop()

try:
    df = process_data(uploaded_file)
    st.success("Data Processed Successfully!")
except Exception as e:
    st.error(f"Error processing file: {e}")
    st.stop()

# --- 5. Visualizations in Tabs ---

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Market Strategy", "Workforce Analysis", "Pay Drivers"])

# === TAB 1: OVERVIEW ===
with tab1:
    st.header("Overview: Pay Ranges & Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Pay Ranges by Job Band")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        sns.boxplot(x='Band', y='Annual_TCC (PPP USD)', data=df, ax=ax1)
        plt.title('Pay Ranges by Job Band (USD)')
        st.pyplot(fig1)

    with col2:
        st.subheader("Headcount by Band")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.countplot(x='Band', data=df, palette='viridis', ax=ax2)
        plt.title('Headcount by Job Band')
        st.pyplot(fig2)

    st.subheader("Distribution of Annual Pay")
    fig3, ax3 = plt.subplots(figsize=(10, 3))
    sns.histplot(df['Annual_TCC (PPP USD)'], bins=100, kde=True, color='skyblue', ax=ax3)
    plt.xlim(0, 250000)
    st.pyplot(fig3)

# === TAB 2: MARKET STRATEGY ===
with tab2:
    st.header("Market Positioning Strategy")
    
    if 'Compa_Ratio' in df.columns:
        st.subheader("Overall Market Positioning")
        fig4, ax4 = plt.subplots(figsize=(10, 3))
        sns.histplot(data=df, x='Compa_Ratio', kde=True, bins=100, color='teal', ax=ax4)
        ax4.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Market Median (1.0)')
        ax4.axvline(0.8, color='orange', linestyle=':', linewidth=2)
        ax4.axvline(1.2, color='orange', linestyle=':', linewidth=2)
        ax4.set_xlim(0, 4.5)
        ax4.legend()
        st.pyplot(fig4)

        st.subheader("Individual Positioning by Band")
        fig5, ax5 = plt.subplots(figsize=(10, 4))
        sns.scatterplot(
            data=df, x='Band', y='Compa_Ratio', hue='Positioning_Status',
            style='Positioning_Status', s=100, palette='viridis', ax=ax5
        )
        ax5.axhline(1.0, color='red', linestyle='--', alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig5)

        st.subheader("Median Positioning (Strategy Line)")
        market_strategy = df.groupby('Band')[['Compa_Ratio']].median().reset_index()
        fig6, ax6 = plt.subplots(figsize=(12, 6))
        sns.lineplot(
            data=market_strategy, x='Band', y='Compa_Ratio', marker='o',
            markersize=10, linewidth=2.5, color='darkblue', label='Wipro Median Pay', ax=ax6
        )
        ax6.axhline(1.0, color='red', linestyle='--', linewidth=2)
        ax6.fill_between(market_strategy['Band'], 0.95, 1.05, color='green', alpha=0.1)
        ax6.set_ylim(0.55, 1.25)
        for x, y in zip(market_strategy['Band'], market_strategy['Compa_Ratio']):
            if pd.notna(y):
                ax6.text(x, y + 0.01, f'{y:.2f}', ha='center', fontweight='bold')
        st.pyplot(fig6)

# === TAB 3: WORKFORCE ANALYSIS ===
with tab3:
    st.header("Workforce Capabilities & Strategy")
    
    # Re-setup specific df for visualization to ensure clean data
    df_viz = df.copy()
    
    st.subheader("Population, Ratings, and Tenure Check")
    fig7, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    if 'Clean_Skill' in df_viz.columns:
        sns.countplot(data=df_viz, x='Clean_Skill', palette='viridis', ax=axes[0])
        axes[0].tick_params(axis='x', rotation=45)
    
    if 'Performance_Rating' in df_viz.columns:
        sns.histplot(data=df_viz, x='Performance_Rating', bins=10, kde=True, color='orange', ax=axes[1])
    
    if 'Clean_Tenure' in df_viz.columns:
        sns.histplot(data=df_viz, x='Clean_Tenure', bins=20, kde=True, color='teal', ax=axes[2])
    
    st.pyplot(fig7)

    st.subheader("Strategic Deep Dive")
    fig8, axes2 = plt.subplots(2, 2, figsize=(15, 12))
    
    # Skill Mix
    sns.countplot(data=df_viz, x='Band', hue='Clean_Skill', palette='viridis', ax=axes2[0, 0])
    axes2[0, 0].set_title('Skill Mix by Band')
    
    # Performance
    if 'Performance_Rating' in df_viz.columns:
        sns.boxplot(data=df_viz, x='Band', y='Performance_Rating', palette='Oranges', ax=axes2[0, 1])
        axes2[0, 1].set_title('Performance Distribution')

    # Experience
    if 'Clean_Experience' in df_viz.columns:
        sns.boxplot(data=df_viz, x='Band', y='Clean_Experience', palette='Blues', ax=axes2[1, 0])
        axes2[1, 0].set_title('Total Experience Profile')

    # Hiring Strategy
    if 'Hiring_Source' in df_viz.columns:
        hiring_strategy = pd.crosstab(df_viz['Band'], df_viz['Hiring_Source'], normalize='index')
        # Ensure columns exist before plotting
        cols_to_plot = [c for c in ['Outside Hire (Buy)', 'Home Grown (Build)'] if c in hiring_strategy.columns]
        if cols_to_plot:
            hiring_strategy = hiring_strategy[cols_to_plot]
            hiring_strategy.plot(kind='bar', stacked=True, color=['#d73027', '#4575b4'], width=0.8, ax=axes2[1, 1])
            axes2[1, 1].set_title('Build vs Buy Ratio')
    
    st.pyplot(fig8)

# === TAB 4: PAY DRIVERS ===
with tab4:
    st.header("What drives pay at Wipro?")
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.subheader("Pay vs Performance")
        if 'Performance_Rating' in df.columns:
            fig9, ax9 = plt.subplots(figsize=(10, 4))
            sns.regplot(data=df, x='Performance_Rating', y='Annual_TCC (PPP USD)', 
                       scatter_kws={'alpha': 0.3, 'color': 'gray'}, line_kws={'color': 'red'}, ax=ax9)
            st.pyplot(fig9)
            
    with col_d2:
        st.subheader("Pay vs Experience")
        if 'Clean_Experience' in df.columns:
            fig10, ax10 = plt.subplots(figsize=(10, 4))
            sns.scatterplot(data=df, x='Clean_Experience', y='Annual_TCC (PPP USD)', 
                           hue='Band', palette='coolwarm', alpha=0.6, ax=ax10)
            st.pyplot(fig10)

    st.subheader("Gender Pay Equity Check")
    if 'Gender' in df.columns:
        df['Clean_Gender'] = df['Gender'].astype(str).str.upper().str.strip()
        fig11, ax11 = plt.subplots(figsize=(12, 4))
        sns.boxplot(data=df, x='Band', y='Annual_TCC (PPP USD)', hue='Clean_Gender', palette='pastel', ax=ax11)
        st.pyplot(fig11)

    st.subheader("Primary Pay Driver Correlation")
    # Prepare correlation data
    cols_map = {
        'Annual_TCC (PPP USD)': 'Annual Pay',
        'Clean_Experience': 'Total Experience',
        'Clean_Tenure': 'Tenure',
        'Performance_Rating': 'Performance'
    }
    
    # Filter columns that actually exist
    valid_cols = [k for k in cols_map.keys() if k in df.columns]
    
    if valid_cols:
        if 'Band' in df.columns:
            df['Band_Code'] = df['Band'].cat.codes
            valid_cols.append('Band_Code')
            cols_map['Band_Code'] = 'Job Band Level'

        corr_df = df[valid_cols].rename(columns=cols_map)
        
        fig12, ax12 = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            corr_df.corr()[['Annual Pay']].sort_values(by='Annual Pay', ascending=False),
            annot=True, cmap='RdYlGn', vmin=-1, vmax=1, ax=ax12
        )
        st.pyplot(fig12)
