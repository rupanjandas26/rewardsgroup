import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set page configuration
st.set_page_config(page_title="Wipro Workforce Rewards Analysis", layout="wide")

# Title and Introduction
st.title("Wipro Analysis: Group 13")
st.markdown("Version 1 of Wipro Worksforce Dashboard: This contains insights on Wipro's Market Positioning and Internal Equity Drivers")

# --- SIDEBAR: SETTINGS & FILE UPLOAD ---
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("IMPORTANT: Please make sure that you rename the file as 'data set' otherwise this tool won't work", type=['xlsb', 'xlsx'])

if uploaded_file is not None:
    # --- 1. DATA LOADING & PREPROCESSING ---
    @st.cache_data
    def load_data(file):
        # Read the file
        df = pd.read_excel(file, engine='pyxlsb')
        
        # Clean Currency
        df['Currency'] = df['Currency'].astype(str).str.strip().str.upper()
        
        # Currency Rates
        market_rates = {'USD': 1.0, 'INR': 0.0119, 'PHP': 0.0172}
        ppp_rates = {'USD': 1.0, 'INR': 1 / 22.54, 'PHP': 1 / 19.16}
        
        def convert_currency(row, col_name, rate_dict):
            currency = row['Currency']
            amount = row[col_name]
            if currency in rate_dict and pd.notnull(amount):
                return amount * rate_dict[currency]
            else:
                return None

        # Convert Target Columns
        target_cols = ['Annual_TCC', 'P10', 'P25', 'P50', 'P75', 'P90']
        for col in target_cols:
            if col in df.columns:
                df[f'{col} (Nominal USD)'] = df.apply(lambda row: convert_currency(row, col, market_rates), axis=1)
                df[f'{col} (PPP USD)'] = df.apply(lambda row: convert_currency(row, col, ppp_rates), axis=1)
        
        # Band Ordering
        df.columns = df.columns.str.strip()
        hierarchy_order = ['AA', 'A3', 'B1', 'B2', 'B3', 'C1', 'C3', 'D1', 'D2', 'D3']
        band_type = pd.CategoricalDtype(categories=hierarchy_order, ordered=True)
        df['Band'] = df['Band'].astype(band_type)
        
        # Cleaning Skills
        skill_col = 'Skill'
        if skill_col in df.columns:
            df[skill_col] = df[skill_col].astype(str).str.upper().str.strip()
            df[skill_col] = df[skill_col].replace(['NAN', 'NONE', '', 'nan'], 'NON PREMIUM')
            skill_order = ['NON PREMIUM', 'PREMIUM', 'SUPER PREMIUM', 'ULTRA PREMIUM']
            df['Clean_Skill'] = pd.Categorical(df[skill_col], categories=skill_order, ordered=True)
            
        # Cleaning Ratings
        rating_cols = [col for col in df.columns if 'Rating' in col or 'Perf' in col]
        for col in rating_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        if rating_cols:
            df['Clean_Rating'] = df[rating_cols].mean(axis=1)
            
        # Cleaning Tenure/Experience
        if 'Tenure' in df.columns:
            df['Clean_Tenure'] = pd.to_numeric(df['Tenure'], errors='coerce')
        if 'Experience' in df.columns:
            df['Clean_Experience'] = pd.to_numeric(df['Experience'], errors='coerce')
            
        # Hiring Source (Build vs Buy)
        if 'Clean_Experience' in df.columns and 'Clean_Tenure' in df.columns:
            df['Prior_Experience'] = (df['Clean_Experience'] - df['Clean_Tenure']).clip(lower=0)
            df['Hiring_Source'] = np.where(df['Prior_Experience'] < 2.0, 'Home Grown (Build)', 'Outside Hire (Buy)')
            
        # Clean Job Family & Gender
        if 'Job_Family' in df.columns:
            df['Clean_Job_Family'] = df['Job_Family'].astype(str).str.upper().str.strip()
        if 'Gender' in df.columns:
            df['Clean_Gender'] = df['Gender'].astype(str).str.upper().str.strip()

        # Compa-Ratio & Positioning
        market_col = 'P50 (PPP USD)'
        if market_col in df.columns:
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
            
        return df

    df = load_data(uploaded_file)
    st.success("Data successfully loaded!")

    # --- TABS FOR ANALYSIS ---
    tab1, tab2, tab3 = st.tabs(["Overview", "Market Positioning", "Pay Drivers"])

    # === TAB 1: OVERVIEW ===
    with tab1:
        st.header("Workforce Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Headcount by Job Band")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.countplot(x='Band', data=df, palette='viridis', ax=ax)
            ax.set_ylabel("Employees")
            st.pyplot(fig)

        with col2:
            st.subheader("Pay Distribution")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df['Annual_TCC (PPP USD)'], bins=50, kde=True, color='skyblue', ax=ax)
            ax.set_xlim(0, 250000)
            st.pyplot(fig)

        st.subheader("Pay Ranges by Job Band")
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.boxplot(x='Band', y='Annual_TCC (PPP USD)', data=df, palette="viridis", ax=ax)
        st.pyplot(fig)

    # === TAB 2: MARKET POSITIONING ===
    with tab2:
        st.header("Market Positioning Analysis")
        
        st.subheader("1. Overall Market Positioning (Compa-Ratio)")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(data=df, x='Compa_Ratio', kde=True, bins=100, color='teal', ax=ax)
        ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Market Median (1.0)')
        ax.axvline(0.8, color='orange', linestyle=':', linewidth=2, label='Low (0.8)')
        ax.axvline(1.2, color='orange', linestyle=':', linewidth=2, label='High (1.2)')
        ax.set_xlim(0, 4.5)
        ax.legend()
        st.pyplot(fig)

        st.subheader("2. Market Strategy (Median Pay vs Market)")
        market_strategy = df.groupby('Band')[['Compa_Ratio']].median().reset_index()
        market_strategy.rename(columns={'Compa_Ratio': 'Median_Compa_Ratio'}, inplace=True)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=market_strategy, x='Band', y='Median_Compa_Ratio', marker='o', 
                     markersize=10, linewidth=2.5, color='darkblue', label='Wipro Median Pay', ax=ax)
        ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Market Median')
        ax.fill_between(market_strategy['Band'], 0.95, 1.05, color='green', alpha=0.1, label='Competitive Range')
        ax.set_ylim(0.55, 1.25)
        ax.legend()
        
        # Add labels
        for x, y in zip(market_strategy['Band'], market_strategy['Median_Compa_Ratio']):
            if pd.notna(y):
                ax.text(x, y + 0.02, f'{y:.2f}', ha='center', fontweight='bold')
        st.pyplot(fig)
        
        st.subheader("3. Individual Positioning (Outlier Check)")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.scatterplot(data=df, x='Band', y='Compa_Ratio', hue='Positioning_Status', 
                        style='Positioning_Status', s=100, palette='viridis', ax=ax)
        ax.axhline(1.0, color='red', linestyle='--', alpha=0.5)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)

    # === TAB 3: PAY DRIVERS ===
    with tab3:
        st.header("Phase 3: Internal Equity & Pay Drivers")
        
        st.subheader("1. Capabilities & Strategy Check")
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        
        # Skill Mix
        sns.countplot(data=df, x='Band', hue='Clean_Skill', palette='viridis', ax=axes[0, 0])
        axes[0, 0].set_title('Skill Mix by Band')
        
        # Performance
        sns.boxplot(data=df, x='Band', y='Clean_Rating', palette='Oranges', ax=axes[0, 1])
        axes[0, 1].set_title('Performance Distribution')
        
        # Experience
        sns.boxplot(data=df, x='Band', y='Clean_Experience', palette='Blues', ax=axes[1, 0])
        axes[1, 0].set_title('Total Experience Profile')
        
        # Hiring Strategy
        hiring_strategy = pd.crosstab(df['Band'], df['Hiring_Source'], normalize='index')
        if not hiring_strategy.empty:
            cols = [c for c in ['Outside Hire (Buy)', 'Home Grown (Build)'] if c in hiring_strategy.columns]
            hiring_strategy = hiring_strategy[cols]
            hiring_strategy.plot(kind='bar', stacked=True, color=['#d73027', '#4575b4'], width=0.8, ax=axes[1, 1])
            axes[1, 1].set_title('Hiring Strategy: Build vs Buy')
            axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        st.pyplot(fig)
        
        st.markdown("---")
        st.subheader("2. Deep Dive: Pay Correlations")
        
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            st.markdown("**Pay vs. Skill Premium**")
            skill_pay = df.dropna(subset=['Annual_TCC (PPP USD)', 'Clean_Skill'])
            skill_summary = skill_pay.groupby(['Band', 'Clean_Skill'])['Annual_TCC (PPP USD)'].mean().reset_index()
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(data=skill_summary, x='Band', y='Annual_TCC (PPP USD)', hue='Clean_Skill', palette='viridis', ax=ax)
            ax.legend(title='Skill', bbox_to_anchor=(1, 1))
            st.pyplot(fig)

        with col_d2:
            st.markdown("**Pay vs. Performance**")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.regplot(data=df, x='Clean_Rating', y='Annual_TCC (PPP USD)', 
                        scatter_kws={'alpha': 0.3, 'color': 'gray'}, line_kws={'color': 'red'}, ax=ax)
            corr = df['Clean_Rating'].corr(df['Annual_TCC (PPP USD)'])
            ax.set_title(f"Correlation: {corr:.2f}")
            st.pyplot(fig)

        st.markdown("---")
        st.subheader("3. Pay Driver Heatmap")
        
        df['Band_Code'] = df['Band'].cat.codes
        cols_analyze = ['Annual_TCC (PPP USD)', 'Clean_Experience', 'Clean_Tenure', 'Clean_Rating', 'Band_Code']
        rename_map = {
            'Annual_TCC (PPP USD)': 'Annual Pay',
            'Clean_Experience': 'Total Experience',
            'Clean_Tenure': 'Tenure',
            'Clean_Rating': 'Performance',
            'Band_Code': 'Band Level'
        }
        
        drivers = df[cols_analyze].rename(columns=rename_map).corr()
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(drivers[['Annual Pay']].sort_values(by='Annual Pay', ascending=False), 
                    annot=True, cmap='RdYlGn', vmin=-1, vmax=1, fmt='.2f', cbar=False, ax=ax)
        st.pyplot(fig)

else:
    st.info("👈 Please upload your **data set.xlsb** file in the sidebar to start the analysis.")

