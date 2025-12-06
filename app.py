import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os  # Added for persistent file handling

# Page Config 
st.set_page_config(page_title="Rewards Group 13", layout="wide")
st.title("Wipro Dashboard v1")

# --- SIDEBAR ADDITIONS ---
st.sidebar.header("Hi!")
uploaded_file = st.sidebar.file_uploader(
    "Please upload the Wipro File here (.xlsb or .xlsx)", 
    type=['xlsb', 'xlsx']
)

if uploaded_file is not None:
    uploaded_file.name = 'data set.xlsb'

# Contact Box
st.sidebar.markdown("---")
st.sidebar.subheader("Contact Us")
st.sidebar.info("For queries, please reach out to Rewards Management: Group 13")

# --- END SIDEBAR ADDITIONS ---

# Data Processing Function
@st.cache_data
def process_data(file):
    df = pd.read_excel(file, engine='pyxlsb')
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
    existing_cols = [c for c in target_cols if c in df.columns]
    
    for col in existing_cols:
        df[f'{col} (Nominal USD)'] = df.apply(lambda row: convert_currency(row, col, market_rates), axis=1)
        df[f'{col} (PPP USD)'] = df.apply(lambda row: convert_currency(row, col, ppp_rates), axis=1)

    # 2. Band Sorting
    hierarchy_order = ['TEAMRBOW', 'A3', 'AA', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3', 'D1', 'D2', 'D3']
    if 'Band' in df.columns:
        band_type = pd.CategoricalDtype(categories=hierarchy_order, ordered=True)
        df['Band'] = df['Band'].astype(band_type)
        df = df.sort_values(by='Band')

    # 3. Compa-Ratio
    market_col = 'P50 (PPP USD)'
    if market_col in df.columns:
        df[market_col] = pd.to_numeric(df[market_col], errors='coerce')
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
    
    # 7. Build vs Buy Logic
    if 'Clean_Experience' in df.columns and 'Clean_Tenure' in df.columns:
        df['Prior_Experience'] = (df['Clean_Experience'] - df['Clean_Tenure']).clip(lower=0)
        df['Hiring_Source'] = np.where(df['Prior_Experience'] < 2.0, 'Home Grown (Build)', 'Outside Hire (Buy)')
        
    return df

# Indian Currency Formatting
def format_indian_currency(value):
    if pd.isna(value):
        return "₹0"
    
    value = int(value)
    s_val = str(value)
    
    if len(s_val) <= 3:
        return f"₹{s_val}"
    
    last_three = s_val[-3:]
    rest = s_val[:-3]
    
    groups = []
    while len(rest) > 2:
        groups.insert(0, rest[-2:])
        rest = rest[:-2]
    groups.insert(0, rest)
    
    formatted = ",".join(groups) + "," + last_three
    return f"₹{formatted}"

# Main App Logic

if uploaded_file is None:
    st.info("⬅️ Please use the sidebar to upload the appropriate file.")
    st.stop()

try:
    df = process_data(uploaded_file)
    st.success("Success! All Currency Figues in the visualizations are in PPP USD")
except Exception as e:
    st.error(f"Error processing file: {e}")
    st.stop()

# Visualizations in Tabs

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Market Strategy", "Workforce Analysis", "Pay Drivers", "Tools & Calculators"])

# TAB 1: OVERVIEW
with tab1:
    st.header("Overview: Pay Ranges & Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Pay Ranges by Job Band")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        # Updated Legend: Top Right
        sns.boxplot(x='Band', y='Annual_TCC (PPP USD)', data=df, ax=ax1, hue='Band', dodge=False)
        plt.title('Pay Ranges by Job Band (USD)')
        ax1.legend(title='Job Band', bbox_to_anchor=(1, 1), loc='upper right')
        st.pyplot(fig1)

    with col2:
        st.subheader("Headcount by Band")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        # Updated Legend: Top Right
        sns.countplot(x='Band', data=df, palette='viridis', ax=ax2, hue='Band', dodge=False)
        plt.title('Headcount by Job Band')
        ax2.legend(title='Job Band', bbox_to_anchor=(1, 1), loc='upper right')
        st.pyplot(fig2)

    st.subheader("Distribution of Annual Pay")
    fig3, ax3 = plt.subplots(figsize=(10, 3))
    # Updated Legend: Top Right
    sns.histplot(df['Annual_TCC (PPP USD)'], bins=100, kde=True, color='skyblue', ax=ax3, label='Pay Distribution')
    plt.xlim(0, 250000)
    ax3.legend(bbox_to_anchor=(1, 1), loc='upper right')
    st.pyplot(fig3)

# TAB 2: MARKET STRATEGY
with tab2:
    st.header("External Equity Overview")
    
    if 'Compa_Ratio' in df.columns:
        st.subheader("Overall Market Positioning")
        fig4, ax4 = plt.subplots(figsize=(10, 3))
        # Updated Legend: Top Right
        sns.histplot(data=df, x='Compa_Ratio', kde=True, bins=100, color='teal', ax=ax4, label='Population Distribution')
        ax4.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Market Median (1.0)')
        ax4.axvline(0.8, color='orange', linestyle=':', linewidth=2, label='Threshold (0.8)')
        ax4.axvline(1.2, color='orange', linestyle=':', linewidth=2, label='Threshold (1.2)')
        ax4.set_xlim(0, 4.5)
        ax4.legend(bbox_to_anchor=(1, 1), loc='upper right')
        st.pyplot(fig4)

        st.subheader("Individual Positioning by Band")
        fig5, ax5 = plt.subplots(figsize=(10, 4))
        sns.scatterplot(
            data=df, x='Band', y='Compa_Ratio', hue='Positioning_Status',
            style='Positioning_Status', s=100, palette='viridis', ax=ax5
        )
        ax5.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Market Median')
        # Legend anchored top right
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right')
        st.pyplot(fig5)

        st.subheader("Median Positioning By Band")
        market_strategy = df.groupby('Band')[['Compa_Ratio']].median().reset_index()
        fig6, ax6 = plt.subplots(figsize=(12, 6))
        sns.lineplot(
            data=market_strategy, x='Band', y='Compa_Ratio', marker='o',
            markersize=10, linewidth=2.5, color='darkblue', label='Wipro Median Pay', ax=ax6
        )
        ax6.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Market Median')
        ax6.fill_between(market_strategy['Band'], 0.95, 1.05, color='green', alpha=0.1, label='Competitive Zone')
        ax6.set_ylim(0.55, 1.25)
        # Legend anchored top right
        ax6.legend(bbox_to_anchor=(1, 1), loc='upper right')
        for x, y in zip(market_strategy['Band'], market_strategy['Compa_Ratio']):
            if pd.notna(y):
                ax6.text(x, y + 0.01, f'{y:.2f}', ha='center', fontweight='bold')
        st.pyplot(fig6)

# TAB 3: WORKFORCE ANALYSIS
with tab3:
    st.header("Internal Equity Overview")
    
    df_viz = df.copy()
    
    st.subheader("Population, Ratings, and Tenure Check")
    fig7, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    if 'Clean_Skill' in df_viz.columns:
        # Updated Legend: Top Right
        sns.countplot(data=df_viz, x='Clean_Skill', palette='viridis', ax=axes[0], hue='Clean_Skill', dodge=False)
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].legend(bbox_to_anchor=(1, 1), loc='upper right', title='Skill')
    
    if 'Performance_Rating' in df_viz.columns:
        # Updated Legend: Top Right
        sns.histplot(data=df_viz, x='Performance_Rating', bins=10, kde=True, color='orange', ax=axes[1], label='Distribution')
        axes[1].legend(bbox_to_anchor=(1, 1), loc='upper right')
    
    if 'Clean_Tenure' in df_viz.columns:
        # Updated Legend: Top Right
        sns.histplot(data=df_viz, x='Clean_Tenure', bins=20, kde=True, color='teal', ax=axes[2], label='Distribution')
        axes[2].legend(bbox_to_anchor=(1, 1), loc='upper right')
    
    st.pyplot(fig7)

    st.subheader("Exploratory insights of current workforce")
    fig8, axes2 = plt.subplots(2, 2, figsize=(15, 12))
    
    # Updated Legend: Top Right
    sns.countplot(data=df_viz, x='Band', hue='Clean_Skill', palette='viridis', ax=axes2[0, 0])
    axes2[0, 0].set_title('Skill Mix by Band')
    axes2[0, 0].legend(bbox_to_anchor=(1, 1), loc='upper right', title='Skill')
    
    if 'Performance_Rating' in df_viz.columns:
        # Updated Legend: Top Right
        sns.boxplot(data=df_viz, x='Band', y='Performance_Rating', palette='Oranges', ax=axes2[0, 1], hue='Band', dodge=False)
        axes2[0, 1].set_title('Performance Distribution')
        axes2[0, 1].legend(bbox_to_anchor=(1, 1), loc='upper right', title='Band')

    if 'Clean_Experience' in df_viz.columns:
        # Updated Legend: Top Right
        sns.boxplot(data=df_viz, x='Band', y='Clean_Experience', palette='Blues', ax=axes2[1, 0], hue='Band', dodge=False)
        axes2[1, 0].set_title('Total Experience Profile')
        axes2[1, 0].legend(bbox_to_anchor=(1, 1), loc='upper right', title='Band')

    if 'Hiring_Source' in df_viz.columns:
        hiring_strategy = pd.crosstab(df_viz['Band'], df_viz['Hiring_Source'], normalize='index')
        cols_to_plot = [c for c in ['Outside Hire (Buy)', 'Home Grown (Build)'] if c in hiring_strategy.columns]
        if cols_to_plot:
            hiring_strategy = hiring_strategy[cols_to_plot]
            hiring_strategy.plot(kind='bar', stacked=True, color=['#d73027', '#4575b4'], width=0.8, ax=axes2[1, 1])
            axes2[1, 1].set_title('Build vs Buy Ratio')
            # Legend anchored top right
            axes2[1, 1].legend(bbox_to_anchor=(1, 1), loc='upper right', title='Source')
    
    st.pyplot(fig8)

# TAB 4: PAY DRIVERS
with tab4:
    st.header("What drives pay at Wipro?")
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.subheader("Pay vs Performance")
        if 'Performance_Rating' in df.columns:
            fig9, ax9 = plt.subplots(figsize=(10, 4))
            # Updated Legend: Top Right
            sns.regplot(data=df, x='Performance_Rating', y='Annual_TCC (PPP USD)', 
                        scatter_kws={'alpha': 0.3, 'color': 'gray', 'label': 'Employee Data'}, 
                        line_kws={'color': 'red', 'label': 'Trend Line'}, ax=ax9)
            ax9.legend(bbox_to_anchor=(1, 1), loc='upper right')
            st.pyplot(fig9)
            
    with col_d2:
        st.subheader("Pay vs Experience")
        if 'Clean_Experience' in df.columns:
            fig10, ax10 = plt.subplots(figsize=(10, 4))
            sns.scatterplot(data=df, x='Clean_Experience', y='Annual_TCC (PPP USD)', 
                            hue='Band', palette='coolwarm', alpha=0.6, ax=ax10)
            # Updated Legend: Top Right
            ax10.legend(bbox_to_anchor=(1, 1), loc='upper right', title='Band')
            st.pyplot(fig10)

    st.subheader("Gender Pay Equity Check")
    if 'Gender' in df.columns:
        df['Clean_Gender'] = df['Gender'].astype(str).str.upper().str.strip()
        fig11, ax11 = plt.subplots(figsize=(12, 4))
        sns.boxplot(data=df, x='Band', y='Annual_TCC (PPP USD)', hue='Clean_Gender', palette='pastel', ax=ax11)
        # Updated Legend: Top Right
        ax11.legend(bbox_to_anchor=(1, 1), loc='upper right', title='Gender')
        st.pyplot(fig11)

    st.subheader("Primary Pay Driver Correlation")
    cols_map = {
        'Annual_TCC (PPP USD)': 'Annual Pay',
        'Clean_Experience': 'Total Experience',
        'Clean_Tenure': 'Tenure',
        'Performance_Rating': 'Performance'
    }
    
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

# TAB 5: TOOLS
with tab5:
    st.header("Tools")
    st.markdown("These tools are designed to make data-driven decisions based on the anaylis drawn from uploaded file.")

    PPP_INR_RATE = 22.54 

    with st.expander("Recruitment Salary Fitment Calculator", expanded=True):
        st.write("Use this tool to determine the appropriate offer range for a new hire.")
        
        calc_col1, calc_col2 = st.columns(2)
        
        with calc_col1:
            c_band = st.selectbox("Select Candidate Job Band", options=df['Band'].unique())
        
        with calc_col2:
            c_exp = st.number_input("Candidate Years of Experience", min_value=0.0, step=0.5)

        if st.button("Calculate Recommended Offer"):
            
            band_data = df[df['Band'] == c_band]
            
            stats = band_data['Annual_TCC (PPP USD)'].describe()
            p25 = stats['25%']
            p50 = stats['50%']
            p75 = stats['75%']
            
            st.markdown(f"### Market Benchmarks for Band: {c_band}")
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            def fmt_currency(usd_val):
                inr_val = usd_val * PPP_INR_RATE
                return f"${usd_val:,.0f} / {format_indian_currency(inr_val)}"

            metric_col1.metric("Low End (25th %)", fmt_currency(p25))
            metric_col2.metric("Median (50th %)", fmt_currency(p50))
            metric_col3.metric("High End (75th %)", fmt_currency(p75))
            
            st.markdown("#### Recommendation:")
            if c_exp > 5:
                st.success(f"Candidate is experienced ({c_exp} years). \n\n**Target Offer:** {fmt_currency(p50)} - {fmt_currency(p75)}")
            else:
                st.info(f"Candidate is junior/mid-level ({c_exp} years). \n\n**Target Offer:** {fmt_currency(p25)} - {fmt_currency(p50)}")

    # Tool 2: Flight Risk Detector
    with st.expander("Flight Risk Detector"):
        st.write("High performers who are underpaid")
        
        # Logic: High Performance (> 4) AND Low Compa-Ratio (< 0.8)
        if 'Performance_Rating' in df.columns and 'Compa_Ratio' in df.columns:
            risk_df = df[
                (df['Performance_Rating'] >= 4.0) & 
                (df['Compa_Ratio'] < 0.85)
            ][['ID', 'Band', 'Performance_Rating', 'Compa_Ratio', 'Annual_TCC (PPP USD)']]
            
            st.error(f"⚠️ Found {len(risk_df)} High-Risk Employees!")
            st.dataframe(risk_df.sort_values(by='Performance_Rating', ascending=False))
        else:
            st.warning("Performance or Market data missing.")

    # Tool 3: Pay Equity Auditor
    with st.expander("Individual Pay Equity Checker"):
        st.write("Check specific employee for internal/external equity.")
        
        emp_id = st.text_input("Enter Employee ID to Audit:")
        
        if emp_id:
            record = df[df['ID'].astype(str) == str(emp_id)]
            
            if not record.empty:
                r = record.iloc[0]
                usd_pay = r['Annual_TCC (PPP USD)']
                inr_pay = usd_pay * PPP_INR_RATE
                
                col1, col2 = st.columns(2)
                col1.metric("Current Pay (PPP Adjusted)", f"${usd_pay:,.0f} / {format_indian_currency(inr_pay)}")
                col2.metric("Compa-Ratio", f"{r['Compa_Ratio']:.2f}")
                
                if r['Compa_Ratio'] < 0.8:
                    st.error("❌ Equity Alert: Employee is significantly underpaid compared to market.")
                elif r['Compa_Ratio'] > 1.2:
                    st.warning("⚠️ Equity Alert: Employee is paid significantly above market.")
                else:
                    st.success("✅ Employee pay is equitable.")
            else:
                st.error("Employee ID not found.")
