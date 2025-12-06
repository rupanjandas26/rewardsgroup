import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Page Config ---
st.set_page_config(page_title="Rewards Management: Group 13", layout="wide")
st.title("Total Rewards & Workforce Analytics Dashboard")

# --- 2. File Uploader ---
st.sidebar.header("Upload your files here")
uploaded_file = st.sidebar.file_uploader(
    "**IMPORTANT:** Upload your file here, make sure the name of the file is **'data set'** and the format is **.xlsb**", 
    type=['xlsb', 'xlsx']
)

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
        df['Skill'] = df['Skill'].replace(['NAN', 'NONE', '', 'nan'], 'NON
