# --- CSS FOR PROFESSIONAL STYLING ---
st.markdown("""
    <style>
    /* 1. Force Main Background to White */
    .main {
        background-color: #ffffff;
    }
    
    /* 2. FORCE METRIC TEXT TO BLACK (Fixes the unreadable issue) */
    div[data-testid="stMetricValue"] {
        color: #000000 !important; /* The numbers (e.g. 34,539) */
    }
    div[data-testid="stMetricLabel"] {
        color: #333333 !important; /* The labels (e.g. Total Employees) */
    }
    
    /* 3. Header Colors */
    h1, h2, h3 {
        color: #002e6e; 
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
    }
    
    /* 4. Tab Styling */
    .stTabs [data-baseweb="tab"] {
        color: #333333; /* Tab text color */
    }
    .stTabs [aria-selected="true"] {
        color: #002e6e; /* Selected tab color */
        border-bottom-color: #002e6e;
    }
    </style>
""", unsafe_allow_html=True)
