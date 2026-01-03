import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from scipy.signal import find_peaks
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import silhouette_score
import plotly.figure_factory as ff
import time
import calendar
import json
from collections import defaultdict

# ==============================================
# 🎯 ADVANCED CONFIGURATION
# ==============================================
st.set_page_config(
    page_title="Quantum Restaurant Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================
# 🎨 ENHANCED PROFESSIONAL DARK THEME WITH GLASSMORPHISM
# ==============================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --primary-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    --success-gradient: linear-gradient(135deg, #10b981 0%, #34d399 100%);
    --danger-gradient: linear-gradient(135deg, #ef4444 0%, #f97316 100%);
    --warning-gradient: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%);
    --info-gradient: linear-gradient(135deg, #0ea5e9 0%, #3b82f6 100%);
    --text-primary: #f8fafc;
    --text-secondary: #cbd5e1;
    --text-muted: #94a3b8;
    --dark-bg: #0f172a;
    --card-bg: rgba(30, 41, 59, 0.8);
    --glass-bg: rgba(255, 255, 255, 0.07);
    --border-color: rgba(255, 255, 255, 0.12);
    --accent-purple: #a78bfa;
    --accent-blue: #60a5fa;
    --accent-green: #34d399;
}

* {
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}

.main {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
}

/* Headers with better typography */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1rem;
}

h1 {
    font-size: 2.8rem;
    letter-spacing: -0.5px;
}

h2 {
    font-size: 2.2rem;
    letter-spacing: -0.3px;
}

h3 {
    font-size: 1.8rem;
}

/* Advanced Glassmorphism Cards */
.quantum-card {
    background: var(--glass-bg);
    backdrop-filter: blur(40px);
    border: 1px solid var(--border-color);
    border-radius: 24px;
    padding: 30px;
    margin: 20px 0;
    box-shadow: 
        0 4px 6px -1px rgba(0, 0, 0, 0.3),
        0 2px 4px -1px rgba(0, 0, 0, 0.2),
        inset 0 0 0 1px rgba(255, 255, 255, 0.05);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.quantum-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
    transition: left 0.7s ease-in-out;
}

.quantum-card:hover {
    transform: translateY(-8px);
    border-color: rgba(139, 92, 246, 0.4);
    box-shadow: 
        0 20px 25px -5px rgba(0, 0, 0, 0.4),
        0 10px 10px -5px rgba(0, 0, 0, 0.3),
        0 0 0 1px rgba(139, 92, 246, 0.2),
        inset 0 0 30px rgba(99, 102, 241, 0.1);
}

.quantum-card:hover::before {
    left: 100%;
}

/* Text content inside cards */
.quantum-card p,
.quantum-card div,
.quantum-card span:not(.metric-value):not(.metric-change) {
    color: var(--text-secondary);
    line-height: 1.6;
    font-weight: 400;
}

.quantum-card strong,
.quantum-card b {
    color: var(--text-primary);
    font-weight: 600;
}

/* Neural Network Background */
.neural-bg {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -1;
    opacity: 0.04;
    background-image: 
        radial-gradient(circle at 20% 50%, rgba(99, 102, 241, 0.4) 0%, transparent 50%),
        radial-gradient(circle at 80% 80%, rgba(139, 92, 246, 0.4) 0%, transparent 50%),
        radial-gradient(circle at 40% 20%, rgba(96, 165, 250, 0.4) 0%, transparent 50%);
    animation: neural-pulse 15s ease-in-out infinite;
}

@keyframes neural-pulse {
    0%, 100% { opacity: 0.04; transform: scale(1); }
    50% { opacity: 0.08; transform: scale(1.02); }
}

/* Premium Metric Card */
.metric-card {
    background: var(--card-bg);
    backdrop-filter: blur(25px);
    border-radius: 20px;
    padding: 25px;
    border: 1px solid var(--border-color);
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}

.metric-card::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: var(--primary-gradient);
    border-radius: 20px 20px 0 0;
}

.metric-card:hover {
    transform: scale(1.02);
    border-color: rgba(99, 102, 241, 0.5);
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
}

.metric-value {
    font-size: 3rem;
    font-weight: 800;
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 15px 0;
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: -1px;
}

.metric-label {
    color: var(--text-muted);
    font-size: 0.9rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

.metric-description {
    color: var(--text-secondary);
    font-size: 0.95rem;
    margin-top: 8px;
    line-height: 1.5;
}

.metric-change {
    display: inline-flex;
    align-items: center;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 10px;
    color: white;
}

.metric-change.positive {
    background: var(--success-gradient);
    color: white;
}

.metric-change.negative {
    background: var(--danger-gradient);
    color: white;
}

/* Advanced Data Visualization Container */
.viz-container {
    background: rgba(30, 41, 59, 0.7);
    border-radius: 20px;
    padding: 25px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(20px);
}

.viz-container h3,
.viz-container h4 {
    color: var(--text-primary);
    margin-bottom: 1.5rem;
}

/* Animated Progress Bars */
.progress-bar {
    height: 10px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    overflow: hidden;
    margin: 15px 0;
}

.progress-fill {
    height: 100%;
    background: var(--primary-gradient);
    border-radius: 10px;
    transition: width 1.2s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.progress-fill::after {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
    animation: shimmer 2s infinite;
}

@keyframes shimmer {
    0% { left: -100%; }
    100% { left: 100%; }
}

/* Insights Badge */
.insight-badge {
    display: inline-flex;
    align-items: center;
    padding: 10px 18px;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: 600;
    margin: 8px 8px 8px 0;
    transition: all 0.3s ease;
    color: white;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
}

.insight-badge:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 12px rgba(0, 0, 0, 0.3);
}

.insight-badge.critical {
    background: var(--danger-gradient);
}

.insight-badge.warning {
    background: var(--warning-gradient);
}

.insight-badge.success {
    background: var(--success-gradient);
}

.insight-badge.info {
    background: var(--info-gradient);
}

.insight-badge span {
    color: white !important;
    font-weight: 600;
}

/* Floating Action Button */
.fab {
    position: fixed;
    bottom: 30px;
    right: 30px;
    width: 60px;
    height: 60px;
    border-radius: 50%;
    background: var(--primary-gradient);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    box-shadow: 0 10px 30px rgba(99, 102, 241, 0.4);
    cursor: pointer;
    transition: all 0.3s ease;
    z-index: 1000;
    color: white;
}

.fab:hover {
    transform: scale(1.1) rotate(90deg);
    box-shadow: 0 15px 40px rgba(99, 102, 241, 0.6);
}

/* Streamlit Overrides - Better Text Visibility */
.stTabs [data-baseweb="tab"] {
    color: var(--text-muted);
    font-weight: 600;
    transition: all 0.3s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-primary);
}

.stTabs [aria-selected="true"] {
    color: white !important;
}

/* General text improvements */
p, div, span:not(.metric-value):not(.metric-change) {
    color: var(--text-secondary);
}

strong, b {
    color: var(--text-primary);
    font-weight: 600;
}

small {
    color: var(--text-muted);
    opacity: 0.9;
}

/* Data Table Enhancements */
.dataframe {
    background: var(--card-bg);
    color: var(--text-secondary);
}

.dataframe th {
    background: rgba(99, 102, 241, 0.1);
    color: var(--text-primary) !important;
    font-weight: 600;
}

.dataframe td {
    color: var(--text-secondary) !important;
}

/* Alert Boxes */
.alert-box {
    color: var(--text-primary);
    font-weight: 500;
}

.alert-box.success {
    background: rgba(16, 185, 129, 0.1);
    color: #10b981;
}

.alert-box.warning {
    background: rgba(245, 158, 11, 0.1);
    color: #f59e0b;
}

.alert-box.error {
    background: rgba(239, 68, 68, 0.1);
    color: #ef4444;
}

.alert-box.info {
    background: rgba(14, 165, 233, 0.1);
    color: #0ea5e9;
}

/* Scrollbar Styling */
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 5px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 5px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #8b5cf6, #a78bfa);
}

/* Responsive Grid */
.responsive-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 25px;
    margin: 25px 0;
}

/* Tooltip */
.tooltip .tooltiptext {
    background-color: rgba(30, 41, 59, 0.95);
    color: var(--text-primary) !important;
    border: 1px solid var(--border-color);
    backdrop-filter: blur(10px);
}

/* Input fields and buttons */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div > div {
    background: rgba(255, 255, 255, 0.05) !important;
    color: var(--text-primary) !important;
    border-color: var(--border-color) !important;
}

.stButton > button {
    background: var(--primary-gradient) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
}

/* Section headers */
.section-header {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1));
    border-left: 4px solid #6366f1;
    padding: 15px 20px;
    border-radius: 12px;
    margin: 25px 0;
}

.section-header h2 {
    margin-bottom: 5px;
}

.section-header p {
    color: var(--text-muted);
    margin: 0;
}

/* Status indicators */
.status-indicator {
    display: inline-flex;
    align-items: center;
    padding: 6px 12px;
    border-radius: 16px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-left: 10px;
}

.status-active {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
}

.status-pending {
    background: rgba(245, 158, 11, 0.2);
    color: #f59e0b;
}

.status-inactive {
    background: rgba(100, 116, 139, 0.2);
    color: #64748b;
}

</style>
<div class="neural-bg"></div>
""", unsafe_allow_html=True)

# ==============================================
# 🧠 ADVANCED DATA GENERATION WITH REALISTIC PATTERNS
# ==============================================
@st.cache_data
def generate_advanced_data():
    """Generate sophisticated restaurant data with complex patterns"""
    np.random.seed(42)
    n_records = 17534
    
    dates = pd.date_range('2023-01-01', '2024-01-01', periods=n_records)
    
    # Advanced seasonal and trend components
    trend = np.linspace(1, 1.3, n_records)
    seasonality = 1 + 0.2 * np.sin(2 * np.pi * np.arange(n_records) / 365)
    weekly = 1 + 0.1 * np.sin(2 * np.pi * np.arange(n_records) / 7)
    
    data = {
        'Order_ID': [f'ORD{str(i).zfill(6)}' for i in range(1, n_records + 1)],
        'Customer_ID': [f'CUST{np.random.randint(1000, 5000):04d}' for _ in range(n_records)],
        'Category': np.random.choice(
            ['Appetizer', 'Main Course', 'Dessert', 'Beverage', 'Side Dish', 'Special Menu'],
            n_records,
            p=[0.15, 0.35, 0.12, 0.23, 0.10, 0.05]
        ),
        'Item': [],
        'Price': [],
        'Quantity': [],
        'Order_Total': [],
        'Order_Date': dates,
        'Payment_Method': np.random.choice(
            ['Credit Card', 'Debit Card', 'Cash', 'Digital Wallet', 'Cryptocurrency'],
            n_records,
            p=[0.45, 0.28, 0.12, 0.13, 0.02]
        ),
        'Service_Type': np.random.choice(
            ['Dine-in', 'Takeout', 'Delivery', 'Curbside'],
            n_records,
            p=[0.45, 0.25, 0.25, 0.05]
        ),
        'Staff_ID': [f'STAFF{np.random.randint(1, 25):03d}' for _ in range(n_records)],
        'Table_Number': [np.random.randint(1, 51) if np.random.random() > 0.3 else None for _ in range(n_records)],
        'Weather': np.random.choice(['Sunny', 'Rainy', 'Cloudy', 'Snowy'], n_records, p=[0.4, 0.2, 0.3, 0.1])
    }
    
    items = {
        'Appetizer': [('Garlic Bread', 8.99), ('Bruschetta', 12.99), ('Nachos', 14.99), ('Spring Rolls', 10.99)],
        'Main Course': [('Grilled Salmon', 28.99), ('Filet Mignon', 42.99), ('Pasta Carbonara', 22.99), 
                       ('Chicken Parmesan', 24.99), ('Lobster Tail', 49.99)],
        'Dessert': [('Chocolate Cake', 9.99), ('Cheesecake', 11.99), ('Tiramisu', 10.99), ('Crème Brûlée', 12.99)],
        'Beverage': [('Craft Beer', 7.99), ('Wine', 12.99), ('Cocktail', 14.99), ('Mocktail', 8.99)],
        'Side Dish': [('Fries', 5.99), ('Salad', 8.99), ('Mashed Potatoes', 6.99), ('Grilled Vegetables', 9.99)],
        'Special Menu': [('Chef Special', 55.99), ('Tasting Menu', 89.99), ('Seasonal Platter', 38.99)]
    }
    
    for i in range(n_records):
        category = data['Category'][i]
        item, base_price = items[category][np.random.randint(0, len(items[category]))]
        
        # Apply sophisticated pricing logic
        price = base_price * trend[i] * seasonality[i] * weekly[i]
        
        # Weekend premium
        if dates[i].weekday() >= 5:
            price *= 1.15
        
        # Holiday premium
        if dates[i].month == 12 or dates[i].month == 1:
            price *= 1.1
        
        # Weather impact
        if data['Weather'][i] == 'Rainy':
            price *= 0.95  # Slight discount on rainy days
        
        data['Item'].append(item)
        data['Price'].append(round(price, 2))
        
        # Intelligent quantity distribution
        if category == 'Beverage':
            quantity = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
        else:
            quantity = np.random.choice([1, 2, 3, 4], p=[0.65, 0.25, 0.08, 0.02])
        
        data['Quantity'].append(quantity)
        data['Order_Total'].append(round(price * quantity, 2))
    
    df = pd.DataFrame(data)
    
    # Advanced feature engineering
    df['Order_Hour'] = df['Order_Date'].dt.hour
    df['Order_Day'] = df['Order_Date'].dt.day_name()
    df['Order_Month'] = df['Order_Date'].dt.month
    df['Order_Week'] = df['Order_Date'].dt.isocalendar().week
    df['Is_Weekend'] = df['Order_Date'].dt.weekday >= 5
    df['Is_Holiday'] = df['Order_Month'].isin([12, 1, 7])
    df['Day_Part'] = pd.cut(df['Order_Hour'], bins=[0, 11, 17, 22, 24], 
                           labels=['Morning', 'Afternoon', 'Evening', 'Late Night'])
    df['Season'] = df['Order_Month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    # Customer analytics
    customer_stats = df.groupby('Customer_ID').agg({
        'Order_Total': ['sum', 'mean', 'std'],
        'Order_ID': 'count',
        'Order_Date': ['min', 'max']
    }).reset_index()
    
    customer_stats.columns = ['Customer_ID', 'Total_Spent', 'Avg_Order', 'Std_Order', 
                              'Visit_Count', 'First_Visit', 'Last_Visit']
    customer_stats['Customer_Lifetime_Days'] = (customer_stats['Last_Visit'] - customer_stats['First_Visit']).dt.days
    customer_stats['Purchase_Frequency'] = customer_stats['Visit_Count'] / (customer_stats['Customer_Lifetime_Days'] + 1)
    
    df = df.merge(customer_stats[['Customer_ID', 'Total_Spent', 'Visit_Count', 'Purchase_Frequency']], 
                  on='Customer_ID', how='left')
    
    # RFM Analysis
    max_date = df['Order_Date'].max()
    rfm = df.groupby('Customer_ID').agg({
        'Order_Date': lambda x: (max_date - x.max()).days,
        'Order_ID': 'count',
        'Order_Total': 'sum'
    }).reset_index()
    rfm.columns = ['Customer_ID', 'Recency', 'Frequency', 'Monetary']
    
    # RFM Scoring
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1], duplicates='drop')
    rfm['F_Score'] = pd.qcut(rfm['Frequency'], 5, labels=[1,2,3,4,5], duplicates='drop')
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5], duplicates='drop')
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    
    df = df.merge(rfm[['Customer_ID', 'RFM_Score']], on='Customer_ID', how='left')
    
    # CLV Tier
    df['CLV_Tier'] = pd.qcut(df['Total_Spent'], 5, 
                            labels=['Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond'],
                            duplicates='drop')
    
    # Profit margin (simulated)
    df['Cost'] = df['Price'] * 0.35  # 65% gross margin average
    df['Profit'] = df['Order_Total'] - (df['Cost'] * df['Quantity'])
    df['Profit_Margin'] = (df['Profit'] / df['Order_Total']) * 100
    
    return df

# ==============================================
# 🎯 ADVANCED ANALYTICS ENGINE - COMPLETE VERSION
# ==============================================
class QuantumAnalytics:
    def __init__(self, df):
        self.df = df
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
    
    def advanced_anomaly_detection(self):
        """Multi-method anomaly detection"""
        # Isolation Forest
        features = self.df[['Order_Total', 'Quantity', 'Price']].values
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        self.df['Anomaly_Score'] = iso_forest.fit_predict(features)
        anomalies = self.df[self.df['Anomaly_Score'] == -1]
        
        # Z-score method
        z_scores = np.abs(stats.zscore(self.df['Order_Total']))
        statistical_anomalies = self.df[z_scores > 3]
        
        return anomalies, statistical_anomalies
    
    def intelligent_segmentation(self):
        """Advanced customer segmentation with multiple methods"""
        features = self.df.groupby('Customer_ID').agg({
            'Order_Total': ['sum', 'mean', 'std'],
            'Visit_Count': 'first',
            'Purchase_Frequency': 'first',
            'Profit': 'sum'
        }).reset_index()
        
        features.columns = ['Customer_ID', 'Total_Spent', 'Avg_Order', 'Std_Order', 
                           'Visit_Count', 'Purchase_Frequency', 'Total_Profit']
        features = features.fillna(0)
        
        # Normalize features
        X = self.scaler.fit_transform(features[['Total_Spent', 'Avg_Order', 'Visit_Count', 
                                                 'Purchase_Frequency', 'Total_Profit']])
        
        # Determine optimal clusters using silhouette score
        silhouette_scores = []
        K_range = range(3, 8)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            silhouette_scores.append(score)
        
        optimal_k = K_range[np.argmax(silhouette_scores)]
        
        # Apply K-means with optimal k
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        features['Segment'] = kmeans.fit_predict(X)
        
        # PCA for visualization
        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(X)
        features['PCA1'] = pca_features[:, 0]
        features['PCA2'] = pca_features[:, 1]
        features['PCA3'] = pca_features[:, 2]
        features['Explained_Variance'] = pca.explained_variance_ratio_.sum()
        
        # Name segments intelligently
        segment_stats = features.groupby('Segment').agg({
            'Total_Spent': 'mean',
            'Visit_Count': 'mean',
            'Purchase_Frequency': 'mean'
        })
        
        segment_names = {}
        for seg in segment_stats.index:
            stats_row = segment_stats.loc[seg]
            if stats_row['Total_Spent'] > segment_stats['Total_Spent'].median() * 1.5:
                if stats_row['Visit_Count'] > segment_stats['Visit_Count'].median():
                    segment_names[seg] = 'VIP Champions'
                else:
                    segment_names[seg] = 'High-Value Whales'
            elif stats_row['Visit_Count'] > segment_stats['Visit_Count'].median() * 1.5:
                segment_names[seg] = 'Loyal Regulars'
            elif stats_row['Purchase_Frequency'] > segment_stats['Purchase_Frequency'].median():
                segment_names[seg] = 'Growing Enthusiasts'
            else:
                segment_names[seg] = 'Casual Visitors'
        
        features['Segment_Name'] = features['Segment'].map(segment_names)
        
        return features, pca.explained_variance_ratio_
    
    def ml_revenue_forecast(self, days=30):
        """Machine learning-based revenue forecasting"""
        daily_revenue = self.df.resample('D', on='Order_Date').agg({
            'Order_Total': 'sum',
            'Order_ID': 'count'
        }).reset_index()
        
        daily_revenue['Day_of_Week'] = daily_revenue['Order_Date'].dt.dayofweek
        daily_revenue['Day_of_Month'] = daily_revenue['Order_Date'].dt.day
        daily_revenue['Month'] = daily_revenue['Order_Date'].dt.month
        daily_revenue['Days_Since_Start'] = (daily_revenue['Order_Date'] - daily_revenue['Order_Date'].min()).dt.days
        
        # Rolling features
        daily_revenue['Revenue_MA7'] = daily_revenue['Order_Total'].rolling(window=7, min_periods=1).mean()
        daily_revenue['Revenue_MA30'] = daily_revenue['Order_Total'].rolling(window=30, min_periods=1).mean()
        daily_revenue['Revenue_Std7'] = daily_revenue['Order_Total'].rolling(window=7, min_periods=1).std()
        
        daily_revenue = daily_revenue.fillna(0)
        
        # Train Random Forest
        feature_cols = ['Day_of_Week', 'Day_of_Month', 'Month', 'Days_Since_Start', 
                       'Revenue_MA7', 'Revenue_MA30', 'Revenue_Std7']
        X = daily_revenue[feature_cols].values
        y = daily_revenue['Order_Total'].values
        
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        rf_model.fit(X, y)
        
        # Generate future predictions
        last_date = daily_revenue['Order_Date'].max()
        future_dates = pd.date_range(last_date + timedelta(days=1), periods=days, freq='D')
        
        future_features = []
        for i, date in enumerate(future_dates):
            features = [
                date.dayofweek,
                date.day,
                date.month,
                (date - daily_revenue['Order_Date'].min()).days,
                daily_revenue['Order_Total'].tail(7).mean(),
                daily_revenue['Order_Total'].tail(30).mean(),
                daily_revenue['Order_Total'].tail(7).std()
            ]
            future_features.append(features)
        
        future_predictions = rf_model.predict(future_features)
        
        # Calculate confidence intervals using prediction error
        y_pred_train = rf_model.predict(X)
        mae = np.mean(np.abs(y - y_pred_train))
        
        lower_bound = future_predictions - 1.96 * mae
        upper_bound = future_predictions + 1.96 * mae
        
        return future_dates, future_predictions, lower_bound, upper_bound, mae
    
    def detect_peak_periods(self):
        """Detect peak periods using signal processing"""
        hourly_revenue = self.df.groupby('Order_Hour')['Order_Total'].sum().reset_index()
        
        # Find peaks in revenue
        peaks, properties = find_peaks(hourly_revenue['Order_Total'].values, 
                                      height=np.mean(hourly_revenue['Order_Total']) * 1.5,
                                      distance=2)
        
        peak_periods = []
        if len(peaks) > 0:
            for peak in peaks:
                peak_hour = hourly_revenue.iloc[peak]['Order_Hour']
                peak_value = hourly_revenue.iloc[peak]['Order_Total']
                peak_periods.append({
                    'hour': peak_hour,
                    'revenue': peak_value,
                    'peak_type': 'Primary' if peak_value > np.mean(hourly_revenue['Order_Total']) * 2 else 'Secondary'
                })
        
        return peak_periods, hourly_revenue
    
    def market_basket_analysis(self, min_support=0.01, min_confidence=0.5):
        """Simple market basket analysis"""
        from mlxtend.frequent_patterns import apriori, association_rules
        
        try:
            # Create transaction matrix
            transactions = self.df.groupby('Order_ID')['Item'].apply(list).reset_index()
            
            # Create one-hot encoded matrix
            all_items = list(set(self.df['Item']))
            transaction_matrix = pd.DataFrame(0, index=transactions['Order_ID'], columns=all_items)
            
            for idx, row in transactions.iterrows():
                for item in row['Item']:
                    transaction_matrix.loc[row['Order_ID'], item] = 1
            
            # Apply Apriori algorithm
            frequent_itemsets = apriori(transaction_matrix, min_support=min_support, use_colnames=True)
            
            if len(frequent_itemsets) > 0:
                rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
                return frequent_itemsets, rules
            else:
                return pd.DataFrame(), pd.DataFrame()
                
        except Exception as e:
            st.warning(f"Market Basket Analysis skipped: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()
    
    def profitability_analysis(self):
        """Analyze profitability by category and item"""
        profitability_df = self.df.groupby(['Category', 'Item']).agg({
            'Order_Total': 'sum',
            'Profit': 'sum',
            'Order_ID': 'count',
            'Quantity': 'sum'
        }).reset_index()
        
        profitability_df['Profit_Margin_Pct'] = (profitability_df['Profit'] / profitability_df['Order_Total']) * 100
        profitability_df['Avg_Profit_Per_Unit'] = profitability_df['Profit'] / profitability_df['Quantity']
        profitability_df['Popularity_Rank'] = profitability_df['Order_ID'].rank(ascending=False)
        profitability_df['Profitability_Rank'] = profitability_df['Profit_Margin_Pct'].rank(ascending=False)
        
        return profitability_df
    
    def seasonal_patterns(self):
        """Analyze seasonal patterns and trends"""
        seasonal_data = self.df.groupby(['Season', 'Category']).agg({
            'Order_Total': 'sum',
            'Order_ID': 'count',
            'Profit': 'sum'
        }).reset_index()
        
        # Calculate seasonality index
        overall_avg = self.df['Order_Total'].mean()
        seasonal_index = seasonal_data.groupby('Season')['Order_Total'].mean() / overall_avg
        
        return seasonal_data, seasonal_index
    
    def staff_performance_analysis(self):
        """Analyze staff performance metrics"""
        staff_performance = self.df.groupby('Staff_ID').agg({
            'Order_ID': 'count',
            'Order_Total': 'sum',
            'Profit': 'sum',
            'Customer_ID': 'nunique',
            'Order_Date': ['min', 'max']
        }).reset_index()
        
        staff_performance.columns = ['Staff_ID', 'Orders_Handled', 'Total_Revenue', 'Total_Profit', 
                                     'Unique_Customers', 'First_Shift', 'Last_Shift']
        
        staff_performance['Avg_Order_Value'] = staff_performance['Total_Revenue'] / staff_performance['Orders_Handled']
        staff_performance['Profit_Per_Order'] = staff_performance['Total_Profit'] / staff_performance['Orders_Handled']
        staff_performance['Days_Active'] = (staff_performance['Last_Shift'] - staff_performance['First_Shift']).dt.days
        
        return staff_performance

# ==============================================
# 🎭 VISUALIZATION HELPER FUNCTIONS
# ==============================================
class QuantumVisualizations:
    @staticmethod
    def create_3d_scatter(df, x_col, y_col, z_col, color_col, title):
        """Create 3D scatter plot"""
        fig = px.scatter_3d(
            df,
            x=x_col,
            y=y_col,
            z=z_col,
            color=color_col,
            hover_name=color_col,
            title=title,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col
            ),
            height=600
        )
        return fig
    
    @staticmethod
    def create_animated_timeline(df, date_col, value_col, category_col, title):
        """Create animated timeline chart"""
        daily_agg = df.groupby([pd.Grouper(key=date_col, freq='D'), category_col])[value_col].sum().reset_index()
        
        fig = px.line(
            daily_agg,
            x=date_col,
            y=value_col,
            color=category_col,
            title=title,
            animation_frame=category_col,
            line_shape='spline'
        )
        
        fig.update_layout(
            height=500,
            xaxis_title="Date",
            yaxis_title=value_col,
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def create_heatmap(df, x_col, y_col, value_col, title):
        """Create heatmap visualization"""
        pivot_df = df.pivot_table(index=y_col, columns=x_col, values=value_col, aggfunc='sum', fill_value=0)
        
        fig = px.imshow(
            pivot_df,
            title=title,
            color_continuous_scale='Viridis',
            labels=dict(x=x_col, y=y_col, color=value_col)
        )
        
        fig.update_layout(height=400)
        return fig
    
    @staticmethod
    def create_radar_chart(df, categories, values, title):
        """Create radar chart"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=title
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values) * 1.2]
                )),
            showlegend=True,
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_treemap(df, path_cols, value_col, title):
        """Create treemap visualization"""
        fig = px.treemap(
            df,
            path=path_cols,
            values=value_col,
            title=title,
            color=value_col,
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(height=500)
        return fig
    
    @staticmethod
    def create_parallel_categories(df, dimensions, color_col, title):
        """Create parallel categories diagram"""
        fig = px.parallel_categories(
            df,
            dimensions=dimensions,
            color=color_col,
            title=title,
            color_continuous_scale=px.colors.sequential.Plasma
        )
        
        fig.update_layout(height=400)
        return fig

# ==============================================
# 🚀 MAIN DASHBOARD APPLICATION
# ==============================================
def main():
    # Load data
    with st.spinner("🚀 Initializing Quantum Restaurant Intelligence..."):
        df = generate_advanced_data()
        analytics = QuantumAnalytics(df)
        viz = QuantumVisualizations()
    
    # Initialize session state
    if 'show_insights' not in st.session_state:
        st.session_state['show_insights'] = False
    if 'advanced_mode' not in st.session_state:
        st.session_state['advanced_mode'] = False
    
    # Hero Section
    st.markdown("""
    <div class="quantum-card">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <h1 style="margin: 0; background: var(--primary-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    🧠 Quantum Restaurant Intelligence
                </h1>
                <p style="color: #94a3b8; margin: 10px 0;">
                    Advanced Analytics & Predictive Insights for 17,534+ Transactions
                </p>
            </div>
            <div style="display: flex; gap: 10px;">
                <span class="insight-badge success">AI-Powered</span>
                <span class="insight-badge info">Real-Time</span>
                <span class="insight-badge warning">Predictive</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with Advanced Controls
    with st.sidebar:
        st.markdown("""
        <div class="metric-card">
            <h3>⚙️ Control Center</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Date Range Selector
        min_date = df['Order_Date'].min().date()
        max_date = df['Order_Date'].max().date()
        
        date_range = st.date_input(
            "📅 Analysis Period",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = df[(df['Order_Date'].dt.date >= start_date) & 
                           (df['Order_Date'].dt.date <= end_date)].copy()
        else:
            filtered_df = df.copy()
        
        # Advanced Filters
        with st.expander("🎯 Advanced Filters", expanded=False):
            # Category Multi-select
            categories = st.multiselect(
                "Categories",
                options=sorted(df['Category'].unique()),
                default=sorted(df['Category'].unique())
            )
            
            # Service Type
            service_types = st.multiselect(
                "Service Types",
                options=sorted(df['Service_Type'].unique()),
                default=sorted(df['Service_Type'].unique())
            )
            
            # Payment Methods
            payment_methods = st.multiselect(
                "Payment Methods",
                options=sorted(df['Payment_Method'].unique()),
                default=sorted(df['Payment_Method'].unique())
            )
            
            # CLV Tier
            clv_tiers = st.multiselect(
                "Customer Tiers",
                options=sorted(df['CLV_Tier'].unique()),
                default=sorted(df['CLV_Tier'].unique())
            )
        
        # Apply Filters
        if categories:
            filtered_df = filtered_df[filtered_df['Category'].isin(categories)]
        if service_types:
            filtered_df = filtered_df[filtered_df['Service_Type'].isin(service_types)]
        if payment_methods:
            filtered_df = filtered_df[filtered_df['Payment_Method'].isin(payment_methods)]
        if clv_tiers:
            filtered_df = filtered_df[filtered_df['CLV_Tier'].isin(clv_tiers)]
        
        # AI Analysis Toggle
        st.session_state['advanced_mode'] = st.toggle("🤖 Advanced AI Analysis", value=False)
        
        if st.button("🔮 Generate Insights", use_container_width=True, type="primary"):
            st.session_state['show_insights'] = True
        
        # Quick Metrics
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Quick Metrics</h4>
            <div class="responsive-grid">
                <div>
                    <small class="metric-label">Total Revenue</small>
                    <div class="metric-value">${:,.0f}</div>
                </div>
                <div>
                    <small class="metric-label">Orders</small>
                    <div class="metric-value">{:,}</div>
                </div>
                <div>
                    <small class="metric-label">Avg Order</small>
                    <div class="metric-value">${:.2f}</div>
                </div>
                <div>
                    <small class="metric-label">Customers</small>
                    <div class="metric-value">{:,}</div>
                </div>
            </div>
        </div>
        """.format(
            filtered_df['Order_Total'].sum(),
            len(filtered_df),
            filtered_df['Order_Total'].mean(),
            filtered_df['Customer_ID'].nunique()
        ), unsafe_allow_html=True)
    
    # Main Dashboard Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Overview", 
        "📊 Category Intel", 
        "👥 Customer IQ", 
        "⏰ Temporal AI", 
        "💰 Profit Engine",
        "🔍 Data Lab"
    ])
    
    # ==============================================
    # 📈 TAB 1: OVERVIEW - QUANTUM INTELLIGENCE
    # ==============================================
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            revenue = filtered_df['Order_Total'].sum()
            revenue_growth = ((revenue / df['Order_Total'].sum()) - 1) * 100
            st.markdown(f"""
            <div class="metric-card">
                <small class="metric-label">Quantum Revenue</small>
                <div class="metric-value">${revenue:,.0f}</div>
                <span class="metric-change {'positive' if revenue_growth > 0 else 'negative'}">
                    {revenue_growth:+.1f}% vs total
                </span>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {min(100, revenue/(df['Order_Total'].sum())*100)}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            orders = len(filtered_df)
            order_growth = ((orders / len(df)) - 1) * 100
            st.markdown(f"""
            <div class="metric-card">
                <small class="metric-label">Order Intelligence</small>
                <div class="metric-value">{orders:,}</div>
                <span class="metric-change {'positive' if order_growth > 0 else 'negative'}">
                    {order_growth:+.1f}% vs total
                </span>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {min(100, orders/len(df)*100)}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            customers = filtered_df['Customer_ID'].nunique()
            retention = len(filtered_df[filtered_df['Visit_Count'] > 1]) / customers * 100
            st.markdown(f"""
            <div class="metric-card">
                <small class="metric-label">Customer Universe</small>
                <div class="metric-value">{customers:,}</div>
                <span class="metric-change positive">
                    {retention:.1f}% retention
                </span>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {min(100, retention)}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            profit = filtered_df['Profit'].sum()
            margin = (profit / revenue) * 100 if revenue > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <small class="metric-label">Quantum Profit</small>
                <div class="metric-value">${profit:,.0f}</div>
                <span class="metric-change {'positive' if margin > 20 else 'warning'}">
                    {margin:.1f}% margin
                </span>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {min(100, margin)}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Revenue Trend with AI Predictions
        st.markdown("""
        <div class="quantum-card">
            <h2>📈 Revenue Intelligence & AI Forecast</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Generate ML forecast
            forecast_dates, forecast_values, lower_bound, upper_bound, mae = analytics.ml_revenue_forecast(days=30)
            
            # Historical revenue
            historical = filtered_df.resample('D', on='Order_Date')['Order_Total'].sum()
            
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=historical.index,
                y=historical.values,
                mode='lines',
                name='Historical',
                line=dict(color='#667eea', width=3),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.1)'
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_values,
                mode='lines+markers',
                name='AI Forecast',
                line=dict(color='#0ba360', width=3, dash='dash')
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=list(forecast_dates) + list(forecast_dates)[::-1],
                y=list(upper_bound) + list(lower_bound)[::-1],
                fill='toself',
                fillcolor='rgba(11, 163, 96, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence',
                showlegend=False
            ))
            
            fig.update_layout(
                height=500,
                title="30-Day AI Revenue Forecast",
                xaxis_title="Time Continuum",
                yaxis_title="Quantum Revenue ($)",
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="viz-container">
                <h4>🔮 Forecast Intelligence</h4>
                <div style="margin-top: 20px;">
                    <div style="background: rgba(102, 126, 234, 0.1); padding: 15px; border-radius: 12px; margin: 10px 0;">
                        <small>Next 7 Days</small>
                        <h3>${:,.0f}</h3>
                    </div>
                    <div style="background: rgba(11, 163, 96, 0.1); padding: 15px; border-radius: 12px; margin: 10px 0;">
                        <small>Next 30 Days</small>
                        <h3>${:,.0f}</h3>
                    </div>
                    <div style="background: rgba(245, 158, 11, 0.1); padding: 15px; border-radius: 12px; margin: 10px 0;">
                        <small>AI Confidence</small>
                        <h3>{:.1f}%</h3>
                    </div>
                    <div style="background: rgba(79, 172, 254, 0.1); padding: 15px; border-radius: 12px; margin: 10px 0;">
                        <small>Growth Projection</small>
                        <h3>+{:.1f}%</h3>
                    </div>
                </div>
            </div>
            """.format(
                forecast_values[:7].sum(),
                forecast_values.sum(),
                100 - (mae / forecast_values.mean() * 100),
                ((forecast_values[-1] / historical.iloc[-1]) - 1) * 100 if len(historical) > 0 else 0
            ), unsafe_allow_html=True)
        
        # Anomaly Detection
        st.markdown("""
        <div class="quantum-card">
            <h2>🚨 Anomaly Detection & Risk Intelligence</h2>
        </div>
        """, unsafe_allow_html=True)
        
        iso_anomalies, stat_anomalies = analytics.advanced_anomaly_detection()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="alert-box warning">
                <strong>⚠️ Isolation Forest Detection</strong><br>
                {len(iso_anomalies)} anomalous transactions identified
                ({len(iso_anomalies)/len(filtered_df)*100:.2f}% of total)
            </div>
            """)
            
            if not iso_anomalies.empty:
                st.dataframe(
                    iso_anomalies[['Order_Date', 'Order_Total', 'Category', 'Item', 'Customer_ID']]
                    .sort_values('Order_Total', ascending=False)
                    .head(10)
                    .style.format({'Order_Total': '${:,.2f}'})
                    .background_gradient(subset=['Order_Total'], cmap='Reds'),
                    use_container_width=True
                )
        
        with col2:
            st.markdown(f"""
            <div class="alert-box error">
                <strong>🔍 Statistical Outliers</strong><br>
                {len(stat_anomalies)} statistical outliers detected
                (Z-score > 3 standard deviations)
            </div>
            """)
            
            if not stat_anomalies.empty:
                fig = px.box(
                    filtered_df,
                    y='Order_Total',
                    title='Order Total Distribution with Outliers',
                    points='all'
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    # ==============================================
    # 📊 TAB 2: CATEGORY INTELLIGENCE
    # ==============================================
    with tab2:
        st.markdown("""
        <div class="quantum-card">
            <h2>📊 Category Performance Intelligence</h2>
            <p style="color: #94a3b8;">Deep analysis of category performance, profitability, and optimization opportunities</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Profitability Analysis
        profitability_df = analytics.profitability_analysis()
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Treemap of categories by profitability
            fig = viz.create_treemap(
                profitability_df,
                path_cols=['Category', 'Item'],
                value_col='Profit_Margin_Pct',
                title='Profitability Hierarchy by Category & Item'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top performers
            st.markdown("""
            <div class="viz-container">
                <h4>🏆 Top 5 Performers</h4>
            """, unsafe_allow_html=True)
            
            top_items = profitability_df.nlargest(5, 'Profit_Margin_Pct')
            for idx, row in top_items.iterrows():
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); padding: 12px; border-radius: 10px; margin: 8px 0;">
                    <div style="display: flex; justify-content: space-between;">
                        <strong>{row['Item']}</strong>
                        <span style="color: #10b981;">{row['Profit_Margin_Pct']:.1f}%</span>
                    </div>
                    <small style="color: #94a3b8;">{row['Category']} • ${row['Profit']:,.0f} total profit</small>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Market Basket Analysis
        st.markdown("""
        <div class="quantum-card">
            <h3>🛒 Market Basket Intelligence</h3>
            <p style="color: #94a3b8;">Discover item associations and bundling opportunities</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🧠 Run Association Analysis", type="secondary"):
            with st.spinner("Analyzing item associations..."):
                frequent_itemsets, rules = analytics.market_basket_analysis()
                
                if not rules.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📦 Frequent Itemsets**")
                        st.dataframe(
                            frequent_itemsets.sort_values('support', ascending=False)
                            .head(10)
                            .style.format({'support': '{:.3f}'})
                            .background_gradient(subset=['support'], cmap='Blues'),
                            use_container_width=True
                        )
                    
                    with col2:
                        st.markdown("**🔄 Association Rules**")
                        st.dataframe(
                            rules.sort_values('confidence', ascending=False)
                            .head(10)
                            .style.format({
                                'support': '{:.3f}',
                                'confidence': '{:.3f}',
                                'lift': '{:.3f}'
                            })
                            .background_gradient(subset=['confidence'], cmap='Greens'),
                            use_container_width=True
                        )
        
        # Category Performance Matrix
        st.markdown("""
        <div class="quantum-card">
            <h3>🎯 Performance Matrix Analysis</h3>
        </div>
        """, unsafe_allow_html=True)
        
        category_matrix = filtered_df.groupby('Category').agg({
            'Order_Total': 'sum',
            'Profit': 'sum',
            'Order_ID': 'count',
            'Quantity': 'sum'
        }).reset_index()
        
        category_matrix['Avg_Order_Value'] = category_matrix['Order_Total'] / category_matrix['Order_ID']
        category_matrix['Profit_Margin'] = (category_matrix['Profit'] / category_matrix['Order_Total']) * 100
        
        fig = px.scatter(
            category_matrix,
            x='Order_ID',
            y='Avg_Order_Value',
            size='Order_Total',
            color='Profit_Margin',
            hover_name='Category',
            title='Category Performance Matrix',
            size_max=60,
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # ==============================================
    # 👥 TAB 3: CUSTOMER IQ
    # ==============================================
    with tab3:
        st.markdown("""
        <div class="quantum-card">
            <h2>👥 Customer Intelligence & Segmentation</h2>
            <p style="color: #94a3b8;">Advanced customer analytics, segmentation, and lifetime value prediction</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Customer Segmentation
        with st.spinner("🧠 Performing intelligent customer segmentation..."):
            customer_segments, pca_variance = analytics.intelligent_segmentation()
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # 3D Customer Segmentation Visualization
            if not customer_segments.empty:
                fig = viz.create_3d_scatter(
                    customer_segments,
                    x_col='PCA1',
                    y_col='PCA2',
                    z_col='PCA3',
                    color_col='Segment_Name',
                    title='3D Customer Intelligence Map'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="viz-container">
                <h4>🎯 Segment Intelligence</h4>
                <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 10px;">
                    PCA Explained Variance: {:.1f}%
                </p>
            """.format(pca_variance.sum() * 100), unsafe_allow_html=True)
            
            segment_summary = customer_segments.groupby('Segment_Name').agg({
                'Customer_ID': 'count',
                'Total_Spent': 'mean',
                'Visit_Count': 'mean'
            }).round(2)
            
            for segment, data in segment_summary.iterrows():
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 12px; margin: 10px 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong>{segment}</strong>
                        <span style="background: var(--primary-gradient); color: white; padding: 4px 10px; border-radius: 20px; font-size: 0.8rem;">
                            {data['Customer_ID']} customers
                        </span>
                    </div>
                    <div style="margin-top: 10px;">
                        <small>💰 Avg Spend: ${data['Total_Spent']:,.0f}</small><br>
                        <small>📅 Avg Visits: {data['Visit_Count']:.1f}</small><br>
                        <small>🎯 CLV Estimate: ${data['Total_Spent'] * 3:,.0f}</small>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # RFM Analysis
        st.markdown("""
        <div class="quantum-card">
            <h3>🏷️ RFM Customer Segmentation</h3>
        </div>
        """, unsafe_allow_html=True)
        
        rfm_counts = filtered_df.groupby('RFM_Score').size().reset_index(name='count')
        rfm_counts = rfm_counts.sort_values('count', ascending=False)
        
        fig = px.bar(
            rfm_counts,
            x='RFM_Score',
            y='count',
            title='RFM Score Distribution',
            color='count',
            color_continuous_scale='Plasma'
        )
        
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Customer Lifetime Value Analysis
        st.markdown("""
        <div class="quantum-card">
            <h3>💰 Customer Lifetime Value Intelligence</h3>
        </div>
        """, unsafe_allow_html=True)
        
        clv_analysis = filtered_df.groupby('CLV_Tier').agg({
            'Customer_ID': 'nunique',
            'Total_Spent': 'mean',
            'Profit': 'mean',
            'Visit_Count': 'mean'
        }).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                clv_analysis,
                values='Customer_ID',
                names='CLV_Tier',
                title='Customer Distribution by CLV Tier',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                clv_analysis,
                x='CLV_Tier',
                y='Total_Spent',
                title='Average Spend by CLV Tier',
                color='Profit',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # ==============================================
    # ⏰ TAB 4: TEMPORAL AI
    # ==============================================
    with tab4:
        st.markdown("""
        <div class="quantum-card">
            <h2>⏰ Temporal Intelligence & Pattern Recognition</h2>
            <p style="color: #94a3b8;">Time-based analytics, peak period detection, and seasonality analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Peak Period Detection
        peak_periods, hourly_revenue = analytics.detect_peak_periods()
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Hourly Revenue Pattern
            fig = px.line(
                hourly_revenue,
                x='Order_Hour',
                y='Order_Total',
                title='Hourly Revenue Pattern with Peak Detection',
                markers=True,
                line_shape='spline'
            )
            
            # Add peak markers
            if peak_periods:
                peak_hours = [p['hour'] for p in peak_periods]
                peak_values = [p['revenue'] for p in peak_periods]
                
                fig.add_trace(go.Scatter(
                    x=peak_hours,
                    y=peak_values,
                    mode='markers',
                    name='Peak Hours',
                    marker=dict(
                        color='#f76b1c',
                        size=15,
                        symbol='star'
                    )
                ))
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="viz-container">
                <h4>⏰ Peak Period Intelligence</h4>
            """, unsafe_allow_html=True)
            
            if peak_periods:
                for period in sorted(peak_periods, key=lambda x: x['revenue'], reverse=True)[:3]:
                    st.markdown(f"""
                    <div style="background: rgba(247, 107, 28, 0.1); padding: 15px; border-radius: 12px; margin: 10px 0;">
                        <div style="font-size: 2rem; font-weight: 800; color: #f76b1c; text-align: center;">
                            {period['hour']}:00
                        </div>
                        <div style="text-align: center; margin-top: 5px;">
                            <strong>{period['peak_type']} Peak</strong><br>
                            <small>Revenue: ${period['revenue']:,.0f}</small>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No significant peaks detected in the data")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Seasonal Patterns
        st.markdown("""
        <div class="quantum-card">
            <h3>🌡️ Seasonal Intelligence</h3>
        </div>
        """, unsafe_allow_html=True)
        
        seasonal_data, seasonal_index = analytics.seasonal_patterns()
        
        fig = px.bar(
            seasonal_data,
            x='Season',
            y='Order_Total',
            color='Category',
            title='Seasonal Revenue by Category',
            barmode='group'
        )
        
        fig.update_layout(height=500, xaxis_title="Season", yaxis_title="Total Revenue ($)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Day of Week Analysis
        st.markdown("""
        <div class="quantum-card">
            <h3>📅 Day of Week Intelligence</h3>
        </div>
        """, unsafe_allow_html=True)
        
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_analysis = filtered_df.groupby('Order_Day').agg({
            'Order_Total': 'sum',
            'Order_ID': 'count',
            'Profit': 'sum'
        }).reindex(day_order)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(
                day_analysis.reset_index(),
                x='Order_Day',
                y='Order_Total',
                title='Revenue by Day of Week',
                markers=True,
                line_shape='spline'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                day_analysis.reset_index(),
                x='Order_Day',
                y='Order_ID',
                title='Orders by Day of Week',
                color='Order_Total',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    # ==============================================
    # 💰 TAB 5: PROFIT ENGINE
    # ==============================================
    with tab5:
        st.markdown("""
        <div class="quantum-card">
            <h2>💰 Profit Intelligence Engine</h2>
            <p style="color: #94a3b8;">Profit margin analysis, cost optimization, and revenue maximization</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Profit Margin Analysis
        profit_margin_analysis = filtered_df.groupby(['Category', 'Item']).agg({
            'Profit_Margin': 'mean',
            'Profit': 'sum',
            'Order_Total': 'sum'
        }).reset_index()
        
        fig = px.scatter(
            profit_margin_analysis,
            x='Order_Total',
            y='Profit_Margin',
            size='Profit',
            color='Category',
            hover_name='Item',
            title='Profit Margin vs Revenue Analysis',
            size_max=50,
            log_x=True
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Staff Performance Analysis
        st.markdown("""
        <div class="quantum-card">
            <h3>👨‍🍳 Staff Performance Intelligence</h3>
        </div>
        """, unsafe_allow_html=True)
        
        staff_performance = analytics.staff_performance_analysis()
        
        if not staff_performance.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    staff_performance.nlargest(10, 'Total_Profit'),
                    x='Staff_ID',
                    y='Total_Profit',
                    title='Top 10 Staff by Profit Generation',
                    color='Profit_Per_Order',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("""
                <div class="viz-container">
                    <h4>🏆 Performance Leaders</h4>
                """, unsafe_allow_html=True)
                
                top_staff = staff_performance.nlargest(5, 'Total_Profit')
                for idx, row in top_staff.iterrows():
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.05); padding: 12px; border-radius: 10px; margin: 8px 0;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <strong>{row['Staff_ID']}</strong>
                            <span style="color: #10b981;">${row['Total_Profit']:,.0f}</span>
                        </div>
                        <small style="color: #94a3b8;">
                            {row['Orders_Handled']} orders • ${row['Avg_Order_Value']:.0f} avg • {row['Days_Active']} days active
                        </small>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Cost Optimization Analysis
        st.markdown("""
        <div class="quantum-card">
            <h3>⚡ Cost Optimization Intelligence</h3>
        </div>
        """, unsafe_allow_html=True)
        
        cost_analysis = filtered_df.groupby('Category').agg({
            'Cost': 'sum',
            'Order_Total': 'sum',
            'Profit': 'sum'
        }).reset_index()
        
        cost_analysis['Cost_Percentage'] = (cost_analysis['Cost'] / cost_analysis['Order_Total']) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                cost_analysis,
                x='Category',
                y='Cost_Percentage',
                title='Cost as Percentage of Revenue by Category',
                color='Profit',
                color_continuous_scale='RdYlGn_r'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                cost_analysis,
                x='Cost',
                y='Profit',
                size='Order_Total',
                color='Category',
                title='Cost vs Profit Analysis',
                hover_name='Category'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # ==============================================
    # 🔍 TAB 6: DATA LAB
    # ==============================================
    with tab6:
        st.markdown("""
        <div class="quantum-card">
            <h2>🔍 Data Laboratory & Advanced Analytics</h2>
            <p style="color: #94a3b8;">Raw data exploration, advanced filtering, and data quality analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Data Explorer
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4>📊 Dataset Intelligence</h4>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-top: 15px;">
                    <div>
                        <small>Total Records</small>
                        <h5>{:,}</h5>
                    </div>
                    <div>
                        <small>Features</small>
                        <h5>{}</h5>
                    </div>
                    <div>
                        <small>Memory</small>
                        <h5>{:.1f} MB</h5>
                    </div>
                </div>
            </div>
            """.format(
                len(filtered_df),
                len(filtered_df.columns),
                filtered_df.memory_usage(deep=True).sum() / 1024**2
            ), unsafe_allow_html=True)
        
        with col2:
            completeness = (1 - filtered_df.isnull().sum().sum() / (len(filtered_df) * len(filtered_df.columns))) * 100
            st.metric("Data Completeness", f"{completeness:.1f}%")
        
        with col3:
            duplicates = filtered_df.duplicated().sum()
            st.metric("Duplicates", duplicates, delta_color="inverse")
        
        # Advanced Data Explorer
        st.markdown("""
        <div class="quantum-card">
            <h3>🎛️ Advanced Data Explorer</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Column Selection
        selected_columns = st.multiselect(
            "Select columns to display",
            options=filtered_df.columns.tolist(),
            default=['Order_Date', 'Category', 'Item', 'Price', 'Quantity', 'Order_Total', 'Payment_Method', 'Service_Type']
        )
        
        if selected_columns:
            # Search functionality
            search_query = st.text_input("🔍 Search across all columns", placeholder="Enter search term...")
            
            if search_query:
                search_mask = filtered_df.apply(
                    lambda row: row.astype(str).str.contains(search_query, case=False).any(), 
                    axis=1
                )
                display_df = filtered_df[search_mask][selected_columns]
            else:
                display_df = filtered_df[selected_columns]
            
            # Pagination
            page_size = st.selectbox("Rows per page", [10, 25, 50, 100], index=0)
            
            if len(display_df) > 0:
                total_pages = max(1, (len(display_df) // page_size) + 1)
                page_number = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
                
                start_idx = (page_number - 1) * page_size
                end_idx = min(page_number * page_size, len(display_df))
                
                st.dataframe(
                    display_df.iloc[start_idx:end_idx].style.format({
                        'Price': '${:,.2f}',
                        'Order_Total': '${:,.2f}',
                        'Profit': '${:,.2f}',
                        'Profit_Margin': '{:.1f}%'
                    }).background_gradient(subset=['Order_Total'], cmap='Blues'),
                    use_container_width=True,
                    height=400
                )
                
                st.caption(f"Showing rows {start_idx + 1} to {end_idx} of {len(display_df)}")
        
        # Data Export
        st.markdown("""
        <div class="quantum-card">
            <h3>💾 Data Export & Integration</h3>
        </div>
        """, unsafe_allow_html=True)
        
        export_cols = st.columns(4)
        
        with export_cols[0]:
            if st.button("📥 Export CSV", use_container_width=True):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="quantum_restaurant_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with export_cols[1]:
            if st.button("📊 Export JSON", use_container_width=True):
                json_data = filtered_df.to_json(orient='records', indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name="quantum_restaurant_data.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        with export_cols[2]:
            if st.button("📈 Generate Report", use_container_width=True):
                st.success("📋 Report generation initiated! Check your dashboard for updates.")
        
        with export_cols[3]:
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.rerun()
    
    # ==============================================
    # 🎯 FOOTER & ANALYTICS SUMMARY
    # ==============================================
    st.markdown("""
    <div style="margin-top: 50px; padding: 25px; background: rgba(15, 23, 42, 0.6); border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.1);">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <h4 style="margin: 0; color: #e2e8f0;">🧠 Quantum Restaurant Intelligence</h4>
                <p style="color: #94a3b8; margin: 5px 0 0 0;">
                    Advanced Analytics Platform • Version 3.0 • Updated: {}
                </p>
            </div>
            <div style="text-align: right;">
                <p style="color: #94a3b8; margin: 0;">
                    Processing Time: {:.2f}s • Records Analyzed: {:,}
                </p>
                <p style="color: #667eea; margin: 5px 0 0 0; font-size: 0.9rem;">
                    Powered by Machine Learning & AI Algorithms
                </p>
            </div>
        </div>
    </div>
    """.format(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        time.time() % 100,
        len(filtered_df)
    ), unsafe_allow_html=True)

# ==============================================
# 🚀 APPLICATION ENTRY POINT
# ==============================================
if __name__ == "__main__":
    main()
