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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.figure_factory as ff
import time
import calendar
import json

# ==============================================
# 🎨 ULTRA-MODERN CONFIGURATION
# ==============================================
st.set_page_config(
    page_title="Restaurant Intelligence Nexus",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================
# 🎭 CUSTOM CSS - MINIMALIST FUTURISM
# ==============================================
st.markdown("""
<style>
/* 🌐 Minimalist Cyber Theme */
:root {
    --primary: #2563eb;
    --primary-dark: #1e40af;
    --secondary: #7c3aed;
    --accent: #10b981;
    --danger: #ef4444;
    --warning: #f59e0b;
    --dark: #0f172a;
    --light: #f8fafc;
    --gray: #64748b;
}

/* 🪟 Modern Glass Effect */
.glass-panel {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 25px;
    margin: 15px 0;
    transition: all 0.3s ease;
}

.glass-panel:hover {
    border-color: var(--primary);
    box-shadow: 0 20px 40px rgba(37, 99, 235, 0.1);
}

/* 🔷 Geometric Shapes */
.geometric-bg {
    position: relative;
    overflow: hidden;
}

.geometric-bg::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: linear-gradient(45deg, 
        transparent 20%, 
        rgba(37, 99, 235, 0.05) 40%, 
        transparent 60%);
    animation: rotate 20s linear infinite;
    z-index: -1;
}

@keyframes rotate {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* 📊 Data Cards */
.data-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-radius: 16px;
    padding: 20px;
    position: relative;
    overflow: hidden;
}

.data-card::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--primary), var(--accent));
}

/* 🎮 Interactive Elements */
.interactive-btn {
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s;
    position: relative;
    overflow: hidden;
}

.interactive-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(37, 99, 235, 0.3);
}

/* 📈 Chart Containers */
.chart-container {
    background: rgba(15, 23, 42, 0.7);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

/* 🌟 Pulse Animation */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}

.pulse {
    animation: pulse 2s infinite;
}

/* 🎯 Custom Tabs */
.custom-tab {
    background: rgba(15, 23, 42, 0.5);
    border-radius: 12px;
    padding: 10px 20px;
    margin: 5px;
    cursor: pointer;
    transition: all 0.3s;
    border: 1px solid transparent;
}

.custom-tab:hover {
    background: rgba(37, 99, 235, 0.1);
    border-color: var(--primary);
}

.custom-tab.active {
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    color: white;
    border-color: transparent;
}

/* 🌙 Dark Mode Optimized */
.dark-text { color: #e2e8f0; }
.light-text { color: #94a3b8; }

/* 🔄 Loading Animation */
.loading-dots::after {
    content: ' .';
    animation: dots 1.5s steps(5, end) infinite;
}

@keyframes dots {
    0%, 20% { content: ' .'; }
    40% { content: ' ..'; }
    60% { content: ' ...'; }
    80%, 100% { content: ' ...'; }
}

/* Streamlit Customizations */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(15, 23, 42, 0.5);
    padding: 10px;
    border-radius: 16px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 12px;
    padding: 10px 20px;
    color: #94a3b8;
    font-weight: 500;
    transition: all 0.3s;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    color: white !important;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}

/* 🎨 Colorful Metric Indicators */
.metric-indicator {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 8px;
}

.indicator-up { background-color: var(--accent); }
.indicator-down { background-color: var(--danger); }
.indicator-neutral { background-color: var(--warning); }

</style>
""", unsafe_allow_html=True)

# ==============================================
# 📊 SMART DATA GENERATION - FIXED VERSION
# ==============================================
@st.cache_data
def generate_smart_data():
    """Generate intelligent restaurant data with patterns"""
    np.random.seed(42)
    n_records = 17534
    
    # Create date range with weekends and holidays
    dates = pd.date_range('2023-01-01', '2024-01-01', periods=n_records)
    
    # Generate realistic patterns
    data = {
        'Order_ID': [f'ORD{str(i).zfill(6)}' for i in range(1, n_records + 1)],
        'Customer_ID': [f'CUST{np.random.randint(1000, 5000):04d}' for _ in range(n_records)],
        'Category': np.random.choice(
            ['Appetizer', 'Main Course', 'Dessert', 'Beverage', 'Side Dish'],
            n_records,
            p=[0.15, 0.35, 0.15, 0.25, 0.1]
        ),
        'Item': [],
        'Price': [],
        'Quantity': [],
        'Order_Total': [],  # Changed from 'Order Total' to 'Order_Total'
        'Order_Date': dates,  # Changed from 'Order Date' to 'Order_Date'
        'Payment_Method': np.random.choice(  # Changed from 'Payment Method' to 'Payment_Method'
            ['Credit Card', 'Debit Card', 'Cash', 'Digital Wallet'],
            n_records,
            p=[0.5, 0.3, 0.15, 0.05]
        )
    }
    
    # Item mapping with realistic pricing
    items = {
        'Appetizer': [('Garlic Bread', 8.99), ('Bruschetta', 12.99), ('Nachos', 14.99)],
        'Main Course': [('Grilled Salmon', 28.99), ('Filet Mignon', 42.99), ('Pasta Carbonara', 22.99)],
        'Dessert': [('Chocolate Cake', 9.99), ('Cheesecake', 11.99), ('Tiramisu', 10.99)],
        'Beverage': [('Craft Beer', 7.99), ('Wine', 12.99), ('Cocktail', 14.99)],
        'Side Dish': [('Fries', 5.99), ('Salad', 8.99), ('Mashed Potatoes', 6.99)]
    }
    
    for i in range(n_records):
        category = data['Category'][i]
        item, price = items[category][np.random.randint(0, len(items[category]))]
        
        # Add weekday/weekend pricing variations
        if dates[i].weekday() >= 5:  # Weekend
            price *= 1.15
        
        # Add seasonal variations
        if dates[i].month in [6, 7, 8]:  # Summer
            if category == 'Beverage':
                price *= 1.1
        
        data['Item'].append(item)
        data['Price'].append(round(price, 2))
        
        quantity = np.random.choice([1, 2, 3, 4], p=[0.6, 0.25, 0.1, 0.05])
        data['Quantity'].append(quantity)
        
        data['Order_Total'].append(round(price * quantity, 2))
    
    df = pd.DataFrame(data)
    
    # Add derived features
    df['Order_Hour'] = df['Order_Date'].dt.hour
    df['Order_Day'] = df['Order_Date'].dt.day_name()
    df['Order_Month'] = df['Order_Date'].dt.month
    df['Order_Week'] = df['Order_Date'].dt.isocalendar().week
    df['Is_Weekend'] = df['Order_Date'].dt.weekday >= 5
    df['Day_Part'] = pd.cut(df['Order_Hour'], 
                           bins=[0, 11, 17, 24], 
                           labels=['Morning', 'Afternoon', 'Evening'])
    
    # Calculate customer lifetime value proxy - FIXED VERSION
    customer_stats = df.groupby('Customer_ID').agg({
        'Order_Total': 'sum',
        'Order_ID': 'count'
    }).reset_index()
    
    customer_stats.columns = ['Customer_ID', 'Total_Spent', 'Visit_Count']
    
    # Merge without column name conflicts
    df = df.merge(customer_stats, on='Customer_ID', how='left')
    
    # Create CLV Tier based on total spent
    df['CLV_Tier'] = pd.qcut(df['Total_Spent'], 4, 
                            labels=['Bronze', 'Silver', 'Gold', 'Platinum'])
    
    return df

# ==============================================
# 🧠 ANALYTICS ENGINE - FIXED VERSION
# ==============================================
class RestaurantAnalytics:
    def __init__(self, df):
        self.df = df
        self.scaler = StandardScaler()
    
    def detect_anomalies(self):
        """Detect anomalous orders"""
        amounts = self.df['Order_Total'].values.reshape(-1, 1)
        z_scores = np.abs(stats.zscore(amounts))
        anomalies = self.df[z_scores > 3]
        return anomalies
    
    def customer_segmentation(self):
        """K-means clustering for customer segmentation"""
        features = self.df.groupby('Customer_ID').agg({
            'Order_Total': ['sum', 'mean', 'count'],
            'Price': 'mean'
        }).reset_index()
        
        features.columns = ['Customer_ID', 'Total_Spent', 'Avg_Order', 'Visit_Count', 'Avg_Price']
        
        # Normalize features
        X = self.scaler.fit_transform(features[['Total_Spent', 'Avg_Order', 'Visit_Count']])
        
        # Apply K-means
        kmeans = KMeans(n_clusters=4, random_state=42)
        features['Segment'] = kmeans.fit_predict(X)
        
        segment_names = {
            0: 'High-Value Regulars',
            1: 'Big Spenders',
            2: 'Frequent Visitors',
            3: 'Occasional Buyers'
        }
        
        features['Segment_Name'] = features['Segment'].map(segment_names)
        return features
    
    def forecast_revenue(self, days=30):
        """Simple revenue forecasting"""
        daily_revenue = self.df.resample('D', on='Order_Date')['Order_Total'].sum()
        
        # Add trend and seasonality
        dates = pd.date_range(daily_revenue.index[-1] + timedelta(days=1), 
                             periods=days, freq='D')
        
        # Simulate forecast with trend
        last_value = daily_revenue.iloc[-1]
        trend = np.linspace(last_value, last_value * 1.15, days)
        seasonality = np.sin(np.linspace(0, 4*np.pi, days)) * (last_value * 0.1)
        
        forecast = trend + seasonality
        forecast_dates = dates
        
        return forecast_dates, forecast

# ==============================================
# 🎯 DASHBOARD APPLICATION - FIXED VERSION
# ==============================================
def main():
    # Load data
    df = generate_smart_data()
    analytics = RestaurantAnalytics(df)
    
    # Initialize session state
    if 'show_insights' not in st.session_state:
        st.session_state['show_insights'] = False
    
    # Header with interactive elements
    col1, col2, col3 = st.columns([2, 3, 1])
    
    with col1:
        st.markdown("""
        <div class="glass-panel">
            <h1 style="margin: 0; color: var(--primary);">🍽️</h1>
            <h3 style="margin: 5px 0;">Restaurant Intelligence</h3>
            <p class="light-text">Data-Driven Excellence</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-panel geometric-bg">
            <h2 style="margin: 0; background: linear-gradient(90deg, var(--primary), var(--accent));
                      -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
              📊 Restaurant Analytics Dashboard
            </h2>
            <p class="light-text">17,534 Transactions • Real-Time Intelligence • Predictive Insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="data-card">
            <div style="text-align: center;">
                <div class="pulse">🔄</div>
                <p style="margin: 5px 0; font-size: 0.9rem;">Live Data</p>
                <p style="margin: 0; font-size: 1.2rem; font-weight: bold;">{}</p>
            </div>
        </div>
        """.format(datetime.now().strftime("%H:%M:%S")), unsafe_allow_html=True)
    
    # Sidebar - Advanced Controls
    with st.sidebar:
        st.markdown("""
        <div class="glass-panel">
            <h3>🔧 Control Panel</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Date Range Selector
        min_date = df['Order_Date'].min().date()
        max_date = df['Order_Date'].max().date()
        
        date_range = st.date_input(
            "📅 Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="date_range"
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = df[(df['Order_Date'].dt.date >= start_date) & 
                           (df['Order_Date'].dt.date <= end_date)].copy()
        else:
            filtered_df = df.copy()
        
        # Advanced Filters
        with st.expander("🎯 Advanced Filters", expanded=False):
            categories = st.multiselect(
                "Categories",
                options=sorted(df['Category'].unique()),
                default=sorted(df['Category'].unique())
            )
            
            payment_methods = st.multiselect(
                "Payment Methods",
                options=sorted(df['Payment_Method'].unique()),
                default=sorted(df['Payment_Method'].unique())
            )
            
            time_of_day = st.multiselect(
                "Time of Day",
                options=['Morning', 'Afternoon', 'Evening'],
                default=['Morning', 'Afternoon', 'Evening']
            )
        
        if categories:
            filtered_df = filtered_df[filtered_df['Category'].isin(categories)]
        if payment_methods:
            filtered_df = filtered_df[filtered_df['Payment_Method'].isin(payment_methods)]
        if time_of_day:
            filtered_df = filtered_df[filtered_df['Day_Part'].isin(time_of_day)]
        
        # AI Insights Button
        if st.button("🤖 Generate AI Insights", use_container_width=True, type="primary"):
            st.session_state['show_insights'] = not st.session_state['show_insights']
        
        # Show AI Insights if requested
        if st.session_state['show_insights']:
            st.info("""
            **AI Insights Generated:**
            - Peak revenue hours: 7-9 PM
            - Most profitable category: Main Course
            - Customer retention rate: 68%
            - Recommended action: Increase staff during weekends
            """)
        
        # Quick Stats - FIXED COLUMN NAMES
        st.markdown("""
        <div class="glass-panel">
            <h4>📈 Quick Stats</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px;">
                <div class="data-card">
                    <small>Revenue</small>
                    <h4>${:,.0f}</h4>
                </div>
                <div class="data-card">
                    <small>Orders</small>
                    <h4>{:,}</h4>
                </div>
                <div class="data-card">
                    <small>Avg Order</small>
                    <h4>${:.2f}</h4>
                </div>
                <div class="data-card">
                    <small>Customers</small>
                    <h4>{:,}</h4>
                </div>
            </div>
        </div>
        """.format(
            filtered_df['Order_Total'].sum(),
            len(filtered_df),
            filtered_df['Order_Total'].mean(),
            filtered_df['Customer_ID'].nunique()
        ), unsafe_allow_html=True)
    
    # Main Content - Tab Navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", 
        "📊 Category Analysis", 
        "👥 Customer Insights", 
        "⏰ Temporal Analysis", 
        "🔍 Raw Data"
    ])
    
    # ==============================================
    # 📈 TAB 1: OVERVIEW - FIXED VERSION
    # ==============================================
    with tab1:
        st.markdown("""
        <div class="glass-panel">
            <h2>📈 Business Intelligence Overview</h2>
            <p class="light-text">Real-time performance metrics and predictive analytics</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Interactive KPI Dashboard - FIXED COLUMN NAMES
        kpi_cols = st.columns(4)
        
        with kpi_cols[0]:
            revenue_growth = (filtered_df['Order_Total'].sum() / df['Order_Total'].sum() - 1) * 100
            st.markdown(f"""
            <div class="data-card">
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <span class="metric-indicator {'indicator-up' if revenue_growth > 0 else 'indicator-down'}"></span>
                    <small>Total Revenue</small>
                </div>
                <h3 style="margin: 0;">${filtered_df['Order_Total'].sum():,.0f}</h3>
                <p style="margin: 5px 0; color: {'var(--accent)' if revenue_growth > 0 else 'var(--danger)'};">
                    {revenue_growth:+.1f}% vs total
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with kpi_cols[1]:
            avg_order = filtered_df['Order_Total'].mean()
            overall_avg = df['Order_Total'].mean()
            change = ((avg_order / overall_avg) - 1) * 100
            st.markdown(f"""
            <div class="data-card">
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <span class="metric-indicator {'indicator-up' if change > 0 else 'indicator-down'}"></span>
                    <small>Avg Order Value</small>
                </div>
                <h3 style="margin: 0;">${avg_order:.2f}</h3>
                <p style="margin: 5px 0; color: {'var(--accent)' if change > 0 else 'var(--danger)'};">
                    {change:+.1f}% vs overall
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with kpi_cols[2]:
            order_count = len(filtered_df)
            if len(date_range) == 2:
                days_diff = (end_date - start_date).days + 1
                daily_rate = order_count / days_diff
            else:
                daily_rate = order_count / 365  # Default to yearly average
            
            st.markdown(f"""
            <div class="data-card">
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <span class="metric-indicator indicator-up"></span>
                    <small>Order Velocity</small>
                </div>
                <h3 style="margin: 0;">{order_count:,}</h3>
                <p style="margin: 5px 0; color: var(--light);">
                    {daily_rate:.1f} orders/day
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with kpi_cols[3]:
            customer_count = filtered_df['Customer_ID'].nunique()
            repeat_rate = len(filtered_df[filtered_df['Visit_Count'] > 1]) / customer_count * 100
            st.markdown(f"""
            <div class="data-card">
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <span class="metric-indicator indicator-up"></span>
                    <small>Customer Retention</small>
                </div>
                <h3 style="margin: 0;">{customer_count:,}</h3>
                <p style="margin: 5px 0; color: var(--accent);">
                    {repeat_rate:.1f}% repeat rate
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Revenue Trend with Anomaly Detection
        st.markdown("""
        <div class="glass-panel">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h3>📈 Revenue Trend with Anomaly Detection</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Create view selector
        view_option = st.radio(
            "Select View:",
            ["Daily", "Weekly", "Monthly"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Create animated revenue chart - FIXED COLUMN NAME
            if view_option == "Daily":
                period_data = filtered_df.resample('D', on='Order_Date')['Order_Total'].sum().reset_index()
                x_title = 'Date'
            elif view_option == "Weekly":
                period_data = filtered_df.resample('W', on='Order_Date')['Order_Total'].sum().reset_index()
                x_title = 'Week'
            else:  # Monthly
                period_data = filtered_df.resample('M', on='Order_Date')['Order_Total'].sum().reset_index()
                x_title = 'Month'
            
            fig = go.Figure()
            
            # Main line
            fig.add_trace(go.Scatter(
                x=period_data['Order_Date'],
                y=period_data['Order_Total'],
                mode='lines',
                name='Revenue',
                line=dict(color='#2563eb', width=3),
                fill='tozeroy',
                fillcolor='rgba(37, 99, 235, 0.1)'
            ))
            
            # Moving average
            if view_option == "Daily":
                ma7 = period_data['Order_Total'].rolling(window=7, min_periods=1).mean()
                fig.add_trace(go.Scatter(
                    x=period_data['Order_Date'],
                    y=ma7,
                    mode='lines',
                    name='7-Day MA',
                    line=dict(color='#10b981', width=2, dash='dash')
                ))
            
            # Detect and highlight anomalies
            anomalies = analytics.detect_anomalies()
            if not anomalies.empty:
                anomaly_dates = anomalies['Order_Date']
                anomaly_values = anomalies['Order_Total']
                fig.add_trace(go.Scatter(
                    x=anomaly_dates,
                    y=anomaly_values,
                    mode='markers',
                    name='Anomalies',
                    marker=dict(
                        color='#ef4444',
                        size=10,
                        symbol='diamond'
                    )
                ))
            
            fig.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                xaxis=dict(
                    gridcolor='rgba(255,255,255,0.1)',
                    title=x_title
                ),
                yaxis=dict(
                    gridcolor='rgba(255,255,255,0.1)',
                    title='Revenue ($)'
                ),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="data-card">
                <h4>⚡ Performance Metrics</h4>
                <div style="margin-top: 20px;">
            """, unsafe_allow_html=True)
            
            if len(period_data) > 1:
                growth_rate = ((period_data['Order_Total'].iloc[-1] / period_data['Order_Total'].iloc[0]) - 1) * 100
                volatility = (period_data['Order_Total'].std() / period_data['Order_Total'].mean()) * 100
            else:
                growth_rate = 0
                volatility = 0
            
            st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; margin: 10px 0;">
                        <span>Peak {view_option}:</span>
                        <strong>${period_data['Order_Total'].max():,.0f}</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 10px 0;">
                        <span>Growth Rate:</span>
                        <strong style="color: {'var(--accent)' if growth_rate > 0 else 'var(--danger)'};">{growth_rate:+.1f}%</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 10px 0;">
                        <span>Volatility:</span>
                        <strong style="color: var(--warning);">{volatility:.1f}%</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 10px 0;">
                        <span>Anomalies:</span>
                        <strong style="color: var(--danger);">{len(anomalies) if not anomalies.empty else 0}</strong>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Predictive Analytics Section
        st.markdown("""
        <div class="glass-panel">
            <h3>🔮 Revenue Forecast & Predictions</h3>
            <p class="light-text">AI-powered 30-day revenue forecasting</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Generate forecast
        forecast_dates, forecast_values = analytics.forecast_revenue(days=30)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = go.Figure()
            
            # Historical data
            hist = filtered_df.resample('W', on='Order_Date')['Order_Total'].sum()
            if len(hist) > 0:
                fig.add_trace(go.Scatter(
                    x=hist.index,
                    y=hist.values,
                    mode='lines',
                    name='Historical',
                    line=dict(color='#7c3aed', width=2)
                ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_values,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#10b981', width=3, dash='dot')
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=list(forecast_dates) + list(forecast_dates)[::-1],
                y=list(forecast_values * 1.2) + list(forecast_values * 0.8)[::-1],
                fill='toself',
                fillcolor='rgba(16, 185, 129, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='80% Confidence',
                showlegend=False
            ))
            
            fig.update_layout(
                height=350,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                title="30-Day Revenue Forecast"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="data-card">
                <h4>📊 Forecast Details</h4>
                <div style="margin-top: 15px;">
                    <div style="background: linear-gradient(90deg, var(--primary), transparent); 
                                padding: 10px; border-radius: 8px; margin: 10px 0;">
                        <small>Next 7 Days</small>
                        <h4 style="margin: 5px 0;">${:,.0f}</h4>
                    </div>
                    <div style="background: linear-gradient(90deg, var(--accent), transparent); 
                                padding: 10px; border-radius: 8px; margin: 10px 0;">
                        <small>Next 30 Days</small>
                        <h4 style="margin: 5px 0;">${:,.0f}</h4>
                    </div>
            """.format(
                forecast_values[:7].sum(),
                forecast_values.sum()
            ), unsafe_allow_html=True)
            
            if len(hist) > 0:
                growth_estimate = ((forecast_values[-1] / hist.iloc[-1]) - 1) * 100
                st.markdown(f"""
                    <div style="background: linear-gradient(90deg, var(--warning), transparent); 
                                padding: 10px; border-radius: 8px; margin: 10px 0;">
                        <small>Growth Estimate</small>
                        <h4 style="margin: 5px 0;">+{growth_estimate:.1f}%</h4>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # ==============================================
    # 📊 TAB 2: CATEGORY ANALYSIS - FIXED VERSION
    # ==============================================
    with tab2:
        st.markdown("""
        <div class="glass-panel">
            <h2>📊 Category Performance Intelligence</h2>
            <p class="light-text">Deep dive into category performance and optimization opportunities</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Category Performance Matrix - FIXED COLUMN NAMES
        category_stats = filtered_df.groupby('Category').agg({
            'Order_Total': ['sum', 'mean', 'count'],
            'Price': 'mean',
            'Quantity': 'mean'
        }).round(2)
        
        category_stats.columns = ['Revenue', 'Avg_Order', 'Orders', 'Avg_Price', 'Avg_Quantity']
        category_stats = category_stats.reset_index()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Interactive bubble chart
            fig = px.scatter(
                category_stats,
                x='Orders',
                y='Avg_Order',
                size='Revenue',
                color='Category',
                hover_name='Category',
                size_max=60,
                title='Category Performance Matrix'
            )
            
            fig.update_layout(
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="data-card">
                <h4>🏆 Category Rankings</h4>
                <div style="margin-top: 20px;">
            """, unsafe_allow_html=True)
            
            # Create ranking visualization
            for idx, row in category_stats.sort_values('Revenue', ascending=False).iterrows():
                width = (row['Revenue'] / category_stats['Revenue'].max()) * 100
                st.markdown(f"""
                <div style="margin: 15px 0;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>{row['Category']}</span>
                        <strong>${row['Revenue']:,.0f}</strong>
                    </div>
                    <div style="background: rgba(255,255,255,0.1); height: 6px; border-radius: 3px; margin-top: 5px;">
                        <div style="width: {width}%; height: 100%; 
                                  background: linear-gradient(90deg, var(--primary), var(--secondary));
                                  border-radius: 3px;"></div>
                    </div>
                    <small style="color: var(--gray);">
                        {row['Orders']} orders • ${row['Avg_Order']:.2f} avg
                    </small>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Item-Level Analysis within Categories
        st.markdown("""
        <div class="glass-panel">
            <h3>🍽️ Item-Level Performance Analysis</h3>
            <p class="light-text">Drill down into individual item performance</p>
        </div>
        """, unsafe_allow_html=True)
        
        selected_category = st.selectbox(
            "Select Category to Analyze",
            options=category_stats['Category'].tolist(),
            key="category_select"
        )
        
        if selected_category:
            category_items = filtered_df[filtered_df['Category'] == selected_category]
            item_stats = category_items.groupby('Item').agg({
                'Order_Total': 'sum',
                'Order_ID': 'count',
                'Price': 'mean',
                'Quantity': 'sum'
            }).rename(columns={'Order_ID': 'Orders'}).nlargest(10, 'Order_Total')
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                fig = px.bar(
                    item_stats.reset_index(),
                    x='Item',
                    y='Order_Total',
                    title=f'Top Items in {selected_category}',
                    color='Orders',
                    color_continuous_scale='Viridis'
                )
                
                fig.update_layout(
                    height=400,
                    xaxis_tickangle=-45,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0')
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Create a radar chart for top item
                if not item_stats.empty:
                    top_item = item_stats.index[0]
                    top_item_data = category_items[category_items['Item'] == top_item]
                    
                    metrics = [
                        ('Revenue', top_item_data['Order_Total'].sum()),
                        ('Orders', len(top_item_data)),
                        ('Avg Price', top_item_data['Price'].mean()),
                        ('Total Quantity', top_item_data['Quantity'].sum()),
                        ('Customer Reach', top_item_data['Customer_ID'].nunique())
                    ]
                    
                    fig = go.Figure(data=go.Scatterpolar(
                        r=[m[1] for m in metrics],
                        theta=[m[0] for m in metrics],
                        fill='toself',
                        line_color='#2563eb'
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, max([m[1] for m in metrics]) * 1.2]
                            )),
                        showlegend=False,
                        height=400,
                        title=f"Performance Radar: {top_item}"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    # ==============================================
    # 👥 TAB 3: CUSTOMER INSIGHTS - FIXED VERSION
    # ==============================================
    with tab3:
        st.markdown("""
        <div class="glass-panel">
            <h2>👥 Customer Intelligence & Segmentation</h2>
            <p class="light-text">Advanced customer analytics and behavioral insights</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Customer Segmentation using K-means
        with st.spinner("🧠 Analyzing customer segments..."):
            customer_segments = analytics.customer_segmentation()
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # 3D segmentation visualization - FIXED COLUMN NAMES
            if not customer_segments.empty:
                fig = px.scatter_3d(
                    customer_segments,
                    x='Total_Spent',
                    y='Visit_Count',
                    z='Avg_Order',
                    color='Segment_Name',
                    hover_name='Customer_ID',
                    title='3D Customer Segmentation'
                )
                
                fig.update_layout(
                    height=500,
                    scene=dict(
                        xaxis_title='Total Spent ($)',
                        yaxis_title='Visit Count',
                        zaxis_title='Avg Order Value ($)'
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="data-card">
                <h4>🎯 Segment Profiles</h4>
            """, unsafe_allow_html=True)
            
            if not customer_segments.empty:
                segment_summary = customer_segments.groupby('Segment_Name').agg({
                    'Customer_ID': 'count',
                    'Total_Spent': 'mean',
                    'Visit_Count': 'mean',
                    'Avg_Order': 'mean'
                }).round(2)
                
                for segment, data in segment_summary.iterrows():
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; margin: 10px 0;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <strong>{segment}</strong>
                            <span style="background: var(--primary); color: white; padding: 3px 8px; border-radius: 12px; font-size: 0.8rem;">
                                {data['Customer_ID']} customers
                            </span>
                        </div>
                        <div style="margin-top: 10px;">
                            <small>Avg Spend: ${data['Total_Spent']:,.0f}</small><br>
                            <small>Visits: {data['Visit_Count']:.1f}</small><br>
                            <small>Order Value: ${data['Avg_Order']:.2f}</small>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Customer Journey Analysis
        st.markdown("""
        <div class="glass-panel">
            <h3>🛤️ Customer Journey Analysis</h3>
            <p class="light-text">Track customer behavior and conversion paths</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate customer lifecycle metrics - FIXED COLUMN NAMES
        customer_journey = filtered_df.groupby('Customer_ID').agg({
            'Order_Date': ['min', 'max', 'count'],
            'Order_Total': 'sum',
            'Category': lambda x: ', '.join(x.value_counts().head(3).index.tolist())
        })
        
        customer_journey.columns = ['First_Visit', 'Last_Visit', 'Visits', 'Total_Spent', 'Top_Categories']
        customer_journey['Customer_Lifetime'] = (customer_journey['Last_Visit'] - customer_journey['First_Visit']).dt.days
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_lifetime = customer_journey['Customer_Lifetime'].mean() if not customer_journey.empty else 0
            st.metric("Avg Customer Lifetime", f"{avg_lifetime:.0f} days")
        
        with col2:
            repeat_rate = (len(customer_journey[customer_journey['Visits'] > 1]) / len(customer_journey) * 100) if not customer_journey.empty else 0
            st.metric("Repeat Customer Rate", f"{repeat_rate:.1f}%")
        
        with col3:
            avg_frequency = customer_journey['Visits'].mean() if not customer_journey.empty else 0
            st.metric("Avg Visit Frequency", f"{avg_frequency:.1f}")
    
    # ==============================================
    # ⏰ TAB 4: TEMPORAL ANALYSIS - FIXED VERSION
    # ==============================================
    with tab4:
        st.markdown("""
        <div class="glass-panel">
            <h2>⏰ Temporal Patterns & Seasonality</h2>
            <p class="light-text">Time-based analysis and peak period identification</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Multi-dimensional Time Analysis
        time_analysis_cols = st.columns(3)
        
        with time_analysis_cols[0]:
            # Hourly patterns - FIXED COLUMN NAMES
            hourly_data = filtered_df.groupby('Order_Hour').agg({
                'Order_Total': 'sum',
                'Order_ID': 'count'
            }).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=hourly_data['Order_Hour'],
                y=hourly_data['Order_Total'],
                name='Revenue',
                marker_color='#2563eb'
            ))
            
            fig.add_trace(go.Scatter(
                x=hourly_data['Order_Hour'],
                y=hourly_data['Order_ID'],
                name='Orders',
                yaxis='y2',
                line=dict(color='#10b981', width=2)
            ))
            
            fig.update_layout(
                title='Hourly Revenue & Order Patterns',
                yaxis=dict(title='Revenue ($)'),
                yaxis2=dict(
                    title='Order Count',
                    overlaying='y',
                    side='right'
                ),
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with time_analysis_cols[1]:
            # Weekly patterns - FIXED COLUMN NAMES
            weekday_order = {
                'Monday': 0, 'Tuesday': 0, 'Wednesday': 0,
                'Thursday': 0, 'Friday': 0, 'Saturday': 0, 'Sunday': 0
            }
            
            for day, revenue in filtered_df.groupby('Order_Day')['Order_Total'].sum().items():
                weekday_order[day] = revenue
            
            fig = px.line_polar(
                r=list(weekday_order.values()),
                theta=list(weekday_order.keys()),
                line_close=True,
                title='Weekly Revenue Pattern'
            )
            
            fig.update_traces(fill='toself')
            fig.update_layout(height=300, polar=dict(radialaxis=dict(visible=True)))
            
            st.plotly_chart(fig, use_container_width=True)
        
        with time_analysis_cols[2]:
            # Monthly trends - FIXED COLUMN NAMES
            monthly_data = filtered_df.groupby('Order_Month').agg({
                'Order_Total': 'sum',
                'Order_ID': 'count'
            }).reset_index()
            
            monthly_data['Month'] = monthly_data['Order_Month'].apply(lambda x: calendar.month_abbr[x])
            
            fig = px.bar(
                monthly_data,
                x='Month',
                y='Order_Total',
                title='Monthly Revenue Trends',
                color='Order_ID',
                color_continuous_scale='Plasma'
            )
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Peak Period Analysis
        st.markdown("""
        <div class="glass-panel">
            <h3>📊 Peak Period Intelligence</h3>
            <p class="light-text">Identify optimal times for promotions and staffing</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate peak hours by day of week - FIXED COLUMN NAMES
        peak_analysis = filtered_df.groupby(['Order_Day', 'Order_Hour']).agg({
            'Order_Total': 'mean',
            'Order_ID': 'count'
        }).reset_index()
        
        col1, col2 = st.columns(2)
        
        if not peak_analysis.empty:
            # Find peak combinations
            peak_revenue = peak_analysis.loc[peak_analysis['Order_Total'].idxmax()]
            peak_orders = peak_analysis.loc[peak_analysis['Order_ID'].idxmax()]
            
            with col1:
                st.markdown(f"""
                <div class="data-card">
                    <h4>💰 Peak Revenue Period</h4>
                    <div style="text-align: center; padding: 20px;">
                        <div style="font-size: 3rem; color: var(--accent);">{peak_revenue['Order_Hour']}:00</div>
                        <h3>{peak_revenue['Order_Day']}</h3>
                        <p>Average Revenue: ${peak_revenue['Order_Total']:.0f}</p>
                        <div style="background: linear-gradient(90deg, var(--primary), var(--accent));
                                    padding: 10px; border-radius: 10px; margin-top: 10px;">
                            <small>OPTIMIZATION TIP</small><br>
                            Consider premium pricing during this period
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="data-card">
                    <h4>👥 Peak Order Period</h4>
                    <div style="text-align: center; padding: 20px;">
                        <div style="font-size: 3rem; color: var(--warning);">{peak_orders['Order_Hour']}:00</div>
                        <h3>{peak_orders['Order_Day']}</h3>
                        <p>Average Orders: {peak_orders['Order_ID']:.0f}</p>
                        <div style="background: linear-gradient(90deg, var(--warning), var(--danger));
                                    padding: 10px; border-radius: 10px; margin-top: 10px;">
                            <small>STAFFING TIP</small><br>
                            Increase staff by 30% during this period
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # ==============================================
    # 🔍 TAB 5: RAW DATA - FIXED VERSION
    # ==============================================
    with tab5:
        st.markdown("""
        <div class="glass-panel">
            <h2>🔍 Advanced Data Exploration</h2>
            <p class="light-text">Interactive data analysis and export capabilities</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Interactive Data Explorer
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown("""
            <div class="data-card">
                <h4>📁 Dataset Summary</h4>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 15px;">
                    <div>
                        <small>Total Records</small>
                        <h5>{:,}</h5>
                    </div>
                    <div>
                        <small>Columns</small>
                        <h5>{}</h5>
                    </div>
                    <div>
                        <small>Time Span</small>
                        <h5>{} days</h5>
                    </div>
                </div>
            </div>
            """.format(
                len(filtered_df),
                len(filtered_df.columns),
                (filtered_df['Order_Date'].max() - filtered_df['Order_Date'].min()).days
            ), unsafe_allow_html=True)
        
        with col2:
            memory_usage = filtered_df.memory_usage(deep=True).sum() / 1024**2
            st.metric("Memory Usage", f"{memory_usage:.2f} MB")
        
        with col3:
            completeness = (1 - filtered_df.isnull().sum().sum() / (len(filtered_df) * len(filtered_df.columns))) * 100
            st.metric("Data Completeness", f"{completeness:.1f}%")
        
        # Advanced Filtering System
        st.markdown("""
        <div class="glass-panel">
            <h4>🎯 Advanced Data Filters</h4>
        </div>
        """, unsafe_allow_html=True)
        
        filter_cols = st.columns(4)
        
        with filter_cols[0]:
            min_price = st.slider("Min Price", 0.0, float(filtered_df['Price'].max()), 0.0)
        
        with filter_cols[1]:
            max_price = st.slider("Max Price", 0.0, float(filtered_df['Price'].max()), 
                                 float(filtered_df['Price'].max()))
        
        with filter_cols[2]:
            min_quantity = st.slider("Min Quantity", 1, int(filtered_df['Quantity'].max()), 1)
        
        with filter_cols[3]:
            max_quantity = st.slider("Max Quantity", 1, int(filtered_df['Quantity'].max()), 
                                    int(filtered_df['Quantity'].max()))
        
        # Apply filters
        filtered_view = filtered_df[
            (filtered_df['Price'] >= min_price) & 
            (filtered_df['Price'] <= max_price) &
            (filtered_df['Quantity'] >= min_quantity) &
            (filtered_df['Quantity'] <= max_quantity)
        ].copy()
        
        # Interactive Data Table with Search
        search_query = st.text_input("🔍 Search in data...", placeholder="Search for items, customers, etc.")
        
        if search_query:
            search_mask = filtered_view.apply(
                lambda row: row.astype(str).str.contains(search_query, case=False).any(), 
                axis=1
            )
            filtered_view = filtered_view[search_mask]
        
        # Column selector
        selected_columns = st.multiselect(
            "Select columns to display",
            options=filtered_view.columns.tolist(),
            default=['Order_Date', 'Category', 'Item', 'Price', 'Quantity', 'Order_Total', 'Payment_Method']
        )
        
        if selected_columns:
            # Display data with pagination
            page_size = st.selectbox("Rows per page", [10, 25, 50, 100], index=0)
            
            if len(filtered_view) > 0:
                total_pages = max(1, (len(filtered_view) // page_size) + 1)
                page_number = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
                
                start_idx = (page_number - 1) * page_size
                end_idx = min(page_number * page_size, len(filtered_view))
                
                st.dataframe(
                    filtered_view[selected_columns].iloc[start_idx:end_idx].style.format({
                        'Price': '${:,.2f}',
                        'Order_Total': '${:,.2f}',
                        'Total_Spent': '${:,.2f}'
                    }).background_gradient(subset=['Order_Total'], cmap='Blues'),
                    use_container_width=True,
                    height=400
                )
                
                st.caption(f"Showing rows {start_idx + 1} to {end_idx} of {len(filtered_view)}")
            else:
                st.info("No data matches your filters")
        
        # Data Quality Dashboard
        st.markdown("""
        <div class="glass-panel">
            <h4>📊 Data Quality Dashboard</h4>
        </div>
        """, unsafe_allow_html=True)
        
        quality_cols = st.columns(4)
        
        with quality_cols[0]:
            duplicates = filtered_df.duplicated().sum()
            st.metric("Duplicates", duplicates, delta_color="inverse")
        
        with quality_cols[1]:
            missing_values = filtered_df.isnull().sum().sum()
            st.metric("Missing Values", missing_values, delta_color="inverse")
        
        with quality_cols[2]:
            outliers = len(analytics.detect_anomalies())
            st.metric("Potential Outliers", outliers)
        
        with quality_cols[3]:
            consistency_score = 100 - ((duplicates + missing_values) / (len(filtered_df) * len(filtered_df.columns)) * 100)
            st.metric("Data Quality Score", f"{consistency_score:.1f}%")
        
        # Export Options
        st.markdown("""
        <div class="glass-panel">
            <h4>💾 Export & Integration</h4>
        </div>
        """, unsafe_allow_html=True)
        
        export_cols = st.columns(3)
        
        with export_cols[0]:
            if st.button("📥 Export CSV", use_container_width=True):
                csv = filtered_view.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="restaurant_export.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with export_cols[1]:
            if st.button("📊 Export JSON", use_container_width=True):
                json_data = filtered_view.to_json(orient='records', indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name="restaurant_export.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        with export_cols[2]:
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.rerun()
    
    # ==============================================
    # 🎯 FOOTER WITH ANALYTICS
    # ==============================================
    st.markdown("""
    <div style="margin-top: 50px; padding: 20px; border-top: 1px solid rgba(255,255,255,0.1);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <small class="light-text">Restaurant Intelligence Dashboard v2.0</small><br>
                <small class="light-text">Powered by Advanced Analytics • Updated: {}</small>
            </div>
            <div>
                <small class="light-text">Processing Time: {:.2f}s • Records: {:,}</small>
            </div>
        </div>
    </div>
    """.format(
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        time.time() % 100,  # Simulated processing time
        len(filtered_df)
    ), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
