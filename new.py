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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestRegressor
import calendar
import json

# ==============================================
# 🎯 ELITE CONFIGURATION
# ==============================================
st.set_page_config(
    page_title="Elite Restaurant Intelligence",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================
# 🎨 ULTRA-PREMIUM DARK THEME
# ==============================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --midnight: #0a0e27;
    --deep-blue: #1a1f3a;
    --accent-gold: #fbbf24;
    --accent-emerald: #10b981;
    --accent-purple: #8b5cf6;
    --accent-cyan: #06b6d4;
    --glass: rgba(255, 255, 255, 0.05);
    --glass-border: rgba(255, 255, 255, 0.1);
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, var(--midnight) 0%, #1a1f3a 50%, #0f1729 100%);
    font-family: 'Inter', sans-serif;
    color: #f1f5f9;
}

/* Animated Background */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: 
        radial-gradient(circle at 20% 50%, rgba(139, 92, 246, 0.15) 0%, transparent 50%),
        radial-gradient(circle at 80% 80%, rgba(16, 185, 129, 0.15) 0%, transparent 50%);
    animation: pulse 15s ease-in-out infinite;
    pointer-events: none;
    z-index: 0;
}

@keyframes pulse {
    0%, 100% { opacity: 0.3; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.05); }
}

/* Premium Cards */
.elite-card {
    background: rgba(26, 31, 58, 0.8);
    backdrop-filter: blur(20px) saturate(180%);
    border: 1px solid var(--glass-border);
    border-radius: 24px;
    padding: 2rem;
    margin: 1.5rem 0;
    box-shadow: 
        0 8px 32px rgba(0, 0, 0, 0.4),
        inset 0 1px 0 rgba(255, 255, 255, 0.1);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.elite-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(251, 191, 36, 0.1), transparent);
    transition: left 0.6s;
}

.elite-card:hover {
    transform: translateY(-4px);
    border-color: rgba(251, 191, 36, 0.3);
    box-shadow: 
        0 20px 60px rgba(0, 0, 0, 0.5),
        0 0 0 1px rgba(251, 191, 36, 0.2),
        inset 0 1px 0 rgba(255, 255, 255, 0.2);
}

.elite-card:hover::before {
    left: 100%;
}

/* Glassmorphic Sidebar */
[data-testid="stSidebar"] {
    background: rgba(10, 14, 39, 0.95) !important;
    backdrop-filter: blur(20px) saturate(180%);
    border-right: 1px solid var(--glass-border);
}

/* Premium Metrics */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(16, 185, 129, 0.1) 100%);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    padding: 1.5rem;
    transition: all 0.3s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-5px) scale(1.02);
    border-color: var(--accent-gold);
    box-shadow: 0 20px 40px rgba(251, 191, 36, 0.2);
}

[data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Elite Buttons */
.stButton > button {
    background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(139, 92, 246, 0.6);
    background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%);
}

/* Data Tables */
.stDataFrame {
    border-radius: 16px;
    overflow: hidden;
}

.stDataFrame th {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(99, 102, 241, 0.2));
    color: white;
    font-weight: 600;
    padding: 1rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stDataFrame td {
    padding: 0.75rem 1rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

.stDataFrame tr:hover {
    background: rgba(139, 92, 246, 0.1);
}

/* Premium Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: rgba(26, 31, 58, 0.6);
    padding: 0.5rem;
    border-radius: 16px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #94a3b8;
    border-radius: 12px;
    padding: 0.75rem 1.5rem;
    font-weight: 500;
    transition: all 0.3s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(139, 92, 246, 0.1);
    color: #e2e8f0;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
    color: white;
    box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4);
}

/* Badges */
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.badge-gold {
    background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
    color: #0a0e27;
}

.badge-purple {
    background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
    color: white;
}

.badge-emerald {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: white;
}

/* Alert Boxes */
.alert {
    padding: 1rem 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    border-left: 4px solid;
}

.alert-success {
    background: rgba(16, 185, 129, 0.1);
    border-color: #10b981;
    color: #6ee7b7;
}

.alert-warning {
    background: rgba(251, 191, 36, 0.1);
    border-color: #fbbf24;
    color: #fcd34d;
}

.alert-info {
    background: rgba(139, 92, 246, 0.1);
    border-color: #8b5cf6;
    color: #c4b5fd;
}

/* Live Indicator */
@keyframes livePulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(1.1); }
}

.live-indicator {
    display: inline-block;
    width: 8px;
    height: 8px;
    background: #10b981;
    border-radius: 50%;
    animation: livePulse 2s ease-in-out infinite;
    margin-right: 8px;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

::-webkit-scrollbar-track {
    background: rgba(15, 23, 42, 0.5);
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
    border-radius: 5px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%);
}

/* Responsive */
@media (max-width: 768px) {
    .elite-card { padding: 1rem; }
    [data-testid="stMetricValue"] { font-size: 2rem; }
}
</style>
""", unsafe_allow_html=True)

# ==============================================
# 📊 DATA GENERATION
# ==============================================
@st.cache_data
def generate_restaurant_data():
    """Generate realistic restaurant sales data"""
    np.random.seed(42)
    n_records = 17534
    
    dates = pd.date_range('2023-01-01', '2024-01-01', periods=n_records)
    
    # Realistic patterns
    trend = np.linspace(1, 1.3, n_records)
    seasonality = 1 + 0.2 * np.sin(2 * np.pi * np.arange(n_records) / 365)
    weekly = 1 + 0.15 * np.sin(2 * np.pi * np.arange(n_records) / 7)
    
    # Menu items with realistic pricing
    menu_items = {
        'Appetizer': [
            ('Garlic Bread', 8.99), ('Bruschetta', 12.99), ('Nachos Supreme', 14.99),
            ('Spring Rolls', 10.99), ('Mozzarella Sticks', 11.99), ('Calamari', 15.99)
        ],
        'Main Course': [
            ('Grilled Salmon', 28.99), ('Filet Mignon', 42.99), ('Pasta Carbonara', 22.99),
            ('Chicken Parmesan', 24.99), ('Lobster Tail', 49.99), ('Ribeye Steak', 38.99),
            ('Seafood Paella', 32.99), ('Lamb Chops', 36.99)
        ],
        'Dessert': [
            ('Chocolate Lava Cake', 9.99), ('Cheesecake', 11.99), ('Tiramisu', 10.99),
            ('Crème Brûlée', 12.99), ('Apple Pie', 8.99), ('Panna Cotta', 9.99)
        ],
        'Beverage': [
            ('Craft Beer', 7.99), ('Premium Wine', 12.99), ('Signature Cocktail', 14.99),
            ('Mocktail', 8.99), ('Fresh Juice', 6.99), ('Specialty Coffee', 5.99)
        ],
        'Side Dish': [
            ('Truffle Fries', 8.99), ('Caesar Salad', 9.99), ('Mashed Potatoes', 6.99),
            ('Grilled Vegetables', 9.99), ('Garlic Bread', 5.99)
        ]
    }
    
    data = []
    for i in range(n_records):
        category = np.random.choice(
            list(menu_items.keys()),
            p=[0.15, 0.40, 0.12, 0.23, 0.10]
        )
        
        item, base_price = menu_items[category][np.random.randint(0, len(menu_items[category]))]
        
        # Dynamic pricing
        price = base_price * trend[i] * seasonality[i] * weekly[i]
        
        # Weekend premium
        if dates[i].weekday() >= 5:
            price *= 1.12
        
        # Peak hours premium
        hour = np.random.randint(10, 23)
        if 18 <= hour <= 21:  # Dinner rush
            price *= 1.08
        
        quantity = np.random.choice([1, 2, 3], p=[0.7, 0.25, 0.05])
        order_total = price * quantity
        
        # Calculate profit (60-70% margin)
        cost = price * np.random.uniform(0.30, 0.40)
        profit = order_total - (cost * quantity)
        
        data.append({
            'Order_ID': f'ORD{str(i+1).zfill(6)}',
            'Customer_ID': f'CUST{np.random.randint(1000, 5000):04d}',
            'Order_Date': dates[i],
            'Order_Hour': hour,
            'Category': category,
            'Item': item,
            'Price': round(price, 2),
            'Quantity': quantity,
            'Order_Total': round(order_total, 2),
            'Cost': round(cost * quantity, 2),
            'Profit': round(profit, 2),
            'Payment_Method': np.random.choice(
                ['Credit Card', 'Debit Card', 'Cash', 'Digital Wallet', 'Crypto'],
                p=[0.45, 0.30, 0.10, 0.13, 0.02]
            ),
            'Service_Type': np.random.choice(
                ['Dine-in', 'Takeout', 'Delivery', 'Curbside'],
                p=[0.50, 0.25, 0.20, 0.05]
            ),
            'Staff_ID': f'STAFF{np.random.randint(1, 25):03d}',
            'Rating': np.random.choice([3, 4, 5], p=[0.1, 0.3, 0.6]),
            'Weather': np.random.choice(['Sunny', 'Rainy', 'Cloudy'], p=[0.5, 0.2, 0.3])
        })
    
    df = pd.DataFrame(data)
    
    # Feature engineering
    df['Day_Name'] = df['Order_Date'].dt.day_name()
    df['Month_Name'] = df['Order_Date'].dt.month_name()
    df['Is_Weekend'] = df['Order_Date'].dt.weekday >= 5
    df['Day_Part'] = pd.cut(df['Order_Hour'], 
                            bins=[0, 11, 17, 22, 24],
                            labels=['Morning', 'Afternoon', 'Evening', 'Late Night'])
    df['Season'] = df['Order_Date'].dt.month.map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    df['Profit_Margin'] = (df['Profit'] / df['Order_Total']) * 100
    
    # Customer analytics
    customer_stats = df.groupby('Customer_ID').agg({
        'Order_Total': ['sum', 'mean', 'count'],
        'Order_Date': ['min', 'max'],
        'Rating': 'mean'
    }).reset_index()
    
    customer_stats.columns = ['Customer_ID', 'Total_Spent', 'Avg_Order', 
                              'Visit_Count', 'First_Visit', 'Last_Visit', 'Avg_Rating']
    
    customer_stats['CLV_Tier'] = pd.qcut(customer_stats['Total_Spent'], 
                                         5, 
                                         labels=['Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond'],
                                         duplicates='drop')
    
    df = df.merge(customer_stats[['Customer_ID', 'Total_Spent', 'Visit_Count', 'CLV_Tier', 'Avg_Rating']], 
                  on='Customer_ID', how='left')
    
    return df

# ==============================================
# 🧠 ANALYTICS ENGINE
# ==============================================
class RestaurantAnalytics:
    def __init__(self, df):
        self.df = df
    
    def get_key_metrics(self):
        """Calculate key performance metrics"""
        total_revenue = self.df['Order_Total'].sum()
        total_orders = len(self.df)
        total_profit = self.df['Profit'].sum()
        avg_order_value = self.df['Order_Total'].mean()
        unique_customers = self.df['Customer_ID'].nunique()
        profit_margin = (total_profit / total_revenue) * 100 if total_revenue > 0 else 0
        
        # Growth metrics (comparing last 30 days to previous 30 days)
        df_sorted = self.df.sort_values('Order_Date')
        last_30_days = df_sorted.tail(int(len(df_sorted) * 0.1))
        prev_30_days = df_sorted.tail(int(len(df_sorted) * 0.2)).head(int(len(df_sorted) * 0.1))
        
        revenue_growth = ((last_30_days['Order_Total'].sum() / prev_30_days['Order_Total'].sum()) - 1) * 100 if len(prev_30_days) > 0 else 0
        
        return {
            'total_revenue': total_revenue,
            'total_orders': total_orders,
            'total_profit': total_profit,
            'avg_order_value': avg_order_value,
            'unique_customers': unique_customers,
            'profit_margin': profit_margin,
            'revenue_growth': revenue_growth
        }
    
    def predict_revenue(self, days=30):
        """ML-based revenue forecasting"""
        daily_revenue = self.df.resample('D', on='Order_Date')['Order_Total'].sum().reset_index()
        daily_revenue['Days'] = (daily_revenue['Order_Date'] - daily_revenue['Order_Date'].min()).dt.days
        
        # Simple trend-based forecast
        X = daily_revenue['Days'].values.reshape(-1, 1)
        y = daily_revenue['Order_Total'].values
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        
        # Forecast
        last_day = daily_revenue['Days'].max()
        future_days = np.arange(last_day + 1, last_day + days + 1).reshape(-1, 1)
        predictions = model.predict(future_days)
        
        future_dates = pd.date_range(daily_revenue['Order_Date'].max() + timedelta(days=1), periods=days)
        
        return future_dates, predictions
    
    def detect_anomalies(self):
        """Detect anomalous transactions"""
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        features = self.df[['Order_Total', 'Quantity']].values
        anomalies = iso_forest.fit_predict(features)
        
        anomaly_df = self.df[anomalies == -1].copy()
        return anomaly_df
    
    def segment_customers(self):
        """Customer segmentation using K-Means"""
        customer_features = self.df.groupby('Customer_ID').agg({
            'Order_Total': 'sum',
            'Visit_Count': 'first',
            'Avg_Rating': 'first'
        }).reset_index()
        
        scaler = StandardScaler()
        X = scaler.fit_transform(customer_features[['Order_Total', 'Visit_Count', 'Avg_Rating']])
        
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        customer_features['Segment'] = kmeans.fit_predict(X)
        
        # Name segments
        segment_map = {
            0: 'High-Value Champions',
            1: 'Loyal Regulars',
            2: 'Occasional Visitors',
            3: 'New Customers'
        }
        customer_features['Segment_Name'] = customer_features['Segment'].map(segment_map)
        
        return customer_features

# ==============================================
# 🚀 MAIN APPLICATION
# ==============================================
def main():
    # Load data
    with st.spinner("🚀 Initializing Elite Restaurant Intelligence..."):
        df = generate_restaurant_data()
        analytics = RestaurantAnalytics(df)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="elite-card">
            <h3 style="margin: 0; color: #fbbf24;">⚙️ Control Center</h3>
            <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.5rem;">
                <span class="live-indicator"></span>Live Dashboard
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Date filter
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
            analytics_filtered = RestaurantAnalytics(filtered_df)
        else:
            filtered_df = df.copy()
            analytics_filtered = RestaurantAnalytics(filtered_df)
        
        # Quick filters
        st.markdown("### 🎯 Quick Filters")
        
        categories = st.multiselect(
            "Categories",
            options=sorted(df['Category'].unique()),
            default=sorted(df['Category'].unique())
        )
        
        service_types = st.multiselect(
            "Service Type",
            options=sorted(df['Service_Type'].unique()),
            default=sorted(df['Service_Type'].unique())
        )
        
        # Apply filters
        if categories:
            filtered_df = filtered_df[filtered_df['Category'].isin(categories)]
        if service_types:
            filtered_df = filtered_df[filtered_df['Service_Type'].isin(service_types)]
        
        analytics_filtered = RestaurantAnalytics(filtered_df)
        
        # Sidebar metrics
        metrics = analytics_filtered.get_key_metrics()
        
        st.markdown(f"""
        <div class="elite-card" style="margin-top: 1.5rem;">
            <h4 style="color: #e2e8f0; margin-bottom: 1rem;">📊 Quick Stats</h4>
            <div style="display: grid; gap: 0.75rem;">
                <div>
                    <small style="color: #94a3b8;">Revenue</small>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #fbbf24;">
                        ${metrics['total_revenue']:,.0f}
                    </div>
                </div>
                <div>
                    <small style="color: #94a3b8;">Orders</small>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #8b5cf6;">
                        {metrics['total_orders']:,}
                    </div>
                </div>
                <div>
                    <small style="color: #94a3b8;">Profit</small>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #10b981;">
                        ${metrics['total_profit']:,.0f}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
    
    # Hero Section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        <div class="elite-card">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="font-size: 3rem;">🏆</div>
                <div>
                    <h1 style="margin: 0; background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                        Elite Restaurant Intelligence
                    </h1>
                    <p style="color: #94a3b8; margin: 0.5rem 0 0 0;">
                        Advanced Analytics Platform • 17,534+ Transactions Analyzed
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        current_time = datetime.now().strftime("%I:%M %p")
        st.markdown(f"""
        <div class="elite-card" style="text-align: center;">
            <div style="font-size: 2rem; font-weight: 700; color: #fbbf24;">
                {current_time}
            </div>
            <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 0.5rem;">
                <span class="live-indicator"></span>Real-time Sync
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Key Metrics
    metrics = analytics_filtered.get_key_metrics()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "💰 Total Revenue",
            f"${metrics['total_revenue']:,.0f}",
            f"{metrics['revenue_growth']:+.1f}%"
        )
    
    with col2:
        st.metric(
            "📦 Total Orders",
            f"{metrics['total_orders']:,}",
            "↗ Trending"
        )
    
    with col3:
        st.metric(
            "💵 Avg Order Value",
            f"${metrics['avg_order_value']:.2f}",
            "+12.3%"
        )
    
    with col4:
        st.metric(
            "👥 Customers",
            f"{metrics['unique_customers']:,}",
            "+8.7%"
        )
    
    with col5:
        st.metric(
            "📊 Profit Margin",
            f"{metrics['profit_margin']:.1f}%",
            "+2.1%"
        )
    
    # Main Dashboard
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview",
        "🎯 Performance",
        "👥 Customers",
        "⚡ Real-time",
        "🔍 Insights"
    ])
    
    # TAB 1: Overview
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="elite-card"><h3>📈 Revenue Trend & Forecast</h3>', unsafe_allow_html=True)
            
            # Historical revenue
            daily_revenue = filtered_df.resample('D', on='Order_Date')['Order_Total'].sum()
            
            # Forecast
            forecast_dates, forecast_values = analytics_filtered.predict_revenue(days=30)
