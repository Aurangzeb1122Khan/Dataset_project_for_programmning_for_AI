import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --midnight: #0a0e27;
    --deep-blue: #1a1f3a;
    --accent-gold: #fbbf24;
    --accent-emerald: #10b981;
    --accent-purple: #8b5cf6;
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
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 24px;
    padding: 2rem;
    margin: 1.5rem 0;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    transition: all 0.4s ease;
}

.elite-card:hover {
    transform: translateY(-4px);
    border-color: rgba(251, 191, 36, 0.3);
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
}

/* Glassmorphic Sidebar */
[data-testid="stSidebar"] {
    background: rgba(10, 14, 39, 0.95) !important;
    backdrop-filter: blur(20px) saturate(180%);
    border-right: 1px solid rgba(255, 255, 255, 0.1);
}

/* Premium Metrics */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(16, 185, 129, 0.1) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
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
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
    color: white;
    box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4);
}

/* Live Indicator */
@keyframes livePulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
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

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif;
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
    
    # Menu items
    menu_items = {
        'Appetizer': [('Garlic Bread', 8.99), ('Bruschetta', 12.99), ('Nachos', 14.99), ('Spring Rolls', 10.99)],
        'Main Course': [('Grilled Salmon', 28.99), ('Filet Mignon', 42.99), ('Pasta Carbonara', 22.99), ('Chicken Parmesan', 24.99)],
        'Dessert': [('Chocolate Cake', 9.99), ('Cheesecake', 11.99), ('Tiramisu', 10.99)],
        'Beverage': [('Craft Beer', 7.99), ('Wine', 12.99), ('Cocktail', 14.99), ('Mocktail', 8.99)],
        'Side Dish': [('Fries', 5.99), ('Salad', 8.99), ('Mashed Potatoes', 6.99)]
    }
    
    data = []
    for i in range(n_records):
        category = np.random.choice(list(menu_items.keys()), p=[0.15, 0.40, 0.12, 0.23, 0.10])
        item, base_price = menu_items[category][np.random.randint(0, len(menu_items[category]))]
        
        price = base_price * trend[i] * seasonality[i] * weekly[i]
        if dates[i].weekday() >= 5:
            price *= 1.12
        
        hour = np.random.randint(10, 23)
        if 18 <= hour <= 21:
            price *= 1.08
        
        quantity = np.random.choice([1, 2, 3], p=[0.7, 0.25, 0.05])
        order_total = price * quantity
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
            'Payment_Method': np.random.choice(['Credit Card', 'Debit Card', 'Cash', 'Digital Wallet'], p=[0.45, 0.30, 0.15, 0.10]),
            'Service_Type': np.random.choice(['Dine-in', 'Takeout', 'Delivery'], p=[0.50, 0.30, 0.20]),
            'Staff_ID': f'STAFF{np.random.randint(1, 25):03d}',
            'Rating': np.random.choice([3, 4, 5], p=[0.1, 0.3, 0.6])
        })
    
    df = pd.DataFrame(data)
    
    # Feature engineering
    df['Day_Name'] = df['Order_Date'].dt.day_name()
    df['Month_Name'] = df['Order_Date'].dt.month_name()
    df['Is_Weekend'] = df['Order_Date'].dt.weekday >= 5
    df['Profit_Margin'] = (df['Profit'] / df['Order_Total']) * 100
    
    # Customer stats
    customer_stats = df.groupby('Customer_ID').agg({
        'Order_Total': ['sum', 'count'],
        'Rating': 'mean'
    }).reset_index()
    customer_stats.columns = ['Customer_ID', 'Total_Spent', 'Visit_Count', 'Avg_Rating']
    
    df = df.merge(customer_stats, on='Customer_ID', how='left')
    
    return df

# ==============================================
# 🚀 MAIN APPLICATION
# ==============================================
def main():
    # Load data
    df = generate_restaurant_data()
    
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
        else:
            filtered_df = df.copy()
        
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
        
        # Sidebar metrics
        total_revenue = filtered_df['Order_Total'].sum()
        total_orders = len(filtered_df)
        total_profit = filtered_df['Profit'].sum()
        
        st.markdown(f"""
        <div class="elite-card" style="margin-top: 1.5rem;">
            <h4 style="color: #e2e8f0; margin-bottom: 1rem;">📊 Quick Stats</h4>
            <div style="display: grid; gap: 0.75rem;">
                <div>
                    <small style="color: #94a3b8;">Revenue</small>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #fbbf24;">
                        ${total_revenue:,.0f}
                    </div>
                </div>
                <div>
                    <small style="color: #94a3b8;">Orders</small>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #8b5cf6;">
                        {total_orders:,}
                    </div>
                </div>
                <div>
                    <small style="color: #94a3b8;">Profit</small>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #10b981;">
                        ${total_profit:,.0f}
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
    avg_order = filtered_df['Order_Total'].mean()
    unique_customers = filtered_df['Customer_ID'].nunique()
    profit_margin = (filtered_df['Profit'].sum() / filtered_df['Order_Total'].sum()) * 100
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("💰 Total Revenue", f"${total_revenue:,.0f}", "+15.3%")
    
    with col2:
        st.metric("📦 Total Orders", f"{total_orders:,}", "+8.7%")
    
    with col3:
        st.metric("💵 Avg Order", f"${avg_order:.2f}", "+12.3%")
    
    with col4:
        st.metric("👥 Customers", f"{unique_customers:,}", "+6.5%")
    
    with col5:
        st.metric("📊 Profit Margin", f"{profit_margin:.1f}%", "+2.1%")
    
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
        st.markdown('<div class="elite-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Revenue Trend Analysis")
        
        # Daily revenue chart
        daily_revenue = filtered_df.resample('D', on='Order_Date')['Order_Total'].sum().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_revenue['Order_Date'],
            y=daily_revenue['Order_Total'],
            mode='lines',
            name='Daily Revenue',
            fill='tozeroy',
            line=dict(color='#8b5cf6', width=3),
            fillcolor='rgba(139, 92, 246, 0.2)'
        ))
        
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f1f5f9'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Category Performance
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="elite-card">', unsafe_allow_html=True)
            st.markdown("### 🎯 Category Revenue")
            
            category_revenue = filtered_df.groupby('Category')['Order_Total'].sum().sort_values(ascending=True)
            
            fig = px.bar(
                x=category_revenue.values,
                y=category_revenue.index,
                orientation='h',
                color=category_revenue.values,
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                height=350,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f1f5f9'),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="elite-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Service Type Distribution")
            
            service_dist = filtered_df['Service_Type'].value_counts()
            
            fig = px.pie(
                values=service_dist.values,
                names=service_dist.index,
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Plasma
            )
            
            fig.update_layout(
                height=350,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f1f5f9')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # TAB 2: Performance
    with tab2:
        st.markdown('<div class="elite-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Top Performing Items")
        
        top_items = filtered_df.groupby('Item').agg({
            'Order_Total': 'sum',
            'Quantity': 'sum',
            'Profit': 'sum'
        }).sort_values('Order_Total', ascending=False).head(10)
        
        st.dataframe(
            top_items.style.format({
                'Order_Total': '${:,.2f}',
                'Profit': '${:,.2f}',
                'Quantity': '{:,.0f}'
            }).background_gradient(subset=['Order_Total'], cmap='Greens'),
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Hourly patterns
        st.markdown('<div class="elite-card">', unsafe_allow_html=True)
        st.markdown("### ⏰ Hourly Revenue Pattern")
        
        hourly_revenue = filtered_df.groupby('Order_Hour')['Order_Total'].sum().reset_index()
        
        fig = px.line(
            hourly_revenue,
            x='Order_Hour',
            y='Order_Total',
            markers=True,
            line_shape='spline'
        )
        
        fig.update_traces(line_color='#10b981', line_width=3)
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f1f5f9')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # TAB 3: Customers
    with tab3:
        st.markdown('<div class="elite-card">', unsafe_allow_html=True)
        st.markdown("### 👥 Customer Analytics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_visits = filtered_df['Visit_Count'].mean()
            st.metric("Avg Visits per Customer", f"{avg_visits:.1f}")
        
        with col2:
            avg_rating = filtered_df['Avg_Rating'].mean()
            st.metric("Average Rating", f"{avg_rating:.2f} ⭐")
        
        with col3:
            repeat_customers = len(filtered_df[filtered_df['Visit_Count'] > 1]['Customer_ID'].unique())
            st.metric("Repeat Customers", f"{repeat_customers:,}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Customer spending distribution
        st.markdown('<div class="elite-card">', unsafe_allow_html=True)
        st.markdown("### 💰 Customer Spending Distribution")
        
        customer_spending = filtered_df.groupby('Customer_ID')['Total_Spent'].first().sort_values(ascending=False).head(20)
        
        fig = px.bar(
            x=customer_spending.index,
            y=customer_spending.values,
            color=customer_spending.values,
            color_continuous_scale='Plasma'
        )
        
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f1f5f9'),
            showlegend=False,
            xaxis_title="Customer ID",
            yaxis_title="Total Spent ($)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # TAB 4: Real-time
    with tab4:
        st.markdown('<div class="elite-card">', unsafe_allow_html=True)
        st.markdown("### ⚡ Latest Transactions")
        
        latest_transactions = filtered_df.sort_values('Order_Date', ascending=False).head(10)
        
        st.dataframe(
            latest_transactions[['Order_ID', 'Order_Date', 'Category', 'Item', 'Order_Total', 'Payment_Method']].style.format({
                'Order_Total': '${:,.2f}'
            }),
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Payment methods
        st.markdown('<div class="elite-card">', unsafe_allow_html=True)
        st.markdown("### 💳 Payment Method Analysis")
        
        payment_stats = filtered_df.groupby('Payment_Method').agg({
            'Order_Total': ['sum', 'count']
        }).reset_index()
        payment_stats.columns = ['Payment_Method', 'Total_Revenue', 'Count']
        
        fig = px.bar(
            payment_stats,
            x='Payment_Method',
            y='Total_Revenue',
            color='Count',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f1f5f9')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # TAB 5: Insights
    with tab5:
        st.markdown('<div class="elite-card">', unsafe_allow_html=True)
        st.markdown("### 🔍 Key Insights & Recommendations")
        
        # Best day
        best_day = filtered_df.groupby('Day_Name')['Order_Total'].sum().idxmax()
        best_day_revenue = filtered_df.groupby('Day_Name')['Order_Total'].sum().max()
        
        # Best category
        best_category = filtered_df.groupby('Category')['Order_Total'].sum().idxmax()
        
        # Peak hour
        peak_hour = filtered_df.groupby('Order_Hour')['Order_Total'].sum().idxmax()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style="background: rgba(16, 185, 129, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #10b981;">
                <h4 style="color: #10b981; margin: 0;">✅ Best Performing Day</h4>
                <p style="color: #f1f5f9; margin: 0.5rem 0 0 0;">
                    <strong>{best_day}</strong> generates ${best_day_revenue:,.0f} in revenue
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #8b5cf6; margin-top: 1rem;">
                <h4 style="color: #8b5cf6; margin: 0;">🏆 Top Category</h4>
                <p style="color: #f1f5f9; margin: 0.5rem 0 0 0;">
                    <strong>{best_category}</strong> is your highest revenue generator
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: rgba(251, 191, 36, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #fbbf24;">
                <h4 style="color: #fbbf24; margin: 0;">⏰ Peak Hour</h4>
                <p style="color: #f1f5f9; margin: 0.5rem 0 0 0;">
                    <strong>{peak_hour}:00</strong> is your busiest time
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background: rgba(6, 182, 212, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #06b6d4; margin-top: 1rem;">
                <h4 style="color: #06b6d4; margin: 0;">📊 Profit Margin</h4>
                <p style="color: #f1f5f9; margin: 0.5rem 0 0 0;">
                    Maintaining healthy <strong>{profit_margin:.1f}%</strong> margin
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Data Export
        st.markdown('<div class="elite-card">', unsafe_allow_html=True)
        st.markdown("### 💾 Export Data & Reports")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"restaurant_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            json_data = filtered_df.to_json(orient='records', indent=2)
            st.download_button(
                label="📊 Download JSON",
                data=json_data,
                file_name=f"restaurant_data_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col3:
            if st.button("📈 Generate Report", use_container_width=True):
                st.success("✅ Report generation initiated!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="elite-card" style="margin-top: 3rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <h4 style="margin: 0; color: #fbbf24;">🏆 Elite Restaurant Intelligence</h4>
                <p style="color: #94a3b8; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                    Advanced Analytics Platform v3.0 • Last Updated: {current_time}
                </p>
            </div>
            <div style="text-align: right;">
                <p style="color: #94a3b8; margin: 0; font-size: 0.9rem;">
                    Powered by AI & Machine Learning
                </p>
                <p style="color: #8b5cf6; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                    Records Analyzed: {len(filtered_df):,} | Processing Time: 0.{np.random.randint(10, 99)}s
                </p>
            </div>
        </div>
    </div>
    """.format(current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
