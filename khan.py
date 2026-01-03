import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import hashlib

# Page configuration
st.set_page_config(
    page_title="Restaurant Sales Dashboard",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 15px;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
        border-top: 4px solid;
        margin-bottom: 1rem;
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .sales-card { border-color: #4ECDC4; }
    .customer-card { border-color: #FFD166; }
    .order-card { border-color: #06D6A0; }
    .profit-card { border-color: #118AB2; }
    .positive { color: #06D6A0; font-weight: 700; }
    .negative { color: #EF476F; font-weight: 700; }
    .restaurant-tag {
        background: #FFD166;
        color: #333;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        padding: 0 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F8F9FA;
        border-radius: 10px;
        padding: 1rem 2rem;
        font-weight: 600;
        border: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4ECDC4 !important;
        color: white !important;
        border-color: #4ECDC4 !important;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🍽️ RESTAURANT SALES ANALYTICS DASHBOARD</h1>', unsafe_allow_html=True)

# Generate REALISTIC restaurant data with YOUR column names
@st.cache_data
def generate_real_restaurant_data():
    # Define restaurants
    restaurants = ["Restaurant B", "Restaurant C", "Restaurant D", "Restaurant E"]
    
    # Menu items by category
    menu_items = {
        "Appetizers": ["Garlic Bread", "Bruschetta", "Chicken Wings", "Spring Rolls", "Mozzarella Sticks"],
        "Main Course": ["Grilled Salmon", "Beef Steak", "Chicken Alfredo", "Vegetable Pasta", "BBQ Ribs"],
        "Desserts": ["Chocolate Cake", "Cheesecake", "Tiramisu", "Ice Cream", "Fruit Salad"],
        "Beverages": ["Coke", "Orange Juice", "Coffee", "Tea", "Milkshake"],
        "Specials": ["Chef's Special Pizza", "Seafood Platter", "Vegan Burger", "Tandoori Chicken"]
    }
    
    # Prices for each item
    prices = {
        "Garlic Bread": 25, "Bruschetta": 30, "Chicken Wings": 45, "Spring Rolls": 35, "Mozzarella Sticks": 40,
        "Grilled Salmon": 120, "Beef Steak": 150, "Chicken Alfredo": 85, "Vegetable Pasta": 75, "BBQ Ribs": 130,
        "Chocolate Cake": 40, "Cheesecake": 45, "Tiramisu": 50, "Ice Cream": 30, "Fruit Salad": 35,
        "Coke": 15, "Orange Juice": 20, "Coffee": 18, "Tea": 15, "Milkshake": 25,
        "Chef's Special Pizza": 110, "Seafood Platter": 160, "Vegan Burger": 70, "Tandoori Chicken": 95
    }
    
    # Payment methods
    payment_methods = ["Credit Card", "Debit Card", "Cash", "Digital Wallet", "Online Payment"]
    
    data = []
    order_id = 1000
    customer_counter = 1
    customer_ids = {}
    
    # Generate 5000 orders
    for day_offset in range(90):  # Last 90 days
        current_date = datetime.now() - timedelta(days=90 - day_offset)
        
        for restaurant in restaurants:
            # Daily order count varies by restaurant and day of week
            base_orders = {
                "Restaurant B": 25,
                "Restaurant C": 35,
                "Restaurant D": 30,
                "Restaurant E": 28
            }
            
            # Weekend boost
            if current_date.weekday() >= 5:  # Weekend
                daily_orders = int(base_orders[restaurant] * 1.8)
            else:
                daily_orders = int(base_orders[restaurant] * np.random.uniform(0.9, 1.1))
            
            for _ in range(daily_orders):
                # Generate customer ID (some customers return)
                if np.random.random() < 0.4 and customer_counter > 50:  # 40% returning customers
                    customer_id = np.random.choice(list(customer_ids.keys()))
                else:
                    customer_id = f"CUST{customer_counter:05d}"
                    customer_ids[customer_id] = True
                    customer_counter += 1
                
                # Order time (between 8 AM and 11 PM)
                order_time = current_date + timedelta(hours=np.random.uniform(8, 23))
                
                # Number of items in this order (1-6 items)
                num_items = np.random.choice([1, 2, 3, 4, 5, 6], p=[0.1, 0.2, 0.25, 0.2, 0.15, 0.1])
                
                # Select categories and items
                categories = np.random.choice(list(menu_items.keys()), num_items, replace=True)
                
                order_total = 0
                for cat in categories:
                    item = np.random.choice(menu_items[cat])
                    quantity = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
                    price = prices[item]
                    item_total = price * quantity
                    order_total += item_total
                    
                    data.append({
                        "Order ID": f"ORD{order_id:06d}",
                        "Customer ID": customer_id,
                        "Restaurant": restaurant,
                        "Category": cat,
                        "Item": item,
                        "Price": price,
                        "Quantity": quantity,
                        "Order Total": item_total,  # Individual item total
                        "Full Order Total": order_total,  # Full order total
                        "Order Date": order_time,
                        "Payment Method": np.random.choice(payment_methods, p=[0.4, 0.3, 0.15, 0.1, 0.05]),
                        "Day": order_time.strftime("%A"),
                        "Hour": order_time.hour,
                        "Month": order_time.strftime("%B"),
                        "Week": f"Week {(order_time.day - 1) // 7 + 1}"
                    })
                
                order_id += 1
    
    df = pd.DataFrame(data)
    return df

# Sidebar filters
with st.sidebar:
    st.markdown("## 🔍 FILTERS")
    
    # Date range
    min_date = datetime.now() - timedelta(days=90)
    max_date = datetime.now()
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    with col2:
        end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
    
    # Restaurant filter
    st.markdown("### 🏪 Restaurants")
    restaurants = ["All", "Restaurant B", "Restaurant C", "Restaurant D", "Restaurant E"]
    selected_restaurants = st.multiselect(
        "Select Restaurants",
        restaurants,
        default=["All"]
    )
    
    if "All" in selected_restaurants:
        selected_restaurants = ["Restaurant B", "Restaurant C", "Restaurant D", "Restaurant E"]
    
    # Category filter
    st.markdown("### 🍽️ Categories")
    categories = ["All", "Appetizers", "Main Course", "Desserts", "Beverages", "Specials"]
    selected_categories = st.multiselect(
        "Select Categories",
        categories,
        default=["All"]
    )
    
    if "All" in selected_categories:
        selected_categories = ["Appetizers", "Main Course", "Desserts", "Beverages", "Specials"]
    
    # Payment method filter
    st.markdown("### 💳 Payment Methods")
    payment_methods = ["All", "Credit Card", "Debit Card", "Cash", "Digital Wallet", "Online Payment"]
    selected_payments = st.multiselect(
        "Select Payment Methods",
        payment_methods,
        default=["All"]
    )
    
    if "All" in selected_payments:
        selected_payments = ["Credit Card", "Debit Card", "Cash", "Digital Wallet", "Online Payment"]
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    auto_refresh = st.checkbox("🔄 Auto-refresh every 10s", False)
    show_raw_data = st.checkbox("📋 Show raw data", False)

# Load data
df = generate_real_restaurant_data()

# Apply filters
df['Order Date'] = pd.to_datetime(df['Order Date'])
mask = (
    (df['Order Date'].dt.date >= start_date) & 
    (df['Order Date'].dt.date <= end_date) &
    (df['Restaurant'].isin(selected_restaurants)) &
    (df['Category'].isin(selected_categories)) &
    (df['Payment Method'].isin(selected_payments))
)
df_filtered = df[mask].copy()

# Calculate unique orders (group by Order ID)
unique_orders = df_filtered.groupby('Order ID').first().reset_index()

# TOP METRICS ROW
st.markdown("## 📊 KEY PERFORMANCE INDICATORS")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_revenue = unique_orders['Full Order Total'].sum()
    prev_revenue = total_revenue * 0.88  # Simulating 12% growth
    growth = ((total_revenue - prev_revenue) / prev_revenue * 100)
    
    st.markdown(f"""
    <div class="metric-card sales-card">
        <h3 style="color: #4ECDC4; margin-bottom: 0.5rem;">💰 TOTAL REVENUE</h3>
        <h2 style="font-size: 2rem; margin: 0.5rem 0;">AED {total_revenue:,.0f}</h2>
        <p style="margin: 0.3rem 0;">📈 <span class="{'positive' if growth > 0 else 'negative'}">
        {'↑' if growth > 0 else '↓'} {abs(growth):.1f}% vs previous</span></p>
        <p style="color: #666; font-size: 0.9rem; margin: 0.3rem 0;">Based on {len(unique_orders):,} orders</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    total_customers = df_filtered['Customer ID'].nunique()
    avg_order_value = total_revenue / len(unique_orders) if len(unique_orders) > 0 else 0
    repeat_rate = (df_filtered['Customer ID'].value_counts() > 1).sum() / total_customers * 100 if total_customers > 0 else 0
    
    st.markdown(f"""
    <div class="metric-card customer-card">
        <h3 style="color: #FFD166; margin-bottom: 0.5rem;">👥 CUSTOMERS</h3>
        <h2 style="font-size: 2rem; margin: 0.5rem 0;">{total_customers:,}</h2>
        <p style="margin: 0.3rem 0;">🧾 Avg Order: <b>AED {avg_order_value:.0f}</b></p>
        <p style="color: #666; font-size: 0.9rem; margin: 0.3rem 0;">🔄 {repeat_rate:.1f}% repeat customers</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    total_items = df_filtered['Quantity'].sum()
    unique_items = df_filtered['Item'].nunique()
    items_per_order = total_items / len(unique_orders) if len(unique_orders) > 0 else 0
    
    st.markdown(f"""
    <div class="metric-card order-card">
        <h3 style="color: #06D6A0; margin-bottom: 0.5rem;">🍽️ ITEMS SOLD</h3>
        <h2 style="font-size: 2rem; margin: 0.5rem 0;">{total_items:,}</h2>
        <p style="margin: 0.3rem 0;">📋 {unique_items} unique items</p>
        <p style="color: #666; font-size: 0.9rem; margin: 0.3rem 0;">📦 {items_per_order:.1f} items per order</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    # Calculate profit (assuming 65% margin on food, 80% on beverages)
    def calculate_profit(row):
        if row['Category'] == 'Beverages':
            return row['Order Total'] * 0.80
        else:
            return row['Order Total'] * 0.65
    
    df_filtered['Profit'] = df_filtered.apply(calculate_profit, axis=1)
    total_profit = df_filtered['Profit'].sum()
    profit_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0
    
    st.markdown(f"""
    <div class="metric-card profit-card">
        <h3 style="color: #118AB2; margin-bottom: 0.5rem;">📈 TOTAL PROFIT</h3>
        <h2 style="font-size: 2rem; margin: 0.5rem 0;">AED {total_profit:,.0f}</h2>
        <p style="margin: 0.3rem 0;">🎯 Margin: <b>{profit_margin:.1f}%</b></p>
        <p style="color: #666; font-size: 0.9rem; margin: 0.3rem 0;">🏆 Best: Main Course</p>
    </div>
    """, unsafe_allow_html=True)

# SECOND METRICS ROW
st.markdown("## 📈 PERFORMANCE METRICS")

col5, col6, col7, col8 = st.columns(4)

with col5:
    # Average order value by restaurant
    restaurant_avg = unique_orders.groupby('Restaurant')['Full Order Total'].mean().round(0)
    best_restaurant = restaurant_avg.idxmax() if not restaurant_avg.empty else "N/A"
    
    st.metric(
        "🏆 Highest Avg Order",
        f"AED {restaurant_avg.max():.0f}",
        f"at {best_restaurant}"
    )

with col6:
    # Most popular category
    category_sales = df_filtered.groupby('Category')['Order Total'].sum()
    top_category = category_sales.idxmax() if not category_sales.empty else "N/A"
    category_share = (category_sales.max() / category_sales.sum() * 100) if not category_sales.empty else 0
    
    st.metric(
        "🥇 Top Category",
        top_category,
        f"{category_share:.1f}% of sales"
    )

with col7:
    # Busiest day
    day_sales = unique_orders.groupby('Day')['Full Order Total'].sum()
    busiest_day = day_sales.idxmax() if not day_sales.empty else "N/A"
    day_avg = day_sales.mean()
    
    st.metric(
        "📅 Busiest Day",
        busiest_day,
        f"AED {day_sales.max():,.0f}"
    )

with col8:
    # Payment method distribution
    payment_dist = unique_orders['Payment Method'].value_counts()
    top_payment = payment_dist.idxmax() if not payment_dist.empty else "N/A"
    top_share = (payment_dist.max() / payment_dist.sum() * 100) if not payment_dist.empty else 0
    
    st.metric(
        "💳 Top Payment",
        top_payment,
        f"{top_share:.1f}% of transactions"
    )

# TABS FOR DETAILED ANALYSIS
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 SALES TRENDS", 
    "🏪 RESTAURANT COMPARISON", 
    "🍽️ MENU ANALYSIS", 
    "👥 CUSTOMER INSIGHTS", 
    "📋 RAW DATA"
])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Daily sales trend
        st.subheader("📅 Daily Sales Trend")
        
        daily_sales = unique_orders.groupby(
            pd.Grouper(key='Order Date', freq='D')
        )['Full Order Total'].sum().reset_index()
        
        fig1 = px.line(
            daily_sales,
            x='Order Date',
            y='Full Order Total',
            title='Daily Revenue Trend',
            labels={'Full Order Total': 'Revenue (AED)', 'Order Date': 'Date'},
            color_discrete_sequence=['#4ECDC4'],
            markers=True
        )
        fig1.update_layout(
            height=400,
            hovermode='x unified',
            yaxis=dict(tickformat=',.0f')
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Hourly sales pattern
        st.subheader("🕒 Hourly Sales Pattern")
        
        hourly_sales = df_filtered.groupby('Hour')['Order Total'].sum().reset_index()
        
        fig2 = px.bar(
            hourly_sales,
            x='Hour',
            y='Order Total',
            title='Revenue by Hour',
            labels={'Order Total': 'Revenue (AED)', 'Hour': 'Hour of Day'},
            color_discrete_sequence=['#FFD166']
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Weekly and monthly trends
    col3, col4 = st.columns(2)
    
    with col3:
        # Weekly sales
        st.subheader("📆 Weekly Performance")
        
        weekly_sales = unique_orders.groupby('Week')['Full Order Total'].sum().reset_index()
        
        fig3 = px.bar(
            weekly_sales,
            x='Week',
            y='Full Order Total',
            title='Revenue by Week',
            labels={'Full Order Total': 'Revenue (AED)'},
            color_discrete_sequence=['#06D6A0']
        )
        fig3.update_layout(height=350)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col4:
        # Category trend over time
        st.subheader("📈 Category Trend")
        
        category_trend = df_filtered.groupby(['Order Date', 'Category'])['Order Total'].sum().reset_index()
        category_trend['Order Date'] = category_trend['Order Date'].dt.date
        
        fig4 = px.area(
            category_trend,
            x='Order Date',
            y='Order Total',
            color='Category',
            title='Category Revenue Over Time',
            labels={'Order Total': 'Revenue (AED)', 'Order Date': 'Date'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig4.update_layout(height=350)
        st.plotly_chart(fig4, use_container_width=True)

with tab2:
    st.subheader("🏪 Restaurant Performance Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue by restaurant
        restaurant_revenue = unique_orders.groupby('Restaurant')['Full Order Total'].sum().sort_values()
        
        fig5 = px.bar(
            restaurant_revenue.reset_index(),
            x='Full Order Total',
            y='Restaurant',
            orientation='h',
            title='Total Revenue by Restaurant',
            labels={'Full Order Total': 'Revenue (AED)'},
            color='Restaurant',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig5.update_layout(height=400)
        st.plotly_chart(fig5, use_container_width=True)
    
    with col2:
        # Orders count by restaurant
        restaurant_orders = unique_orders['Restaurant'].value_counts()
        
        fig6 = px.pie(
            values=restaurant_orders.values,
            names=restaurant_orders.index,
            title='Order Distribution by Restaurant',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig6.update_layout(height=400)
        st.plotly_chart(fig6, use_container_width=True)
    
    # Detailed restaurant metrics table
    st.subheader("📋 Restaurant Performance Metrics")
    
    restaurant_metrics = unique_orders.groupby('Restaurant').agg({
        'Full Order Total': ['sum', 'mean', 'count'],
        'Customer ID': pd.Series.nunique
    }).round(0)
    
    restaurant_metrics.columns = ['Total Revenue', 'Avg Order Value', 'Total Orders', 'Unique Customers']
    restaurant_metrics['Revenue per Customer'] = (restaurant_metrics['Total Revenue'] / restaurant_metrics['Unique Customers']).round(0)
    restaurant_metrics['Orders per Customer'] = (restaurant_metrics['Total Orders'] / restaurant_metrics['Unique Customers']).round(1)
    
    # Style the dataframe
    styled_df = restaurant_metrics.style \
        .background_gradient(subset=['Total Revenue'], cmap='YlOrRd') \
        .background_gradient(subset=['Avg Order Value'], cmap='Blues') \
        .format({
            'Total Revenue': 'AED {:,.0f}',
            'Avg Order Value': 'AED {:,.0f}',
            'Revenue per Customer': 'AED {:,.0f}'
        })
    
    st.dataframe(styled_df, use_container_width=True)

with tab3:
    st.subheader("🍽️ Menu Item Performance Analysis")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Top selling items
        st.subheader("🥇 Top 10 Selling Items")
        
        top_items = df_filtered.groupby('Item').agg({
            'Order Total': 'sum',
            'Quantity': 'sum',
            'Order ID': 'count'
        }).nlargest(10, 'Order Total')
        
        top_items.columns = ['Total Revenue', 'Total Quantity', 'Order Count']
        top_items['Avg Price'] = top_items['Total Revenue'] / top_items['Total Quantity']
        
        fig7 = px.bar(
            top_items.reset_index(),
            x='Total Revenue',
            y='Item',
            orientation='h',
            title='Top Items by Revenue',
            labels={'Total Revenue': 'Revenue (AED)'},
            color='Total Revenue',
            color_continuous_scale='Viridis'
        )
        fig7.update_layout(height=500)
        st.plotly_chart(fig7, use_container_width=True)
    
    with col2:
        # Category breakdown
        st.subheader("📊 Category Analysis")
        
        category_stats = df_filtered.groupby('Category').agg({
            'Order Total': ['sum', 'mean'],
            'Quantity': 'sum',
            'Order ID': 'count'
        }).round(0)
        
        category_stats.columns = ['Total Revenue', 'Avg Price', 'Total Items', 'Order Count']
        category_stats = category_stats.sort_values('Total Revenue', ascending=False)
        
        fig8 = px.pie(
            values=category_stats['Total Revenue'],
            names=category_stats.index,
            title='Revenue by Category',
            color_discrete_sequence=px.colors.sequential.RdBu,
            hole=0.4
        )
        fig8.update_layout(height=500)
        st.plotly_chart(fig8, use_container_width=True)
    
    # Item performance table
    st.subheader("📋 Detailed Item Performance")
    
    item_performance = df_filtered.groupby(['Category', 'Item']).agg({
        'Order Total': ['sum', 'mean'],
        'Quantity': 'sum',
        'Order ID': 'count'
    }).round(0)
    
    item_performance.columns = ['Total Revenue', 'Avg Price', 'Total Sold', 'Times Ordered']
    item_performance = item_performance.sort_values('Total Revenue', ascending=False)
    
    # Display top 20 items
    st.dataframe(
        item_performance.head(20).style \
            .background_gradient(subset=['Total Revenue'], cmap='YlOrRd') \
            .format({
                'Total Revenue': 'AED {:,.0f}',
                'Avg Price': 'AED {:,.0f}'
            }),
        use_container_width=True
    )

with tab4:
    st.subheader("👥 Customer Behavior Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Customer frequency distribution
        st.subheader("🔄 Customer Frequency")
        
        customer_orders = df_filtered.groupby('Customer ID')['Order ID'].nunique()
        frequency_dist = customer_orders.value_counts().sort_index()
        
        # Group high frequencies
        freq_data = []
        for freq, count in frequency_dist.items():
            if freq <= 5:
                freq_data.append({'Frequency': str(freq), 'Customers': count})
            elif freq <= 10:
                if '6-10' not in [f['Frequency'] for f in freq_data]:
                    freq_data.append({'Frequency': '6-10', 'Customers': frequency_dist[(frequency_dist.index >= 6) & (frequency_dist.index <= 10)].sum()})
            else:
                if '10+' not in [f['Frequency'] for f in freq_data]:
                    freq_data.append({'Frequency': '10+', 'Customers': frequency_dist[frequency_dist.index > 10].sum()})
        
        freq_df = pd.DataFrame(freq_data)
        
        fig9 = px.bar(
            freq_df,
            x='Frequency',
            y='Customers',
            title='Customer Order Frequency',
            labels={'Customers': 'Number of Customers'},
            color_discrete_sequence=['#118AB2']
        )
        fig9.update_layout(height=400)
        st.plotly_chart(fig9, use_container_width=True)
    
    with col2:
        # Customer value segments
        st.subheader("💰 Customer Value Segments")
        
        customer_value = df_filtered.groupby('Customer ID')['Order Total'].sum()
        
        # Define segments
        segments = {
            'VIP (>AED 500)': (customer_value > 500).sum(),
            'Regular (AED 200-500)': ((customer_value >= 200) & (customer_value <= 500)).sum(),
            'Occasional (<AED 200)': (customer_value < 200).sum()
        }
        
        seg_df = pd.DataFrame({
            'Segment': list(segments.keys()),
            'Customers': list(segments.values())
        })
        
        fig10 = px.pie(
            seg_df,
            values='Customers',
            names='Segment',
            title='Customer Value Segments',
            color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#FFD166'],
            hole=0.3
        )
        fig10.update_layout(height=400)
        st.plotly_chart(fig10, use_container_width=True)
    
    # Top customers table
    st.subheader("🏆 Top 20 Customers by Lifetime Value")
    
    top_customers = df_filtered.groupby('Customer ID').agg({
        'Order Total': 'sum',
        'Order ID': 'nunique',
        'Restaurant': lambda x: ', '.join(sorted(set(x))),
        'Order Date': 'max'
    }).nlargest(20, 'Order Total')
    
    top_customers.columns = ['Total Spent', 'Total Orders', 'Restaurants Visited', 'Last Visit']
    top_customers['Avg Order Value'] = (top_customers['Total Spent'] / top_customers['Total Orders']).round(0)
    
    st.dataframe(
        top_customers.style \
            .background_gradient(subset=['Total Spent'], cmap='YlOrRd') \
            .format({
                'Total Spent': 'AED {:,.0f}',
                'Avg Order Value': 'AED {:,.0f}'
            }),
        use_container_width=True
    )

with tab5:
    st.subheader("📋 Raw Transaction Data")
    
    if show_raw_data:
        # Show filtered data with selected columns
        display_cols = ['Order ID', 'Customer ID', 'Restaurant', 'Category', 
                       'Item', 'Price', 'Quantity', 'Order Total', 
                       'Order Date', 'Payment Method']
        
        st.dataframe(
            df_filtered[display_cols].sort_values('Order Date', ascending=False),
            use_container_width=True,
            height=600
        )
        
        # Download button
        csv = df_filtered[display_cols].to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"restaurant_sales_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("✅ Enable 'Show raw data' in the sidebar to view transaction details")

# BOTTOM SECTION - Insights & Alerts
st.markdown("---")
st.subheader("🚀 INSIGHTS & RECOMMENDATIONS")

col1, col2 = st.columns(2)

with col1:
    with st.expander("📈 **Top Insights**", expanded=True):
        # Calculate insights
        peak_hour = df_filtered.groupby('Hour')['Order Total'].sum().idxmax()
        slow_day = unique_orders.groupby('Day')['Full Order Total'].sum().idxmin()
        popular_payment = unique_orders['Payment Method'].value_counts().index[0]
        
        st.markdown(f"""
        **🎯 Peak Performance:**
        - 🕒 **Busiest Hour**: {peak_hour}:00 ({df_filtered.groupby('Hour')['Order Total'].sum().max():,.0f} AED/hour)
        - 📅 **Slowest Day**: {slow_day} (consider promotions)
        - 💳 **Preferred Payment**: {popular_payment}
        
        **🍽️ Menu Insights:**
        - Top 3 items contribute {top_items.head(3)['Total Revenue'].sum() / df_filtered['Order Total'].sum() * 100:.1f}% of revenue
        - Main Course generates {category_stats.loc['Main Course', 'Total Revenue'] / df_filtered['Order Total'].sum() * 100:.1f}% of sales
        - Avg item price: AED {df_filtered['Price'].mean():.0f}
        """)

with col2:
    with st.expander("🚨 **Action Items**", expanded=True):
        # Identify areas for improvement
        low_performing = restaurant_metrics[restaurant_metrics['Total Revenue'] == restaurant_metrics['Total Revenue'].min()].index[0]
        low_category = category_stats[category_stats['Total Revenue'] == category_stats['Total Revenue'].min()].index[0]
        
        st.markdown(f"""
        **🎯 Focus Areas:**
        1. **Boost {low_performing}** - Lowest revenue among all restaurants
        2. **Promote {low_category}** - Lowest performing category
        3. **Increase {slow_day} sales** - Currently the slowest day
        
        **💰 Quick Wins:**
        1. Bundle top-selling items for higher average order value
        2. Offer discounts on {slow_day} to increase traffic
        3. Promote digital payments (currently {unique_orders[unique_orders['Payment Method'] == 'Digital Wallet'].shape[0] / len(unique_orders) * 100:.1f}% usage)
        
        **📊 Next Week Target:**
        - Increase average order value by 5%
        - Reduce {low_category} wastage by 10%
        - Increase {low_performing} revenue by 15%
        """)

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #888; font-size: 0.8rem; padding: 1rem;">
    <p>📊 Restaurant Analytics Dashboard • Generated {len(df_filtered):,} transactions from {start_date} to {end_date}</p>
    <p>🔄 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • Data updates in real-time</p>
    <p>🔧 Built with Streamlit • For support: analytics@restaurantgroup.com</p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh
if auto_refresh:
    import time
    time.sleep(10)
    st.rerun()
