import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import calendar
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page configuration for professional appearance
st.set_page_config(
    page_title="Restaurant Sales Intelligence Dashboard",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for professional styling
def apply_custom_css():
    st.markdown("""
    <style>
    /* Main styling */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 10px;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        transition: transform 0.3s ease;
        border-left: 5px solid;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
    }
    
    .metric-title {
        font-size: 0.9rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 0.5rem 0;
    }
    
    .metric-change {
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2c3e50 !important;
        color: white !important;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #4ECDC4, #45B7D1);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 14px rgba(0,0,0,0.1);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    }
    
    /* Custom section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2c3e50;
        border-bottom: 3px solid #4ECDC4;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        color: #6c757d;
        font-size: 0.9rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #eaeaea;
    }
    </style>
    """, unsafe_allow_html=True)

# Generate sample data if no dataset is uploaded
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    
    # Generate dates
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 1, 31)
    date_range = pd.date_range(start_date, end_date, freq='D')
    
    # Generate order IDs
    order_ids = ['ORD' + str(i).zfill(6) for i in range(1, 17535)]
    
    # Generate customer IDs
    customer_ids = ['CUST' + str(i).zfill(5) for i in np.random.randint(1, 1500, 17534)]
    
    # Categories and items
    categories = ['Appetizer', 'Main Course', 'Dessert', 'Beverage', 'Side Dish']
    items_by_category = {
        'Appetizer': ['Garlic Bread', 'Bruschetta', 'Nachos', 'Spring Rolls', 'Soup of the Day'],
        'Main Course': ['Grilled Salmon', 'Filet Mignon', 'Pasta Carbonara', 'Chicken Alfredo', 'Veggie Burger'],
        'Dessert': ['Chocolate Lava Cake', 'Cheesecake', 'Tiramisu', 'Ice Cream Sundae', 'Fruit Tart'],
        'Beverage': ['Craft Beer', 'Wine', 'Cocktail', 'Soft Drink', 'Coffee'],
        'Side Dish': ['French Fries', 'Garden Salad', 'Mashed Potatoes', 'Grilled Vegetables', 'Rice Pilaf']
    }
    
    # Generate data
    data = []
    for i in range(17534):
        category = np.random.choice(categories, p=[0.15, 0.35, 0.2, 0.2, 0.1])
        item = np.random.choice(items_by_category[category])
        
        # Price based on category
        if category == 'Appetizer':
            price = np.random.uniform(5, 15)
        elif category == 'Main Course':
            price = np.random.uniform(15, 45)
        elif category == 'Dessert':
            price = np.random.uniform(6, 18)
        elif category == 'Beverage':
            price = np.random.uniform(3, 12)
        else:  # Side Dish
            price = np.random.uniform(4, 10)
        
        # Quantity
        quantity = np.random.choice([1, 2, 3, 4], p=[0.5, 0.3, 0.15, 0.05])
        
        # Order total
        order_total = price * quantity
        
        # Order date
        order_date = np.random.choice(date_range)
        
        # Payment method
        payment_method = np.random.choice(['Credit Card', 'Debit Card', 'Cash', 'Digital Wallet'], 
                                         p=[0.5, 0.3, 0.15, 0.05])
        
        data.append([
            order_ids[i], customer_ids[i], category, item, round(price, 2), 
            quantity, round(order_total, 2), order_date, payment_method
        ])
    
    df = pd.DataFrame(data, columns=[
        'Order ID', 'Customer ID', 'Category', 'Item', 'Price', 
        'Quantity', 'Order Total', 'Order Date', 'Payment Method'
    ])
    
    return df

# Load data
@st.cache_data
def load_data():
    df = generate_sample_data()
    # Convert Order Date to datetime
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    # Extract additional date features
    df['Order Month'] = df['Order Date'].dt.month
    df['Order Day'] = df['Order Date'].dt.day
    df['Order Weekday'] = df['Order Date'].dt.day_name()
    df['Order Hour'] = df['Order Date'].dt.hour
    df['Order Year'] = df['Order Date'].dt.year
    
    # Calculate derived metrics
    df['Revenue'] = df['Order Total']
    return df

# Main dashboard
def main():
    # Apply custom CSS
    apply_custom_css()
    
    # Header section
    st.markdown('<h1 class="main-header">🍽️ Restaurant Sales Intelligence Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Analytics for Strategic Decision Making | Dataset: 17,534 Transactions</p>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    # Create sidebar filters
    with st.sidebar:
        st.markdown("### 🔍 **Filters & Controls**")
        
        # Date range filter
        min_date = df['Order Date'].min().date()
        max_date = df['Order Date'].max().date()
        date_range = st.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = df[(df['Order Date'].dt.date >= start_date) & (df['Order Date'].dt.date <= end_date)]
        else:
            filtered_df = df.copy()
        
        # Category filter
        categories = st.multiselect(
            "Select Categories",
            options=sorted(df['Category'].unique()),
            default=sorted(df['Category'].unique())
        )
        
        if categories:
            filtered_df = filtered_df[filtered_df['Category'].isin(categories)]
        
        # Payment method filter
        payment_methods = st.multiselect(
            "Select Payment Methods",
            options=sorted(df['Payment Method'].unique()),
            default=sorted(df['Payment Method'].unique())
        )
        
        if payment_methods:
            filtered_df = filtered_df[filtered_df['Payment Method'].isin(payment_methods)]
        
        # Add a reset button
        if st.button("🔄 Reset All Filters", use_container_width=True):
            st.rerun()
        
        # Show data summary in sidebar
        st.markdown("---")
        st.markdown("### 📊 **Data Summary**")
        st.metric("Total Orders", f"{len(filtered_df):,}")
        st.metric("Unique Customers", f"{filtered_df['Customer ID'].nunique():,}")
        st.metric("Total Revenue", f"${filtered_df['Revenue'].sum():,.2f}")
        st.metric("Avg Order Value", f"${filtered_df['Revenue'].mean():,.2f}")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Overview", "📊 Category Analysis", "👥 Customer Insights", "⏰ Temporal Analysis", "🔍 Raw Data"])
    
    with tab1:
        # Key Metrics Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_revenue = filtered_df['Revenue'].sum()
            avg_revenue = df['Revenue'].mean()
            change_pct = ((total_revenue / len(filtered_df)) / avg_revenue - 1) * 100 if len(filtered_df) > 0 else 0
            
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #4ECDC4;">
                <div class="metric-title">Total Revenue</div>
                <div class="metric-value">${total_revenue:,.2f}</div>
                <div class="metric-change" style="color: {'#2ecc71' if change_pct >= 0 else '#e74c3c'}">
                    {change_pct:+.1f}% vs average
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            total_orders = len(filtered_df)
            avg_orders = len(df) / (df['Order Date'].max() - df['Order Date'].min()).days * (end_date - start_date).days if len(date_range) == 2 else len(df)
            change_pct = (total_orders / avg_orders - 1) * 100 if avg_orders > 0 else 0
            
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #45B7D1;">
                <div class="metric-title">Total Orders</div>
                <div class="metric-value">{total_orders:,}</div>
                <div class="metric-change" style="color: {'#2ecc71' if change_pct >= 0 else '#e74c3c'}">
                    {change_pct:+.1f}% vs expected
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_order_value = filtered_df['Revenue'].mean() if len(filtered_df) > 0 else 0
            overall_avg = df['Revenue'].mean()
            change_pct = ((avg_order_value / overall_avg) - 1) * 100 if overall_avg > 0 else 0
            
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #FF6B6B;">
                <div class="metric-title">Avg Order Value</div>
                <div class="metric-value">${avg_order_value:,.2f}</div>
                <div class="metric-change" style="color: {'#2ecc71' if change_pct >= 0 else '#e74c3c'}">
                    {change_pct:+.1f}% vs overall
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            top_category = filtered_df['Category'].mode()[0] if len(filtered_df) > 0 else 'N/A'
            top_category_rev = filtered_df[filtered_df['Category'] == top_category]['Revenue'].sum() if len(filtered_df) > 0 else 0
            cat_pct = (top_category_rev / total_revenue * 100) if total_revenue > 0 else 0
            
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #FFD166;">
                <div class="metric-title">Top Category</div>
                <div class="metric-value">{top_category}</div>
                <div class="metric-change">{cat_pct:.1f}% of revenue</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Revenue Trend Chart
        st.markdown('<div class="section-header">Revenue Trend Over Time</div>', unsafe_allow_html=True)
        
        # Create time series data
        if len(filtered_df) > 0:
            daily_revenue = filtered_df.resample('D', on='Order Date')['Revenue'].sum().reset_index()
            weekly_revenue = filtered_df.resample('W', on='Order Date')['Revenue'].sum().reset_index()
            monthly_revenue = filtered_df.resample('M', on='Order Date')['Revenue'].sum().reset_index()
            
            # Create interactive chart with subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Daily Revenue', 'Weekly Revenue', 'Monthly Revenue', 'Cumulative Revenue'),
                specs=[[{"secondary_y": True}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Daily revenue
            fig.add_trace(
                go.Scatter(x=daily_revenue['Order Date'], y=daily_revenue['Revenue'],
                          mode='lines+markers', name='Daily Revenue',
                          line=dict(color='#4ECDC4', width=3),
                          marker=dict(size=6)),
                row=1, col=1
            )
            
            # 7-day moving average
            daily_revenue['MA7'] = daily_revenue['Revenue'].rolling(window=7, min_periods=1).mean()
            fig.add_trace(
                go.Scatter(x=daily_revenue['Order Date'], y=daily_revenue['MA7'],
                          mode='lines', name='7-Day Moving Avg',
                          line=dict(color='#FF6B6B', width=2, dash='dash')),
                row=1, col=1, secondary_y=False
            )
            
            # Weekly revenue (bar chart)
            fig.add_trace(
                go.Bar(x=weekly_revenue['Order Date'], y=weekly_revenue['Revenue'],
                      name='Weekly Revenue', marker_color='#45B7D1'),
                row=1, col=2
            )
            
            # Monthly revenue (bar chart)
            fig.add_trace(
                go.Bar(x=monthly_revenue['Order Date'], y=monthly_revenue['Revenue'],
                      name='Monthly Revenue', marker_color='#FFD166'),
                row=2, col=1
            )
            
            # Cumulative revenue
            filtered_df_sorted = filtered_df.sort_values('Order Date')
            filtered_df_sorted['Cumulative Revenue'] = filtered_df_sorted['Revenue'].cumsum()
            cumulative_daily = filtered_df_sorted.resample('D', on='Order Date')['Cumulative Revenue'].last().reset_index()
            
            fig.add_trace(
                go.Scatter(x=cumulative_daily['Order Date'], y=cumulative_daily['Cumulative Revenue'],
                          mode='lines', name='Cumulative Revenue',
                          line=dict(color='#2c3e50', width=4),
                          fill='tozeroy', fillcolor='rgba(44, 62, 80, 0.1)'),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                height=700,
                showlegend=True,
                plot_bgcolor='rgba(248, 249, 250, 0.5)',
                paper_bgcolor='rgba(255, 255, 255, 0.1)',
                font=dict(family="Arial, sans-serif"),
                hovermode='x unified'
            )
            
            # Update axes
            fig.update_xaxes(title_text="Date", row=1, col=1)
            fig.update_xaxes(title_text="Week", row=1, col=2)
            fig.update_xaxes(title_text="Month", row=2, col=1)
            fig.update_xaxes(title_text="Date", row=2, col=2)
            
            fig.update_yaxes(title_text="Revenue ($)", row=1, col=1)
            fig.update_yaxes(title_text="Revenue ($)", row=1, col=2)
            fig.update_yaxes(title_text="Revenue ($)", row=2, col=1)
            fig.update_yaxes(title_text="Cumulative Revenue ($)", row=2, col=2)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Top performing items
        st.markdown('<div class="section-header">Top Performing Menu Items</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top items by revenue
            top_items_rev = filtered_df.groupby('Item').agg({
                'Revenue': 'sum',
                'Order ID': 'count',
                'Price': 'mean'
            }).rename(columns={'Order ID': 'Orders'}).nlargest(10, 'Revenue').reset_index()
            
            fig = px.bar(top_items_rev, x='Revenue', y='Item', 
                        orientation='h',
                        title='Top 10 Items by Revenue',
                        color='Revenue',
                        color_continuous_scale='Viridis')
            
            fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top items by quantity sold
            top_items_qty = filtered_df.groupby('Item').agg({
                'Quantity': 'sum',
                'Revenue': 'sum'
            }).nlargest(10, 'Quantity').reset_index()
            
            fig = px.bar(top_items_qty, x='Quantity', y='Item', 
                        orientation='h',
                        title='Top 10 Items by Quantity Sold',
                        color='Revenue',
                        color_continuous_scale='Plasma')
            
            fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown('<div class="section-header">Category Performance Analysis</div>', unsafe_allow_html=True)
        
        # Category metrics
        category_stats = filtered_df.groupby('Category').agg({
            'Revenue': 'sum',
            'Order ID': 'count',
            'Quantity': 'sum',
            'Price': 'mean'
        }).rename(columns={'Order ID': 'Orders'}).reset_index()
        
        # Create 4 columns for category metrics
        cat_cols = st.columns(4)
        for idx, (_, row) in enumerate(category_stats.iterrows()):
            with cat_cols[idx % 4]:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: {['#4ECDC4', '#45B7D1', '#FF6B6B', '#FFD166'][idx % 4]};">
                    <div class="metric-title">{row['Category']}</div>
                    <div class="metric-value">${row['Revenue']:,.0f}</div>
                    <div class="metric-change">{row['Orders']:,} orders</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Category visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue by category (pie chart)
            fig = px.pie(category_stats, values='Revenue', names='Category',
                        title='Revenue Distribution by Category',
                        color_discrete_sequence=px.colors.qualitative.Set3,
                        hole=0.4)
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Orders by category (bar chart)
            fig = px.bar(category_stats, x='Category', y='Orders',
                        title='Number of Orders by Category',
                        color='Revenue',
                        color_continuous_scale='Teal')
            
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
        
        # Item performance within categories
        st.markdown('<div class="section-header">Item Performance Within Categories</div>', unsafe_allow_html=True)
        
        selected_category = st.selectbox("Select Category to Analyze Items", category_stats['Category'].tolist())
        
        category_items = filtered_df[filtered_df['Category'] == selected_category].groupby('Item').agg({
            'Revenue': 'sum',
            'Order ID': 'count',
            'Quantity': 'sum',
            'Price': 'mean'
        }).rename(columns={'Order ID': 'Orders'}).reset_index()
        
        # Display item performance
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(category_items.sort_values('Revenue', ascending=False).head(10),
                        x='Item', y=['Revenue', 'Orders'],
                        title=f'Top Items in {selected_category} Category',
                        barmode='group')
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(
                category_items.sort_values('Revenue', ascending=False).head(10)[
                    ['Item', 'Revenue', 'Orders', 'Quantity', 'Price']
                ].style.format({
                    'Revenue': '${:,.2f}',
                    'Price': '${:,.2f}'
                }).background_gradient(subset=['Revenue'], cmap='Greens'),
                use_container_width=True
            )
    
    with tab3:
        st.markdown('<div class="section-header">Customer Insights & Segmentation</div>', unsafe_allow_html=True)
        
        # Customer metrics
        customer_stats = filtered_df.groupby('Customer ID').agg({
            'Revenue': 'sum',
            'Order ID': 'count',
            'Quantity': 'sum',
            'Order Date': ['min', 'max']
        }).reset_index()
        
        # Flatten multi-index columns
        customer_stats.columns = ['Customer ID', 'Total Revenue', 'Total Orders', 'Total Quantity', 'First Order', 'Last Order']
        
        # Calculate customer lifetime
        customer_stats['Customer Lifetime'] = (customer_stats['Last Order'] - customer_stats['First Order']).dt.days + 1
        customer_stats['Avg Order Value'] = customer_stats['Total Revenue'] / customer_stats['Total Orders']
        
        # RFM Analysis
        current_date = filtered_df['Order Date'].max()
        
        rfm = filtered_df.groupby('Customer ID').agg({
            'Order Date': lambda x: (current_date - x.max()).days,  # Recency
            'Order ID': 'count',  # Frequency
            'Revenue': 'sum'  # Monetary
        }).reset_index()
        
        rfm.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']
        
        # Create RFM segments
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1])
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4])
        
        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
        
        # Define segments
        seg_map = {
            r'111|112|121|131|141|151': 'Lost Customers',
            r'332|322|233|232|223|222|132|123|122|212|211': 'Hibernating Customers',
            r'133|134|143|244|334|343|344|144': 'At Risk Customers',
            r'311|411|331': 'New Customers',
            r'323|333|321|422|332|432': 'Potential Loyalists',
            r'433|434|443|444': 'Loyal Customers',
            r'414|424|425|435|445|455': 'Champions'
        }
        
        rfm['Segment'] = rfm['RFM_Score'].replace(seg_map, regex=True)
        rfm['Segment'] = rfm['Segment'].fillna('Others')
        
        # Display customer segmentation
        col1, col2 = st.columns(2)
        
        with col1:
            # Customer segmentation pie chart
            segment_counts = rfm['Segment'].value_counts().reset_index()
            segment_counts.columns = ['Segment', 'Count']
            
            fig = px.pie(segment_counts, values='Count', names='Segment',
                        title='Customer Segmentation',
                        color_discrete_sequence=px.colors.qualitative.Set2,
                        hole=0.4)
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Customer metrics summary
            st.metric("Total Customers", f"{len(customer_stats):,}")
            st.metric("Avg Orders per Customer", f"{customer_stats['Total Orders'].mean():.1f}")
            st.metric("Avg Customer Lifetime (days)", f"{customer_stats['Customer Lifetime'].mean():.0f}")
            st.metric("Customer Retention Rate", 
                     f"{(len(customer_stats[customer_stats['Total Orders'] > 1]) / len(customer_stats) * 100):.1f}%")
            
            # Top customers table
            st.markdown("### 🏆 Top 5 Customers")
            top_customers = customer_stats.sort_values('Total Revenue', ascending=False).head(5)
            st.dataframe(
                top_customers[['Customer ID', 'Total Revenue', 'Total Orders', 'Avg Order Value']].style.format({
                    'Total Revenue': '${:,.2f}',
                    'Avg Order Value': '${:,.2f}'
                }),
                use_container_width=True
            )
    
    with tab4:
        st.markdown('<div class="section-header">Temporal & Seasonal Analysis</div>', unsafe_allow_html=True)
        
        # Time-based analysis
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Revenue by weekday
            weekday_rev = filtered_df.groupby('Order Weekday')['Revenue'].sum().reindex(
                ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            ).reset_index()
            
            fig = px.bar(weekday_rev, x='Order Weekday', y='Revenue',
                        title='Revenue by Day of Week',
                        color='Revenue',
                        color_continuous_scale='Viridis')
            
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Revenue by hour of day
            if 'Order Hour' in filtered_df.columns:
                hour_rev = filtered_df.groupby('Order Hour')['Revenue'].sum().reset_index()
                
                fig = px.line(hour_rev, x='Order Hour', y='Revenue',
                            title='Revenue by Hour of Day',
                            markers=True)
                
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Revenue by month
            month_rev = filtered_df.groupby('Order Month')['Revenue'].sum().reset_index()
            month_rev['Order Month'] = month_rev['Order Month'].apply(lambda x: calendar.month_abbr[x])
            
            fig = px.bar(month_rev, x='Order Month', y='Revenue',
                        title='Revenue by Month',
                        color='Revenue',
                        color_continuous_scale='Plasma')
            
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        # Payment method analysis
        st.markdown('<div class="section-header">Payment Method Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            payment_stats = filtered_df.groupby('Payment Method').agg({
                'Revenue': 'sum',
                'Order ID': 'count',
                'Price': 'mean'
            }).rename(columns={'Order ID': 'Orders'}).reset_index()
            
            fig = px.pie(payment_stats, values='Revenue', names='Payment Method',
                        title='Revenue by Payment Method',
                        color_discrete_sequence=px.colors.qualitative.Pastel)
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(payment_stats, x='Payment Method', y='Orders',
                        title='Number of Orders by Payment Method',
                        color='Revenue',
                        color_continuous_scale='Blues')
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.markdown('<div class="section-header">Raw Data & Advanced Analytics</div>', unsafe_allow_html=True)
        
        # Data summary
        st.markdown("### Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Total Rows", f"{len(filtered_df):,}")
        col2.metric("Total Columns", f"{len(filtered_df.columns):,}")
        col3.metric("Date Range", f"{filtered_df['Order Date'].min().date()} to {filtered_df['Order Date'].max().date()}")
        col4.metric("Memory Usage", f"{filtered_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Show raw data with filters
        st.markdown("### Filtered Data Preview")
        
        # Add column selection
        columns = st.multiselect(
            "Select columns to display",
            options=filtered_df.columns.tolist(),
            default=['Order Date', 'Category', 'Item', 'Price', 'Quantity', 'Order Total', 'Payment Method']
        )
        
        if columns:
            st.dataframe(
                filtered_df[columns].sort_values('Order Date', ascending=False).head(100).style.format({
                    'Price': '${:,.2f}',
                    'Order Total': '${:,.2f}'
                }).background_gradient(subset=['Order Total'], cmap='YlOrRd'),
                use_container_width=True,
                height=400
            )
        
        # Data statistics
        st.markdown("### Descriptive Statistics")
        st.dataframe(
            filtered_df.describe().style.format('{:,.2f}'),
            use_container_width=True
        )
        
        # Export option
        if st.button("📥 Export Filtered Data to CSV"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="restaurant_sales_filtered.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🍽️ Restaurant Sales Intelligence Dashboard | Created with Streamlit | Data Science Excellence</p>
        <p>This dashboard demonstrates world-class data visualization and analytics for restaurant sales data</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
