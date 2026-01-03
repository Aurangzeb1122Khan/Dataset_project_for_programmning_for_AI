import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==============================================
# 🎨 PROFESSIONAL DASHBOARD STYLING
# ==============================================
st.set_page_config(
    page_title="Restaurant Sales Intelligence",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

/* Main container */
.main {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

/* Cards */
.stMetric {
    background: rgba(255, 255, 255, 0.9) !important;
    backdrop-filter: blur(10px);
    border-radius: 15px !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    padding: 20px !important;
}

/* Headers */
h1, h2, h3 {
    color: #2D3748 !important;
    font-weight: 700 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(255, 255, 255, 0.8) !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-weight: 500 !important;
}

.stTabs [aria-selected="true"] {
    background: #667eea !important;
    color: white !important;
}

/* Dataframe styling */
.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2D3748 0%, #4A5568 100%);
}

/* Progress bars */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #667eea, #764ba2);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 600;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
}
</style>
""", unsafe_allow_html=True)

# ==============================================
# 📊 DATA LOADING AND PROCESSING
# ==============================================
@st.cache_data
def load_and_process_data():
    """Load and process restaurant sales data"""
    # Create sample data (replace with your actual data loading)
    data = {
        'Order ID': [f'ORD_{i:06d}' for i in range(1, 1001)],
        'Customer ID': [f'CUST_{np.random.randint(1, 101):03d}' for _ in range(1000)],
        'Category': np.random.choice(['Side Dishes', 'Main Dishes', 'Drinks', 'Desserts'], 1000),
        'Item': np.random.choice(['Side Salad', 'Mashed Potatoes', 'Grilled Chicken', 'Pasta Alfredo', 
                                  'Soft Drink', 'Ice Cream', 'Garlic Bread', 'Salmon'], 1000),
        'Price': np.random.uniform(3, 20, 1000).round(2),
        'Quantity': np.random.randint(1, 6, 1000),
        'Order Date': pd.date_range('2023-01-01', periods=1000, freq='D'),
        'Payment Method': np.random.choice(['Credit Card', 'Digital Wallet', 'Cash', 'Debit Card'], 1000)
    }
    
    df = pd.DataFrame(data)
    df['Order Total'] = df['Price'] * df['Quantity']
    
    # Introduce some missing values for analysis
    mask = np.random.random(len(df)) < 0.05
    df.loc[mask, 'Item'] = np.nan
    df.loc[np.random.random(len(df)) < 0.03, 'Price'] = np.nan
    df.loc[np.random.random(len(df)) < 0.02, 'Payment Method'] = np.nan
    
    return df

# ==============================================
# 🎯 ANALYTICS FUNCTIONS
# ==============================================
def data_quality_dashboard(df):
    """Display data quality metrics"""
    st.markdown("### 📊 Data Quality Assessment")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_records = len(df)
        st.metric("Total Records", f"{total_records:,}")
    
    with col2:
        missing_values = df.isnull().sum().sum()
        st.metric("Missing Values", missing_values, delta_color="inverse")
    
    with col3:
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Data Completeness", f"{completeness:.1f}%")
    
    with col4:
        duplicates = df.duplicated().sum()
        st.metric("Duplicates", duplicates, delta_color="inverse")
    
    # Missing values matrix
    st.markdown("#### 🔍 Missing Values Analysis")
    fig, ax = plt.subplots(figsize=(12, 6))
    msno.matrix(df, ax=ax, color=(0.2, 0.4, 0.6))
    st.pyplot(fig)

def outlier_analysis(df):
    """Perform outlier analysis"""
    st.markdown("### 📈 Outlier Detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price outliers
        Q1 = df['Price'].quantile(0.25)
        Q3 = df['Price'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        price_outliers = df[(df['Price'] < lower_bound) | (df['Price'] > upper_bound)]
        
        st.metric("Price Outliers", len(price_outliers))
        
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.boxplot(df['Price'].dropna())
        ax1.set_title('Price Distribution with Outliers')
        ax1.set_ylabel('Price ($)')
        st.pyplot(fig1)
    
    with col2:
        # Quantity outliers
        Q1_q = df['Quantity'].quantile(0.25)
        Q3_q = df['Quantity'].quantile(0.75)
        IQR_q = Q3_q - Q1_q
        lower_bound_q = Q1_q - 1.5 * IQR_q
        upper_bound_q = Q3_q + 1.5 * IQR_q
        qty_outliers = df[(df['Quantity'] < lower_bound_q) | (df['Quantity'] > upper_bound_q)]
        
        st.metric("Quantity Outliers", len(qty_outliers))
        
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.boxplot(df['Quantity'].dropna())
        ax2.set_title('Quantity Distribution with Outliers')
        ax2.set_ylabel('Quantity')
        st.pyplot(fig2)

def sales_performance_dashboard(df):
    """Display sales performance metrics"""
    st.markdown("### 💰 Sales Performance Dashboard")
    
    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = df['Order Total'].sum()
        st.metric("Total Revenue", f"${total_revenue:,.2f}")
    
    with col2:
        avg_order_value = df['Order Total'].mean()
        st.metric("Avg Order Value", f"${avg_order_value:.2f}")
    
    with col3:
        total_orders = len(df)
        st.metric("Total Orders", total_orders)
    
    with col4:
        unique_customers = df['Customer ID'].nunique()
        st.metric("Unique Customers", unique_customers)
    
    # Revenue trends over time
    st.markdown("#### 📅 Revenue Trends")
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    daily_revenue = df.groupby(df['Order Date'].dt.date)['Order Total'].sum().reset_index()
    
    fig = px.line(daily_revenue, x='Order Date', y='Order Total',
                  title='Daily Revenue Trends',
                  labels={'Order Total': 'Revenue ($)', 'Order Date': 'Date'})
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Category performance
    st.markdown("#### 📊 Category Performance")
    category_performance = df.groupby('Category').agg({
        'Order Total': 'sum',
        'Order ID': 'count'
    }).reset_index()
    category_performance.columns = ['Category', 'Total Revenue', 'Order Count']
    category_performance['Avg Order Value'] = category_performance['Total Revenue'] / category_performance['Order Count']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(category_performance, x='Category', y='Total Revenue',
                     title='Revenue by Category',
                     color='Category')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(category_performance, values='Total Revenue', names='Category',
                     title='Revenue Distribution by Category',
                     hole=0.3)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def customer_analytics(df):
    """Customer analytics and segmentation"""
    st.markdown("### 👥 Customer Intelligence")
    
    # Customer segmentation
    customer_stats = df.groupby('Customer ID').agg({
        'Order Total': ['sum', 'mean', 'count'],
        'Order Date': ['min', 'max']
    }).reset_index()
    
    customer_stats.columns = ['Customer_ID', 'Total_Spent', 'Avg_Order_Value', 
                              'Visit_Count', 'First_Visit', 'Last_Visit']
    
    # RFM Analysis
    current_date = df['Order Date'].max()
    customer_stats['Recency'] = (current_date - customer_stats['Last_Visit']).dt.days
    customer_stats['Frequency'] = customer_stats['Visit_Count']
    customer_stats['Monetary'] = customer_stats['Total_Spent']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏆 Top Customers by Revenue")
        top_customers = customer_stats.sort_values('Total_Spent', ascending=False).head(10)
        st.dataframe(top_customers[['Customer_ID', 'Total_Spent', 'Visit_Count', 'Avg_Order_Value']]
                     .style.format({'Total_Spent': '${:,.2f}', 'Avg_Order_Value': '${:,.2f}'}),
                     use_container_width=True)
    
    with col2:
        st.markdown("#### 📊 Customer Value Segments")
        
        # Create segments based on spending
        conditions = [
            customer_stats['Total_Spent'] > customer_stats['Total_Spent'].quantile(0.75),
            customer_stats['Total_Spent'] > customer_stats['Total_Spent'].quantile(0.5),
            customer_stats['Total_Spent'] > customer_stats['Total_Spent'].quantile(0.25)
        ]
        choices = ['VIP', 'Loyal', 'Regular', 'New']
        
        customer_stats['Segment'] = np.select(conditions, choices[:3], default=choices[3])
        
        segment_counts = customer_stats['Segment'].value_counts()
        
        fig = px.pie(values=segment_counts.values, names=segment_counts.index,
                     title='Customer Segmentation',
                     color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def item_analytics(df):
    """Item-level analytics"""
    st.markdown("### 🍽️ Menu Item Analytics")
    
    item_performance = df.groupby(['Category', 'Item']).agg({
        'Order Total': ['sum', 'count', 'mean'],
        'Quantity': 'sum'
    }).reset_index()
    
    item_performance.columns = ['Category', 'Item', 'Total_Revenue', 'Order_Count', 
                                'Avg_Order_Value', 'Total_Quantity']
    
    # Top selling items
    st.markdown("#### 📈 Top Selling Items")
    top_items = item_performance.sort_values('Total_Revenue', ascending=False).head(15)
    
    fig = px.bar(top_items, x='Item', y='Total_Revenue',
                 color='Category',
                 title='Top 15 Items by Revenue',
                 labels={'Total_Revenue': 'Revenue ($)'})
    fig.update_layout(height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Item performance matrix
    st.markdown("#### 🎯 Item Performance Matrix")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Popularity vs Revenue
        fig = px.scatter(item_performance, x='Order_Count', y='Total_Revenue',
                         size='Avg_Order_Value', color='Category',
                         hover_name='Item',
                         title='Popularity vs Revenue Analysis',
                         labels={'Order_Count': 'Number of Orders', 
                                'Total_Revenue': 'Total Revenue ($)'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Item distribution by category
        category_items = df.groupby('Category')['Item'].nunique().reset_index()
        category_items.columns = ['Category', 'Unique_Items']
        
        fig = px.bar(category_items, x='Category', y='Unique_Items',
                     color='Category',
                     title='Number of Unique Items by Category',
                     labels={'Unique_Items': 'Number of Items'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def payment_analytics(df):
    """Payment method analytics"""
    st.markdown("### 💳 Payment Analytics")
    
    payment_stats = df.groupby('Payment Method').agg({
        'Order Total': ['sum', 'count', 'mean'],
        'Customer ID': 'nunique'
    }).reset_index()
    
    payment_stats.columns = ['Payment_Method', 'Total_Revenue', 'Transaction_Count', 
                            'Avg_Transaction_Value', 'Unique_Customers']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue by payment method
        fig = px.bar(payment_stats, x='Payment_Method', y='Total_Revenue',
                     color='Payment_Method',
                     title='Revenue by Payment Method',
                     labels={'Total_Revenue': 'Revenue ($)'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Customer preference by payment method
        fig = px.pie(payment_stats, values='Transaction_Count', names='Payment_Method',
                     title='Payment Method Distribution',
                     hole=0.3)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Payment method trends
    st.markdown("#### 📅 Payment Method Trends Over Time")
    df['Month'] = df['Order Date'].dt.to_period('M')
    monthly_payments = df.groupby(['Month', 'Payment Method']).size().reset_index(name='Count')
    monthly_payments['Month'] = monthly_payments['Month'].dt.to_timestamp()
    
    fig = px.line(monthly_payments, x='Month', y='Count', color='Payment Method',
                  title='Monthly Payment Method Trends',
                  labels={'Count': 'Number of Transactions'})
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def export_data(df):
    """Data export functionality"""
    st.markdown("### 💾 Export Analysis Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Export Cleaned Data (CSV)"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="restaurant_sales_cleaned.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📊 Export Summary Report (JSON)"):
            # Create summary report
            summary = {
                "total_records": len(df),
                "total_revenue": float(df['Order Total'].sum()),
                "avg_order_value": float(df['Order Total'].mean()),
                "unique_customers": int(df['Customer ID'].nunique()),
                "categories": df['Category'].value_counts().to_dict(),
                "top_items": df.groupby('Item')['Order Total'].sum().nlargest(10).to_dict()
            }
            import json
            json_data = json.dumps(summary, indent=2)
            st.download_button(
                label="Download JSON Report",
                data=json_data,
                file_name="restaurant_sales_summary.json",
                mime="application/json"
            )
    
    with col3:
        if st.button("🔄 Refresh Data"):
            st.rerun()

# ==============================================
# 🚀 MAIN APPLICATION
# ==============================================
def main():
    # Title and description
    st.title("🍽️ Restaurant Sales Intelligence Dashboard")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin: 20px 0;'>
        <h3 style='color: white; margin: 0;'>Advanced Analytics for 17,534+ Restaurant Transactions</h3>
        <p style='color: rgba(255,255,255,0.9); margin: 10px 0 0 0;'>
            Real-time insights, anomaly detection, and predictive analytics for restaurant operations
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading and processing restaurant data..."):
        df = load_and_process_data()
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("## ⚙️ Control Panel")
        
        # Date range filter
        st.markdown("### 📅 Date Range")
        min_date = df['Order Date'].min().date()
        max_date = df['Order Date'].max().date()
        date_range = st.date_input(
            "Select period",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            df_filtered = df[(df['Order Date'].dt.date >= start_date) & 
                           (df['Order Date'].dt.date <= end_date)].copy()
        else:
            df_filtered = df.copy()
        
        # Category filter
        st.markdown("### 🍽️ Categories")
        categories = st.multiselect(
            "Select categories",
            options=sorted(df['Category'].unique()),
            default=sorted(df['Category'].unique())
        )
        
        if categories:
            df_filtered = df_filtered[df_filtered['Category'].isin(categories)]
        
        # Price range filter
        st.markdown("### 💰 Price Range")
        min_price, max_price = st.slider(
            "Select price range",
            float(df['Price'].min()),
            float(df['Price'].max()),
            (float(df['Price'].min()), float(df['Price'].max()))
        )
        df_filtered = df_filtered[(df_filtered['Price'] >= min_price) & 
                                (df_filtered['Price'] <= max_price)]
        
        # Quick stats in sidebar
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        st.metric("Filtered Records", len(df_filtered))
        st.metric("Total Revenue", f"${df_filtered['Order Total'].sum():,.2f}")
        st.metric("Avg Order", f"${df_filtered['Order Total'].mean():.2f}")
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🏠 Overview", 
        "🔍 Data Quality", 
        "📈 Sales Analytics", 
        "👥 Customer Insights",
        "🍽️ Menu Analysis",
        "💳 Payments",
        "💾 Export"
    ])
    
    with tab1:
        # Hero metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_rev = df_filtered['Order Total'].sum()
            st.markdown(f"""
            <div style='background: rgba(102, 126, 234, 0.1); padding: 20px; border-radius: 10px; border-left: 4px solid #667eea;'>
                <h3 style='margin: 0; color: #667eea;'>💰 Total Revenue</h3>
                <h1 style='margin: 10px 0; color: #2D3748;'>${total_rev:,.2f}</h1>
                <p style='color: #718096; margin: 0;'>From {len(df_filtered):,} orders</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_order = df_filtered['Order Total'].mean()
            st.markdown(f"""
            <div style='background: rgba(118, 75, 162, 0.1); padding: 20px; border-radius: 10px; border-left: 4px solid #764ba2;'>
                <h3 style='margin: 0; color: #764ba2;'>📊 Avg Order Value</h3>
                <h1 style='margin: 10px 0; color: #2D3748;'>${avg_order:.2f}</h1>
                <p style='color: #718096; margin: 0;'>Per transaction</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            customers = df_filtered['Customer ID'].nunique()
            st.markdown(f"""
            <div style='background: rgba(56, 178, 172, 0.1); padding: 20px; border-radius: 10px; border-left: 4px solid #38B2AC;'>
                <h3 style='margin: 0; color: #38B2AC;'>👥 Unique Customers</h3>
                <h1 style='margin: 10px 0; color: #2D3748;'>{customers:,}</h1>
                <p style='color: #718096; margin: 0;'>Active in period</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            top_category = df_filtered.groupby('Category')['Order Total'].sum().idxmax()
            st.markdown(f"""
            <div style='background: rgba(237, 137, 54, 0.1); padding: 20px; border-radius: 10px; border-left: 4px solid #ED8936;'>
                <h3 style='margin: 0; color: #ED8936;'>🏆 Top Category</h3>
                <h1 style='margin: 10px 0; color: #2D3748; font-size: 1.8rem;'>{top_category}</h1>
                <p style='color: #718096; margin: 0;'>By revenue</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Top charts overview
        st.markdown("## 📈 Performance Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily revenue trend
            daily_rev = df_filtered.groupby(df_filtered['Order Date'].dt.date)['Order Total'].sum().reset_index()
            fig = px.line(daily_rev, x='Order Date', y='Order Total',
                         title='Daily Revenue Trend',
                         labels={'Order Total': 'Revenue ($)', 'Order Date': 'Date'})
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Category distribution
            cat_dist = df_filtered['Category'].value_counts().reset_index()
            cat_dist.columns = ['Category', 'Count']
            fig = px.pie(cat_dist, values='Count', names='Category',
                        title='Order Distribution by Category',
                        hole=0.3)
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent transactions
        st.markdown("## 📋 Recent Transactions")
        st.dataframe(
            df_filtered[['Order Date', 'Customer ID', 'Category', 'Item', 'Price', 'Quantity', 'Order Total', 'Payment Method']]
            .sort_values('Order Date', ascending=False)
            .head(20)
            .style.format({'Price': '${:.2f}', 'Order Total': '${:.2f}'})
            .background_gradient(subset=['Order Total'], cmap='Blues'),
            use_container_width=True,
            height=400
        )
    
    with tab2:
        data_quality_dashboard(df_filtered)
        outlier_analysis(df_filtered)
    
    with tab3:
        sales_performance_dashboard(df_filtered)
    
    with tab4:
        customer_analytics(df_filtered)
    
    with tab5:
        item_analytics(df_filtered)
    
    with tab6:
        payment_analytics(df_filtered)
    
    with tab7:
        export_data(df_filtered)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #718096; padding: 20px;'>
        <p>🧠 <strong>Restaurant Sales Intelligence Dashboard</strong> • Version 1.0 • Updated: {}</p>
        <p>Processing {} records • Powered by Streamlit & Plotly</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), len(df_filtered)), unsafe_allow_html=True)

# ==============================================
# 🚀 RUN THE APPLICATION
# ==============================================
if __name__ == "__main__":
    main()
