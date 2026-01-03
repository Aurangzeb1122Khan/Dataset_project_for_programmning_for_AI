import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="Restaurant Sales Analytics",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ===================================
# FILE UPLOAD SECTION (IF FILES DON'T EXIST)
# ===================================
@st.cache_data
def load_data():
    """Try to load data from multiple possible filenames"""
    possible_filenames = [
        'restaurant_sales_cleaned.csv',
        'restaurant_sales_featured.csv',
        'restaurant_sales.csv'
    ]
    
    for filename in possible_filenames:
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                # Ensure order_date is datetime
                if 'order_date' in df.columns:
                    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
                
                # Add time features if they don't exist
                if 'order_date' in df.columns and 'hour' not in df.columns:
                    df['day_of_week'] = df['order_date'].dt.dayofweek
                    df['hour'] = df['order_date'].dt.hour
                    df['month'] = df['order_date'].dt.month
                    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
                
                st.sidebar.success(f"✅ Loaded: {filename}")
                return df
            except Exception as e:
                st.sidebar.error(f"Error loading {filename}: {e}")
                continue
    
    return None

# ===================================
# MAIN APP
# ===================================
st.markdown('<p class="main-header">🍽️ Restaurant Sales Analytics System</p>', unsafe_allow_html=True)

# Try to load data
df = load_data()

# If no data file found, show upload option
if df is None:
    st.warning("⚠️ No data file found. Please upload your CSV file.")
    
    st.info("""
    **Looking for these files:**
    - restaurant_sales_cleaned.csv
    - restaurant_sales_featured.csv
    - restaurant_sales.csv
    
    **Make sure your CSV file is in the same folder as this app.py file!**
    """)
    
    uploaded_file = st.file_uploader("Upload your restaurant sales CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Process the uploaded data
            st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
            
            # Show columns
            st.write("**Columns in your data:**", list(df.columns))
            
            # Clean and process data
            with st.spinner("Processing data..."):
                # Handle missing values
                if 'price' in df.columns:
                    df['price'] = df.groupby('category')['price'].transform(
                        lambda x: x.fillna(x.median())
                    )
                
                if 'quantity' in df.columns:
                    df['quantity'].fillna(df['quantity'].mode()[0], inplace=True)
                
                if 'order_total' in df.columns:
                    df['order_total'] = df['price'] * df['quantity']
                
                # Convert date
                if 'order_date' in df.columns:
                    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
                    df['day_of_week'] = df['order_date'].dt.dayofweek
                    df['hour'] = df['order_date'].dt.hour
                    df['month'] = df['order_date'].dt.month
                    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
                
                st.success("✅ Data processed successfully!")
            
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

# ===================================
# SIDEBAR
# ===================================
if df is not None:
    st.sidebar.header("📊 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Dashboard", "Data Analysis", "Make Predictions", "Data Table"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"""
    **Data Overview**
    
    📊 Records: {len(df):,}
    
    📁 Columns: {len(df.columns)}
    
    📅 Date Range: {df['order_date'].min().date() if 'order_date' in df.columns else 'N/A'} to {df['order_date'].max().date() if 'order_date' in df.columns else 'N/A'}
    """)

    # ===================================
    # PAGE 1: DASHBOARD
    # ===================================
    if page == "Dashboard":
        st.header("📊 Sales Dashboard")
        
        # KPI Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_revenue = df['order_total'].sum()
            st.metric("💰 Total Revenue", f"${total_revenue:,.2f}")
        
        with col2:
            total_orders = len(df)
            st.metric("📦 Total Orders", f"{total_orders:,}")
        
        with col3:
            avg_order = df['order_total'].mean()
            st.metric("📈 Avg Order Value", f"${avg_order:.2f}")
        
        with col4:
            unique_customers = df['customer_id'].nunique()
            st.metric("👥 Unique Customers", f"{unique_customers:,}")
        
        st.markdown("---")
        
        # Charts Row 1
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Sales by Category")
            category_sales = df.groupby('category')['order_total'].sum().reset_index()
            category_sales = category_sales.sort_values('order_total', ascending=False)
            
            fig = px.bar(category_sales, x='category', y='order_total',
                        color='order_total', 
                        color_continuous_scale='Blues',
                        labels={'order_total': 'Total Sales ($)', 'category': 'Category'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("💳 Payment Methods")
            payment_counts = df['payment_method'].value_counts().reset_index()
            payment_counts.columns = ['payment_method', 'count']
            
            fig = px.pie(payment_counts, values='count', names='payment_method',
                        hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig, use_container_width=True)
        
        # Charts Row 2
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📅 Daily Sales Trend")
            if 'order_date' in df.columns:
                daily_sales = df.groupby(df['order_date'].dt.date)['order_total'].sum().reset_index()
                daily_sales.columns = ['date', 'sales']
                
                fig = px.line(daily_sales, x='date', y='sales', markers=True)
                fig.update_traces(line_color='#00CC96')
                fig.update_layout(xaxis_title="Date", yaxis_title="Sales ($)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Order date column not available")
        
        with col2:
            st.subheader("💵 Price Distribution")
            fig = px.histogram(df, x='price', nbins=30, 
                             color_discrete_sequence=['#636EFA'])
            fig.update_layout(xaxis_title="Price ($)", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)
        
        # Top Items
        st.subheader("🏆 Top 10 Best Selling Items")
        top_items = df.groupby('item').agg({
            'order_total': 'sum',
            'order_id': 'count'
        }).sort_values('order_total', ascending=False).head(10)
        top_items.columns = ['Total Revenue', 'Number of Orders']
        st.dataframe(top_items, use_container_width=True)

    # ===================================
    # PAGE 2: DATA ANALYSIS
    # ===================================
    elif page == "Data Analysis":
        st.header("🔍 Exploratory Data Analysis")
        
        tab1, tab2, tab3 = st.tabs(["📊 Statistics", "📈 Visualizations", "🎯 Insights"])
        
        with tab1:
            st.subheader("Basic Statistics")
            st.dataframe(df.describe(), use_container_width=True)
            
            st.subheader("Category Analysis")
            category_stats = df.groupby('category').agg({
                'order_total': ['sum', 'mean', 'count'],
                'price': 'mean',
                'quantity': 'mean'
            }).round(2)
            st.dataframe(category_stats, use_container_width=True)
            
            st.subheader("Payment Method Analysis")
            payment_stats = df.groupby('payment_method').agg({
                'order_total': ['sum', 'mean', 'count']
            }).round(2)
            st.dataframe(payment_stats, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Price vs Order Total")
                sample_df = df.sample(min(1000, len(df)))
                fig = px.scatter(sample_df, x='price', y='order_total',
                               color='category', size='quantity',
                               hover_data=['item'])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Quantity Distribution")
                fig = px.histogram(df, x='quantity', nbins=20,
                                 color_discrete_sequence=['#EF553B'])
                st.plotly_chart(fig, use_container_width=True)
            
            if 'hour' in df.columns:
                st.subheader("Orders by Hour of Day")
                hourly_orders = df.groupby('hour')['order_id'].count().reset_index()
                hourly_orders.columns = ['hour', 'orders']
                
                fig = px.bar(hourly_orders, x='hour', y='orders',
                            color='orders', color_continuous_scale='Viridis')
                fig.update_layout(xaxis_title="Hour of Day", yaxis_title="Number of Orders")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("🎯 Key Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("**📊 Top Performing Category**")
                top_category = df.groupby('category')['order_total'].sum().idxmax()
                top_revenue = df.groupby('category')['order_total'].sum().max()
                st.write(f"🏆 **{top_category}**")
                st.write(f"Revenue: ${top_revenue:,.2f}")
                
                st.info("**💳 Most Popular Payment**")
                top_payment = df['payment_method'].value_counts().idxmax()
                payment_pct = (df['payment_method'].value_counts().iloc[0] / len(df)) * 100
                st.write(f"💳 **{top_payment}**")
                st.write(f"Usage: {payment_pct:.1f}%")
            
            with col2:
                st.info("**💰 Price Range**")
                st.write(f"Min: ${df['price'].min():.2f}")
                st.write(f"Max: ${df['price'].max():.2f}")
                st.write(f"Average: ${df['price'].mean():.2f}")
                
                st.info("**📦 Order Quantity**")
                st.write(f"Average: {df['quantity'].mean():.2f}")
                st.write(f"Most Common: {df['quantity'].mode()[0]}")

    # ===================================
    # PAGE 3: MAKE PREDICTIONS
    # ===================================
    elif page == "Make Predictions":
        st.header("🤖 Predict Order Total")
        
        st.info("💡 **Simple Prediction**: We'll calculate order_total = price × quantity")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Features")
            
            # Get unique categories from data
            categories = df['category'].unique().tolist()
            category = st.selectbox("Category", categories)
            
            # Get price range from data
            price_min = float(df['price'].min())
            price_max = float(df['price'].max())
            price = st.number_input(
                "Price ($)",
                min_value=price_min,
                max_value=price_max,
                value=float(df['price'].mean()),
                step=0.5
            )
            
            quantity = st.number_input(
                "Quantity",
                min_value=1,
                max_value=10,
                value=2,
                step=1
            )
            
            # Get unique payment methods from data
            payment_methods = df['payment_method'].unique().tolist()
            payment_method = st.selectbox("Payment Method", payment_methods)
        
        with col2:
            st.subheader("Category Statistics")
            
            # Show stats for selected category
            category_data = df[df['category'] == category]
            
            st.metric("Average Price", f"${category_data['price'].mean():.2f}")
            st.metric("Average Quantity", f"{category_data['quantity'].mean():.2f}")
            st.metric("Average Total", f"${category_data['order_total'].mean():.2f}")
        
        if st.button("🔮 Calculate Prediction", type="primary"):
            # Simple prediction
            prediction = price * quantity
            
            st.success(f"### Predicted Order Total: ${prediction:.2f}")
            
            # Compare with category average
            category_avg = df[df['category'] == category]['order_total'].mean()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Your Prediction", f"${prediction:.2f}")
            with col2:
                st.metric("Category Average", f"${category_avg:.2f}")
            with col3:
                difference = prediction - category_avg
                st.metric("Difference", f"${difference:.2f}", 
                         delta=f"{(difference/category_avg)*100:.1f}%")

    # ===================================
    # PAGE 4: DATA TABLE
    # ===================================
    elif page == "Data Table":
        st.header("📋 Data Table")
        
        st.subheader("Filter Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_categories = st.multiselect(
                "Filter by Category",
                options=df['category'].unique().tolist(),
                default=df['category'].unique().tolist()
            )
        
        with col2:
            selected_payments = st.multiselect(
                "Filter by Payment Method",
                options=df['payment_method'].unique().tolist(),
                default=df['payment_method'].unique().tolist()
            )
        
        with col3:
            min_price = st.number_input("Min Price", value=float(df['price'].min()))
            max_price = st.number_input("Max Price", value=float(df['price'].max()))
        
        # Apply filters
        filtered_df = df[
            (df['category'].isin(selected_categories)) &
            (df['payment_method'].isin(selected_payments)) &
            (df['price'] >= min_price) &
            (df['price'] <= max_price)
        ]
        
        st.write(f"**Showing {len(filtered_df):,} of {len(df):,} records**")
        
        # Display data
        st.dataframe(filtered_df, use_container_width=True, height=400)
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name="filtered_restaurant_sales.csv",
            mime="text/csv"
        )

else:
    st.info("👆 Please upload your data file to continue")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>🍽️ Restaurant Sales Analytics System</p>
        <p>Built with Streamlit & Python</p>
    </div>
    """,
    unsafe_allow_html=True
)
