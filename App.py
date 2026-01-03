import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ==============================================
# 🎨 SIMPLE KID-FRIENDLY THEME
# ==============================================
st.set_page_config(
    page_title="Restaurant Game Dashboard",
    page_icon="🍕",
    layout="wide"
)

st.markdown("""
<style>
/* SIMPLE KID-FRIENDLY THEME */
@import url('https://fonts.googleapis.com/css2?family=Comic+Neue:wght@400;700&display=swap');

* {
    font-family: 'Comic Neue', cursive;
}

body {
    background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
    color: #333333;
}

.main {
    background: transparent !important;
}

/* BIG HEADERS */
h1, h2, h3 {
    color: #ff6b6b !important;
    text-shadow: 2px 2px 0px #ffeaa7;
    margin-bottom: 20px !important;
}

h1 {
    font-size: 3rem !important;
    color: #0984e3 !important;
}

h2 {
    font-size: 2.5rem !important;
    color: #00b894 !important;
}

/* CARD BOXES */
.card-box {
    background: white;
    border-radius: 25px;
    padding: 25px;
    margin: 20px 0;
    border: 5px solid #74b9ff;
    box-shadow: 10px 10px 0px #a29bfe;
}

/* BUTTONS */
.stButton > button {
    background: linear-gradient(135deg, #81ecec 0%, #74b9ff 100%) !important;
    color: #2d3436 !important;
    font-size: 1.2rem !important;
    font-weight: bold !important;
    border: 4px solid #0984e3 !important;
    border-radius: 20px !important;
    padding: 15px 30px !important;
    box-shadow: 5px 5px 0px #a29bfe !important;
}

.stButton > button:hover {
    transform: translateY(-5px) !important;
    box-shadow: 10px 10px 0px #a29bfe !important;
}

/* METRIC CARDS */
.metric-card-kid {
    background: white;
    border-radius: 20px;
    padding: 20px;
    margin: 10px;
    border: 3px dashed #55efc4;
    text-align: center;
}

.metric-value-kid {
    font-size: 3rem !important;
    font-weight: bold;
    color: #e84393 !important;
    margin: 10px 0;
}

.metric-label-kid {
    font-size: 1.2rem !important;
    color: #636e72 !important;
    font-weight: bold;
}

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background: transparent !important;
}

.stTabs [data-baseweb="tab"] {
    background: white !important;
    color: #2d3436 !important;
    border-radius: 15px !important;
    padding: 15px 25px !important;
    border: 3px solid #a29bfe !important;
    font-size: 1.2rem !important;
    font-weight: bold !important;
}

.stTabs [aria-selected="true"] {
    background: #ffeaa7 !important;
    color: #2d3436 !important;
    border: 3px solid #fdcb6e !important;
}

/* SIMPLE CHARTS */
.js-plotly-plot {
    background: white !important;
    border-radius: 20px;
    padding: 20px;
    border: 3px solid #fd79a8;
}

/* EMOJI SECTION */
.emoji-section {
    font-size: 2rem;
    text-align: center;
    margin: 20px 0;
}

/* FUN FACTS BOX */
.fun-fact {
    background: #fff9c4;
    border: 3px solid #fdcb6e;
    border-radius: 15px;
    padding: 15px;
    margin: 15px 0;
    font-size: 1.3rem;
}

/* ANIMATIONS */
@keyframes bounce {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-10px); }
}

.bounce-emoji {
    display: inline-block;
    animation: bounce 1s infinite;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #a29bfe 0%, #81ecec 100%) !important;
    border-right: 5px solid #6c5ce7 !important;
}

[data-testid="stSidebar"] h3 {
    color: #2d3436 !important;
}
</style>
""", unsafe_allow_html=True)

# ==============================================
# 🍕 SIMPLE RESTAURANT DATA
# ==============================================
@st.cache_data
def get_simple_data():
    # Super simple restaurant data a 10-year-old would understand
    data = {
        'Day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        'Pizza_Sold': [50, 60, 55, 70, 120, 150, 100],
        'Burger_Sold': [40, 45, 50, 55, 90, 110, 80],
        'Ice_Cream_Sold': [30, 35, 40, 45, 80, 95, 70],
        'Money_Made': [1000, 1200, 1100, 1400, 2500, 3000, 2000],
        'Happy_Customers': [45, 50, 48, 55, 85, 95, 75]
    }
    return pd.DataFrame(data)

# ==============================================
# 🎮 MAIN GAME DASHBOARD
# ==============================================
def main():
    # HEADER WITH BIG EMOJIS
    st.markdown("""
    <div class="card-box">
        <div class="emoji-section">
            🍕🍔🍦🎮🍟🥤
        </div>
        <h1>🍕 RESTAURANT GAME DASHBOARD 🎮</h1>
        <div style="font-size: 1.5rem; color: #636e72; text-align: center;">
            Let's run a pretend restaurant and see how much money we can make!
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # GET SIMPLE DATA
    df = get_simple_data()
    
    # SIDEBAR - SIMPLE CONTROLS
    with st.sidebar:
        st.markdown("""
        <div class="card-box">
            <h2>🎮 Game Controls</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### What food do you want to see?")
        
        show_pizza = st.checkbox("🍕 Pizza", value=True)
        show_burger = st.checkbox("🍔 Burgers", value=True)
        show_icecream = st.checkbox("🍦 Ice Cream", value=True)
        
        st.markdown("---")
        
        # SIMPLE SLIDER
        st.markdown("### How many days to show?")
        days = st.slider("Days", 1, 7, 7)
        
        # BIG COLORFUL BUTTON
        if st.button("🔄 PLAY AGAIN / RESET GAME", type="primary"):
            st.rerun()
    
    # TOP METRICS - BIG AND COLORFUL
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card-kid">
            <div class="metric-label-kid">💰 TOTAL MONEY</div>
            <div class="metric-value-kid">${df['Money_Made'].sum():,}</div>
            <div style="font-size: 1.5rem;">🍕🍔🍦</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card-kid">
            <div class="metric-label-kid">😊 HAPPY PEOPLE</div>
            <div class="metric-value-kid">{df['Happy_Customers'].sum():,}</div>
            <div style="font-size: 1.5rem;">😀😄😁</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card-kid">
            <div class="metric-label-kid">🍕 PIZZA SOLD</div>
            <div class="metric-value-kid">{df['Pizza_Sold'].sum():,}</div>
            <div style="font-size: 1.5rem;">🍕🍕🍕</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card-kid">
            <div class="metric-label-kid">🏆 BEST DAY</div>
            <div class="metric-value-kid">{df.loc[df['Money_Made'].idxmax(), 'Day']}</div>
            <div style="font-size: 1.5rem;">⭐🥇🎉</div>
        </div>
        """, unsafe_allow_html=True)
    
    # FUN FACTS
    st.markdown("""
    <div class="card-box">
        <h2>📊 FUN FACTS ABOUT OUR RESTAURANT</h2>
        <div class="fun-fact">💰 We made ${} on Saturday - That's our BEST DAY EVER!</div>
        <div class="fun-fact">🍕 We sold {} pizzas - That's like {} pizza parties!</div>
        <div class="fun-fact">😊 {} people smiled at our food - We're making people HAPPY!</div>
        <div class="fun-fact">📈 Friday to Saturday, our money went up by ${} - WOW!</div>
    </div>
    """.format(
        df.loc[df['Day'] == 'Saturday', 'Money_Made'].values[0],
        df['Pizza_Sold'].sum(),
        df['Pizza_Sold'].sum() // 10,
        df['Happy_Customers'].sum(),
        df.loc[df['Day'] == 'Saturday', 'Money_Made'].values[0] - df.loc[df['Day'] == 'Friday', 'Money_Made'].values[0]
    ), unsafe_allow_html=True)
    
    # TABS FOR DIFFERENT VIEWS
    tab1, tab2, tab3 = st.tabs(["📈 CHARTS", "🍕 FOOD GAME", "🏆 LEADERBOARD"])
    
    # TAB 1: SIMPLE CHARTS
    with tab1:
        st.markdown("""
        <div class="card-box">
            <h2>📈 MONEY MADE EACH DAY</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # SIMPLE BAR CHART - Money each day
        fig1 = px.bar(
            df.head(days),
            x='Day',
            y='Money_Made',
            color='Money_Made',
            color_continuous_scale='sunset',
            title="💰 Money We Made Each Day"
        )
        
        fig1.update_layout(
            height=400,
            xaxis_title="Day of Week",
            yaxis_title="Money Made ($)",
            plot_bgcolor='white'
        )
        
        # Add emoji annotations
        max_day = df.loc[df['Money_Made'].idxmax(), 'Day']
        max_money = df['Money_Made'].max()
        
        fig1.add_annotation(
            x=max_day,
            y=max_money,
            text="🏆 BEST DAY!",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#FF6B6B"
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # FOOD SOLD CHART
        st.markdown("""
        <div class="card-box">
            <h2>🍕 FOOD SOLD EACH DAY</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Simple line chart
        fig2 = go.Figure()
        
        if show_pizza:
            fig2.add_trace(go.Scatter(
                x=df['Day'].head(days),
                y=df['Pizza_Sold'].head(days),
                mode='lines+markers',
                name='🍕 Pizza',
                line=dict(color='#FF6B6B', width=4)
            ))
        
        if show_burger:
            fig2.add_trace(go.Scatter(
                x=df['Day'].head(days),
                y=df['Burger_Sold'].head(days),
                mode='lines+markers',
                name='🍔 Burgers',
                line=dict(color='#00B894', width=4)
            ))
        
        if show_icecream:
            fig2.add_trace(go.Scatter(
                x=df['Day'].head(days),
                y=df['Ice_Cream_Sold'].head(days),
                mode='lines+markers',
                name='🍦 Ice Cream',
                line=dict(color='#74B9FF', width=4)
            ))
        
        fig2.update_layout(
            height=400,
            title="Food Items Sold Each Day",
            xaxis_title="Day of Week",
            yaxis_title="Number Sold",
            plot_bgcolor='white',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # TAB 2: FOOD GAME
    with tab2:
        st.markdown("""
        <div class="card-box">
            <h2>🍕 FOOD GAME - WHAT SHOULD WE SELL?</h2>
            <div style="font-size: 1.5rem; color: #636e72;">
                Try changing the numbers to see what happens to our money!
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pizza_price = st.number_input("🍕 Pizza Price ($)", min_value=5, max_value=30, value=15, step=1)
            st.write(f"${pizza_price} per pizza")
        
        with col2:
            burger_price = st.number_input("🍔 Burger Price ($)", min_value=3, max_value=20, value=10, step=1)
            st.write(f"${burger_price} per burger")
        
        with col3:
            icecream_price = st.number_input("🍦 Ice Cream Price ($)", min_value=2, max_value=15, value=5, step=1)
            st.write(f"${icecream_price} per ice cream")
        
        # CALCULATE NEW MONEY
        new_money = (
            df['Pizza_Sold'].sum() * pizza_price +
            df['Burger_Sold'].sum() * burger_price +
            df['Ice_Cream_Sold'].sum() * icecream_price
        )
        
        old_money = df['Money_Made'].sum()
        difference = new_money - old_money
        
        st.markdown(f"""
        <div class="card-box">
            <h3>🎮 GAME RESULTS:</h3>
            <div class="fun-fact">
                With these prices, we would make: <span style="color:#e84393; font-size: 2rem;">${new_money:,}</span>
            </div>
            <div class="fun-fact">
                That's <span style="color:#00b894; font-size: 1.8rem;">${difference:,}</span> {"more" if difference > 0 else "less"} than before!
            </div>
            <div class="fun-fact">
                <span style="font-size: 2rem;">{"🎉 YAY! GOOD PRICES!" if difference > 0 else "🤔 LET'S TRY AGAIN"}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # SIMPLE PIE CHART
        st.markdown("### 📊 WHERE DOES OUR MONEY COME FROM?")
        
        pie_data = {
            'Food': ['Pizza', 'Burgers', 'Ice Cream'],
            'Money': [
                df['Pizza_Sold'].sum() * pizza_price,
                df['Burger_Sold'].sum() * burger_price,
                df['Ice_Cream_Sold'].sum() * icecream_price
            ]
        }
        
        pie_df = pd.DataFrame(pie_data)
        
        fig3 = px.pie(
            pie_df,
            values='Money',
            names='Food',
            color='Food',
            color_discrete_map={'Pizza':'#FF6B6B', 'Burgers':'#00B894', 'Ice Cream':'#74B9FF'},
            title="Money from Each Food"
        )
        
        fig3.update_traces(textposition='inside', textinfo='percent+label')
        fig3.update_layout(height=400)
        
        st.plotly_chart(fig3, use_container_width=True)
    
    # TAB 3: LEADERBOARD
    with tab3:
        st.markdown("""
        <div class="card-box">
            <h2>🏆 RESTAURANT LEADERBOARD</h2>
            <div style="font-size: 1.5rem; color: #636e72;">
                Which day was the BEST? Let's find out!
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # CREATE LEADERBOARD
        leaderboard = df.copy()
        leaderboard['Rank'] = leaderboard['Money_Made'].rank(ascending=False, method='first').astype(int)
        leaderboard = leaderboard.sort_values('Rank')
        
        # DISPLAY AS BIG MEDALS
        st.markdown("### 🥇🥈🥉 TOP 3 DAYS")
        
        for idx, row in leaderboard.head(3).iterrows():
            medal = "🥇" if row['Rank'] == 1 else "🥈" if row['Rank'] == 2 else "🥉"
            
            st.markdown(f"""
            <div class="card-box" style="background: {'#FFF9C4' if row['Rank'] == 1 else '#E3F2FD' if row['Rank'] == 2 else '#F3E5F5'};">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div style="font-size: 3rem;">{medal}</div>
                    <div style="text-align: center;">
                        <div style="font-size: 2.5rem; font-weight: bold; color: #2D3436;">{row['Day']}</div>
                        <div style="font-size: 1.5rem; color: #636E72;">Rank #{row['Rank']}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 2rem; color: #E84393; font-weight: bold;">${row['Money_Made']:,}</div>
                        <div style="font-size: 1.2rem; color: #636E72;">
                            🍕{row['Pizza_Sold']} 🍔{row['Burger_Sold']} 🍦{row['Ice_Cream_Sold']}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # SIMPLE TABLE FOR ALL DAYS
        st.markdown("### 📋 ALL DAYS SCOREBOARD")
        
        # Make table colorful
        styled_df = leaderboard[['Rank', 'Day', 'Money_Made', 'Pizza_Sold', 'Burger_Sold', 'Ice_Cream_Sold', 'Happy_Customers']]
        
        def color_rank(val):
            if val == 1:
                return 'background-color: #FFF9C4; font-weight: bold;'
            elif val == 2:
                return 'background-color: #E3F2FD;'
            elif val == 3:
                return 'background-color: #F3E5F5;'
            return ''
        
        st.dataframe(
            styled_df.style
            .applymap(color_rank, subset=['Rank'])
            .format({'Money_Made': '${:,.0f}'})
            .set_properties(**{
                'font-size': '1.2rem',
                'text-align': 'center'
            }),
            use_container_width=True,
            height=400
        )
    
    # GAME TIPS FOR KIDS
    st.markdown("""
    <div class="card-box">
        <h2>💡 GAME TIPS FOR RESTAURANT BOSSES</h2>
        <div class="emoji-section">💡🌟✨🎯📚</div>
        
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-top: 20px;">
            <div class="fun-fact">
                <span style="font-size: 2rem;">💰</span><br>
                <strong>More food sold = More money!</strong><br>
                Try selling popular foods
            </div>
            
            <div class="fun-fact">
                <span style="font-size: 2rem;">😊</span><br>
                <strong>Happy customers come back!</strong><br>
                Good food = More smiles
            </div>
            
            <div class="fun-fact">
                <span style="font-size: 2rem;">📈</span><br>
                <strong>Weekends are busy!</strong><br>
                Saturday = Best money day
            </div>
            
            <div class="fun-fact">
                <span style="font-size: 2rem;">🎮</span><br>
                <strong>Try different prices!</strong><br>
                See what makes most money
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # FINAL FUN EMOJI
    st.markdown("""
    <div style="text-align: center; margin-top: 50px;">
        <div style="font-size: 4rem;" class="emoji-section">
            <span class="bounce-emoji">🎮</span>
            <span class="bounce-emoji">🍕</span>
            <span class="bounce-emoji">💰</span>
            <span class="bounce-emoji">😊</span>
            <span class="bounce-emoji">⭐</span>
        </div>
        <div style="font-size: 2rem; color: #636e72; margin-top: 20px;">
            Thanks for playing Restaurant Game! 🎉
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================
# 🚀 RUN THE GAME
# ==============================================
if __name__ == "__main__":
    main()
