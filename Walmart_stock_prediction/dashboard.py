import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8000/forecast"
REPLENISHMENT_CSV = "Csv_data\Final_Replenishment_Report.csv"
FEATURE_DATA_CSV = "Csv_data\Walmart_Final_FE.csv" # Needed to get features to send to API
HISTORICAL_DATA_CSV = "Csv_data\Walmart.csv" # Needed for the line chart history

# Inventory Constants (Must match your training logic)
LEAD_TIME = 2
ALPHA = 0.5
HOLDING_COST = 0.25

# --- PAGE SETUP ---
st.set_page_config(page_title="Walmart AI Supply Chain", layout="wide")

# --- DATA LOADING (Cached for Speed) ---
@st.cache_data
def load_static_data():
    """Loads the CSV reports and historical data."""
    rep_df = pd.read_csv(REPLENISHMENT_CSV)
    feat_df = pd.read_csv(FEATURE_DATA_CSV)
    hist_df = pd.read_csv(HISTORICAL_DATA_CSV)
    
    # Pre-processing dates
    feat_df['Date'] = pd.to_datetime(feat_df['Date'])
    hist_df['Date'] = pd.to_datetime(hist_df['Date'], format='%d-%m-%Y')
    
    # Cleanup Currency Strings if present
    cols_to_clean = ['Prediction_T+1', 'Current_Inventory', 'AI_ROP', 'Order_Qty', 'Prediction_T+1']
    for col in cols_to_clean:
        if col in rep_df.columns:
            # Check if column contains currency strings before conversion
            if rep_df[col].dtype == 'object' or rep_df[col].astype(str).str.contains(r'[\$,]').any():
                 rep_df[col] = rep_df[col].astype(str).str.replace(r'[\$,]', '', regex=True).str.replace(',', '').astype(float)
    
    return rep_df, feat_df, hist_df

try:
    df_rep, df_features, df_history = load_static_data()
except FileNotFoundError as e:
    st.error(f"❌ Missing Data File: {e}. Please run your simulation scripts first.")
    st.stop()


# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Supply Chain AI")
page = st.sidebar.radio("Navigation", ["📊 Executive Summary", "🔍 Store Explorer (Live AI)"])

st.sidebar.divider()
st.sidebar.info("System Status: **Online** 🟢")


# ==========================================
# PAGE 1: EXECUTIVE SUMMARY (Uses CSV)
# ==========================================
if page == "📊 Executive Summary":
    st.title("📊 Stock Replenishment Status")
    st.markdown("Overview of inventory health across all 45 stores using AI-driven forecasts and replenishment logic.")

    # 1. KPI Metrics
    total_restock_needed = df_rep[df_rep['Action'] == 'REORDER_NEEDED'].shape[0]
    total_cost_exposure = df_rep[df_rep['Action'] == 'REORDER_NEEDED']['Order_Qty'].sum()
    avg_inventory_health = (df_rep['Current_Inventory'] / df_rep['AI_ROP']).mean() * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Stores Requiring Restock", f"{total_restock_needed} / 45", delta_color="inverse")
    c2.metric("Total Restock Value",f"${total_cost_exposure / 1_000_000:,.2f}M")
    # --- UPDATED METRIC WITH TOOLTIP ---
    c3.metric(
        "Avg Inventory Health", 
        f"{avg_inventory_health:.1f}%", 
        help="""
        **Definition:** The average ratio of Current Inventory to the AI-Calculated Reorder Point (ROP).
        
        **How to Read:**
        * **100%+:** HEALTHY. Stores have enough stock to cover lead time.
        * **< 100%:** CRITICAL. On average, stores are below their safety threshold.
        * **> 200%:** OVERSTOCKED. Too much capital tied up in inventory.
        """
    )

    st.divider()

    # 2. Visualizations
    col1, col2 = st.columns([2, 1])

    # ... inside the "Executive Summary" section of dashboard.py ...

    with col1:
        st.subheader("Inventory Health Status (Current Stock vs. ROP)")
        
        # 1. Plot Current Stock on Y-axis
        fig = px.bar(
            df_rep, 
            x='Store', 
            y='Current_Inventory', # <-- NOW PLOTTING CURRENT STOCK
            color='Buffer %', 
            # Use Diverging Scale: Red (Negative Buffer) to Green (Positive Buffer)
            color_continuous_scale=px.colors.diverging.RdYlGn, 
            range_color=[-10, 20], # Focus color scale on the 0-30% danger zone
            title="Current Stock Level Relative to Reorder Point (ROP)",
            labels={'Buffer %': 'Safety Buffer %'}
        )
        
        # 3. Add the AI_ROP as a critical threshold line (overlayed)
        # This shows where the stock needs to be
        fig.add_trace(go.Scatter(
            x=df_rep['Store'], 
            y=df_rep['AI_ROP'], 
            mode='lines',
            name='AI Reorder Point (ROP)',
            line=dict(color='darkred', dash='dot', width=3)
        ))
        
        # 4. Remove the legend title for clarity
        fig.update_layout(coloraxis_colorbar_title_text='Buffer %')
        
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Action Distribution (Prioritized)")
        
        # 3. Define Colors for the Tiers
        color_map = {
            "INVENTORY OK": "#2BDD66",      # Green
            "Low Warning": "#FFC44B",     # Yellow/Light Orange
            "URGENT ACTION": "#FF8C4B", # Orange
            "CRITICAL DANGER": "#FF4B4B"        # Deep Red
        }

        # 4. Generate the Prioritized Pie Chart
        fig_pie = px.pie(
            df_rep, 
            names='Restock_Level', 
            color='Restock_Level',
            color_discrete_map=color_map,
            title="Action Priority Breakdown",
            hole=0.4
        )
        
        # Ensure the 'OK' slice is always at the top/start
        fig_pie.update_traces(sort=False) 
        
        st.plotly_chart(fig_pie, use_container_width=True)

    # 3. Detailed Data Table
    st.subheader("Detailed Store Data")
    st.dataframe(
        df_rep[['Store', 'Prediction_T+1', 'Current_Inventory', 'AI_ROP', 'Order_Qty', 'Buffer %']].style.format({
            "Prediction_T+1": lambda x: f"${x / 1_000_000:,.2f}M",
            "Current_Inventory": lambda x: f"${x / 1_000_000:,.2f}M",
            "AI_ROP": lambda x: f"${x / 1_000_000:,.2f}M",
            "Order_Qty": lambda x: f"${x / 1_000_000:,.2f}M",
            "Buffer %": lambda x: f"{x:.2f}%"
        }),
        use_container_width=True
    )


# ==========================================
# PAGE 2: STORE EXPLORER (Uses FastAPI)
# ==========================================
elif page == "🔍 Store Explorer (Live AI)":
    st.title("🔍 Live Store Explorer")
    st.markdown("""
    Select a store to pull **real-time AI forecasts** from the API and generate a custom replenishment plan.
    *This module communicates directly with the FastAPI backend.*
    """)

    # 1. Controls
    selected_store = st.selectbox("Select Store ID", df_rep['Store'].unique())
    
    # Get Static Policy Data for this store (for calculation)
    store_policy = df_rep[df_rep['Store'] == selected_store].iloc[0]
    
    # 2. Prepare API Data
    # Get last 20 weeks of features for this store to send to API
    # Columns MUST match what model was trained on: 
    # ['Weekly_Sales', 'Holiday_Flag', 'Sales_Lag_1', 'Sales_Rolling_Mean_4']
    input_cols = ['Weekly_Sales', 'Holiday_Flag', 'Sales_Lag_1', 'Sales_Rolling_Mean_4']
    
    store_features = df_features[df_features['Store'] == selected_store].sort_values('Date').tail(20)
    
    if len(store_features) < 20:
        st.error("Insufficient historical data for this store to make a prediction.")
    else:
        # Create list of lists for JSON payload
        input_sequence = store_features[input_cols].values.tolist()

        # 3. Action Button
        if st.button(f"📡 Request AI Forecast for Store {selected_store}"):
            with st.spinner("Connecting to Neural Network via FastAPI..."):
                try:
                    # --- CALL THE API ---
                    payload = {"store_id": int(selected_store), "input_sequence": input_sequence}
                    response = requests.post(API_URL, json=payload)
                    
                    if response.status_code == 200:
                        api_data = response.json()
                        forecast_value = api_data['weekly_sales_forecast']
                        
                        st.success("Analysis Complete!")
                        
                        # --- DYNAMIC CALCULATION ---
                        # We use the LIVE forecast to calculate ROP
                        # Re-calculating SS and EOQ from the CSV for consistency
                        # (In reality, we'd pull these from the Policy DB)
                        
                        # Reverse engineer SS from the CSV ROP for demo accuracy or calculate fresh
                        # Let's calculate fresh ROP using the formula
                        # ROP = (Forecast * LeadTime * Alpha) + Safety Stock (We need SS)
                        
                        # Estimate SS from the report (ROP - (Forecast*Lead*Alpha)) is tricky if Forecast changed.
                        # For this demo, let's grab the SS/EOQ from the generation script logic if possible.
                        # Simplified: We will recalculate ROP based on the NEW Forecast and the implied SS.
                        # Implied SS = Old_ROP - (Old_Forecast * Lead * Alpha)
                        old_forecast = store_policy['Prediction_T+1']
                        implied_ss = store_policy['AI_ROP'] - (old_forecast * LEAD_TIME * ALPHA)
                        
                        # NEW AI ROP
                        new_ai_rop = (forecast_value * LEAD_TIME * ALPHA) + implied_ss
                        
                        # Live Decision
                        current_inv = store_policy['Current_Inventory'] # Keeping inventory static for demo
                        
                        if current_inv <= new_ai_rop:
                            status = "REORDER NEEDED"
                            color = "red"
                            qty = (new_ai_rop + (forecast_value*3)) - current_inv # Approx EOQ
                        else:
                            status = "HEALTHY"
                            color = "green"
                            qty = 0.0

                        # --- DISPLAY RESULTS ---
                        
                        # Metrics Row
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("AI Forecast (Next Week)", f"${forecast_value / 1_000_000:,.2f}M")
                        m2.metric("Dynamic ROP", f"${new_ai_rop / 1_000_000:,.2f}M")
                        m3.metric("Current Stock (ERP)", f"${current_inv / 1_000_000:,.2f}M")
                        m4.metric("Recommendation", status, delta_color="off" if color=="red" else "normal")

                        # Visualization: History + Forecast
                        st.subheader("Sales Trajectory")
                        
                        # Filter history
                        history_plot = df_history[df_history['Store'] == selected_store].tail(52).copy()
                        
                        # Add the forecast point
                        last_date = history_plot['Date'].max()
                        next_date = last_date + pd.Timedelta(weeks=1)
                        
                        fig = px.line(history_plot, x='Date', y='Weekly_Sales', title=f"Store {selected_store} Performance")
                        
                        # Add Prediction Marker
                        fig.add_scatter(
                            x=[next_date], 
                            y=[forecast_value], 
                            mode='markers+text', 
                            marker=dict(color='red', size=12),
                            name='AI Prediction',
                            text=['Forecast'],
                            textposition='top center'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # JSON Response (For technical marks)
                        with st.expander("View Raw API Response (JSON)"):
                            st.json(api_data)

                    else:
                        st.error(f"API Error {response.status_code}: {response.text}")
                
                except Exception as e:
                    st.error(f"Connection Failed: {e}. Is 'api_server.py' running?")