import pandas as pd
import numpy as np
import os
import sys

# --- CONFIGURATION ---
LEAD_TIME = 2        
ALPHA = 0.5          
FORECAST_FILE = 'Csv_data\lstm_per_store_evaluation_metrics.csv' # From LSTM Model
POLICY_FILE = 'Csv_data\store_inventory_policies.csv'     # From Policy Generator

# --- 1. LOAD DATA SOURCES ---

# A. Load AI Forecasts
if not os.path.exists(FORECAST_FILE):
    sys.exit(f"❌ Missing {FORECAST_FILE}. Run LSTM script first.")
df_forecast = pd.read_csv(FORECAST_FILE)
print(f"Loaded Forecasts: {len(df_forecast)} stores")

# B. Load Inventory Policies (Replaces Hardcoded List)
if not os.path.exists(POLICY_FILE):
    sys.exit(f"❌ Missing {POLICY_FILE}. Run generate_policy_data.py first.")
df_policy = pd.read_csv(POLICY_FILE)
print(f"Loaded Policies:  {len(df_policy)} stores")

# --- 2. MERGE DATA ---
# This mimics a SQL JOIN between your "Forecast" table and "Policy" table
df_merged = pd.merge(df_policy, df_forecast[['Store', 'Prediction_T+1']], on='Store', how='left')

# --- 3. RUN SIMULATION ---
# Simulate ERP Snapshot (Current Inventory)
df_merged['Current_Inventory'] = df_merged['Mean_Weekly_Revenue'] * 1.3  # Example: 1.3 weeks of sales

def run_replenishment_logic(row):
    # Inputs from merged data
    forecast = row['Prediction_T+1'] # AI Input
    ss = row['SS_Revenue']              # Policy Input
    eoq = row['EOQ_Revenue']            # Policy Input
    current_inv = row['Current_Inventory'] # ERP Input
    
    # AI-Driven ROP Calculation
    rop_ai = (forecast * LEAD_TIME * ALPHA) + ss
    
    # Decision
    if current_inv <= rop_ai:
        status = "REORDER_NEEDED"
        qty = (rop_ai + eoq) - current_inv
        qty = max(0, qty)
    else:
        status = "INVENTORY_OK"
        qty = 0.00
        
    return pd.Series([round(rop_ai, 2), status, round(qty, 2)])

print("Running A.R.E. logic...")
df_merged[['AI_ROP', 'Action', 'Order_Qty']] = df_merged.apply(run_replenishment_logic, axis=1)

# A. Calculate Buffer Percentage
# Positive means good, Negative means danger (below ROP)
df_merged['Buffer %'] = ((df_merged['Current_Inventory'] - df_merged['AI_ROP']) / df_merged['AI_ROP']) * 100

# B. Define Restock Tiers (Must match the Streamlit logic for consistency)
def categorize_restock(pct):
    if pct >= 0:
        return "INVENTORY OK"
    elif pct >= -5:
        return "Low Warning"  # Buffer -5% to 0%
    elif pct >= -15:
        return "URGENT ACTION" # Buffer -15% to -5%
    else:
        return "CRITICAL DANGER" # Buffer less than -15%

df_merged['Restock_Level'] = df_merged['Buffer %'].apply(categorize_restock)



# --- 4. EXPORT REPORT ---
output_filename = 'Csv_data\Final_Replenishment_Report.csv'
final_cols = ['Store', 'Prediction_T+1', 'Current_Inventory', 'AI_ROP', 'Order_Qty','Buffer %','Action', 'Restock_Level']

df_merged[final_cols].to_csv(output_filename, index=False)

print(f"\n✅ SUCCESS: Generated {output_filename}")
print(df_merged[final_cols].to_string())