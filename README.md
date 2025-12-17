# 🛒 AI-Driven Automated Supply Planning System 

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Flask](https://img.shields.io/badge/Flask-Dashboard-yellow)


## 📖 Overview
This project is an end-to-end **Digital Supply Chain Transformation** solution designed for Walmart Stores and Retail Stores. It replaces static, manual inventory rules with a **Predictive AI Operating Model**.

By leveraging **Deep Learning (LSTM)** for demand forecasting and an **Automated Reorder Engine (A.R.E.)** for replenishment, the system optimizes inventory levels, reducing holding costs by ~15-20% while preventing stockouts during peak seasons.

---
## 🔬 Methodology: Dual-Data Strategy
To ensure maximum accuracy and operational relevance, this project implements a robust **Two-Pronged Approach**:

### 1. 📈 Demand-Side Approach (Sales Data)
* **Source:** Walmart Weekly Sales Historical Data (2010-2012).
* **Purpose:** We trained 45 Per-Store LSTM Models on this dataset. By analyzing past sales trends alongside macroeconomic indicators (CPI, Unemployment, Fuel Price), the model predicts future consumer demand (T+1 Week) with high precision.
* **Outcome:** A highly accurate "Demand Signal" that accounts for seasonality and holiday spikes.

### 2. 📦 Supply-Side Approach (Inventory Data)
* **Source:** Retail Store Inventory Level Data (Real-time Snapshot).
* **Purpose:** The system addresses critical inventory management challenges faced by retail, manufacturing, and supply chain organizations.
* **Outcome:** The system Shift from reactive to proactive inventory management and cross-functional alignment between sales, operations and finance.
---
## 🚀 Key Features

## A) Demand-Side Approach (Sales Data):
### 1. 🧠 AI-Powered Demand Forecasting
* **Model:** Per-Store LSTM (Long Short-Term Memory) Recurrent Neural Networks.
* **Logic:** Learns non-linear patterns from historical weekly sales (2010-2012), considering seasonality, holidays, CPI, and Unemployment trends.
* **Performance:** Achieved a **MAPE of 4.2 %**, significantly outperforming the statistical baseline by 28.6%.

### 2. 📦 Automated Reorder Engine (A.R.E.)
* **Dynamic Policy:** Moves away from fixed "Safety Stock" to dynamic buffers based on predicted volatility.
* **Replenishment Logic:** Automatically calculates `Order_Qty = (Forecast_Demand + Safety_Stock) - Current_Inventory`.
* **Risk Classification:** Flags stores as **"CRITICAL"** (Stockout Risk) or **"OVERSTOCK"** (High Holding Cost) for immediate management intervention.

### 3. 📊 Interactive Dashboard 
* **Operational View:** A Streamlit-based interface for Store Managers to view live forecasts and Replenish status of stores.
* **Strategic View:** A Power BI Executive Summary for Walmart historical data .

## B) Supply-Side Approach (Inventory Data)
### 1. 🧠 Intelligent Forecasting Engine
* **Multi-Product Support:** Simultaneous forecasting for all products
* **Confidence Intervals:** 95% prediction intervals
* **Trend Analysis:** Upward/downward trend identification

### 2. 📦 Automated Reorder System
* **Smart Thresholds:** Dynamic reorder points based on lead time
* **Urgency Classification:**
    * HIGH: Immediate action required (< lead time)
    * MEDIUM: Monitor closely (< 2× lead time)
    * LOW: Normal monitoring
* **Safety Stock Calculation:** Risk-based buffer inventory
  
### 3. 📊 Interactive Dashboard
* **Real-time Metrics:** A flask based interface for 7 key performance indicators,Time-series charts,Product Selection and Export Capabilities.
---

## 🏗️ System Architecture
## A) Demand-Side Approach (Sales Data):
The solution follows a decoupled **Microservices Architecture** :

| Layer | Tech Stack | Role |
| :--- | :--- | :--- |
| **Presentation** | Streamlit | Interactive Dashboard for Store Managers (Port 8501) |
| **Application** | FastAPI | High-performance REST API serving LSTM inference (Port 8000) |
| **Model** | TensorFlow/Keras | 45 trained LSTM models (one per store) |
| **Data** | Pandas/NumPy | Feature Engineering (Lags, Rolling Means) |


## B) Supply-Side Approach (Inventory Data):

| Layer | Tech Stack | Role |
| :--- | :--- | :--- |
| **Presentation** | Flask/HTML/CSS | Interactive Dashboard for Store Managers |
| **Application** |Flask/Flask-CORS | Web framework/Cross-origin Support |
| **Model** | TensorFlow/Keras | 20 trained LSTM models (one per product) |
| **Data** | Python 3.8+ | Core language |

---

## 🛠️ Installation & Local Setup:

### 1. Create Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Run the Application

#### A) Walmart Data
#### Backend (FastAPI):
uvicorn api_server:app --reload --port 8000

#### Frontend (Streamlit):
streamlit run dashboard.py

#### B) Retail Store Inventory
python app.py

---
## Screenshot:
## A) Demand-Side Approach (Sales Data):
<img width="1007" height="563" alt="Screenshot 2025-12-11 194608" src="https://github.com/user-attachments/assets/c9c16a08-aaeb-4e9e-a3a1-90a0bd212cc2" />
<img width="1592" height="824" alt="Screenshot 2025-12-11 at 14-09-17 Walmart AI Supply Chain" src="https://github.com/user-attachments/assets/3395fa76-6a7e-4749-910b-909e64288d15" />
<img width="1544" height="806" alt="Screenshot 2025-12-11 at 14-13-56 Walmart AI Supply Chain" src="https://github.com/user-attachments/assets/3bc32fce-3ef1-40ed-ba8e-24acf85f4e86" />

## B) Supply-Side Approach (Inventory Data):
<img width="1383" height="876" alt="Screenshot 2025-12-11 at 14-08-40 🧠 Neural Inventory Intelligence Dashboard" src="https://github.com/user-attachments/assets/8dbb6b96-9833-44b7-b538-6f8949687d16" />

---


