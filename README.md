# 🥬 FrischMarkt Expiry Loss Analysis: Optimizing Fresh Food Inventory

## 📊 Project Overview

This repository presents a comprehensive data analysis project aimed at mitigating expiry and markdown losses for **FrischMarkt**, a retail chain specializing in perishable goods. Using operational data from 2023, we identify major loss drivers, build a robust demand forecasting model, and propose actionable strategies to help FrischMarkt shift from **reactive loss management** to a **proactive, data-driven inventory optimization** approach.

> 💸 **The Problem:**
> FrischMarkt incurred **€5.78 million** in combined expiry and markdown losses in 2023 — a staggering **39.20%** of its total revenue. This highlights critical inefficiencies in current inventory and markdown management practices.

> 🧠 **The Solution:**
> A data-driven system powered by demand forecasting to reduce overstocking, improve markdown effectiveness, and guide smarter inventory decisions.

---

## ✨ Key Features & Components

* **🔧 Data Generation**
  A custom Python script (`enhanced_frischgenerator.py`) simulates realistic retail data — products, stores, weather, holidays, daily transactions — with a focus on replicating high-expiry scenarios.

* **📈 Exploratory Data Analysis (EDA)**
  Deep dives into product/store-level losses, category trends, correlation studies (e.g., shelf life vs. losses), and external factor impacts.

* **📊 Demand Forecasting Model**
  Built and compared machine learning models:

  * `RandomForestRegressor` (with `RandomizedSearchCV` tuning)
  * `LightGBM Regressor`
    Predictions made at the **product-store-day level**.

* **✅ Actionable Recommendations**
  Strategies for:

  * Inventory right-sizing
  * Smarter markdown timing
  * Supplier improvement and operational tweaks
    All backed by **quantified potential impact**.

---

## 🔍 Key Findings

* **💰 Financial Leakage:**
  Total losses: **€5.78M (39.20% of revenue)**

  * Expiry: €4.70M
  * Markdown: €1.08M

* **📦 Top Loss Categories:**

  * Meat (`Fleisch`)
  * Fresh Produce (`Frischware`)
  * Baked Goods (`Backwaren`)

* **🚨 Problem Products:**

  * **Beef Mince (`Rinderhackfleisch`)**: > €1.3M
  * **Strawberries (`Erdbeeren`)**: > €0.6M
  * **Pork Chops** (`Schweinekoteletts`) > €0.6M

  > *These three products alone dominate loss volumes.*

* **🏬 Underperforming Stores:**

  * `FrischMarkt Kreuzberg`
  * `FrischMarkt Prenzlauer Berg`

  > Each incurred nearly **€2.0M** in total losses.

* **📉 Management Impact:**

  * Poor-quality stores show **expiry rates >40%**
  * In contrast:

    * Good: \~30%
    * Excellent: \~24%

* **📈 Model Performance:**

  * **LightGBM**:

    * R² ≈ **0.69**
    * MAE ≈ **11.9 units**

* **🧠 Key Sales Drivers (Feature Importance):**

  * `received_inventory`
  * `beginning_inventory`
  * Lagged sales
  * Day of week, month
  * Temperature, precipitation

  > Overstocking is strongly tied to future loss.

---

## 📈 Quantified Impact (Estimates)
By implementing the proposed data-driven strategies, FrischMarkt can expect substantial financial improvements:

| Strategy                   | Estimated Savings                                       |
| -------------------------- | ------------------------------------------------------- |
| **Demand-Driven Ordering** | An estimated 10-15% reduction in current expiry losses, potentially saving €500,000 - €800,000 annually |
| **Optimized Markdowns**    | An additional €200,000 - €400,000 in recovered revenue annually from items that would otherwise expire completely |
| **Enhanced Operations & Supplier Review**  | A projected 5-8% reduction in overall expiry losses, translating to €250,000 - €450,000 annually |

---

## 🛠️ Technical Details

* **Language:** Python
* **Libraries:**

  * `pandas`, `numpy` – data handling
  * `matplotlib`, `seaborn` – visualization
  * `scikit-learn` – ML models
  * `lightgbm` – gradient boosting
  * `scipy.stats` – for randint, uniform in tuning
* **Models Used:**

  * Tuned `RandomForestRegressor`
  * `LightGBM Regressor`
* **Feature Engineering:**

  * Time-series lags
  * Rolling averages
  * External factor encoding
* **Evaluation Strategy:**

  * Chronological train-test split
  * Metrics: MAE, R²

---

## 🚀 How to Run This Project

### 1. Clone the Repository

```bash
git clone [https://github.com/Godwin-Oti/Expiry-Loss-Minimization-Project]
cd [Expiry-Loss-Minimization-Project]
```

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm faker
```

### 3. Generate Data

```bash
python enhanced_frischgenerator.py
```

Creates a `frischmarkt_data/` folder with all necessary CSV files.

### 4. Run the Analysis Notebook

```bash
jupyter lab frischmarkt_expiry_analysis_notebook_v2.ipynb
```

Run all cells sequentially to:

* Load and explore data
* Visualize loss patterns
* Train models
* View recommendations

---

## 💡 Future Work & Improvements

* **📆 Advanced Forecasting Models**
  Try `Prophet`, `ARIMA`, or LSTM models for deeper time-series understanding.

* **🧾 Markdown Optimization Engine**
  Build a pricing model to recommend markdown timing/amounts based on demand and perishability.

* **📦 Inventory Simulation**
  Simulate stock policies to test before deploying in stores.

* **📡 Deployment**
  Serve forecasts via API (Flask/FastAPI) for live integration with ordering systems.

* **📊 Store Manager Dashboard**
  Use Streamlit/Dash to deliver easy-to-use visual forecasts and KPIs.

---

## ✍️ Author

**Godwin Oti**
\[[My LinkedIn](https://www.linkedin.com/in/godwin-oti/)]\[[My Portfolio](https://www.datascienceportfol.io/godwinotigo)]
