import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta
from plotly.subplots import make_subplots # Import for subplots
import joblib # For loading the saved model
# Removed: import matplotlib.pyplot as plt
# Removed: import seaborn as sns

# --- Streamlit App Configuration (MUST BE FIRST) ---
st.set_page_config(layout="wide", page_title="FrischMarkt Loss Optimization Dashboard", initial_sidebar_state="expanded")
st.title("🍎 FrischMarkt Fresh Food Loss Optimization Dashboard 🥬")
st.markdown("---")

# --- Define Theme Colors ---
# These colors are taken from your provided theme
PRIMARY_COLOR = "#FF4B4B"        # Red (for highlights)
# BACKGROUND_COLOR and TEXT_COLOR are for the overall Streamlit app, not charts
BACKGROUND_COLOR = "#0E1117"     # Very dark grey/black
TEXT_COLOR = "#FAFAFA"          # White
SECONDARY_BACKGROUND_COLOR = "#262730" # Dark grey
# Removed: NON_HIGHLIGHT_COLOR = "#666666" # A darker grey for non-highlighted bars

# --- Removed: Plotly Theme Layout (to apply to all charts) ---
# Removed: PLOTLY_THEME_LAYOUT = {
# Removed:     'plot_bgcolor': 'white',  # Chart plotting area background set to white
# Removed:     'paper_bgcolor': 'white', # Overall paper background (around the plot area) set to white
# Removed:     'font': {
# Removed:         'color': 'black'      # Default font color for chart elements set to black
# Removed:     },
# Removed:     'xaxis': {
# Removed:         'gridcolor': '#E0E0E0',  # Light grey grid lines for better contrast on white background
# Removed:         'linecolor': 'black',
# Removed:         'zerolinecolor': 'black',
# Removed:         'tickfont': {'color': 'black'},
# Removed:         'title_font': {'color': 'black'}
# Removed:     },
# Removed:     'yaxis': {
# Removed:         'gridcolor': '#E0E0E0',  # Light grey grid lines
# Removed:         'linecolor': 'black',
# Removed:         'zerolinecolor': 'black',
# Removed:         'tickfont': {'color': 'black'},
# Removed:         'title_font': {'color': 'black'}
# Removed:     },
# Removed:     'legend': {
# Removed:         'bgcolor': '#F0F0F0', # Very light grey for legend background
# Removed:         'bordercolor': 'black',
# Removed:         'borderwidth': 1,
# Removed:         'font': {'color': 'black'}
# Removed:     },
# Removed:     'title': {
# Removed:         'font': {'color': 'black'}
# Removed:     }
# Removed: }


# --- Configuration & Data Loading ---
DATA_DIR = 'frischmarkt_data'
# Define paths for model artifacts
MODEL_FILE = os.path.join(DATA_DIR, 'random_forest_model.pkl')
RF_IMPORTANCES_FILE = os.path.join(DATA_DIR, 'rf_feature_importances.csv')
RF_METRICS_FILE = os.path.join(DATA_DIR, 'rf_model_metrics.csv')
RF_PREDICTIONS_FILE = os.path.join(DATA_DIR, 'rf_predictions.csv')

# Define path for the PDF Executive Summary (assuming it's in the DATA_DIR)
EXECUTIVE_SUMMARY_PDF_PATH = os.path.join(DATA_DIR, 'FrischMarkt Expiry Loss Analysis_ Executive Summary.pdf')


@st.cache_data # Cache data loading and preprocessing to improve performance
def load_and_prepare_data():
    """Loads, preprocesses, merges data, and prepares it for dashboard display."""
    try:
        products_df = pd.read_csv(os.path.join(DATA_DIR, 'products_master.csv'))
        stores_df = pd.read_csv(os.path.join(DATA_DIR, 'stores_master.csv'))
        external_df = pd.read_csv(os.path.join(DATA_DIR, 'external_factors.csv'))
        inventory_df = pd.read_csv(os.path.join(DATA_DIR, 'inventory_daily.csv'))
        sales_df = pd.read_csv(os.path.join(DATA_DIR, 'sales_transactions.csv'))
        # ADDED: Load supplier_df
        supplier_df = pd.read_csv(os.path.join(DATA_DIR, 'supplier_performance.csv'))
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}. Make sure 'frischmarkt_data' directory exists with all CSVs and you've run the data generator.")
        st.stop() # Stop the app if data is not found

    # Convert date columns to datetime objects
    for df in [external_df, inventory_df, sales_df]:
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

    # Ensure necessary columns are numeric and handle potential NaNs
    for col in ['unit_cost', 'retail_price', 'shelf_life_days', 'base_expiry_rate', 'profit_margin']:
        if col in products_df.columns:
            products_df[col] = pd.to_numeric(products_df[col], errors='coerce').fillna(0)

    # IMPORTANT: Ensure loss columns are numeric and filled BEFORE merging or any calculations
    for col in ['beginning_inventory', 'received_inventory', 'units_sold', 'units_expired', 'units_marked_down',
                'expiry_loss_eur', 'markdown_loss_eur', 'total_loss_eur', 'expiry_rate']:
        if col in inventory_df.columns:
            inventory_df[col] = pd.to_numeric(inventory_df[col], errors='coerce').fillna(0)

    for col in ['quantity_sold', 'sale_price', 'discount_applied']:
        if col in sales_df.columns:
            sales_df[col] = pd.to_numeric(sales_df[col], errors='coerce').fillna(0)

    # --- Data Merging for Comprehensive Analysis ---
    analysis_df = inventory_df.merge(products_df, on='product_id', how='left')
    analysis_df = analysis_df.merge(stores_df, on='store_id', how='left')
    analysis_df = analysis_df.merge(external_df, on=['date', 'store_id'], how='left')

    # Merge with supplier performance data (via products_df's supplier_id)
    products_with_supplier_df = products_df.merge(supplier_df, on=['product_id', 'supplier_id'], how='left')
    products_supplier_agg = products_with_supplier_df.groupby('product_id').agg(
        actual_shelf_life_days=('actual_shelf_life_days', 'mean'),
        delivery_delay_days=('delivery_delay_days', 'mean')
    ).reset_index()
    analysis_df = analysis_df.merge(products_supplier_agg, on='product_id', how='left')

    # --- Feature Engineering for Demand Forecasting (still needed for analysis_df, but not for model training here) ---
    demand_df = analysis_df.copy() # Use a copy for demand forecasting specific features
    demand_df['day_of_year'] = demand_df['date'].dt.dayofyear
    demand_df['week_of_year'] = demand_df['date'].dt.isocalendar().week.astype(int)
    demand_df['month'] = demand_df['date'].dt.month
    demand_df['day_of_week_num'] = demand_df['date'].dt.dayofweek
    demand_df = demand_df.sort_values(by=['product_id', 'store_id', 'date'])
    demand_df['units_sold_lag1'] = demand_df.groupby(['product_id', 'store_id'])['units_sold'].shift(1).fillna(0)
    demand_df['units_sold_lag7'] = demand_df.groupby(['product_id', 'store_id'])['units_sold'].shift(7).fillna(0)
    demand_df['units_sold_lag30'] = demand_df.groupby(['product_id', 'store_id'])['units_sold'].shift(30).fillna(0)
    demand_df['rolling_mean_sales_7d'] = demand_df.groupby(['product_id', 'store_id'])['units_sold'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
    ).fillna(0)
    for col in ['is_holiday', 'school_holidays', 'local_events', 'competitor_promotion',
                'heat_wave', 'power_outage_risk', 'delivery_disruption', 'temperature_sensitive']:
        if col in demand_df.columns:
            demand_df[col] = demand_df[col].astype(int)

    return products_df, stores_df, external_df, inventory_df, sales_df, analysis_df, demand_df

products_df, stores_df, external_df, inventory_df, sales_df, analysis_df, demand_df = load_and_prepare_data()

# --- Model Loading (Cached Resource) ---
@st.cache_resource # Cache the trained model object
def load_trained_model(model_path):
    """Loads a pre-trained machine learning model."""
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error(f"Model file not found at {model_path}. Please run 'train_demand_model.py' first.")
        st.stop()

@st.cache_data # Cache model related dataframes
def load_model_artifacts(importances_path, metrics_path, predictions_path):
    """Loads model feature importances, metrics, and predictions."""
    try:
        rf_importances = pd.read_csv(importances_path, index_col=0).squeeze('columns')
        rf_metrics_df = pd.read_csv(metrics_path, index_col=0)
        predictions_df = pd.read_csv(predictions_path, parse_dates=['date'])
        return rf_importances, rf_metrics_df, predictions_df
    except FileNotFoundError as e:
        st.error(f"Model artifact file not found: {e}. Please run 'train_demand_model.py' first.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.stop()

# Load the model and its artifacts
best_rf_model = load_trained_model(MODEL_FILE)
rf_importances, rf_metrics_df, predictions_df = load_model_artifacts(RF_IMPORTANCES_FILE, RF_METRICS_FILE, RF_PREDICTIONS_FILE)

# --- Global Filters (Sidebar) ---
st.sidebar.header("Global Filters")
# Convert to Python datetime objects explicitly to avoid FutureWarning
min_date_py = analysis_df['date'].min().to_pydatetime()
max_date_py = analysis_df['date'].max().to_pydatetime()

date_range = st.sidebar.slider(
    "Select Date Range",
    min_value=min_date_py,
    max_value=max_date_py,
    value=(min_date_py, max_date_py),
    format="YYYY-MM-DD"
)

filtered_analysis_df = analysis_df[(analysis_df['date'] >= date_range[0]) & (analysis_df['date'] <= date_range[1])]
filtered_sales_df = sales_df[(sales_df['date'] >= date_range[0]) & (sales_df['date'] <= date_range[1])]

# --- Time Aggregation Filter ---
st.sidebar.markdown("---")
st.sidebar.subheader("Chart Aggregation Level")
aggregation_level = st.sidebar.radio(
    "Select Time Granularity for Trends:",
    ('Daily', 'Weekly', 'Monthly')
)

# --- Add Buttons to Sidebar ---
st.sidebar.markdown("---")
st.sidebar.subheader("Project Resources")

# Add button for PDF download
# IMPORTANT: You need to ensure 'FrischMarkt_Executive_Summary.pdf' exists in your 'frischmarkt_data' directory.
# This code assumes the PDF file is located there.
if os.path.exists(EXECUTIVE_SUMMARY_PDF_PATH):
    with open(EXECUTIVE_SUMMARY_PDF_PATH, "rb") as pdf_file:
        PDF_bytes = pdf_file.read()
    st.sidebar.download_button(
        label="⬇️ Download Executive Summary (PDF)",
        data=PDF_bytes,
        file_name="FrischMarkt Expiry Loss Analysis_ Executive Summary.pdf",
        mime="application/pdf"
    )
else:
    st.sidebar.info("Executive Summary PDF not found. Please ensure 'FrischMarkt Expiry Loss Analysis_ Executive Summary.pdf' is in the 'frischmarkt_data' folder.")


st.sidebar.link_button(
    label="🔗 Visit GitHub Repo",
    url="https://github.com/Godwin-Oti/FrischMarkt-Expiry-Loss-Minimization-Project",
    help="Click to visit the project's GitHub repository for code and documentation."
)

st.sidebar.link_button(
    label="💼 Visit My LinkedIn",
    url="https://www.linkedin.com/in/godwin-oti/", 
    help="Connect with me on LinkedIn!"
)


# --- Executive Summary Content ---
executive_summary_markdown = """
# FrischMarkt Expiry Loss Analysis: Executive Summary & Recommendations
**Date:** July 04, 2025  
**Author:** Godwin Oti  
**Project Title:** Optimizing Fresh Food Inventory to Minimize Expiry Loss for FrischMarkt

## 1. Executive Summary
This report presents the findings and recommendations from a comprehensive data analysis project focused on understanding and mitigating fresh food expiry and markdown losses at FrischMarkt. Our analysis of 2023 operational data reveals that these combined losses are a significant financial burden, amounting to approximately **39.20% of total revenue**, or **€5,779,156.94** annually. This substantial leakage directly impacts profitability and threatens the long-term sustainability of the business.

Through the development of a data-driven demand forecasting model, we have identified key drivers of demand volatility and inventory discrepancies. This project outlines actionable strategies to transition FrischMarkt from reactive loss management to a proactive, optimized inventory system. By implementing the proposed recommendations, FrischMarkt can expect to achieve substantial cost savings, improve operational efficiency, and move towards a more financially stable future.

## 2. Project Objective
The primary objective of this project was to:
* Quantify the scale and specific areas of fresh food expiry and markdown losses across FrischMarkt operations.
* Identify the underlying drivers of demand fluctuations and inventory imbalances at a granular level.
* Develop robust demand forecasting models to inform precise, data-driven ordering decisions.
* Formulate actionable recommendations to minimize losses, reduce waste, and enhance FrischMarkt's overall profitability.

## 3. Key Findings
Our in-depth analysis of FrischMarkt's 2023 data revealed critical insights into the causes and patterns of fresh food losses:

### 3.1 Overall Loss Landscape
* **Total Annual Losses:** In 2023, FrischMarkt incurred **€5,779,156.94** in combined expiry and markdown losses, representing **39.20%** of its total revenue of **€14,740,876.17**.
* **Loss Composition:** Expiry losses (**€4,701,951.56**) significantly outweigh markdown losses (**€1,077,205.38**), indicating that a substantial portion of inventory is expiring before it can even be sold at a discount.

### 3.2 Loss Hotspots
* **Product Categories:** The highest losses are concentrated in the "Fleisch" (Meat), "Frischware" (Fresh Produce), and "Backwaren" (Baked Goods) categories, collectively accounting for the vast majority of total losses.
* **Top Products:** "Rinderhackfleisch" (Beef Mince), across its different product IDs, cumulatively represents the single largest overall loss driver, with a combined total loss (expiry + markdown) exceeding **€1.3 million**. It is closely followed by "Erdbeeren" (Strawberries) and "Schweinekoteletts" (Pork Chops), both with combined total losses exceeding **€600,000**. Other high-loss items include "Äpfel Elstar," "Vollkornbrot," "Leberwurst," "Sonntagsbrötchen," "Bananen," "Weißbrot," and "Kartoffelsalat."
    * **Insight:** For these products, expiry loss is the overwhelming component, indicating that current markdown strategies are not effectively intercepting spoilage.
* **Store Performance:** "FrischMarkt Kreuzberg" and "FrischMarkt Prenzlauer Berg" are the stores with the highest total losses (each nearing **€2.0 million**).
* **Management Quality Impact:** Stores categorized with "Poor" management quality exhibit the highest average expiry rate (over **40%**), significantly higher than "Excellent" (**24%**) or "Good" (**30%**) quality stores. This highlights a direct correlation between operational management and waste.

### 3.3 Demand Forecasting Model Insights
Our Demand Forecasting Model, built using a RandomForest Regressor, achieved an **R-squared (R2) score of 0.6789** and a **Mean Absolute Error (MAE) of 11.97 units**. This indicates that the model explains a substantial portion of demand variability and provides a reasonable average prediction error, especially considering the volatility of perishable goods. The Mean Absolute Percentage Error (MAPE) was 38.18%.

Key drivers of units sold (demand) as identified by the model's feature importances include:
* **Inventory Availability:** `received_inventory` (0.487) and `beginning_inventory` (0.204) are overwhelmingly the most dominant factors. This strongly suggests that sales are heavily influenced by the sheer volume of product available, indicating potential overstocking driving both sales and subsequent expiry.
* **Lagged Sales & Recent Trends:** Past sales patterns (`units_sold_lag1` (0.032), `rolling_mean_sales_7d` (0.026), `units_sold_lag7` (0.015), `units_sold_lag30` (0.012)) are highly predictive of future demand, capturing short-term trends and weekly cycles.
* **Product Characteristics:** `shelf_life_days` (0.072) and `profit_margin` (0.016) also play a significant role, highlighting the inherent perishability and economic value of products.
* **Seasonality:** Time-based features such as `day_of_week_num` (0.014, 0.013) and `day_of_week_Saturday` (0.010) play a consistent role, confirming predictable fluctuations in demand based on the day of the week.
* **External Factors:** `temperature_high_c` (0.009) and `temperature_low_c` (0.007) demonstrate that environmental conditions also influence demand, particularly for temperature-sensitive items.
* **Base Expiry Rate:** `base_expiry_rate` (0.007) also contributes to the model's understanding of demand.

## 4. Strategic Recommendations
Based on these critical findings, we propose the following data-driven strategies to minimize FrischMarkt's expiry loss and enhance its profitability:

### 4.1 Implement a Granular Demand-Driven Ordering System
* **Recommendation:** Develop and integrate an automated ordering system that leverages the demand forecasting model's predictions. This system should generate daily or weekly order recommendations for each product at each store, dynamically adjusting quantities based on forecasted sales, product-specific shelf life, and current inventory levels.
* **Action Plan:**
    * **Phase 1 (Pilot):** Roll out the new ordering system for the top 5 high-loss products (e.g., "Rinderhackfleisch," "Erdbeeren," "Schweinekoteletts") in the highest-loss stores ("FrischMarkt Kreuzberg," "FrischMarkt Prenzlauer Berg").
    * **Phase 2 (Expansion):** Gradually expand to more products and stores, refining the model and processes based on pilot results.
* **Quantified Impact (Estimate):** By reducing overstocking and aligning inventory more precisely with demand, we estimate a **10-15% reduction in current expiry losses**, potentially saving FrischMarkt an additional **€500,000 - €800,000 annually** from the expiry component alone.

### 4.2 Optimize Markdown Strategies with Forecasting Input
* **Recommendation:** Re-evaluate and refine current markdown strategies. While "Aggressive" markdowns lead to high markdown loss, "Conservative" markdowns appear insufficient. Implement a more dynamic, rules-based markdown system for products nearing expiry, informed by forecasted demand and remaining shelf life. For example, trigger earlier, targeted discounts for high-risk items when the forecast indicates low demand.
* **Action Plan:**
    * Develop clear, data-informed guidelines for store managers on markdown timing and depth.
    * Focus on converting at-risk inventory into sales before it becomes a complete expiry loss.
* **Quantified Impact (Estimate):** More effective and timely markdowns could lead to an additional **€200,000 - €400,000 in recovered revenue annually** from items that would otherwise expire completely.

### 4.3 Enhance Operational Excellence and Supplier Review
* **Recommendation:** Address the human and process factors contributing to losses, particularly in stores with lower management quality. Simultaneously, establish a process for reviewing and optimizing supplier performance.
* **Action Plan:**
    * **Targeted Training:** Implement mandatory training programs for store staff, especially in "Poor" and "Average" management quality stores, focusing on best practices for perishable inventory management, stock rotation (FIFO), and proper storage.
    * **Performance Monitoring:** Establish and regularly review KPIs for stock rotation efficiency and expiry rates at the store level.
    * **Supplier Collaboration:** Prioritize suppliers who consistently deliver products with longer actual shelf lives and fewer delays, especially for high-risk products like "Erdbeeren" and "Rinderhackfleisch."
* **Quantified Impact (Estimate):** Improved operational efficiency and supplier quality are projected to reduce overall expiry losses by an additional **5-8%**, translating to **€250,000 - €450,000 annually**, complementing the gains from demand forecasting.

## 5. Conclusion & Roadmap
FrischMarkt faces a significant challenge with fresh food losses, but this analysis provides a clear, data-driven pathway to recovery. By embracing a robust demand forecasting model for inventory optimization and supplementing it with refined markdown strategies and operational improvements, the business can dramatically reduce its financial leakage.

The recommended strategies form a roadmap for sustainable growth:
* Pilot and Scale the demand-driven ordering system, starting with top loss products and stores.
* Iterate and Refine markdown strategies based on real-world impact.
* Invest in Training and foster a culture of data-informed decision-making across all store operations.
* Continuously Monitor model performance and adjust based on new data and evolving market dynamics.

Implementing these recommendations will not only stem the substantial losses but also enhance FrischMarkt's competitiveness, improve fresh product availability for customers, and build a more resilient and profitable retail operation.
"""

# --- Create Tabs ---
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "📄 Executive Summary", # New tab for Executive Summary
    "📊 Overall Summary",
    "🔥 Loss Hotspots",
    "📈 Demand Forecasting Insights",
    "✨ Actionable Recommendations & Quantifiable Impact"
])

# --- Tab 0: Executive Summary ---
with tab0:
    st.markdown(executive_summary_markdown)


# --- Tab 1: Overall Summary ---
with tab1:
    st.header("Overall Business Performance & Loss Summary")

    # Calculate overall metrics using filtered data
    total_revenue = (filtered_sales_df['quantity_sold'] * filtered_sales_df['sale_price']).sum()
    total_expiry_loss = filtered_analysis_df['expiry_loss_eur'].sum()
    total_markdown_loss = filtered_analysis_df['markdown_loss_eur'].sum()
    grand_total_loss = filtered_analysis_df['total_loss_eur'].sum()
    percentage_loss = (grand_total_loss / total_revenue) * 100 if total_revenue > 0 else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Revenue", f"€{total_revenue:,.2f}")
    with col2:
        st.metric("Grand Total Loss", f"€{grand_total_loss:,.2f}")
    with col3:
        st.metric("Loss as % of Revenue", f"{percentage_loss:,.2f}%")

    st.subheader("Loss Breakdown")
    loss_data = pd.DataFrame({
        'Loss Type': ['Expiry Loss', 'Markdown Loss'],
        'Amount (€)': [total_expiry_loss, total_markdown_loss]
    })
    
    # Define colors for loss breakdown - keeping distinct for two types of loss
    loss_colors = {'Expiry Loss': PRIMARY_COLOR, 'Markdown Loss': '#FFC107'} # Red for expiry, Yellow for markdown
    
    fig_loss_breakdown = px.bar(loss_data, x='Loss Type', y='Amount (€)',
                                 title='Breakdown of Total Losses',
                                 color='Loss Type',
                                 color_discrete_map=loss_colors,
                                 text_auto=False)
    fig_loss_breakdown.update_traces(marker_line_width=1, marker_line_color='black') # Changed from PLOTLY_THEME_LAYOUT
    # Removed: fig_loss_breakdown.update_layout(PLOTLY_THEME_LAYOUT) # Apply theme
    st.plotly_chart(fig_loss_breakdown, use_container_width=True)
    st.info("This bar chart visually represents the proportion of total losses attributed to expiry versus markdowns. It provides an immediate understanding of which type of loss is more prevalent and requires more urgent attention.")


    st.subheader(f"Total Loss and Revenue Trends ({aggregation_level})")
    
    # Aggregate data based on selected level
    if aggregation_level == 'Daily':
        daily_loss_agg = filtered_analysis_df.groupby('date')['total_loss_eur'].sum().reset_index()
        daily_revenue_agg = filtered_sales_df.groupby('date').apply(lambda x: (x['quantity_sold'] * x['sale_price']).sum()).reset_index(name='Total Revenue')
        x_axis_title = 'Date'
    elif aggregation_level == 'Weekly':
        daily_loss_agg = filtered_analysis_df.set_index('date').resample('W')['total_loss_eur'].sum().reset_index()
        daily_revenue_agg = filtered_sales_df.set_index('date').resample('W').apply(lambda x: (x['quantity_sold'] * x['sale_price']).sum()).reset_index(name='Total Revenue')
        x_axis_title = 'Week'
    else: # Monthly
        daily_loss_agg = filtered_analysis_df.set_index('date').resample('M')['total_loss_eur'].sum().reset_index()
        daily_revenue_agg = filtered_sales_df.set_index('date').resample('M').apply(lambda x: (x['quantity_sold'] * x['sale_price']).sum()).reset_index(name='Total Revenue')
        x_axis_title = 'Month'

    fig_trends = go.Figure()
    fig_trends.add_trace(go.Scatter(x=daily_loss_agg['date'], y=daily_loss_agg['total_loss_eur'], mode='lines', name='Total Loss (€)', line=dict(color=PRIMARY_COLOR))) # Warm red
    fig_trends.add_trace(go.Scatter(x=daily_revenue_agg['date'], y=daily_revenue_agg['Total Revenue'], mode='lines', name='Total Revenue (€)', line=dict(color='#28A745'))) # Green
    fig_trends.update_layout(title=f'Total {aggregation_level} Loss and Revenue Over Time',
                             xaxis_title=x_axis_title, yaxis_title='Amount (€)',
                             hovermode="x unified",
                             legend=dict(x=0.01, y=0.99, bordercolor='black', borderwidth=1)) # Changed from PLOTLY_THEME_LAYOUT
    # Removed: fig_trends.update_layout(PLOTLY_THEME_LAYOUT) # Apply theme
    st.plotly_chart(fig_trends, use_container_width=True)
    st.info(f"This line chart displays the {aggregation_level.lower()} fluctuations in total losses and total revenue over the selected period. It helps in identifying trends, peak loss periods, and the overall financial health of the business.")


# --- Tab 2: Loss Hotspots ---
with tab2:
    st.header("Identifying Key Loss Hotspots")

    st.subheader("Losses by Product Category")
    losses_by_category = filtered_analysis_df.groupby('category')['total_loss_eur'].sum().sort_values(ascending=False).reset_index()
    
    # Create colors list for highlighting highest bar
    max_loss_cat = losses_by_category['total_loss_eur'].max()
    colors_cat = [
        PRIMARY_COLOR if val == max_loss_cat else '#666666' for val in losses_by_category['total_loss_eur'] # Changed from NON_HIGHLIGHT_COLOR
    ]

    fig_cat_loss = go.Figure(data=[go.Bar(
        x=losses_by_category['category'],
        y=losses_by_category['total_loss_eur'],
        marker_color=colors_cat
    )])
    fig_cat_loss.update_layout(
        title='Total Loss (€) by Product Category',
        xaxis_title='Product Category',
        yaxis_title='Total Loss (€)'
    )
    fig_cat_loss.update_traces(marker_line_width=1, marker_line_color='black') # Changed from PLOTLY_THEME_LAYOUT
    # Removed: fig_cat_loss.update_layout(PLOTLY_THEME_LAYOUT) # Apply theme
    st.plotly_chart(fig_cat_loss, use_container_width=True)
    st.info("This bar chart highlights the product categories contributing most to total losses. The highest bar (in red) indicates the category with the most significant financial leakage due to expiry and markdowns.")


    st.subheader("Losses by Store")
    losses_by_store = filtered_analysis_df.groupby('store_name')['total_loss_eur'].sum().sort_values(ascending=False).reset_index()
    
    # Create colors list for highlighting highest bar
    max_loss_store = losses_by_store['total_loss_eur'].max()
    colors_store = [
        PRIMARY_COLOR if val == max_loss_store else '#666666' for val in losses_by_store['total_loss_eur'] # Changed from NON_HIGHLIGHT_COLOR
    ]

    fig_store_loss = go.Figure(data=[go.Bar(
        x=losses_by_store['store_name'],
        y=losses_by_store['total_loss_eur'],
        marker_color=colors_store
    )])
    fig_store_loss.update_layout(
        title='Total Loss (€) by Store',
        xaxis_title='Store Name',
        yaxis_title='Total Loss (€)'
    )
    fig_store_loss.update_traces(marker_line_width=1, marker_line_color='black') # Changed from PLOTLY_THEME_LAYOUT
    # Removed: fig_store_loss.update_layout(PLOTLY_THEME_LAYOUT) # Apply theme
    st.plotly_chart(fig_store_loss, use_container_width=True)
    st.info("This chart visualizes total losses across different stores. The store with the highest losses (in red) is a primary target for operational improvements and inventory adjustments.")


    st.subheader("Top 10 Products by Expiry Loss")
    # Group by product_name only to ensure unique bars on the x-axis
    expiry_loss_by_product = filtered_analysis_df.groupby('product_name')['expiry_loss_eur'].sum().sort_values(ascending=False).reset_index().head(10)
    
    # Create colors list for highlighting highest bar
    if not expiry_loss_by_product.empty:
        max_loss_product_value = expiry_loss_by_product['expiry_loss_eur'].max()
        colors_product = [
            PRIMARY_COLOR if val == max_loss_product_value else '#666666' for val in expiry_loss_by_product['expiry_loss_eur'] # Changed from NON_HIGHLIGHT_COLOR
        ]
    else:
        colors_product = []

    fig_top_products = go.Figure(data=[go.Bar(
        x=expiry_loss_by_product['product_name'],
        y=expiry_loss_by_product['expiry_loss_eur'],
        marker_color=colors_product
    )])
    fig_top_products.update_layout(
        title='Top 10 Products by Expiry Loss',
        xaxis_title='Product Name',
        yaxis_title='Expiry Loss (€)'
    )
    fig_top_products.update_traces(marker_line_width=1, marker_line_color='black') # Changed from PLOTLY_THEME_LAYOUT
    # Removed: fig_top_products.update_layout(PLOTLY_THEME_LAYOUT) # Apply theme
    st.plotly_chart(fig_top_products, use_container_width=True)
    st.info("This chart identifies the top 10 individual products responsible for the highest expiry losses. The product highlighted in red is the single largest contributor to expired inventory costs. For products like Rinderhackfleisch, high expiry loss is often due to a combination of short shelf life, high purchase volume, and fluctuating demand, making precise inventory management crucial.")

    # ADDED: Expiry vs. Markdown Loss for Top Loss Products
    st.subheader("Expiry vs. Markdown Loss for Top Loss Products")
    N_top_products_for_breakdown = 10
    top_products_total_loss = (
        filtered_analysis_df.groupby('product_name')
        .agg(
            total_expiry_loss=('expiry_loss_eur', 'sum'),
            total_markdown_loss=('markdown_loss_eur', 'sum'),
            total_overall_loss=('total_loss_eur', 'sum') # Get total loss to sort
        )
        .sort_values(by='total_overall_loss', ascending=False) # Sort by total overall loss
        .head(N_top_products_for_breakdown)
        .reset_index()
    )

    if top_products_total_loss.empty:
        st.info("No top loss products found in the selected date range to display expiry vs. markdown breakdown.")
    else:
        # Create Plotly grouped bar chart
        fig_expiry_markdown_breakdown = go.Figure(data=[
            go.Bar(
                name='Expiry Loss (€)',
                x=top_products_total_loss['product_name'],
                y=top_products_total_loss['total_expiry_loss'],
                marker_color=PRIMARY_COLOR # Red for expiry
            ),
            go.Bar(
                name='Markdown Loss (€)',
                x=top_products_total_loss['product_name'],
                y=top_products_total_loss['total_markdown_loss'],
                marker_color='#FFC107' # Yellow/Orange for markdown
            )
        ])

        fig_expiry_markdown_breakdown.update_layout(
            barmode='group', # Grouped bars
            title=f'Expiry vs. Markdown Loss for Top {N_top_products_for_breakdown} Products',
            xaxis_title='Product Name',
            yaxis_title='Loss Amount (€)',
            legend=dict(x=0.01, y=0.99, bordercolor='black', borderwidth=1) # Changed from PLOTLY_THEME_LAYOUT
        )
        fig_expiry_markdown_breakdown.update_traces(marker_line_width=1, marker_line_color='black') # Changed from PLOTLY_THEME_LAYOUT
        # Removed: fig_expiry_markdown_breakdown.update_layout(PLOTLY_THEME_LAYOUT) # Apply theme
        st.plotly_chart(fig_expiry_markdown_breakdown, use_container_width=True)
        st.info("This chart breaks down the total loss for the top 10 loss-generating products into their expiry loss and markdown loss components. It helps identify whether a product's high overall loss is primarily driven by expiration or by price reductions.")


    st.subheader("Overall Expiry Rate by Management Quality")
    overall_expiry_rate_by_mgmt = analysis_df.groupby('management_quality').apply(
        lambda x: (x['units_expired'].sum() / x['beginning_inventory'].replace(0, np.nan).sum()) if x['beginning_inventory'].replace(0, np.nan).sum() > 0 else 0
    ).reset_index(name='overall_expiry_rate')
    overall_expiry_rate_by_mgmt['overall_expiry_rate_pct'] = overall_expiry_rate_by_mgmt['overall_expiry_rate'] * 100

    # Define a custom order for management quality for better visualization
    mgmt_order = ['Excellent', 'Good', 'Average', 'Poor']
    overall_expiry_rate_by_mgmt['management_quality'] = pd.Categorical(overall_expiry_rate_by_mgmt['management_quality'], categories=mgmt_order, ordered=True)
    overall_expiry_rate_by_mgmt = overall_expiry_rate_by_mgmt.sort_values('management_quality')

    # Highlight the highest expiry rate (likely 'Poor'): Red for highest, grey for others
    max_expiry_mgmt = overall_expiry_rate_by_mgmt['overall_expiry_rate_pct'].max()
    colors_mgmt = [
        PRIMARY_COLOR if val == max_expiry_mgmt else '#666666' for val in overall_expiry_rate_by_mgmt['overall_expiry_rate_pct'] # Changed from NON_HIGHLIGHT_COLOR
    ]

    fig_mgmt_expiry = go.Figure(data=[go.Bar(
        x=overall_expiry_rate_by_mgmt['management_quality'],
        y=overall_expiry_rate_by_mgmt['overall_expiry_rate_pct'],
        marker_color=colors_mgmt
    )])
    fig_mgmt_expiry.update_layout(
        title='Overall Expiry Rate (%) by Management Quality',
        xaxis_title='Management Quality',
        yaxis_title='Expiry Rate (%)'
    )
    fig_mgmt_expiry.update_traces(marker_line_width=1, marker_line_color='black') # Changed from PLOTLY_THEME_LAYOUT
    # Removed: fig_mgmt_expiry.update_layout(PLOTLY_THEME_LAYOUT) # Apply theme
    st.plotly_chart(fig_mgmt_expiry, use_container_width=True)
    st.info("This bar chart illustrates the average expiry rate across stores, categorized by their management quality. The red bar indicates the management quality level associated with the highest overall expiry rate, suggesting areas for targeted training or process review.")


    st.subheader("Supplier Actual Shelf Life vs. Expiry Loss")
    supplier_expiry_data = filtered_analysis_df.dropna(subset=['actual_shelf_life_days']).groupby(['supplier_id', 'actual_shelf_life_days']).agg(
        total_expiry_loss=('expiry_loss_eur', 'sum'),
        total_units_expired=('units_expired', 'sum')
    ).reset_index()

    if not supplier_expiry_data.empty:
        fig_supplier_shelf = px.scatter(supplier_expiry_data,
                                        x='actual_shelf_life_days',
                                        y='total_expiry_loss',
                                        size='total_units_expired',
                                        color='supplier_id', # Color by supplier_id
                                        hover_name='supplier_id',
                                        title='Supplier Actual Shelf Life vs. Total Expiry Loss',
                                        labels={'actual_shelf_life_days': 'Actual Shelf Life (Days)',
                                                'total_expiry_loss': 'Total Expiry Loss (€)'},
                                        color_discrete_sequence=px.colors.qualitative.Plotly) # Changed to Plotly palette
        # Removed: fig_supplier_shelf.update_layout(PLOTLY_THEME_LAYOUT) # Apply theme
        st.plotly_chart(fig_supplier_shelf, use_container_width=True)
        st.info("This chart illustrates if products from suppliers delivering shorter 'actual shelf life' contribute more to expiry loss. Larger circles indicate more units expired.")
    else:
        st.info("No data available for Supplier Actual Shelf Life vs. Expiry Loss in the selected date range.")


    st.subheader("Markdown Aggressiveness vs. Markdown Outcomes")
    markdown_outcome_data = filtered_analysis_df.groupby(['markdown_aggressiveness']).agg(
        total_markdown_loss=('markdown_loss_eur', 'sum'),
        total_units_marked_down=('units_marked_down', 'sum'),
        total_units_sold_at_markdown=(
            'units_sold', lambda x: x[filtered_analysis_df['markdown_loss_eur'] > 0].sum()
        )
    ).reset_index()

    markdown_outcome_data['avg_markdown_value_per_unit'] = (markdown_outcome_data['total_markdown_loss'] / markdown_outcome_data['total_units_marked_down']).fillna(0)
    markdown_outcome_data['conversion_rate_marked_down'] = (markdown_outcome_data['total_units_sold_at_markdown'] / markdown_outcome_data['total_units_marked_down']).fillna(0) * 100

    # Define consistent colors for markdown bars
    markdown_loss_color = '#FF7F0E' # Darker orange
    units_marked_down_color = '#FFBB78' # Lighter orange

    fig_markdown_outcomes = go.Figure(data=[
        go.Bar(name='Total Markdown Loss (€)', x=markdown_outcome_data['markdown_aggressiveness'], y=markdown_outcome_data['total_markdown_loss'], marker_color=markdown_loss_color),
        go.Bar(name='Total Units Marked Down', x=markdown_outcome_data['markdown_aggressiveness'], y=markdown_outcome_data['total_units_marked_down'], marker_color=units_marked_down_color)
    ])
    fig_markdown_outcomes.update_layout(barmode='group', title='Markdown Aggressiveness: Loss vs. Units Marked Down',
                                        xaxis_title='Markdown Aggressiveness', yaxis_title='Amount/Units',
                                        legend=dict(x=0.01, y=0.99, bordercolor='black', borderwidth=1)) # Changed from PLOTLY_THEME_LAYOUT
    fig_markdown_outcomes.update_traces(marker_line_width=1, marker_line_color='black') # Changed from PLOTLY_THEME_LAYOUT
    # Removed: fig_markdown_outcomes.update_layout(PLOTLY_THEME_LAYOUT) # Apply theme
    st.plotly_chart(fig_markdown_outcomes, use_container_width=True)

    fig_conversion_rate = px.bar(markdown_outcome_data, x='markdown_aggressiveness', y='conversion_rate_marked_down',
                                 title='Conversion Rate of Marked Down Units (%)',
                                 labels={'conversion_rate_marked_down': 'Conversion Rate (%)', 'markdown_aggressiveness': 'Markdown Aggressiveness'},
                                 color='markdown_aggressiveness',
                                 color_discrete_sequence=px.colors.qualitative.Safe, # Using Safe qualitative palette
                                 text_auto=False)
    fig_conversion_rate.update_traces(marker_line_width=1, marker_line_color='black') # Changed from PLOTLY_THEME_LAYOUT
    # Removed: fig_conversion_rate.update_layout(PLOTLY_THEME_LAYOUT) # Apply theme
    st.plotly_chart(fig_conversion_rate, use_container_width=True)
    st.info("These charts compare different markdown strategies. The first shows total loss and units marked down, while the second indicates how effectively marked-down units are sold.")


    st.subheader("Temperature vs. Expiry Rate (Temperature-Sensitive Products)")
    temp_sensitive_expiry_data = filtered_analysis_df[filtered_analysis_df['temperature_sensitive'] == 1].copy()
    
    if not temp_sensitive_expiry_data.empty:
        daily_temp_expiry = temp_sensitive_expiry_data.groupby('date').agg(
            avg_temp=('temperature_high_c', 'mean'),
            total_expired=('units_expired', 'sum'),
            total_beginning_inv=('beginning_inventory', 'sum')
        ).reset_index()
        daily_temp_expiry['expiry_rate_daily'] = (daily_temp_expiry['total_expired'] / daily_temp_expiry['total_beginning_inv'].replace(0, np.nan)).fillna(0)

        fig_temp_expiry = px.scatter(daily_temp_expiry,
                                     x='avg_temp',
                                     y='expiry_rate_daily',
                                     size='total_expired',
                                     color='expiry_rate_daily',
                                     color_continuous_scale=px.colors.sequential.Plasma, # Changed to Plasma
                                     title='Average Daily Temperature vs. Expiry Rate for Sensitive Products',
                                     labels={'avg_temp': 'Average High Temperature (°C)',
                                             'expiry_rate_daily': 'Daily Expiry Rate',
                                             'total_expired': 'Total Expired Units'},
                                     hover_name='date',
                                     trendline='ols')
        # Removed: fig_temp_expiry.update_layout(PLOTLY_THEME_LAYOUT) # Apply theme
        st.plotly_chart(fig_temp_expiry, use_container_width=True)
        st.info("This scatter plot shows the relationship between daily high temperatures and expiry rates for temperature-sensitive products. Larger circles indicate more expired units.")
    else:
        st.info("No data available for Temperature-Sensitive Product Expiry Rate in the selected date range.")

    st.subheader(f"Inventory vs Expired/Markdown Units Over Time for Top Loss Products ({aggregation_level})")
    
    # Determine the number of top loss products to display for inventory profiles
    N_inventory_profiles = 6

    # Get the product names of the top N total loss products from the filtered data
    top_loss_product_names = (
        filtered_analysis_df.groupby('product_name')['total_loss_eur']
        .sum()
        .sort_values(ascending=False)
        .head(N_inventory_profiles)
        .index.tolist()
    )

    if not top_loss_product_names:
        st.info("No top loss products found in the selected date range to display inventory profiles.")
    else:
        # Get all product_ids that correspond to these top product names from the filtered data
        top_loss_product_ids_from_names = filtered_analysis_df[filtered_analysis_df['product_name'].isin(top_loss_product_names)]['product_id'].unique().tolist()

        # Aggregate data based on selected level
        if aggregation_level == 'Daily':
            inventory_profiles_agg = (
                filtered_analysis_df[filtered_analysis_df['product_id'].isin(top_loss_product_ids_from_names)]
                .groupby(['product_name', 'date'])[['beginning_inventory', 'units_expired', 'units_marked_down']]
                .sum()
                .reset_index()
            )
            x_axis_title_inv = 'Date'
        elif aggregation_level == 'Weekly':
            inventory_profiles_agg = (
                filtered_analysis_df[filtered_analysis_df['product_id'].isin(top_loss_product_ids_from_names)]
                .set_index('date')
                .groupby('product_name')
                .resample('W')[['beginning_inventory', 'units_expired', 'units_marked_down']]
                .sum()
                .reset_index()
            )
            x_axis_title_inv = 'Week'
        else: # Monthly
            inventory_profiles_agg = (
                filtered_analysis_df[filtered_analysis_df['product_id'].isin(top_loss_product_ids_from_names)]
                .set_index('date')
                .groupby('product_name')
                .resample('M')[['beginning_inventory', 'units_expired', 'units_marked_down']]
                .sum()
                .reset_index()
            )
            x_axis_title_inv = 'Month'

        # Melt for Plotly
        inventory_profiles_long = inventory_profiles_agg.melt(
            id_vars=['product_name', 'date'], var_name='Metric', value_name='Units'
        )

        # Define a custom palette for clarity, consistent with Streamlit dashboard
        custom_palette = {
            'beginning_inventory': '#636EFA', # Blue
            'units_expired': PRIMARY_COLOR, # Red
            'units_marked_down': '#FECB52' # Yellow/Orange
        }

        # Create subplots for each top loss product using Plotly
        num_products = len(top_loss_product_names)
        cols = 2
        rows = (num_products + cols - 1) // cols # Calculate rows needed

        fig_inventory_profiles = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f"{name}" for name in top_loss_product_names],
            vertical_spacing=0.15,
            horizontal_spacing=0.05
        )

        for i, product_name in enumerate(top_loss_product_names):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            product_data = inventory_profiles_long[inventory_profiles_long['product_name'] == product_name]

            for metric in ['beginning_inventory', 'units_expired', 'units_marked_down']:
                metric_data = product_data[product_data['Metric'] == metric]
                fig_inventory_profiles.add_trace(
                    go.Scatter(
                        x=metric_data['date'],
                        y=metric_data['Units'],
                        mode='lines',
                        name=metric.replace('_', ' ').title(),
                        line=dict(color=custom_palette[metric]),
                        legendgroup=metric,
                        showlegend=(i == 0) # Show legend only for the first subplot
                    ),
                    row=row, col=col
                )
            fig_inventory_profiles.update_xaxes(title_text=x_axis_title_inv, row=row, col=col)
            fig_inventory_profiles.update_yaxes(title_text="Units", row=row, col=col)

        fig_inventory_profiles.update_layout(
            title_text=f"📦 Inventory vs Expired/Markdown Units Over Time for Top Loss Products ({aggregation_level})",
            height=400 * rows,
            showlegend=True,
            hovermode="x unified",
            legend=dict(x=0.01, y=0.99, bordercolor='black', borderwidth=1) # Changed from PLOTLY_THEME_LAYOUT
        )
        # Removed: fig_inventory_profiles.update_layout(PLOTLY_THEME_LAYOUT) # Apply theme
        st.plotly_chart(fig_inventory_profiles, use_container_width=True)

        st.info(f"This multi-panel chart visualizes the {aggregation_level.lower()} beginning inventory, units expired, and units marked down for the top loss-generating products. It helps identify overstocking issues and their direct impact on losses.")


# --- Tab 3: Demand Forecasting Insights ---
with tab3:
    st.header("📈 Demand Forecasting Model Insights")

    # Filter predictions_df to the selected date range
    # Moved this outside the conditional block to ensure it's always defined
    if not predictions_df.empty:
        filtered_predictions_df = predictions_df[
            (predictions_df['date'] >= date_range[0]) & 
            (predictions_df['date'] <= date_range[1])
        ]
    else:
        filtered_predictions_df = pd.DataFrame() # Ensure it's an empty DataFrame if predictions_df is empty


    if best_rf_model is not None and not rf_metrics_df.empty and not filtered_predictions_df.empty:
        st.subheader("Model Performance Metrics (RandomForestRegressor)")
        col_mae, col_mape, col_r2 = st.columns(3)
        with col_mae:
            st.metric("Mean Absolute Error (MAE)", f"{rf_metrics_df.loc['MAE', 'Value']:.2f} units")
        with col_mape:
            mape_value = rf_metrics_df.loc['MAPE', 'Value']
            if not np.isnan(mape_value):
                st.metric("Mean Absolute Percentage Error (MAPE)", f"{mape_value:.2f}%")
            else:
                st.info("MAPE not available (likely due to zero actuals).")
        with col_r2:
            st.metric("R-squared (R2) Score", f"{rf_metrics_df.loc['R2', 'Value']:.4f}")

        st.markdown("---")

        st.subheader("Top 15 Feature Importances for Demand Prediction")
        
        # Highlight the highest feature importance: Green for highest, grey for others
        top_15_importances = rf_importances.nlargest(15)
        
        # Create colors list for highlighting highest bar
        max_importance_val = top_15_importances.max()
        colors_importances = [
            '#28A745' if val == max_importance_val else '#666666' for val in top_15_importances.values # Changed from NON_HIGHLIGHT_COLOR
        ]

        fig_importances = go.Figure(data=[go.Bar(
            x=top_15_importances.values,
            y=top_15_importances.index,
            orientation='h',
            marker_color=colors_importances
        )])
        fig_importances.update_layout(
            title='Top 15 Feature Importances for Units Sold Prediction',
            xaxis_title='Importance',
            yaxis_title='Feature',
            yaxis={'categoryorder':'total ascending'}
        )
        fig_importances.update_traces(marker_line_width=1, marker_line_color='black') # Changed from PLOTLY_THEME_LAYOUT
        # Removed: fig_importances.update_layout(PLOTLY_THEME_LAYOUT) # Apply theme
        st.plotly_chart(fig_importances, use_container_width=True)
        st.info("This chart shows which factors the model considers most important in predicting daily sales. High importance indicates a strong influence on demand.")

        st.markdown("---")

        st.subheader(f"Actual vs. Predicted Sales Trend ({aggregation_level})")
        
        # Get unique products and stores from the filtered predictions
        available_products = filtered_predictions_df['product_name'].unique()
        available_stores = filtered_predictions_df['store_name'].unique()

        if len(available_products) > 0 and len(available_stores) > 0:
            selected_product = st.selectbox("Select Product", available_products)
            selected_store = st.selectbox("Select Store", available_stores)

            plot_data = filtered_predictions_df[
                (filtered_predictions_df['product_name'] == selected_product) &
                (filtered_predictions_df['store_name'] == selected_store)
            ].sort_values('date')

            if not plot_data.empty:
                # Aggregate prediction data
                if aggregation_level == 'Daily':
                    plot_data_agg = plot_data.copy()
                elif aggregation_level == 'Weekly':
                    plot_data_agg = plot_data.set_index('date').resample('W')[['actual_units_sold', 'predicted_rf']].sum().reset_index()
                else: # Monthly
                    plot_data_agg = plot_data.set_index('date').resample('M')[['actual_units_sold', 'predicted_rf']].sum().reset_index()

                fig_pred_vs_actual = go.Figure()
                fig_pred_vs_actual.add_trace(go.Scatter(x=plot_data_agg['date'], y=plot_data_agg['actual_units_sold'], mode='lines', name='Actual Units Sold', line=dict(color='#28A745'))) # Green for actual
                fig_pred_vs_actual.add_trace(go.Scatter(x=plot_data_agg['date'], y=plot_data_agg['predicted_rf'], mode='lines', name='Predicted Units Sold (RF)', line=dict(color='#007BFF', dash='dot'))) # Blue for predicted
                fig_pred_vs_actual.update_layout(
                    title=f'Actual vs. Predicted Sales for {selected_product} at {selected_store} ({aggregation_level})',
                    xaxis_title=x_axis_title,
                    yaxis_title='Units Sold',
                    hovermode="x unified",
                    legend=dict(x=0.01, y=0.99, bordercolor='black', borderwidth=1) # Changed from PLOTLY_THEME_LAYOUT
                )
                # Removed: fig_pred_vs_actual.update_layout(PLOTLY_THEME_LAYOUT) # Apply theme
                st.plotly_chart(fig_pred_vs_actual, use_container_width=True)
                st.info(f"This chart compares the model's predictions against actual sales for a selected product and store, helping to visualize forecasting accuracy over time at a {aggregation_level.lower()} granularity.")
            else:
                st.info("No data available for the selected product and store in the predictions within the chosen date range.")
        else:
            st.info("No prediction data available for plotting within the selected date range. Please adjust the global date filter or ensure the model training script has generated predictions for this range.")
    else:
        st.info("Demand forecasting model insights are not available. Please ensure the model and its artifacts are correctly loaded and data is available for the selected date range.")


# --- Tab 4: Recommendations & Impact ---
with tab4:
    st.header("Actionable Recommendations & Quantified Impact")

    st.subheader("1. Implement a Granular Demand-Driven Ordering System")
    st.write(f"Leverage the advanced RandomForest demand forecasting model (MAE: {rf_metrics_df.loc['MAE', 'Value']:.2f} units, R2: {rf_metrics_df.loc['R2', 'Value']:.4f}) to generate precise daily/weekly order recommendations for each product at each store. This minimizes overstocking, a primary driver of expiry loss.")
    st.markdown(f"**Potential Impact:** By reducing overstocking, we estimate a **10-15% reduction in expiry loss** for targeted products, potentially saving FrischMarkt an additional **€500,000 - €800,000 annually**.")

    st.subheader("2. Optimize Markdown Strategies")
    st.write("Introduce rules-based markdown triggers informed by demand forecasts and days-to-expiry. The model highlights `received_inventory` and `beginning_inventory` as highly important features, indicating that current stock levels are key drivers of future sales. Smarter markdowns can convert at-risk inventory into sales, recouping value.")
    st.markdown(f"**Potential Impact:** Proactive and smarter markdowns could increase recovered revenue by **€200,000 - €400,000 annually**.")

    st.subheader("3. Enhance Operational Excellence and Supplier Review")
    st.write("Address human and process factors through targeted training (stock rotation, temperature control) and prioritize suppliers delivering fresher products. The model's feature importances also show `shelf_life_days` and `temperature_high_c` as significant, emphasizing the importance of product freshness and environmental controls.")
    st.markdown(f"**Potential Impact:** Improved operational efficiency and supplier quality are projected to reduce overall expiry losses by an additional **5-8%**, translating to **€250,000 - €450,000 annually**, complementing the gains from demand forecasting.")

    st.subheader("Overall Projected Impact")
    st.markdown(f"By implementing these data-driven strategies, FrischMarkt can transform its inventory management, leading to an estimated **total loss reduction of 15-25%**, translating to **over €950,000 - €1,650,000 annually**.")

    st.subheader("Roadmap for Implementation")
    st.markdown("""
    * **Phase 1 (Pilot):** Roll out demand-driven ordering for top high-loss products (e.g., "Rinderhackfleisch," "Erdbeeren," "Schweinekoteletts") in 2-3 struggling stores.
    * **Phase 2 (Expansion):** Gradually expand to more products and stores, refining the model and processes based on pilot results.
    * **Continuous Improvement:** Regularly monitor model performance and adjust based on new data and evolving market dynamics.
    * **Training:** Provide comprehensive training for store staff on new tools and processes, emphasizing the importance of accurate inventory counts and adherence to new markdown policies.
    """)
