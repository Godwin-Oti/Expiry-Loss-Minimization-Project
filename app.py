import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta
# Commenting out model-related imports for now
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
# import lightgbm as lgb
# from scipy.stats import randint, uniform

# --- Streamlit App Configuration (MUST BE FIRST) ---
st.set_page_config(layout="wide", page_title="FrischMarkt Loss Optimization Dashboard", initial_sidebar_state="expanded")
st.title("🍎 FrischMarkt Fresh Food Loss Optimization Dashboard")
st.markdown("---")

# --- Configuration & Data Loading ---
DATA_DIR = 'frischmarkt_data'
# Commenting out model file paths for now
# PREDICTIONS_FILE = os.path.join(DATA_DIR, 'predictions.csv')
# RF_IMPORTANCES_FILE = os.path.join(DATA_DIR, 'rf_importances.csv')
# LGBM_IMPORTANCES_FILE = os.path.join(DATA_DIR, 'lgbm_importances.csv')
# MODEL_METRICS_FILE = os.path.join(DATA_DIR, 'model_metrics.csv')


@st.cache_data # Cache data loading and preprocessing to improve performance
def load_and_prepare_data():
    """Loads, preprocesses, merges data, and prepares it for dashboard display."""
    st.info("Loading and preparing data...")
    try:
        products_df = pd.read_csv(os.path.join(DATA_DIR, 'products_master.csv'))
        stores_df = pd.read_csv(os.path.join(DATA_DIR, 'stores_master.csv'))
        external_df = pd.read_csv(os.path.join(DATA_DIR, 'external_factors.csv'))
        inventory_df = pd.read_csv(os.path.join(DATA_DIR, 'inventory_daily.csv'))
        sales_df = pd.read_csv(os.path.join(DATA_DIR, 'sales_transactions.csv'))
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}. Make sure 'frischmarkt_data' directory exists with all CSVs.")
        st.stop() # Stop the app if data is not found

    # Convert date columns to datetime objects
    for df in [external_df, inventory_df, sales_df]:
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

    # Ensure necessary columns are numeric and handle potential NaNs
    for col in ['unit_cost', 'retail_price', 'shelf_life_days', 'base_expiry_rate', 'profit_margin']:
        if col in products_df.columns:
            products_df[col] = pd.to_numeric(products_df[col], errors='coerce').fillna(0)

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

# --- Model Training / Loading (Cached Resource) - COMMENTED OUT FOR NOW ---
# @st.cache_resource # Cache the trained model results
# def get_model_results(demand_df_input):
#     """
#     Trains and evaluates the demand forecasting models, or loads pre-calculated results.
#     Assumes model training happens in a separate notebook and results are saved.
#     """
#     if os.path.exists(PREDICTIONS_FILE) and os.path.exists(RF_IMPORTANCES_FILE) and \
#        os.path.exists(LGBM_IMPORTANCES_FILE) and os.path.exists(MODEL_METRICS_FILE):
#         st.info("Loading pre-calculated model predictions and importances...")
#         predictions_df = pd.read_csv(PREDICTIONS_FILE, parse_dates=['date'])
#         rf_importances = pd.read_csv(RF_IMPORTANCES_FILE, index_col=0).squeeze('columns')
#         lgbm_importances = pd.read_csv(LGBM_IMPORTANCES_FILE, index_col=0).squeeze('columns')
#         model_metrics_df = pd.read_csv(MODEL_METRICS_FILE, index_col=0)

#         rf_metrics = {'mae': model_metrics_df.loc['RandomForestRegressor (Tuned)', 'MAE'],
#                       'mape': model_metrics_df.loc['RandomForestRegressor (Tuned)', 'MAPE'],
#                       'r2': model_metrics_df.loc['RandomForestRegressor (Tuned)', 'R2'],
#                       'importances': rf_importances}
#         lgbm_metrics = {'mae': model_metrics_df.loc['LGBMRegressor', 'MAE'],
#                         'mape': model_metrics_df.loc['LGBMRegressor', 'MAPE'],
#                         'r2': model_metrics_df.loc['LGBMRegressor', 'R2'],
#                         'importances': lgbm_importances}

#         return {
#             'predictions_df': predictions_df,
#             'rf_metrics': rf_metrics,
#             'lgbm_metrics': lgbm_metrics
#         }
#     else:
#         st.warning("Model prediction files not found. Training models now. This might take a moment.")
#         # --- Model Training Logic (as in your Jupyter Notebook) ---
#         numerical_features = [
#             'beginning_inventory', 'received_inventory', 'unit_cost', 'retail_price',
#             'shelf_life_days', 'base_expiry_rate', 'profit_margin',
#             'temperature_high_c', 'temperature_low_c', 'precipitation_mm', 'expiry_risk_multiplier',
#             'demographics_score', 'refrigeration_capacity', 'staff_efficiency_score',
#             'distance_from_warehouse_km', 'day_of_year', 'week_of_year', 'month', 'day_of_week_num',
#             'units_sold_lag1', 'units_sold_lag7', 'units_sold_lag30', 'rolling_mean_sales_7d'
#         ]

#         categorical_features_for_model = [
#             'category', 'subcategory', 'brand', 'location_type', 'foot_traffic_level',
#             'management_quality', 'markdown_aggressiveness', 'day_of_week',
#         ]

#         boolean_features = [
#             'is_holiday', 'school_holidays', 'local_events', 'competitor_promotion',
#             'heat_wave', 'power_outage_risk', 'delivery_disruption', 'temperature_sensitive'
#         ]

#         # Filter features to ensure they exist
#         all_features_to_include = numerical_features + categorical_features_for_model + boolean_features
#         current_features = [col for col in all_features_to_include if col in demand_df_input.columns]

#         # Identify the split date
#         split_date = demand_df_input['date'].max() - timedelta(days=90)

#         # Separate test set data BEFORE encoding to retain original product_id and store_id for plotting
#         test_data_original_ids = demand_df_input[demand_df_input['date'] > split_date][['date', 'product_id', 'store_id', 'units_sold']].copy()

#         # Create dummy variables for categorical features, including product_id and store_id for model training
#         demand_df_encoded = pd.get_dummies(demand_df_input, columns=[f for f in categorical_features_for_model if f in demand_df_input.columns] + ['product_id', 'store_id'], drop_first=True)

#         # Combine all feature column names
#         X_cols = numerical_features + boolean_features + [col for col in demand_df_encoded.columns if any(col.startswith(f"{cat}_") for cat in categorical_features_for_model + ['product_id', 'store_id'])]

#         # Filter X_cols to ensure only columns present in demand_df_encoded are used
#         X_cols = [col for col in X_cols if col in demand_df_encoded.columns]

#         X = demand_df_encoded[X_cols]
#         y = demand_df_encoded['units_sold']

#         X = X.fillna(0)
#         y = y.fillna(0)

#         X_train = X[demand_df_encoded['date'] <= split_date]
#         y_train = y[demand_df_encoded['date'] <= split_date]

#         X_test = X[demand_df_encoded['date'] > split_date]
#         y_test = y[demand_df_encoded['date'] > split_date]

#         # --- Model 1: RandomForestRegressor with RandomizedSearchCV ---
#         param_dist = {
#             'n_estimators': randint(50, 200),
#             'max_features': uniform(0.6, 0.3),
#             'max_depth': randint(5, 20),
#             'min_samples_split': randint(2, 10),
#             'min_samples_leaf': randint(1, 5)
#         }
#         rf = RandomForestRegressor(random_state=42)
#         rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_dist,
#                                        n_iter=5, cv=3, verbose=0, random_state=42, n_jobs=-1, scoring='neg_mean_absolute_error')
#         rf_random.fit(X_train, y_train)
#         best_rf_model = rf_random.best_estimator_
#         y_pred_rf = best_rf_model.predict(X_test)

#         # --- Model 2: LightGBM Regressor ---
#         lgbm_model = lgb.LGBMRegressor(objective='regression', metric='mae', n_estimators=100, learning_rate=0.05, num_leaves=31, max_depth=-1, random_state=42, n_jobs=-1)
#         lgbm_model.fit(X_train, y_train)
#         y_pred_lgbm = lgbm_model.predict(X_test)

#         # Combine predictions with original IDs for plotting
#         predictions_df = test_data_original_ids.copy()
#         predictions_df['actual_units_sold'] = y_test.values
#         predictions_df['predicted_rf'] = y_pred_rf
#         predictions_df['predicted_lgbm'] = y_pred_lgbm

#         # Calculate metrics
#         mae_rf = mean_absolute_error(y_test, y_pred_rf)
#         mape_rf = np.mean(np.abs((y_test - y_pred_rf) / y_test.replace(0, np.nan))) * 100
#         r2_rf = r2_score(y_test, y_pred_rf)

#         mae_lgbm = mean_absolute_error(y_test, y_pred_lgbm)
#         mape_lgbm = np.mean(np.abs((y_test - y_pred_lgbm) / y_test.replace(0, np.nan))) * 100
#         r2_lgbm = r2_score(y_test, y_pred_lgbm)

#         # Feature Importances
#         feature_importances_rf = pd.Series(best_rf_model.feature_importances_, index=X.columns)
#         feature_importances_lgbm = pd.Series(lgbm_model.feature_importances_, index=X.columns)

#         # Save results for faster loading next time
#         predictions_df.to_csv(PREDICTIONS_FILE, index=False)
#         rf_importances.to_csv(RF_IMPORTANCES_FILE, index=True) # Save index as feature name
#         lgbm_importances.to_csv(LGBM_IMPORTANCES_FILE, index=True) # Save index as feature name
#         model_metrics_data = pd.DataFrame({
#             'Model': ['RandomForestRegressor (Tuned)', 'LGBMRegressor'],
#             'MAE': [mae_rf, mae_lgbm],
#             'R2': [r2_rf, r2_lgbm],
#             'MAPE': [mape_rf, mape_lgbm]
#         }).set_index('Model')
#         model_metrics_data.to_csv(MODEL_METRICS_FILE, index=True)

#         return {
#             'predictions_df': predictions_df,
#             'rf_metrics': {'mae': mae_rf, 'mape': mape_rf, 'r2': r2_rf, 'importances': feature_importances_rf},
#             'lgbm_metrics': {'mae': mae_lgbm, 'mape': mape_lgbm, 'r2': r2_lgbm, 'importances': lgbm_importances}
#         }

# # --- Run Model Training / Loading (cached) - COMMENTED OUT FOR NOW ---
# model_results = get_model_results(demand_df)
# predictions_df = model_results['predictions_df']
# rf_metrics = model_results['rf_metrics']
# lgbm_metrics = model_results['lgbm_metrics']

# --- Global Filters (Sidebar) ---
st.sidebar.header("Global Filters")
min_date = analysis_df['date'].min().to_pydatetime()
max_date = analysis_df['date'].max().to_pydatetime()

date_range = st.sidebar.slider(
    "Select Date Range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM-DD"
)

filtered_analysis_df = analysis_df[(analysis_df['date'] >= date_range[0]) & (analysis_df['date'] <= date_range[1])]
filtered_sales_df = sales_df[(sales_df['date'] >= date_range[0]) & (sales_df['date'] <= date_range[1])]


# --- Create Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overall Summary",
    "🔥 Loss Hotspots",
    "📈 Demand Forecasting Insights",
    "💡 Recommendations & Impact"
])

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
    fig_loss_breakdown = px.bar(loss_data, x='Loss Type', y='Amount (€)',
                                 title='Breakdown of Total Losses',
                                 color='Loss Type',
                                 color_discrete_map={'Expiry Loss': 'firebrick', 'Markdown Loss': 'darkorange'})
    st.plotly_chart(fig_loss_breakdown, use_container_width=True)

    st.subheader("Daily Loss and Revenue Trends")
    daily_loss = filtered_analysis_df.groupby('date')['total_loss_eur'].sum().reset_index()
    daily_revenue = filtered_sales_df.groupby('date').apply(lambda x: (x['quantity_sold'] * x['sale_price']).sum()).reset_index(name='Total Revenue')

    fig_trends = go.Figure()
    fig_trends.add_trace(go.Scatter(x=daily_loss['date'], y=daily_loss['total_loss_eur'], mode='lines', name='Total Daily Loss (€)', line=dict(color='red')))
    fig_trends.add_trace(go.Scatter(x=daily_revenue['date'], y=daily_revenue['Total Revenue'], mode='lines', name='Total Daily Revenue (€)', line=dict(color='green')))
    fig_trends.update_layout(title='Total Daily Loss and Revenue Over Time',
                             xaxis_title='Date', yaxis_title='Amount (€)',
                             hovermode="x unified")
    st.plotly_chart(fig_trends, use_container_width=True)


# --- Tab 2: Loss Hotspots ---
with tab2:
    st.header("Identifying Key Loss Hotspots")

    st.subheader("Losses by Product Category")
    losses_by_category = filtered_analysis_df.groupby('category')['total_loss_eur'].sum().sort_values(ascending=False).reset_index()
    fig_cat_loss = px.bar(losses_by_category, x='category', y='total_loss_eur',
                          title='Total Loss (€) by Product Category',
                          labels={'total_loss_eur': 'Total Loss (€)', 'category': 'Product Category'},
                          color='category',
                          color_discrete_sequence=px.colors.qualitative.Vivid)
    st.plotly_chart(fig_cat_loss, use_container_width=True)

    st.subheader("Losses by Store")
    losses_by_store = filtered_analysis_df.groupby('store_name')['total_loss_eur'].sum().sort_values(ascending=False).reset_index()
    fig_store_loss = px.bar(losses_by_store, x='store_name', y='total_loss_eur',
                            title='Total Loss (€) by Store',
                            labels={'total_loss_eur': 'Total Loss (€)', 'store_name': 'Store Name'},
                            color='store_name',
                            color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_store_loss, use_container_width=True)

    st.subheader("Top 10 Products by Expiry Loss")
    expiry_loss_by_product = filtered_analysis_df.groupby(['product_id', 'product_name'])['expiry_loss_eur'].sum().sort_values(ascending=False).reset_index().head(10)
    fig_top_products = px.bar(expiry_loss_by_product, x='product_name', y='expiry_loss_eur',
                              title='Top 10 Products by Expiry Loss',
                              labels={'expiry_loss_eur': 'Expiry Loss (€)', 'product_name': 'Product Name'},
                              color='product_name',
                              color_discrete_sequence=px.colors.qualitative.Set1)
    st.plotly_chart(fig_top_products, use_container_width=True)

    st.subheader("Overall Expiry Rate by Management Quality")
    # Using analysis_df directly here as it's for overall rates, not filtered by date range slider
    overall_expiry_rate_by_mgmt = analysis_df.groupby('management_quality').apply(
        lambda x: (x['units_expired'].sum() / x['beginning_inventory'].replace(0, np.nan).sum()) if x['beginning_inventory'].replace(0, np.nan).sum() > 0 else 0
    ).reset_index(name='overall_expiry_rate')
    overall_expiry_rate_by_mgmt['overall_expiry_rate_pct'] = overall_expiry_rate_by_mgmt['overall_expiry_rate'] * 100

    fig_mgmt_expiry = px.bar(overall_expiry_rate_by_mgmt, x='management_quality', y='overall_expiry_rate_pct',
                             title='Overall Expiry Rate (%) by Management Quality',
                             labels={'overall_expiry_rate_pct': 'Expiry Rate (%)', 'management_quality': 'Management Quality'},
                             color='management_quality',
                             color_discrete_sequence=px.colors.qualitative.Dark2)
    st.plotly_chart(fig_mgmt_expiry, use_container_width=True)


# --- Tab 3: Demand Forecasting Insights (Model Disabled) ---
with tab3:
    st.header("Demand Forecasting Model Insights")
    st.warning("The Demand Forecasting Model section is currently disabled. Please enable model training/loading in the script to view these insights.")
    st.info("Once enabled, this section will display model performance metrics, feature importances, and interactive actual vs. predicted sales plots.")


# --- Tab 4: Recommendations & Impact ---
with tab4:
    st.header("Actionable Recommendations & Quantified Impact")

    st.subheader("1. Implement a Granular Demand-Driven Ordering System")
    st.write("Leverage the demand forecasting model to generate precise daily/weekly order recommendations for each product at each store. This minimizes overstocking, a primary driver of expiry loss.")
    st.markdown(f"**Potential Impact:** By reducing overstocking, we estimate a **[Your Estimated Percentage Reduction]% reduction in expiry loss** for targeted products, potentially saving FrischMarkt an additional **€[Your Estimated Annual Savings] annually**.")

    st.subheader("2. Optimize Markdown Strategies")
    st.write("Introduce rules-based markdown triggers informed by demand forecasts and days-to-expiry. This helps convert at-risk inventory into sales, recouping value.")
    st.markdown(f"**Potential Impact:** Proactive and smarter markdowns could increase recovered revenue by **€[Your Estimated Markdown Recovery] annually**.")

    st.subheader("3. Enhance Operational Excellence and Supplier Review")
    st.write("Address human and process factors through targeted training (stock rotation, temperature control) and prioritize suppliers delivering fresher products.")
    st.markdown(f"**Potential Impact:** Improved efficiency and supplier quality projected to reduce overall expiry losses by an additional **[Your Estimated %]%**, translating to **€[Your Estimated Savings from Operational/Supplier Improvements] annually**.")

    st.subheader("Overall Projected Impact")
    st.markdown(f"By implementing these strategies, FrischMarkt can transform its inventory management, leading to an estimated **total loss reduction of [Sum of your estimated reductions]%**, translating to **over €[Sum of your estimated savings] annually**.")

    st.subheader("Roadmap for Implementation")
    st.markdown("""
    * **Phase 1 (Pilot):** Roll out demand-driven ordering for top high-loss products in 2-3 struggling stores.
    * **Phase 2 (Expansion):** Gradually expand to more products/stores, refining the model and processes.
    * **Continuous Improvement:** Regularly monitor model performance, update data, and adapt strategies.
    * **Training:** Provide comprehensive training for store staff on new tools and processes.
    """)
