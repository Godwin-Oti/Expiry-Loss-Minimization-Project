import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from scipy.stats import randint, uniform
import joblib # For saving/loading models
import matplotlib.pyplot as plt # For plotting in notebook (optional, but good for local verification)
import seaborn as sns # For plotting in notebook (optional, but good for local verification)

# --- Configuration ---
DATA_DIR = 'frischmarkt_data'

print("📈 Starting Demand Forecasting Model Training...")
print("=" * 60)

def load_and_prepare_data_for_model():
    """Loads, preprocesses, and merges data specifically for model training."""
    print("1. Loading raw data for model preparation...")
    try:
        products_df = pd.read_csv(os.path.join(DATA_DIR, 'products_master.csv'))
        stores_df = pd.read_csv(os.path.join(DATA_DIR, 'stores_master.csv'))
        external_df = pd.read_csv(os.path.join(DATA_DIR, 'external_factors.csv'))
        inventory_df = pd.read_csv(os.path.join(DATA_DIR, 'inventory_daily.csv'))
        print("✅ Raw data loaded.")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Make sure 'frischmarkt_data' directory exists with all CSVs and you've run enhanced_frischgenerator.py.")
        exit() # Exit if essential data is missing

    # Convert date columns to datetime objects
    for df in [external_df, inventory_df]:
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

    # --- Data Merging for Comprehensive Analysis ---
    # Merge inventory data with product and store master data
    demand_df = inventory_df.merge(products_df, on='product_id', how='left')
    demand_df = demand_df.merge(stores_df, on='store_id', how='left')
    # Merge with external factors
    demand_df = demand_df.merge(external_df, on=['date', 'store_id'], how='left')

    print("✅ Data merged for model preparation.")
    return demand_df, products_df, stores_df # Return products_df and stores_df for later use

def feature_engineer_data(demand_df):
    """Performs feature engineering on the merged demand data."""
    print("2. Performing feature engineering...")

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

    print("✅ Feature engineering complete.")
    return demand_df

def train_and_save_model(demand_df, products_df, stores_df):
    """Trains the RandomForest model, evaluates it, and saves artifacts."""
    print("3. Training RandomForestRegressor model...")

    # Define features (X) and target (y)
    numerical_features = [
        'beginning_inventory', 'received_inventory', 'unit_cost', 'retail_price',
        'shelf_life_days', 'base_expiry_rate', 'profit_margin',
        'temperature_high_c', 'temperature_low_c', 'precipitation_mm', 'expiry_risk_multiplier',
        'demographics_score', 'refrigeration_capacity', 'staff_efficiency_score',
        'distance_from_warehouse_km', 'day_of_year', 'week_of_year', 'month', 'day_of_week_num',
        'units_sold_lag1', 'units_sold_lag7', 'units_sold_lag30', 'rolling_mean_sales_7d'
    ]
    categorical_features_for_model = [
        'category', 'subcategory', 'brand', 'location_type', 'foot_traffic_level',
        'management_quality', 'markdown_aggressiveness', 'day_of_week',
        'product_id', 'store_id'
    ]
    boolean_features = [
        'is_holiday', 'school_holidays', 'local_events', 'competitor_promotion',
        'heat_wave', 'power_outage_risk', 'delivery_disruption', 'temperature_sensitive'
    ]

    # Ensure all selected features exist in the DataFrame
    all_features_to_include = numerical_features + categorical_features_for_model + boolean_features
    
    # Create dummy variables for categorical features
    demand_df_encoded = pd.get_dummies(demand_df, columns=[f for f in categorical_features_for_model if f in demand_df.columns], drop_first=True)

    # Combine all feature column names, ensuring they exist in the encoded DF
    X_cols = numerical_features + boolean_features + [col for col in demand_df_encoded.columns if any(col.startswith(f"{cat}_") for cat in categorical_features_for_model)]
    X_cols = [col for col in X_cols if col in demand_df_encoded.columns] # Final filter

    X = demand_df_encoded[X_cols]
    y = demand_df_encoded['units_sold']

    X = X.fillna(0)
    y = y.fillna(0)

    # Split data chronologically (last 3 months for testing)
    split_date = demand_df['date'].max() - timedelta(days=90)
    X_train = X[demand_df_encoded['date'] <= split_date]
    y_train = y[demand_df_encoded['date'] <= split_date]
    X_test = X[demand_df_encoded['date'] > split_date]
    y_test = y[demand_df_encoded['date'] > split_date]

    # Store original IDs and actual units sold for predictions_df
    test_data_original_ids = demand_df[demand_df['date'] > split_date][['date', 'product_id', 'store_id', 'units_sold']].copy()

    # --- RandomForestRegressor with RandomizedSearchCV ---
    param_dist = {
        'n_estimators': randint(50, 200),
        'max_features': uniform(0.6, 0.3),
        'max_depth': randint(5, 20),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5)
    }
    rf = RandomForestRegressor(random_state=42)
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_dist,
                                   n_iter=10, cv=3, verbose=1, random_state=42, n_jobs=-1, scoring='neg_mean_absolute_error')
    rf_random.fit(X_train, y_train)
    best_rf_model = rf_random.best_estimator_
    y_pred_rf = best_rf_model.predict(X_test)

    # Calculate metrics
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    mape_rf = np.mean(np.abs((y_test - y_pred_rf) / y_test.replace(0, np.nan))) * 100
    r2_rf = r2_score(y_test, y_pred_rf)

    # Feature Importances
    feature_importances_rf = pd.Series(best_rf_model.feature_importances_, index=X.columns)

    # Save model and artifacts
    joblib.dump(best_rf_model, os.path.join(DATA_DIR, 'random_forest_model.pkl'))
    feature_importances_rf.to_csv(os.path.join(DATA_DIR, 'rf_feature_importances.csv'), header=True)
    
    model_metrics_data = pd.DataFrame({
        'Metric': ['MAE', 'MAPE', 'R2'],
        'Value': [mae_rf, mape_rf, r2_rf]
    })
    model_metrics_data.to_csv(os.path.join(DATA_DIR, 'rf_model_metrics.csv'), index=False)

    # Prepare predictions_df with product_name and store_name for dashboard
    predictions_df = test_data_original_ids.copy()
    predictions_df['actual_units_sold'] = y_test.values
    predictions_df['predicted_rf'] = y_pred_rf

    # Merge product_name and store_name into predictions_df
    predictions_df = predictions_df.merge(products_df[['product_id', 'product_name']], on='product_id', how='left')
    predictions_df = predictions_df.merge(stores_df[['store_id', 'store_name']], on='store_id', how='left')

    predictions_df.to_csv(os.path.join(DATA_DIR, 'rf_predictions.csv'), index=False)

    print("✅ RandomForest model, feature importances, metrics, and predictions saved.")
    print(f"Model MAE: {mae_rf:,.2f}, R2: {r2_rf:.4f}")

    # Optional: Plot feature importances locally for verification
    plt.figure(figsize=(10, 7))
    sns.barplot(x=feature_importances_rf.nlargest(15).values, y=feature_importances_rf.nlargest(15).index, palette='viridis')
    plt.title('Top 15 Feature Importances for RandomForest (Units Sold Prediction)')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

    print("\n🎉 Model Training and Saving Complete! Check 'frischmarkt_data/' directory for model artifacts.")


if __name__ == "__main__":
    full_demand_df, products_df_main, stores_df_main = load_and_prepare_data_for_model()
    full_demand_df_fe = feature_engineer_data(full_demand_df)
    train_and_save_model(full_demand_df_fe, products_df_main, stores_df_main)
