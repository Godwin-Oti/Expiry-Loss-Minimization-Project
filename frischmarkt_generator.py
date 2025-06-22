import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker
import os

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
fake = Faker('de_DE')  # German locale

print("🥨 Generating FrischMarkt Dataset - German Retail Analytics")
print("=" * 60)

# Create output directory
os.makedirs('frischmarkt_data', exist_ok=True)

# German product data with FICTIONAL brand names
GERMAN_PRODUCTS = {
    'Molkereiprodukte': {
        'subcategories': ['Milch', 'Joghurt', 'Quark', 'Käse', 'Butter'],
        'products': [
            'Vollmilch 3.8%', 'Fettarme Milch 1.5%', 'Naturjoghurt', 'Fruchtjoghurt Erdbeere',
            'Griechischer Joghurt', 'Magerquark', 'Speisequark', 'Gouda Jung', 'Camembert',
            'Frischkäse', 'Deutsche Markenbutter', 'Saure Sahne'
        ],
        'brands': ['Alpenhof', 'Bergwiese', 'Landlust', 'Frische Quelle', 'FrischMarkt'],
        'shelf_life': (3, 14), 'cost_range': (0.89, 4.50)
    },
    'Backwaren': {
        'subcategories': ['Brot', 'Brötchen', 'Kuchen', 'Gebäck'],
        'products': [
            'Schwarzbrot', 'Vollkornbrot', 'Weißbrot', 'Sonntagsbrötchen', 'Körnerbrötchen',
            'Laugenbrezeln', 'Apfelkuchen', 'Streuselkuchen', 'Croissant', 'Berliner'
        ],
        'brands': ['Bäckerei Goldkorn', 'Kornmühle', 'Backstube Hansen', 'FrischMarkt Bäckerei'],
        'shelf_life': (1, 5), 'cost_range': (0.65, 3.20)
    },
    'Frischware': {
        'subcategories': ['Obst', 'Gemüse', 'Salate', 'Kräuter'],
        'products': [
            'Äpfel Elstar', 'Bananen', 'Erdbeeren', 'Spargel Weiß', 'Kartoffeln Festkochend',
            'Tomaten', 'Gurken', 'Möhren', 'Kopfsalat', 'Rucola', 'Petersilie', 'Basilikum'
        ],
        'brands': ['Regional', 'Bio Sonnenhof', 'Naturkost Weber', 'FrischMarkt'],
        'shelf_life': (2, 10), 'cost_range': (0.99, 5.99)
    },
    'Fleisch': {
        'subcategories': ['Rind', 'Schwein', 'Geflügel', 'Wurst'],
        'products': [
            'Rinderhackfleisch', 'Schweinekoteletts', 'Hähnchenbrust', 'Bratwurst',
            'Leberwurst', 'Salami', 'Schinken Gekocht', 'Wiener Würstchen'
        ],
        'brands': ['Fleischerei Meister', 'Landmetzger Koch', 'Geflügelhof Grün', 'FrischMarkt Metzgerei'],
        'shelf_life': (2, 8), 'cost_range': (2.99, 12.99)
    },
    'Feinkost': {
        'subcategories': ['Aufschnitt', 'Salate', 'Antipasti'],
        'products': [
            'Kartoffelsalat', 'Nudelsalat', 'Thunfischsalat', 'Oliven Mix',
            'Antipasti Gemüse', 'Hummus', 'Tzatziki', 'Leberpastete'
        ],
        'brands': ['Delikatessen Richter', 'Feinkost Bergmann', 'FrischMarkt Feinkost'],
        'shelf_life': (3, 12), 'cost_range': (1.49, 6.99)
    }
}

GERMAN_BRANDS = ['Alpenhof', 'Bergwiese', 'Kornmühle', 'Fleischerei Meister', 'Bio Sonnenhof', 'FrischMarkt', 'Naturkost Weber']

def generate_products_master():
    """Generate German product master data"""
    print("📦 Generating Product Master...")
    
    products = []
    product_id = 1
    
    for category, data in GERMAN_PRODUCTS.items():
        for subcategory in data['subcategories']:
            # Get products for this subcategory
            category_products = [p for p in data['products'] if 
                               any(word in p.lower() for word in subcategory.lower().split())]
            if not category_products:
                category_products = data['products'][:3]  # fallback
            
            for product_name in category_products[:6]:  # Limit products per subcategory
                shelf_life = random.randint(*data['shelf_life'])
                cost = round(random.uniform(*data['cost_range']), 2)
                
                # German retail margins typically 20-50%
                margin = random.uniform(0.2, 0.5)
                retail_price = round(cost * (1 + margin), 2)
                
                # MHD vs Verbrauchsdatum logic
                mhd_vs_verbrauch = "Verbrauchsdatum" if category in ['Fleisch', 'Molkereiprodukte'] else "MHD"
                
                products.append({
                    'product_id': f'P{product_id:03d}',
                    'product_name': product_name,
                    'category': category,
                    'subcategory': subcategory,
                    'brand': random.choice(data['brands']),
                    'unit_cost': cost,
                    'retail_price': retail_price,
                    'shelf_life_days': shelf_life,
                    'mhd_vs_verbrauch': mhd_vs_verbrauch,
                    'seasonality_factor': round(random.uniform(0.7, 1.5), 2),
                    'temperature_sensitive': category in ['Molkereiprodukte', 'Fleisch', 'Feinkost'],
                    'supplier_id': f'SUP{random.randint(1, 20):02d}'
                })
                product_id += 1
    
    df = pd.DataFrame(products)
    df.to_csv('frischmarkt_data/products_master.csv', index=False)
    print(f"✅ Generated {len(products)} products")
    return df

def generate_stores_master():
    """Generate German store master data"""
    print("🏪 Generating Store Master...")
    
    berlin_areas = ['Mitte', 'Charlottenburg', 'Prenzlauer Berg', 'Kreuzberg', 'Schöneberg', 'Friedrichshain']
    brandenburg_towns = ['Potsdam', 'Brandenburg', 'Cottbus', 'Frankfurt Oder']
    
    stores = []
    for i in range(1, 7):
        if i <= 2:
            location_type = 'Berlin City Center'
            area = random.choice(berlin_areas[:3])
            size_sqm = random.randint(800, 1500)
            demo_score = random.randint(6, 9)
            traffic = random.choice(['Mittel', 'Hoch'])
        elif i <= 4:
            location_type = 'Berlin Suburbs'
            area = random.choice(berlin_areas[3:])
            size_sqm = random.randint(1200, 2000)
            demo_score = random.randint(5, 8)
            traffic = random.choice(['Mittel', 'Hoch'])
        else:
            location_type = 'Brandenburg Towns'
            area = random.choice(brandenburg_towns)
            size_sqm = random.randint(1500, 2500)
            demo_score = random.randint(3, 7)
            traffic = random.choice(['Niedrig', 'Mittel'])
        
        stores.append({
            'store_id': f'S{i:03d}',
            'store_name': f'FrischMarkt {area}',
            'location_type': location_type,
            'square_meters': size_sqm,
            'demographics_score': demo_score,
            'foot_traffic_level': traffic,
            'refrigeration_capacity': random.randint(6, 10),
            'staff_efficiency_score': random.randint(5, 9),
            'distance_from_warehouse_km': random.randint(5, 120),
            'opening_year': random.randint(2010, 2020),
            'sunday_closed': True
        })
    
    df = pd.DataFrame(stores)
    df.to_csv('frischmarkt_data/stores_master.csv', index=False)
    print(f"✅ Generated {len(stores)} stores")
    return df

def generate_external_factors():
    """Generate German external factors (weather, holidays, etc.)"""
    print("🌤️ Generating External Factors...")
    
    # German holidays 2023
    german_holidays = [
        '2023-01-01', '2023-04-07', '2023-04-10', '2023-05-01', '2023-05-18',
        '2023-05-29', '2023-10-03', '2023-12-25', '2023-12-26'
    ]
    
    # Berlin school holidays 2023 (simplified)
    school_holidays = [
        ('2023-02-06', '2023-02-11'), ('2023-04-03', '2023-04-14'),
        ('2023-07-13', '2023-08-25'), ('2023-10-23', '2023-11-04'),
        ('2023-12-23', '2023-12-31')
    ]
    
    external_data = []
    stores = [f'S{i:03d}' for i in range(1, 7)]
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    current_date = start_date
    
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        
        # Berlin weather simulation (realistic for Germany)
        if current_date.month in [12, 1, 2]:  # Winter
            temp_high = random.randint(-2, 8)
            temp_low = random.randint(-8, 3)
            precip = random.uniform(0, 15) if random.random() < 0.4 else 0
        elif current_date.month in [3, 4, 5]:  # Spring
            temp_high = random.randint(8, 20)
            temp_low = random.randint(2, 12)
            precip = random.uniform(0, 12) if random.random() < 0.35 else 0
        elif current_date.month in [6, 7, 8]:  # Summer
            temp_high = random.randint(18, 32)
            temp_low = random.randint(12, 20)
            precip = random.uniform(0, 25) if random.random() < 0.3 else 0
        else:  # Fall
            temp_high = random.randint(5, 18)
            temp_low = random.randint(0, 10)
            precip = random.uniform(0, 18) if random.random() < 0.45 else 0
        
        # Check if school holidays
        in_school_holidays = any(
            datetime.strptime(start, '%Y-%m-%d') <= current_date <= datetime.strptime(end, '%Y-%m-%d')
            for start, end in school_holidays
        )
        
        # Local events
        local_events = False
        if current_date.month == 12 and current_date.day >= 15:  # Christmas markets
            local_events = True
        elif current_date.month == 10 and current_date.day <= 7:  # Oktoberfest influence
            local_events = True
        
        for store_id in stores:
            external_data.append({
                'date': date_str,
                'store_id': store_id,
                'temperature_high_c': temp_high,
                'temperature_low_c': temp_low,
                'precipitation_mm': round(precip, 1),
                'day_of_week': current_date.strftime('%A'),
                'is_holiday': date_str in german_holidays,
                'school_holidays': in_school_holidays,
                'local_events': local_events,
                'competitor_promotion': random.random() < 0.15  # 15% chance
            })
        
        current_date += timedelta(days=1)
    
    df = pd.DataFrame(external_data)
    df.to_csv('frischmarkt_data/external_factors.csv', index=False)
    print(f"✅ Generated external factors for {len(df)} store-days")
    return df

def generate_inventory_and_sales(products_df, stores_df, external_df):
    """Generate the main inventory and sales data"""
    print("📊 Generating Inventory & Sales Data (this may take a moment)...")
    
    inventory_data = []
    sales_data = []
    
    stores = stores_df['store_id'].tolist()
    
    # Group external factors by date for easier lookup
    external_by_date = external_df.groupby('date').first().to_dict('index')
    
    transaction_id = 1
    
    for store_id in stores:
        print(f"  Processing {store_id}...")
        store_info = stores_df[stores_df['store_id'] == store_id].iloc[0]
        
        # Each store carries 70-90% of products
        store_products = products_df.sample(frac=random.uniform(0.7, 0.9))
        
        for _, product in store_products.iterrows():
            current_stock = 0
            
            for date_str in pd.date_range('2023-01-01', '2023-12-31').strftime('%Y-%m-%d'):
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                
                # Skip Sundays (stores closed)
                if date_obj.weekday() == 6:  # Sunday
                    continue
                
                external_info = external_by_date.get(date_str, {})
                
                # Calculate demand factors
                base_demand = random.uniform(5, 50)  # Base daily demand
                
                # Seasonal factor
                seasonal_mult = product['seasonality_factor']
                if product['category'] == 'Frischware':
                    if date_obj.month in [5, 6, 7, 8] and 'Spargel' in product['product_name']:
                        seasonal_mult = 2.5  # Spargel season
                    elif date_obj.month in [12, 1] and any(x in product['product_name'] for x in ['Äpfel', 'Kartoffeln']):
                        seasonal_mult = 1.3
                
                # Weather effects
                weather_mult = 1.0
                if external_info.get('temperature_high_c', 15) > 25 and 'Joghurt' in product['product_name']:
                    weather_mult = 1.4
                elif external_info.get('temperature_high_c', 15) < 5 and product['category'] == 'Backwaren':
                    weather_mult = 1.2
                
                # Store type effects
                location_mult = 1.0
                if store_info['location_type'] == 'Berlin City Center':
                    location_mult = 1.3
                elif store_info['location_type'] == 'Brandenburg Towns':
                    location_mult = 0.8
                
                # Holiday effects
                holiday_mult = 1.0
                if external_info.get('is_holiday', False):
                    holiday_mult = 0.3  # Stores closed or reduced hours
                elif external_info.get('local_events', False):
                    holiday_mult = 1.2
                
                # Weekend effect (Saturday)
                weekend_mult = 1.5 if date_obj.weekday() == 5 else 1.0
                
                # Calculate expected demand
                expected_demand = int(base_demand * seasonal_mult * weather_mult * 
                                    location_mult * holiday_mult * weekend_mult)
                expected_demand = max(0, expected_demand)
                
                # Delivery logic (not every day, realistic patterns)
                receives_delivery = False
                received_quantity = 0
                expiry_date = None
                
                if date_obj.weekday() in [0, 2, 4]:  # Mon, Wed, Fri deliveries
                    if current_stock < expected_demand * 2:  # Reorder point
                        receives_delivery = True
                        received_quantity = random.randint(int(expected_demand * 1.5), 
                                                         int(expected_demand * 4))
                        
                        # Calculate expiry date
                        shelf_life_variation = random.uniform(0.8, 1.2)  # ±20% variation
                        actual_shelf_life = int(product['shelf_life_days'] * shelf_life_variation)
                        expiry_date = (date_obj + timedelta(days=actual_shelf_life)).strftime('%Y-%m-%d')
                
                # Update stock
                beginning_inventory = current_stock
                if receives_delivery:
                    current_stock += received_quantity
                
                # Calculate actual sales (can't sell more than available)
                actual_sales = min(expected_demand, current_stock)
                actual_sales += random.randint(-2, 5)  # Add some randomness
                actual_sales = max(0, min(actual_sales, current_stock))
                
                # Check for expiries and markdowns
                units_expired = 0
                units_marked_down = 0
                markdown_price = None
                markdown_date = None
                
                # Simple expiry simulation (products expire based on FIFO)
                if random.random() < 0.05 and current_stock > 0:  # 5% chance of some expiry
                    units_expired = random.randint(1, min(5, current_stock))
                
                # Markdown logic (day before expiry)
                if random.random() < 0.08 and current_stock > 0:  # 8% chance of markdown
                    units_marked_down = random.randint(1, min(10, current_stock))
                    markdown_price = round(product['retail_price'] * 0.5, 2)  # 50% off
                    markdown_date = date_str
                
                # Update stock after sales and waste
                current_stock = current_stock - actual_sales - units_expired
                current_stock = max(0, current_stock)
                
                # Record inventory data
                inventory_data.append({
                    'date': date_str,
                    'store_id': store_id,
                    'product_id': product['product_id'],
                    'beginning_inventory': beginning_inventory,
                    'received_inventory': received_quantity if receives_delivery else 0,
                    'expiry_date': expiry_date if receives_delivery else None,
                    'units_sold': actual_sales,
                    'units_expired': units_expired,
                    'units_marked_down': units_marked_down,
                    'markdown_price': markdown_price,
                    'markdown_date': markdown_date,
                    'current_stock': current_stock
                })
                
                # Generate individual sales transactions
                if actual_sales > 0:
                    # Break sales into realistic transactions
                    remaining_sales = actual_sales
                    while remaining_sales > 0:
                        transaction_qty = min(remaining_sales, random.randint(1, 5))
                        
                        # Determine if markdown sale
                        is_markdown = units_marked_down > 0 and random.random() < 0.7
                        sale_price = markdown_price if is_markdown else product['retail_price']
                        discount = 50 if is_markdown else 0
                        
                        sales_data.append({
                            'transaction_id': f'T{transaction_id:06d}',
                            'date': date_str,
                            'store_id': store_id,
                            'product_id': product['product_id'],
                            'quantity_sold': transaction_qty,
                            'sale_price': sale_price,
                            'discount_applied': discount,
                            'customer_segment': random.choice(['Stammkunde', 'Gelegenheitskunde', 'Tourist'])
                        })
                        
                        transaction_id += 1
                        remaining_sales -= transaction_qty
    
    # Save datasets
    inventory_df = pd.DataFrame(inventory_data)
    sales_df = pd.DataFrame(sales_data)
    
    inventory_df.to_csv('frischmarkt_data/inventory_daily.csv', index=False)
    sales_df.to_csv('frischmarkt_data/sales_transactions.csv', index=False)
    
    print(f"✅ Generated {len(inventory_data):,} inventory records")
    print(f"✅ Generated {len(sales_data):,} sales transactions")
    
    return inventory_df, sales_df

def generate_supplier_performance(products_df):
    """Generate supplier performance data"""
    print("🚚 Generating Supplier Performance...")
    
    suppliers = products_df['supplier_id'].unique()
    supplier_data = []
    
    for supplier_id in suppliers:
        # Generate 20-50 deliveries per supplier in 2023
        num_deliveries = random.randint(20, 50)
        
        for _ in range(num_deliveries):
            # Generate random date in 2023 using datetime
            start_date = datetime(2023, 1, 1)
            end_date = datetime(2023, 12, 31)
            time_between = end_date - start_date
            days_between = time_between.days
            random_days = random.randrange(days_between)
            delivery_date = start_date + timedelta(days=random_days)
            
            supplier_products = products_df[products_df['supplier_id'] == supplier_id]
            product = supplier_products.sample(1).iloc[0]
            
            planned_shelf_life = product['shelf_life_days']
            actual_shelf_life = max(1, int(planned_shelf_life * random.uniform(0.7, 1.3)))
            
            supplier_data.append({
                'delivery_date': delivery_date.strftime('%Y-%m-%d'),
                'supplier_id': supplier_id,
                'store_id': f'S{random.randint(1, 6):03d}',
                'product_id': product['product_id'],
                'planned_shelf_life_days': planned_shelf_life,
                'actual_shelf_life_days': actual_shelf_life,
                'delivery_delay_days': max(0, random.randint(-1, 5)),  # Mostly on time
                'quantity_delivered': random.randint(10, 200),
                'quality_score': random.randint(6, 10)
            })
    
    df = pd.DataFrame(supplier_data)
    df.to_csv('frischmarkt_data/supplier_performance.csv', index=False)
    print(f"✅ Generated {len(supplier_data)} supplier deliveries")
    return df

def main():
    """Main function to generate all datasets"""
    print("🇩🇪 Starting FrischMarkt Dataset Generation...")
    print(f"📅 Generating data for year 2023")
    print()
    
    # Generate all datasets
    products_df = generate_products_master()
    stores_df = generate_stores_master()
    external_df = generate_external_factors()
    inventory_df, sales_df = generate_inventory_and_sales(products_df, stores_df, external_df)
    supplier_df = generate_supplier_performance(products_df)
    
    # Generate summary
    print("\n" + "="*60)
    print("📈 DATASET SUMMARY")
    print("="*60)
    print(f"Products: {len(products_df):,}")
    print(f"Stores: {len(stores_df):,}")
    print(f"Daily External Factors: {len(external_df):,}")
    print(f"Inventory Records: {len(inventory_df):,}")
    print(f"Sales Transactions: {len(sales_df):,}")
    print(f"Supplier Deliveries: {len(supplier_df):,}")
    print()
    
    # Calculate some key metrics
    total_sales_value = (sales_df['quantity_sold'] * sales_df['sale_price']).sum()
    total_expired_units = inventory_df['units_expired'].sum()
    total_markdown_units = inventory_df['units_marked_down'].sum()
    
    print("🏷️ KEY BUSINESS METRICS")
    print("-" * 30)
    print(f"Total Sales Revenue: €{total_sales_value:,.2f}")
    print(f"Total Units Expired: {total_expired_units:,}")
    print(f"Total Units Marked Down: {total_markdown_units:,}")
    print(f"Estimated Loss from Expiry: €{total_expired_units * 2.5:,.2f}")
    print()
    
    print("✅ Dataset generation complete!")
    print(f"📁 All files saved in 'frischmarkt_data' directory")
    print("\n🚀 Ready for your expiry loss minimization analysis!")

if __name__ == "__main__":
    main()