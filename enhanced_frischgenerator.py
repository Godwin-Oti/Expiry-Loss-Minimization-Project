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

print("🥨 Generating REALISTIC FrischMarkt Dataset - High Expiry Loss Patterns")
print("=" * 70)

# Create output directory
os.makedirs('frischmarkt_data', exist_ok=True)

# Enhanced German product data with REALISTIC expiry characteristics
GERMAN_PRODUCTS = {
    'Molkereiprodukte': {
        'subcategories': ['Milch', 'Joghurt', 'Quark', 'Käse', 'Butter'],
        'products': [
            'Vollmilch 3.8%', 'Fettarme Milch 1.5%', 'Naturjoghurt', 'Fruchtjoghurt Erdbeere',
            'Griechischer Joghurt', 'Magerquark', 'Speisequark', 'Gouda Jung', 'Camembert',
            'Frischkäse', 'Deutsche Markenbutter', 'Saure Sahne'
        ],
        'brands': ['Alpenhof', 'Bergwiese', 'Landlust', 'Frische Quelle', 'FrischMarkt'],
        'shelf_life': (3, 14),
        'cost_range': (0.89, 4.50),
        'expiry_risk': 'HIGH',  # 15-25% expiry rate
        'temperature_sensitive': True
    },
    'Backwaren': {
        'subcategories': ['Brot', 'Brötchen', 'Kuchen', 'Gebäck'],
        'products': [
            'Schwarzbrot', 'Vollkornbrot', 'Weißbrot', 'Sonntagsbrötchen', 'Körnerbrötchen',
            'Laugenbrezeln', 'Apfelkuchen', 'Streuselkuchen', 'Croissant', 'Berliner'
        ],
        'brands': ['Bäckerei Goldkorn', 'Kornmühle', 'Backstube Hansen', 'FrischMarkt Bäckerei'],
        'shelf_life': (1, 5),
        'cost_range': (0.65, 3.20),
        'expiry_risk': 'VERY_HIGH',  # 20-35% expiry rate
        'temperature_sensitive': False
    },
    'Frischware': {
        'subcategories': ['Obst', 'Gemüse', 'Salate', 'Kräuter'],
        'products': [
            'Äpfel Elstar', 'Bananen', 'Erdbeeren', 'Spargel Weiß', 'Kartoffeln Festkochend',
            'Tomaten', 'Gurken', 'Möhren', 'Kopfsalat', 'Rucola', 'Petersilie', 'Basilikum'
        ],
        'brands': ['Regional', 'Bio Sonnenhof', 'Naturkost Weber', 'FrischMarkt'],
        'shelf_life': (2, 10),
        'cost_range': (0.99, 5.99),
        'expiry_risk': 'VERY_HIGH',  # 25-40% expiry rate
        'temperature_sensitive': True
    },
    'Fleisch': {
        'subcategories': ['Rind', 'Schwein', 'Geflügel', 'Wurst'],
        'products': [
            'Rinderhackfleisch', 'Schweinekoteletts', 'Hähnchenbrust', 'Bratwurst',
            'Leberwurst', 'Salami', 'Schinken Gekocht', 'Wiener Würstchen'
        ],
        'brands': ['Fleischerei Meister', 'Landmetzger Koch', 'Geflügelhof Grün', 'FrischMarkt Metzgerei'],
        'shelf_life': (2, 8),
        'cost_range': (2.99, 12.99),
        'expiry_risk': 'EXTREME',  # 30-50% expiry rate
        'temperature_sensitive': True
    },
    'Feinkost': {
        'subcategories': ['Aufschnitt', 'Salate', 'Antipasti'],
        'products': [
            'Kartoffelsalat', 'Nudelsalat', 'Thunfischsalat', 'Oliven Mix',
            'Antipasti Gemüse', 'Hummus', 'Tzatziki', 'Leberpastete'
        ],
        'brands': ['Delikatessen Richter', 'Feinkost Bergmann', 'FrischMarkt Feinkost'],
        'shelf_life': (3, 12),
        'cost_range': (1.49, 6.99),
        'expiry_risk': 'HIGH',  # 12-20% expiry rate
        'temperature_sensitive': True
    }
}

# Define expiry risk multipliers (ADJUSTED FOR REALISM)
EXPIRY_RISK_MULTIPLIERS = {
    'LOW': (0.02, 0.08),       # 2-8% base expiry rate
    'MEDIUM': (0.08, 0.15),    # 8-15% base expiry rate
    'HIGH': (0.10, 0.18),      # Adjusted from (0.15, 0.25)
    'VERY_HIGH': (0.18, 0.28), # Adjusted from (0.25, 0.35)
    'EXTREME': (0.25, 0.40)    # Adjusted from (0.35, 0.50)
}

def generate_products_master():
    """Generate German product master data with realistic expiry characteristics"""
    print("📦 Generating Product Master with Expiry Risk Profiles...")

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

                # Get expiry risk range for this category
                expiry_min, expiry_max = EXPIRY_RISK_MULTIPLIERS[data['expiry_risk']]
                base_expiry_rate = round(random.uniform(expiry_min, expiry_max), 3)

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
                    'temperature_sensitive': data['temperature_sensitive'],
                    'supplier_id': f'SUP{random.randint(1, 20):02d}',
                    'expiry_risk_level': data['expiry_risk'],
                    'base_expiry_rate': base_expiry_rate,
                    'profit_margin': round(margin, 3)
                })
                product_id += 1

    df = pd.DataFrame(products)
    df.to_csv('frischmarkt_data/products_master.csv', index=False)
    print(f"✅ Generated {len(products)} products with realistic expiry risk profiles")
    return df

def generate_stores_master():
    """Generate German store master data with performance characteristics"""
    print("🏪 Generating Store Master with Management Efficiency Scores...")

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
            management_quality = random.choice(['Excellent', 'Good'])
        elif i <= 4:
            location_type = 'Berlin Suburbs'
            area = random.choice(berlin_areas[3:])
            size_sqm = random.randint(1200, 2000)
            demo_score = random.randint(5, 8)
            traffic = random.choice(['Mittel', 'Hoch'])
            management_quality = random.choice(['Good', 'Average'])
        else:
            location_type = 'Brandenburg Towns'
            area = random.choice(brandenburg_towns)
            size_sqm = random.randint(1500, 2500)
            demo_score = random.randint(3, 7)
            traffic = random.choice(['Niedrig', 'Mittel'])
            management_quality = random.choice(['Average', 'Poor'])

        # Management efficiency affects expiry rates
        mgmt_multipliers = {
            'Excellent': 0.6,  # 40% reduction in expiry
            'Good': 0.8,       # 20% reduction in expiry
            'Average': 1.0,    # Normal expiry rates
            'Poor': 1.4        # 40% increase in expiry
        }

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
            'sunday_closed': True,
            'management_quality': management_quality,
            'expiry_multiplier': mgmt_multipliers[management_quality],
            'markdown_aggressiveness': random.choice(['Conservative', 'Moderate', 'Aggressive'])
        })

    df = pd.DataFrame(stores)
    df.to_csv('frischmarkt_data/stores_master.csv', index=False)
    print(f"✅ Generated {len(stores)} stores with management efficiency profiles")
    return df

def generate_external_factors():
    """Generate German external factors with expiry impact"""
    print("🌤 Generating External Factors with Expiry Impact...")

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

        # Calculate expiry impact factors
        heat_wave = temp_high > 30  # Extreme heat accelerates spoilage
        power_outage_risk = random.random() < 0.02  # 2% chance of power issues
        delivery_disruption = random.random() < 0.05  # 5% chance of delivery delays

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
                'competitor_promotion': random.random() < 0.15,  # 15% chance
                'heat_wave': heat_wave,
                'power_outage_risk': power_outage_risk,
                'delivery_disruption': delivery_disruption,
                'expiry_risk_multiplier': 1.5 if heat_wave else (1.3 if power_outage_risk else 1.0)
            })

        current_date += timedelta(days=1)

    df = pd.DataFrame(external_data)
    df.to_csv('frischmarkt_data/external_factors.csv', index=False)
    print(f"✅ Generated external factors with expiry impact for {len(df)} store-days")
    return df

def generate_inventory_and_sales(products_df, stores_df, external_df):
    """Generate realistic inventory and sales data with HIGH expiry rates"""
    print("📊 Generating Inventory & Sales Data with REALISTIC Expiry Patterns...")

    inventory_data = []
    sales_data = []

    stores = stores_df['store_id'].tolist()

    # Group external factors by date and store for easier lookup
    external_by_date_store = external_df.set_index(['date', 'store_id']).to_dict('index')

    transaction_id = 1

    for store_id in stores:
        print(f"  Processing {store_id}...")
        store_info = stores_df[stores_df['store_id'] == store_id].iloc[0]

        # Each store carries 70-90% of products
        store_products = products_df.sample(frac=random.uniform(0.7, 0.9))

        # Track inventory aging for FIFO expiry calculation
        inventory_batches = {}  # product_id -> [(quantity, expiry_date), ...]

        for _, product in store_products.iterrows():
            inventory_batches[product['product_id']] = []

            for date_str in pd.date_range('2023-01-01', '2023-12-31').strftime('%Y-%m-%d'):
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')

                # Skip Sundays (stores closed)
                if date_obj.weekday() == 6:  # Sunday
                    continue

                external_info = external_by_date_store.get((date_str, store_id), {})

                # Calculate current stock from batches
                current_batches = inventory_batches[product['product_id']]
                current_stock = sum(batch[0] for batch in current_batches)

                # Calculate demand factors
                base_demand = random.uniform(8, 80)  # Higher base demand

                # Enhanced seasonal factors
                seasonal_mult = product['seasonality_factor']
                if product['category'] == 'Frischware':
                    if date_obj.month in [5, 6, 7, 8] and 'Spargel' in product['product_name']:
                        seasonal_mult = 2.5
                    elif date_obj.month in [12, 1] and any(x in product['product_name'] for x in ['Äpfel', 'Kartoffeln']):
                        seasonal_mult = 1.3

                # Weather effects on demand AND spoilage (ADJUSTED FOR REALISM)
                weather_mult = 1.0
                spoilage_mult = 1.0

                if external_info.get('temperature_high_c', 15) > 25:
                    if 'Joghurt' in product['product_name'] or 'Milch' in product['product_name']:
                        weather_mult = 1.4
                    if product['temperature_sensitive']:
                        spoilage_mult = 1.3
                
                # Reduced impact of extreme factors for more realism
                if external_info.get('heat_wave', False):
                    spoilage_mult *= 1.2  # Adjusted from 1.5
                
                if external_info.get('power_outage_risk', False):
                    spoilage_mult *= 1.5  # Adjusted from 2.0


                # Store management effect on expiry
                mgmt_expiry_mult = store_info['expiry_multiplier']

                # Holiday effects
                holiday_mult = 1.0
                if external_info.get('is_holiday', False):
                    holiday_mult = 0.3
                elif external_info.get('local_events', False):
                    holiday_mult = 1.2

                # Weekend effect
                weekend_mult = 1.5 if date_obj.weekday() == 5 else 1.0

                # Calculate expected demand
                expected_demand = int(base_demand * seasonal_mult * weather_mult *
                                    holiday_mult * weekend_mult)
                expected_demand = max(0, expected_demand)

                # Delivery logic (ADJUSTED FOR REALISM)
                receives_delivery = False
                received_quantity = 0

                if date_obj.weekday() in [0, 2, 4]:  # Mon, Wed, Fri deliveries
                    if current_stock < expected_demand * 1.5:  # Lower reorder point = more frequent small orders
                        receives_delivery = True
                        # Adjusted upper bound for received quantity to reduce overstocking
                        received_quantity = random.randint(int(expected_demand * 0.8),
                                                         int(expected_demand * 2.0))


                        # Calculate expiry date with realistic variation
                        shelf_life_variation = random.uniform(0.6, 1.2)  # More variation
                        if external_info.get('delivery_disruption', False):
                            shelf_life_variation *= 0.8  # Disrupted deliveries = shorter shelf life

                        actual_shelf_life = max(1, int(product['shelf_life_days'] * shelf_life_variation))
                        expiry_date = date_obj + timedelta(days=actual_shelf_life)

                        # Add to inventory batches
                        inventory_batches[product['product_id']].append((received_quantity, expiry_date))

                beginning_inventory = current_stock

                # Process expiries FIRST (FIFO basis)
                units_expired = 0
                new_current_batches = []
                for batch_qty, batch_expiry in current_batches:
                    if batch_expiry <= date_obj:
                        units_expired += batch_qty
                    else:
                        new_current_batches.append((batch_qty, batch_expiry))
                current_batches = new_current_batches # Update the list after processing expiries

                # Additional spoilage due to conditions
                if current_stock > 0:
                    # Calculate realistic expiry rate
                    combined_expiry_rate = (product['base_expiry_rate'] *
                                          spoilage_mult *
                                          mgmt_expiry_mult *
                                          external_info.get('expiry_risk_multiplier', 1.0))

                    # Apply expiry rate to remaining stock
                    additional_expired = int(current_stock * combined_expiry_rate * random.uniform(0.5, 1.5))
                    additional_expired = min(additional_expired, current_stock)

                    # Remove from oldest batches first
                    remaining_to_expire = additional_expired
                    temp_batches_after_spoilage = [] # Use a temporary list to build the new state
                    for batch_qty, batch_expiry in current_batches:
                        if remaining_to_expire <= 0:
                            temp_batches_after_spoilage.append((batch_qty, batch_expiry)) # Keep the rest as is
                        else:
                            expire_from_batch = min(batch_qty, remaining_to_expire)
                            units_expired += expire_from_batch # Add to total expired
                            remaining_to_expire -= expire_from_batch
                            if batch_qty - expire_from_batch > 0:
                                temp_batches_after_spoilage.append((batch_qty - expire_from_batch, batch_expiry))
                    current_batches = temp_batches_after_spoilage # Update the list after spoilage


                # Update current stock after expiry
                current_stock = sum(batch[0] for batch in current_batches)

                # Add new delivery
                if receives_delivery:
                    # Add new delivery as a new batch
                    if received_quantity > 0:
                        # Re-calculate expiry date for the new delivery, as shelf_life_variation might have changed
                        shelf_life_variation_new_delivery = random.uniform(0.6, 1.2)
                        if external_info.get('delivery_disruption', False):
                            shelf_life_variation_new_delivery *= 0.8
                        actual_shelf_life_new_delivery = max(1, int(product['shelf_life_days'] * shelf_life_variation_new_delivery))
                        expiry_date_new_delivery = date_obj + timedelta(days=actual_shelf_life_new_delivery)
                        current_batches.append((received_quantity, expiry_date_new_delivery))
                    current_stock += received_quantity # Update total stock

                # Calculate sales (can't sell more than available after expiry)
                actual_sales = min(expected_demand, current_stock)
                actual_sales += random.randint(-3, 8)  # Add randomness
                actual_sales = max(0, min(actual_sales, current_stock))

                # Process sales (FIFO)
                remaining_to_sell = actual_sales
                temp_batches_after_sales = [] # Use a temporary list to build the new state
                for batch_qty, batch_expiry in current_batches:
                    if remaining_to_sell <= 0:
                        temp_batches_after_sales.append((batch_qty, batch_expiry)) # Keep the rest as is
                    else:
                        sell_from_batch = min(batch_qty, remaining_to_sell)
                        remaining_to_sell -= sell_from_batch
                        if batch_qty - sell_from_batch > 0:
                            temp_batches_after_sales.append((batch_qty - sell_from_batch, batch_expiry))
                current_batches = temp_batches_after_sales # Update the list after sales

                # Markdown logic - more aggressive near expiry (ADJUSTED FOR REALISM)
                units_marked_down = 0
                markdown_price = None
                markdown_date = None

                # Check for items expiring soon
                items_expiring_soon = 0
                for batch_qty, batch_expiry in current_batches:
                    days_to_expiry = (batch_expiry - date_obj).days
                    if days_to_expiry <= 2:  # Mark down items expiring in 2 days
                        items_expiring_soon += batch_qty

                if items_expiring_soon > 0:
                    markdown_aggressiveness = store_info['markdown_aggressiveness']
                    if markdown_aggressiveness == 'Aggressive':
                        units_marked_down = int(items_expiring_soon * random.uniform(0.7, 1.0))
                        markdown_price = round(product['retail_price'] * random.uniform(0.5, 0.6), 2)  # 40-50% off
                    elif markdown_aggressiveness == 'Moderate':
                        units_marked_down = int(items_expiring_soon * random.uniform(0.4, 0.7))
                        markdown_price = round(product['retail_price'] * random.uniform(0.65, 0.75), 2)  # 25-35% off
                    else:  # Conservative
                        units_marked_down = int(items_expiring_soon * random.uniform(0.1, 0.4))
                        markdown_price = round(product['retail_price'] * random.uniform(0.8, 0.9), 2)  # 10-20% off

                    if units_marked_down > 0:
                        markdown_date = date_str

                # Final stock calculation (after all operations)
                final_stock = sum(batch[0] for batch in current_batches)
                inventory_batches[product['product_id']] = current_batches # Ensure the main tracking dict is updated

                # Calculate financial impact
                expiry_loss = units_expired * product['unit_cost']
                markdown_loss = units_marked_down * (product['retail_price'] - markdown_price) if markdown_price else 0

                # Record inventory data
                inventory_data.append({
                    'date': date_str,
                    'store_id': store_id,
                    'product_id': product['product_id'],
                    'beginning_inventory': beginning_inventory,
                    'received_inventory': received_quantity,
                    'units_sold': actual_sales,
                    'units_expired': units_expired,
                    'units_marked_down': units_marked_down,
                    'markdown_price': markdown_price,
                    'markdown_date': markdown_date,
                    'current_stock': final_stock,
                    'expiry_loss_eur': round(expiry_loss, 2),
                    'markdown_loss_eur': round(markdown_loss, 2),
                    'total_loss_eur': round(expiry_loss + markdown_loss, 2),
                    'expiry_rate': round(units_expired / max(1, beginning_inventory), 3),
                    'days_until_next_expiry': min([(batch_expiry - date_obj).days for _, batch_expiry in current_batches], default=999)
                })

                # Generate sales transactions
                if actual_sales > 0:
                    remaining_sales = actual_sales
                    while remaining_sales > 0:
                        transaction_qty = min(remaining_sales, random.randint(1, 8))

                        # Determine if markdown sale
                        is_markdown = units_marked_down > 0 and random.random() < 0.6
                        sale_price = markdown_price if is_markdown else product['retail_price']
                        discount = round((1 - sale_price/product['retail_price']) * 100) if is_markdown else 0

                        sales_data.append({
                            'transaction_id': f'T{transaction_id:06d}',
                            'date': date_str,
                            'store_id': store_id,
                            'product_id': product['product_id'],
                            'quantity_sold': transaction_qty,
                            'sale_price': sale_price,
                            'discount_applied': discount,
                            'customer_segment': random.choice(['Stammkunde', 'Gelegenheitskunde', 'Tourist']),
                            'is_markdown_sale': is_markdown
                        })

                        transaction_id += 1
                        remaining_sales -= transaction_qty

    # Save datasets
    inventory_df = pd.DataFrame(inventory_data)
    sales_df = pd.DataFrame(sales_data)

    inventory_df.to_csv('frischmarkt_data/inventory_daily.csv', index=False)
    sales_df.to_csv('frischmarkt_data/sales_transactions.csv', index=False)

    print(f"✅ Generated {len(inventory_data):,} inventory records with realistic expiry patterns")
    print(f"✅ Generated {len(sales_data):,} sales transactions")

    return inventory_df, sales_df

def generate_supplier_performance(products_df):
    """Generate supplier performance data with quality impact on expiry"""
    print("🚚 Generating Supplier Performance with Quality Impact...")

    suppliers = products_df['supplier_id'].unique()
    supplier_data = []

    for supplier_id in suppliers:
        # Assign supplier quality tier
        quality_tier = random.choices(['Premium', 'Standard', 'Budget'], weights=[20, 60, 20])[0]

        # Quality affects shelf life consistency
        quality_multipliers = {
            'Premium': (0.9, 1.1),    # Very consistent shelf life
            'Standard': (0.7, 1.2),   # Moderate variation
            'Budget': (0.5, 1.3)      # High variation, sometimes very short
        }

        # Generate 30-70 deliveries per supplier
        num_deliveries = random.randint(30, 70)

        for _ in range(num_deliveries):
            # Generate random date in 2023
            start_date = datetime(2023, 1, 1)
            end_date = datetime(2023, 12, 31)
            time_between = end_date - start_date
            days_between = time_between.days
            random_days = random.randrange(days_between)
            delivery_date = start_date + timedelta(days=random_days)

            supplier_products = products_df[products_df['supplier_id'] == supplier_id]
            product = supplier_products.sample(1).iloc[0]

            planned_shelf_life = product['shelf_life_days']

            # Apply quality tier impact on actual shelf life
            quality_min, quality_max = quality_multipliers[quality_tier]
            actual_shelf_life = max(1, int(planned_shelf_life * random.uniform(quality_min, quality_max)))

            # Delivery delays more common with budget suppliers
            delay_probability = {'Premium': 0.05, 'Standard': 0.15, 'Budget': 0.30}
            max_delay = {'Premium': 1, 'Standard': 3, 'Budget': 7}

            delivery_delay = 0
            if random.random() < delay_probability[quality_tier]:
                delivery_delay = random.randint(1, max_delay[quality_tier])

            # Quality scores vary by tier
            quality_ranges = {
                'Premium': (8, 10),
                'Standard': (6, 9),
                'Budget': (4, 7)
            }
            quality_score = random.randint(*quality_ranges[quality_tier])

            supplier_data.append({
                'delivery_date': delivery_date.strftime('%Y-%m-%d'),
                'supplier_id': supplier_id,
                'store_id': f'S{random.randint(1, 6):03d}',
                'product_id': product['product_id'],
                'supplier_quality_tier': quality_tier,
                'planned_shelf_life_days': planned_shelf_life,
                'actual_shelf_life_days': actual_shelf_life,
                'shelf_life_variance': round((actual_shelf_life - planned_shelf_life) / planned_shelf_life, 3),
                'delivery_delay_days': delivery_delay,
                'quantity_delivered': random.randint(20, 300),
                'quality_score': quality_score,
                'temperature_maintained': random.random() > (0.05 if quality_tier == 'Premium' else 0.15),
                'packaging_condition': random.choices(['Excellent', 'Good', 'Fair', 'Poor'],
                                                   weights=[50, 30, 15, 5] if quality_tier == 'Premium' else
                                                          [20, 40, 30, 10] if quality_tier == 'Standard' else
                                                          [10, 30, 40, 20])[0]
            })

    df = pd.DataFrame(supplier_data)
    df.to_csv('frischmarkt_data/supplier_performance.csv', index=False)
    print(f"✅ Generated {len(supplier_data)} supplier deliveries with quality impact")
    return df

def generate_loss_summary():
    """Generate a summary of expiry and financial losses"""
    print("💰 Generating Loss Impact Summary...")

    # Read the generated data
    inventory_df = pd.read_csv('frischmarkt_data/inventory_daily.csv')
    products_df = pd.read_csv('frischmarkt_data/products_master.csv')
    stores_df = pd.read_csv('frischmarkt_data/stores_master.csv')
    sales_df = pd.read_csv('frischmarkt_data/sales_transactions.csv') # Load sales data

    # Calculate total revenue
    total_revenue = (sales_df['quantity_sold'] * sales_df['sale_price']).sum()

    # Merge to get product and store details
    inventory_df = inventory_df.merge(products_df[['product_id', 'product_name', 'category', 'expiry_risk_level', 'unit_cost']], on='product_id', how='left')
    inventory_df = inventory_df.merge(stores_df[['store_id', 'store_name', 'management_quality']], on='store_id', how='left')

    # Aggregate total losses
    total_expiry_loss = inventory_df['expiry_loss_eur'].sum()
    total_markdown_loss = inventory_df['markdown_loss_eur'].sum()
    total_loss = inventory_df['total_loss_eur'].sum()

    print(f"\n--- Overall Loss and Revenue Summary (2023) ---")
    print(f"Total Revenue: €{total_revenue:,.2f}") # Added total revenue
    print(f"Total Expiry Loss: €{total_expiry_loss:,.2f}")
    print(f"Total Markdown Loss: €{total_markdown_loss:,.2f}")
    print(f"Grand Total Loss: €{total_loss:,.2f}")

    # Calculate and print percentage loss
    if total_revenue > 0:
        percentage_loss = (total_loss / total_revenue) * 100
        print(f"Percentage Loss (of Revenue): {percentage_loss:,.2f}%")
    else:
        print("Cannot calculate percentage loss: Total Revenue is zero.")


    # Losses by Category
    losses_by_category = inventory_df.groupby('category')['total_loss_eur'].sum().sort_values(ascending=False).reset_index()
    print(f"\n--- Losses by Product Category ---")
    print(losses_by_category.to_string(index=False))

    # Losses by Store
    losses_by_store = inventory_df.groupby(['store_id', 'store_name'])['total_loss_eur'].sum().sort_values(ascending=False).reset_index()
    print(f"\n--- Losses by Store ---")
    print(losses_by_store.to_string(index=False))

    # Top 10 Products by Expiry Loss
    expiry_loss_by_product = inventory_df.groupby(['product_id', 'product_name'])['expiry_loss_eur'].sum().sort_values(ascending=False).reset_index().head(10)
    print(f"\n--- Top 10 Products by Expiry Loss ---")
    print(expiry_loss_by_product.to_string(index=False))

    # Expiry Rates by Management Quality
    # Ensure 'beginning_inventory' column is not empty or contains non-numeric values
    inventory_df['beginning_inventory_safe'] = inventory_df['beginning_inventory'].replace(0, np.nan)
    expiry_rate_by_mgmt = inventory_df.groupby('management_quality').apply(
        lambda x: (x['units_expired'].sum() / x['beginning_inventory_safe'].sum()) if x['beginning_inventory_safe'].sum() > 0 else 0
    ).reset_index(name='overall_expiry_rate')
    expiry_rate_by_mgmt['overall_expiry_rate'] = expiry_rate_by_mgmt['overall_expiry_rate'].apply(lambda x: f"{x:.2%}")
    print(f"\n--- Overall Expiry Rate by Management Quality ---")
    print(expiry_rate_by_mgmt.to_string(index=False))

    print(f"✅ Generated Loss Impact Summary")


def main():
    products_df = generate_products_master()
    stores_df = generate_stores_master()
    external_df = generate_external_factors()

    # Generate inventory and sales data
    inventory_df, sales_df = generate_inventory_and_sales(products_df, stores_df, external_df)


    # Generate supplier performance data
    supplier_performance_df = generate_supplier_performance(products_df)

    # Generate loss summary (requires inventory_daily.csv to be written)
    generate_loss_summary()

    print("\n🎉 Data Generation Complete! Check 'frischmarkt_data/' directory for CSV files.")
    print("Next steps: Analyze the generated data to identify key drivers of expiry loss and develop mitigation strategies.")


if __name__ == "__main__":
    main()
