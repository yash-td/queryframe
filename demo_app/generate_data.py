"""Generate realistic sample sales data for the QueryFrame demo."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_sales_data(n_rows: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic sales DataFrame with multiple dimensions."""
    rng = np.random.default_rng(seed)

    regions = ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East"]
    region_weights = [0.35, 0.30, 0.20, 0.10, 0.05]

    categories = {
        "Electronics": ["Laptop", "Smartphone", "Tablet", "Headphones", "Smartwatch", "Camera"],
        "Home & Kitchen": ["Coffee Maker", "Blender", "Air Fryer", "Vacuum Cleaner", "Microwave"],
        "Clothing": ["T-Shirt", "Jeans", "Jacket", "Dress", "Sneakers", "Hoodie"],
        "Books": ["Fiction", "Non-Fiction", "Cookbook", "Biography", "Children's Book"],
        "Sports & Outdoors": ["Yoga Mat", "Dumbbells", "Tent", "Bike", "Running Shoes"],
    }

    category_prices = {
        "Electronics": (200, 2000),
        "Home & Kitchen": (30, 500),
        "Clothing": (15, 150),
        "Books": (8, 35),
        "Sports & Outdoors": (20, 800),
    }

    payment_methods = ["Credit Card", "PayPal", "Bank Transfer", "Apple Pay", "Google Pay"]
    customer_segments = ["Consumer", "SMB", "Enterprise"]
    segment_weights = [0.60, 0.30, 0.10]

    dates = pd.date_range("2024-01-01", "2026-03-31", freq="D")

    rows = []
    for i in range(n_rows):
        date = rng.choice(dates)
        month = pd.Timestamp(date).month

        # Seasonality: Q4 and summer are hot
        seasonal_boost = 1.5 if month in (11, 12) else (1.2 if month in (6, 7, 8) else 1.0)

        category = rng.choice(list(categories.keys()))
        product = rng.choice(categories[category])
        price_low, price_high = category_prices[category]
        unit_price = round(rng.uniform(price_low, price_high), 2)

        quantity = int(rng.integers(1, 10) * seasonal_boost)
        discount_pct = round(rng.choice([0, 0, 0, 5, 10, 15, 20]), 0)
        revenue = round(quantity * unit_price * (1 - discount_pct / 100), 2)

        region = rng.choice(regions, p=region_weights)
        segment = rng.choice(customer_segments, p=segment_weights)
        payment = rng.choice(payment_methods)

        # Customer satisfaction influenced by discount (happier with bigger discounts)
        rating = round(min(5.0, rng.normal(4.0 + discount_pct * 0.02, 0.6)), 1)
        rating = max(1.0, rating)

        # Shipping
        shipping_days = int(rng.integers(1, 14))
        shipping_cost = round(rng.uniform(5, 50), 2)

        rows.append({
            "order_id": f"ORD-{i + 100000:06d}",
            "date": date,
            "region": region,
            "category": category,
            "product": product,
            "quantity": quantity,
            "unit_price": unit_price,
            "discount_pct": discount_pct,
            "revenue": revenue,
            "customer_segment": segment,
            "payment_method": payment,
            "shipping_days": shipping_days,
            "shipping_cost": shipping_cost,
            "customer_rating": rating,
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_sales_data(2000)
    df.to_csv("sales_data.csv", index=False)
    print(f"Generated {len(df)} rows")
    print(df.head())
    print(f"\nColumns: {list(df.columns)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total revenue: ${df['revenue'].sum():,.2f}")
