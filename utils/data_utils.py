import pandas as pd
import numpy as np
from geopy.distance import geodesic
import missingno as msno

def change_column_names(data: pd.DataFrame):
    return (
        data.rename(str.lower, axis=1)
        .rename({
            "delivery_person_id": "rider_id",
            "delivery_person_age": "age",
            "delivery_person_ratings": "ratings",
            "delivery_location_latitude": "delivery_latitude",
            "delivery_location_longitude": "delivery_longitude",
            "time_orderd": "order_time",
            "time_order_picked": "order_picked_time",
            "weatherconditions": "weather",
            "road_traffic_density": "traffic",
            "city": "city_type",
            "time_taken(min)": "time_taken"
        }, axis=1)
    )

def clean_data(dataframe):
    """
    This function cleans the whole data.
    """

    # Step 1: Make a copy
    df = dataframe.copy()

    # ✅ Step 2: Fix location columns properly (lat 1–90, lon 1–180)
    location_cols = ["restaurant_latitude", "restaurant_longitude",
                     "delivery_latitude", "delivery_longitude"]

    for col in location_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")   # convert to numeric
        df[col] = df[col].abs()                             # absolute values
        if "latitude" in col:
            df[col] = df[col].where((df[col] >= 1) & (df[col] <= 90), np.nan)
        else:  # longitude
            df[col] = df[col].where((df[col] >= 1) & (df[col] <= 180), np.nan)

    # Step 3: Continue rest of cleaning
    return (df.
            drop(columns='id').
            drop(df[df['age'].astype(float) < 18].index).
            drop(df[df['ratings'].astype(float) > 5].index).
            replace(to_replace=['NaN ', 'NaN'], value=np.nan).
            assign(
                extracted_city=lambda x: x['rider_id'].astype(str).str.split("RES").str.get(0),
                age=lambda x: x['age'].astype(float),
                ratings=lambda x: x['ratings'].astype(float),
                order_date=lambda d: pd.to_datetime(d['order_date'], dayfirst=True),
                order_day=lambda x: extract_datetime(x['order_date'])['day'],
                order_month=lambda x: extract_datetime(x['order_date'])['month'],
                order_day_of_week=lambda x: extract_datetime(x['order_date'])['day_of_week'],
                order_is_weekend=lambda x: extract_datetime(x['order_date'])['is_weekend'],
                order_time=lambda x: pd.to_datetime(x['order_time'], format='mixed'),
                order_picked_time=lambda x: pd.to_datetime(x['order_picked_time'], format='mixed'),
                pickup_time_minutes=lambda x: np.where(
                    (x["order_picked_time"] - x["order_time"]).dt.total_seconds() / 60 < 0,
                    (x["order_picked_time"] - x["order_time"]).dt.total_seconds() / 60 + 1440,
                    (x["order_picked_time"] - x["order_time"]).dt.total_seconds() / 60),
                order_time_hour=lambda x: x['order_time'].dt.hour,
                time_of_day=lambda x: time_of_day(x['order_time_hour']),
                weather=lambda x: (x['weather'].str.replace("conditions ", "").str.lower().replace("nan", np.nan)),
                traffic=lambda x: x['traffic'].str.strip().str.lower(),
                type_of_order=lambda x: x['type_of_order'].str.strip().str.lower(),
                type_of_vehicle=lambda x: x['type_of_vehicle'].str.strip().str.lower(),
                festival=lambda x: x['festival'].str.strip().str.lower(),
                city_type=lambda x: x['city_type'].str.strip().str.lower(),
                multiple_deliveries=lambda x: x['multiple_deliveries'].astype(float),
                time_taken=lambda x: (x['time_taken'].str.replace("(min) ", "").astype(int)),
                distance_km=lambda x: x.apply(
                    lambda row: geodesic(
                        (row["restaurant_latitude"], row["restaurant_longitude"]),
                        (row["delivery_latitude"], row["delivery_longitude"])
                    ).km if pd.notnull(row["restaurant_latitude"]) and pd.notnull(row["restaurant_longitude"]) 
                           and pd.notnull(row["delivery_latitude"]) and pd.notnull(row["delivery_longitude"])
                    else np.nan,
                    axis=1
                )
            ).
            drop(columns=['order_time', 'order_picked_time'])
           )

def extract_datetime(series):
    date_col = pd.to_datetime(series, dayfirst=True)
    return pd.DataFrame({
        "day": date_col.dt.day,
        "month": date_col.dt.month,
        "year": date_col.dt.year,
        "day_of_week": date_col.dt.day_name(),
        "is_weekend": date_col.dt.day_name().isin(["Saturday", "Sunday"]).astype(int)
    })

def time_of_day(series):
    hours = pd.to_datetime(series, format="mixed").dt.hour
    return(
        pd.cut(series, bins = [0, 6, 12, 17, 20, 24], right = True,
               labels = ["after_midnight", "morning", "afternoon", "evening", "night"])
        
    )

def create_distance_type(data: pd.DataFrame):
    return (
        data.assign(
            distance_type=pd.cut(data["distance_km"], bins=[0, 5, 10, 15, 25],
                                 right=False, labels=["short", "medium", "long", "very_long"])
        )
    )

def perform_data_cleaning(data: pd.DataFrame, saved_data_path="clean_data.csv"):
    cleaned_data = (
        data
        .pipe(change_column_names)
        .pipe(clean_data)
        .pipe(create_distance_type)
    )
    cleaned_data.to_csv(saved_data_path, index=False)
    return cleaned_data

if __name__ == "__main__":
    DATA_PATH = "notebooks/swiggy.csv"
    data = pd.read_csv(DATA_PATH)
    print('Data loaded successfully')
    cleaned = perform_data_cleaning(data)
    print('Cleaned data saved successfully!')
