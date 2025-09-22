import numpy as np
import pandas as pd
from pathlib import Path
import logging
from geopy.distance import geodesic

# create logger
logger = logging.getLogger("data_cleaning")
logger.setLevel(logging.INFO)

# console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# add handler to logger
logger.addHandler(handler)

# create a fomratter
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to handler
handler.setFormatter(formatter)

columns_to_drop =  ['rider_id',
                    'restaurant_latitude',
                    'restaurant_longitude',
                    'delivery_latitude',
                    'delivery_longitude',
                    'order_date',
                    "order_time_hour",
                    "order_day",
                    "extracted_city",
                    "order_day_of_week",
                    "order_month"]


def load_data(data_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
    
    except FileNotFoundError:
        logger.error("The file to load does not exist")
    
    return df


def change_column_names(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.rename(str.lower,axis=1)
        .rename({
            "delivery_person_id" : "rider_id",
            "delivery_person_age": "age",
            "delivery_person_ratings": "ratings",
            "delivery_location_latitude": "delivery_latitude",
            "delivery_location_longitude": "delivery_longitude",
            "time_orderd": "order_time",
            "time_order_picked": "order_picked_time",
            "weatherconditions": "weather",
            "road_traffic_density": "traffic",
            "city": "city_type",
            "time_taken(min)": "time_taken"},axis=1)
    )


def data_cleaning(data: pd.DataFrame) -> pd.DataFrame:
    location_cols = ["restaurant_latitude", "restaurant_longitude",
                     "delivery_latitude", "delivery_longitude"]

    for col in location_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")   # convert to numeric
        data[col] = data[col].abs()                             # absolute values
        if "latitude" in col:
            data[col] = data[col].where((data[col] >= 1) & (data[col] <= 90), np.nan)
        else:  # longitude
            data[col] = data[col].where((data[col] >= 1) & (data[col] <= 180), np.nan)

    # Continue rest of cleaning
    return (data.
            drop(columns='id').
            drop(data[data['age'].astype(float) < 18].index).
            drop(data[data['ratings'].astype(float) > 5].index).
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
    
    
# extract day, day name, month and year
def extract_datetime(series: pd.Series) -> pd.DataFrame:
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


def create_distance_type(data: pd.DataFrame) -> pd.DataFrame:
    return(
        data
        .assign(
                distance_type = pd.cut(data["distance_km"],bins=[0,5,10,15,25],
                                        right=False,labels=["short","medium","long","very_long"])
    ))



def drop_columns(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    df = data.drop(columns=columns)
    return df
 
    
    
def perform_data_cleaning(data: pd.DataFrame, saved_data_path: Path) -> None:
    
    cleaned_data = (
        data
        .pipe(change_column_names)
        .pipe(data_cleaning)
        .pipe(create_distance_type)
        .pipe(drop_columns,columns=columns_to_drop)
    )
    
    # save the data
    cleaned_data.to_csv(saved_data_path,index=False)
    
    

if __name__ == "__main__":
    # root path
    root_path = Path(__file__).parent.parent.parent
    # data save directory
    cleaned_data_save_dir = root_path / "data" / "cleaned"
    # make directory if not exits
    cleaned_data_save_dir.mkdir(exist_ok=True,parents=True)
    # cleaned data file name
    cleaned_data_filename = "clean_data.csv"
    # data save path
    cleaned_data_save_path = cleaned_data_save_dir / cleaned_data_filename
    # data load path
    data_load_path = root_path / "data" / "raw" / "swiggy.csv"
    
    # load the data
    df = load_data(data_load_path)
    logger.info("Data read successfully")
    
    # clean the data and save
    perform_data_cleaning(data=df, saved_data_path=cleaned_data_save_path)
    logger.info("Data cleaned and saved")