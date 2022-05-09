# %%
import vaex
import json
import pandas as pd

# %%
GARMIN_ACTIVITY_REGEX = "/projects/bdata/datasets/gatesfoundation/raw/COVID_Signals/*/all_other_datafiles/garmin/activities/*.parquet"
garmin_activity = vaex.open(GARMIN_ACTIVITY_REGEX)

# %%
list(garmin_activity.columns)

# %%
garmin_activity["activityType"].value_counts()

# %%
len(garmin_activity)

# %%
garmin_activity["id_participant_external"].nunique()

# %%
garmin_activity.column_names

# %%
GARMIN_SLEEPS_REGEX = "/projects/bdata/datasets/gatesfoundation/raw/COVID_Signals/*/all_other_datafiles/garmin/sleeps/*.parquet"
garmin_sleeps = vaex.open(GARMIN_SLEEPS_REGEX)

# %%
garmin_sleeps.column_names

# %%
garmin_sleeps = garmin_sleeps.to_pandas_df()

# %%
# Based off of the FitBit "classic" sleep api: https://dev.fitbit.com/build/reference/web-api/sleep/get-sleep-log-by-date/
GARMIN_SLEEP_STAGE_MAP={
    "light": 1,
    "deep":2,
    "rem":2,
    "awake":3
}

# %%
garmin_sleeps["sleepLevelsMap"] = garmin_sleeps["sleepLevelsMap"].map(json.loads)

# %%
exploded_sleep = (pd.melt(
    pd.concat([garmin_sleeps,pd.json_normalize(garmin_sleeps["sleepLevelsMap"])],axis=1),
    id_vars=["id_participant_external","startTimeOffsetInSeconds"],
    value_vars=["awake","light","rem","deep"],
    var_name="sleep_type"
    ).explode("value").dropna(subset=["value"])
)


# %%
processed_sleep = pd.concat([exploded_sleep.drop(columns="value").reset_index(drop=True),pd.json_normalize(exploded_sleep["value"])],axis=1)

# %%
processed_sleep["endTime"] = pd.to_datetime(processed_sleep["endTimeInSeconds"], unit="s") + pd.to_timedelta(processed_sleep["startTimeOffsetInSeconds"])
processed_sleep["startTime"] = pd.to_datetime(processed_sleep["startTimeInSeconds"], unit="s") + pd.to_timedelta(processed_sleep["startTimeOffsetInSeconds"])
processed_sleep["sleep_classic"] = processed_sleep["sleep_type"].map(GARMIN_SLEEP_STAGE_MAP) 
processed_sleep["id_participant_external"] = processed_sleep["id_participant_external"].astype("category") 
processed_sleep["durationInSeconds"] = (processed_sleep["endTime"] - processed_sleep["startTime"]).dt.seconds
processed_sleep["timestamp"] = processed_sleep["startTime"] 
processed_sleep = processed_sleep.dropna().set_index("id_participant_external")

# %%
processed_sleep.head(10)

# %%
garmin_activity.column_names

# %%
processed_heart_rate = garmin_activity[["id_participant_external","startTimeInSeconds","startTimeOffsetInSeconds",
                                        'averageHeartRateInBeatsPerMinute',"durationInSeconds"]].to_pandas_df()
processed_heart_rate = processed_heart_rate.dropna(subset=["durationInSeconds"])                                        
processed_heart_rate["timestamp"] = pd.to_datetime(processed_heart_rate["startTimeInSeconds"],unit="s") +\
                                    pd.to_timedelta(processed_heart_rate["startTimeOffsetInSeconds"].astype(int), unit="s")
processed_heart_rate["durationInSeconds"] = processed_heart_rate["durationInSeconds"].astype(int)
processed_heart_rate["id_participant_external"] = processed_heart_rate["id_participant_external"].astype("category") 
processed_heart_rate = processed_heart_rate.dropna().set_index("id_participant_external")
processed_heart_rate.head(10)

# %%
processed_steps = garmin_activity[["id_participant_external","startTimeInSeconds","startTimeOffsetInSeconds",
                                        'steps',"durationInSeconds"]].to_pandas_df()
processed_steps = processed_steps.dropna(subset=["durationInSeconds"])  
processed_steps["timestamp"] = pd.to_datetime(processed_steps["startTimeInSeconds"],unit="s") +\
                                    pd.to_timedelta(processed_steps["startTimeOffsetInSeconds"].astype(int), unit="s")
processed_steps["durationInSeconds"] = processed_steps["durationInSeconds"].astype(int)
processed_steps["id_participant_external"] = processed_steps["id_participant_external"].astype("category")
processed_steps = processed_steps.dropna().set_index("id_participant_external")
processed_steps.head(10)

# %%
from src.data.make_dataset import explode_str_column, get_new_index, safe_loc
from src.data.utils import  process_minute_level_pandas
from tqdm import tqdm

# %%
processed_out_path = "/projects/bdata/datasets/gatesfoundation/processed/COVID_Signals/garmin_minute_level_activity"
users_with_steps = processed_steps.index.unique()
all_results = []

for user in tqdm(users_with_steps.values):
    exploded_sleep = explode_str_column(safe_loc(processed_sleep,user),
                                target_col = "sleep_classic",
                                # rename_target_column="sleep_classic",
                                start_col="startTime",
                                dur_col = "durationInSeconds",
                                single_val=True,
                                dtype=pd.Int8Dtype())
    exploded_hr =  explode_str_column(safe_loc(processed_heart_rate,user),
                                        target_col = "averageHeartRateInBeatsPerMinute",
                                        rename_target_column="heart_rate",
                                        start_col="timestamp",
                                        single_val=True,
                                        dur_col = "durationInSeconds",
                                        dtype=pd.Int8Dtype())
    exploded_steps = explode_str_column(safe_loc(processed_steps,user),
                                        target_col = "steps",
                                        # rename_target_column="averageHeartRateInBeatsPerMinute",
                                        start_col="timestamp",
                                        single_val=True,
                                        dur_col = "durationInSeconds",
                                        dtype=pd.Int8Dtype())
    steps_and_hr = exploded_steps.join(exploded_hr,how = "left") 
    merged = steps_and_hr.join(exploded_sleep,how="left")                        

    
    processed = process_minute_level_pandas(minute_level_df = merged)

    # Keep datatypes in check
    processed["heart_rate"] = processed["heart_rate"].astype(pd.Int16Dtype())
    processed["participant_id"] = user
    all_results.append(processed)

all_results = pd.concat(all_results)
all_results["sleep_classic_0"] = all_results["sleep_classic_0"].fillna(False)
all_results["sleep_classic_1"] = all_results["sleep_classic_1"].fillna(False)
all_results["sleep_classic_2"] = all_results["sleep_classic_2"].fillna(False)
all_results["sleep_classic_3"] = all_results["sleep_classic_3"].fillna(False)

all_results.to_parquet(path = processed_out_path, partition_cols=["date"], engine="fastparquet")

# %%
processed_heart_rate.dtypes

# %%
processed_sleep["timestamp"].median()

# %%



