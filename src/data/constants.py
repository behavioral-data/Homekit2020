SURVEY_NAMES = ["baseline",
                "follow_up_a",
                "follow_up_b",
                "recovery",
                "screener"]

MTL_NAMES = [
    "mtl_lab_order_results",
    "mtl_lab_order_updates"
]

ACTIVITY_NAMES = [
    "fitbit_minute_level_activity",
    "fitbit_day_level_activity"
]

PROCESSED_DATASETS = [
    "lab_results",
    "lab_updates",
    "lab_results_with_triggerdate",
    "baseline_screener_survey",
    "daily_surveys_onehot",
    "fitbit_day_level_activity"
]

PARQUET_DATASETS = [
    "processed_fitbit_minute_level_activity"
]

DAILY_SLEEP_FEATURES = ['main_in_bed_minutes', 
                        'main_efficiency',
                        'nap_count', 
                        'total_asleep_minutes', 
                        'total_in_bed_minutes'
]

DAILY_STEP_FEATURES =  ['activityCalories',
                        'caloriesOut', 
                        'caloriesBMR', 
                        'marginalCalories', 
                        'sedentaryMinutes', 
                        'lightlyActiveMinutes', 
                        'fairlyActiveMinutes', 
                        'veryActiveMinutes']

