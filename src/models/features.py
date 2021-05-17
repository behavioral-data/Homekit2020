import sys

def get_feature_with_name(name):
    try:
        identifier = getattr(sys.modules[__name__], name)
    except AttributeError:
        raise NameError(f"{name} is not a valid feature.")
    return identifier

def get_rested(partition):
    not_moving = partition[(partition["steps"]==0) & ~(partition["missing_steps"].astype(bool))]
    return not_moving

def get_awake(partition):
    awake = partition[partition["sleep_classic_0"].astype(bool)]
    return awake

# Features:

def resting_hr_95th_percentile(partition):
    rested = get_rested(partition)
    return rested["heart_rate"].quantile(0.95)

def resting_hr_50th_percentile(partition):
    rested = get_rested(partition)
    return rested["heart_rate"].quantile(0.5)

def resting_hr_std(partition):
    rested = get_rested(partition)
    return rested["heart_rate"].std()

def hr_awake_95th_percentile(partition):
    awake = get_awake(partition)
    return awake["heart_rate"].quantile(0.95)

def steps_moving_streak_95th_percentile(partition):
    not_moving = (partition["steps"] == 0 )
    moving_spans = (not_moving != not_moving.shift(-1)).cumsum()
    return moving_spans.groupby(moving_spans).apply(len).quantile(0.95)

def steps_moving_streak_50th_percentile(partition):
    not_moving = (partition["steps"] == 0 )
    moving_spans = (not_moving != not_moving.shift(-1)).cumsum()
    return moving_spans.groupby(moving_spans).apply(len).quantile(0.50)


    


