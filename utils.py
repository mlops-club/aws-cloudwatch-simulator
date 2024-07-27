import random
from datetime import datetime, timedelta, date, time
import pandas as pd
import numpy as np
from typing import List, Iterator, Dict, Any, Optional

def generate_random_data(
    date: date,
    start_time: time,
    end_time: time,
    count: int,
    response_time_range: (int, int),
    null_percentage: float
) -> pd.DataFrame:
    start_datetime: datetime = datetime.combine(date, start_time)
    end_datetime: datetime = datetime.combine(date, end_time)

    random_timestamps: List[datetime] = [
        start_datetime + timedelta(seconds=random.randint(0, int((end_datetime - start_datetime).total_seconds())))
        for _ in range(count)
    ]
    random_timestamps.sort()

    random_response_times: List[Optional[int]] = [
        random.randint(response_time_range[0], response_time_range[1]) for _ in range(count)
    ]

    null_count: int = int(null_percentage * count)
    null_indices: List[int] = random.sample(range(count), null_count)
    for idx in null_indices:
        random_response_times[idx] = None

    data: Dict[str, Any] = {
        'Timestamp': random_timestamps,
        'ResponseTime(ms)': random_response_times
    }
    df: pd.DataFrame = pd.DataFrame(data)
    return df

def calculate_percentile(
    df: pd.DataFrame,
    freq: str,
    percentile: float
) -> pd.DataFrame:
    percentile_df: pd.DataFrame = df.groupby(pd.Grouper(key='Timestamp', freq=freq))["ResponseTime(ms)"]                                    .quantile(percentile).reset_index(name=f"p{int(percentile * 100)}_ResponseTime(ms)")
    percentile_df.replace(to_replace=np.nan, value=None, inplace=True)
    return percentile_df

def aggregate_data(
    df: pd.DataFrame,
    period_length: str
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()  # Return an empty DataFrame if input is empty

    aggregation_funcs = {
        'p50': lambda x: np.percentile(x.dropna(), 50) if not x.dropna().empty else np.nan,
        'p95': lambda x: np.percentile(x.dropna(), 95) if not x.dropna().empty else np.nan,
        'p99': lambda x: np.percentile(x.dropna(), 99) if not x.dropna().empty else np.nan,
        'max': lambda x: np.max(x.dropna()) if not x.dropna().empty else np.nan,
        'min': lambda x: np.min(x.dropna()) if not x.dropna().empty else np.nan,
        'average': lambda x: np.mean(x.dropna()) if not x.dropna().empty else np.nan
    }

    summary_df = df.groupby(pd.Grouper(key='Timestamp', freq=period_length)).agg(
        p50=('ResponseTime(ms)', aggregation_funcs['p50']),
        p95=('ResponseTime(ms)', aggregation_funcs['p95']),
        p99=('ResponseTime(ms)', aggregation_funcs['p99']),
        max=('ResponseTime(ms)', aggregation_funcs['max']),
        min=('ResponseTime(ms)', aggregation_funcs['min']),
        average=('ResponseTime(ms)', aggregation_funcs['average']),
    ).reset_index()
    return summary_df

def chunk_list(input_list: List[Any], size: int = 3) -> Iterator[List[Any]]:
    while input_list:
        chunk: List[Any] = input_list[:size]
        yield chunk
        input_list = input_list[size:]

def evaluate_alarm_state(
    summary_df: pd.DataFrame,
    threshold: int,
    datapoints_to_alarm: int,
    evaluation_range: int,
    aggregation_function: str,
    alarm_condition: str
) -> pd.DataFrame:
    data_points: List[Optional[float]] = list(summary_df[aggregation_function].values)

    data_table_dict: Dict[str, List[Any]] = {
        "DataPoints": [],
        "# of data points that must be filled": [],
        "MISSING": [],
        "IGNORE": [],
        "BREACHING": [],
        "NOT BREACHING": []
    }

    def check_condition(value, threshold, condition):
        if condition == '>':
            return value > threshold
        elif condition == '>=':
            return value >= threshold
        elif condition == '<':
            return value < threshold
        elif condition == '<=':
            return value <= threshold

    for chunk in chunk_list(input_list=data_points, size=evaluation_range):
        data_point_repr: str = ''
        num_dp_that_must_be_filled: int = 0

        for dp in chunk:
            if str(dp).lower() == "nan":
                dp_symbol = '⚫️'
            elif check_condition(dp, threshold, alarm_condition):
                dp_symbol = '🔴'
            else:
                dp_symbol = '🟢'
            data_point_repr += dp_symbol

        if len(chunk) < evaluation_range:
            data_point_repr += '⚫️' * (evaluation_range - len(chunk))

        if data_point_repr.count('⚫️') > (evaluation_range - datapoints_to_alarm):
            num_dp_that_must_be_filled = datapoints_to_alarm - sum([data_point_repr.count('🟢'), data_point_repr.count('🔴')])

        data_table_dict["DataPoints"].append(data_point_repr)
        data_table_dict["# of data points that must be filled"].append(num_dp_that_must_be_filled)

        if num_dp_that_must_be_filled > 0:
            data_table_dict["MISSING"].append("INSUFFICIENT_DATA" if data_point_repr.count('⚫️') == evaluation_range else "Retain current state")
            data_table_dict["IGNORE"].append("Retain current state")
            data_table_dict["BREACHING"].append("ALARM")
            data_table_dict["NOT BREACHING"].append("OK")
        else:
            data_table_dict["MISSING"].append("OK")
            data_table_dict["IGNORE"].append("Retain current state")
            data_table_dict["BREACHING"].append("ALARM" if '🔴' * datapoints_to_alarm in data_point_repr else "OK")
            data_table_dict["NOT BREACHING"].append("ALARM" if '🟢' * datapoints_to_alarm not in data_point_repr else "OK")

    return pd.DataFrame(data_table_dict)
