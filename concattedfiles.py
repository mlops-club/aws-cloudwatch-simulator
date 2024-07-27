.
├── streamlit_app.py
└── utils.py

1 directory, 2 files



# File: ./streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, time, date
from typing import List, Dict, Any, Tuple
from utils import generate_random_data, calculate_percentile, evaluate_alarm_state, aggregate_data

# Constants
HARD_CODED_DATE = date(2024, 7, 26)

def main():
    st.title("Streamlit App for Data Generation and Analysis")

    # Initialize session state
    initialize_session_state()

    # Section 1 - Generate random data
    st.header("Section 1 - Generate Random Data")
    generate_data_form()

    if not st.session_state.df.empty:
        display_dataframe("Raw Event Data", st.session_state.df)

    # Section 2 - Calculate Percentile
    st.header("Section 2 - Calculate Percentile")
    percentile_form()

    if not st.session_state.percentile_df.empty:
        display_dataframe("Aggregated Summary Data", st.session_state.percentile_df)

    # Section 3 - Summary Data Aggregated by Period
    st.header("Section 3 - Summary Data Aggregated by Period")
    summary_by_period_form()

    if not st.session_state.summary_by_period_df.empty:
        display_dataframe("Summary Data Aggregated by Period", st.session_state.summary_by_period_df)

    # Section 4 - Evaluate Alarm State
    st.header("Section 4 - Evaluate Alarm State")
    alarm_state_form()

    if not st.session_state.alarm_state_df.empty:
        plot_time_series(st.session_state.summary_by_period_df, st.session_state.threshold_input, st.session_state.alarm_condition_input, st.session_state.evaluation_range_input)
        display_alarm_state_evaluation(st.session_state.alarm_state_df)

    display_key_tables()

def initialize_session_state() -> None:
    if 'df' not in st.session_state:
        st.session_state.df = pd.DataFrame()
    if 'percentile_df' not in st.session_state:
        st.session_state.percentile_df = pd.DataFrame()
    if 'summary_by_period_df' not in st.session_state:
        st.session_state.summary_by_period_df = pd.DataFrame()
    if 'alarm_state_df' not in st.session_state:
        st.session_state.alarm_state_df = pd.DataFrame()

def generate_data_form() -> None:
    with st.form(key='generate_data_form'):
        start_time_input = st.time_input("Start Time", time(12, 0), help="Select the start time for generating random data.")
        end_time_input = st.time_input("End Time", time(12, 30), help="Select the end time for generating random data.")
        count_input = st.slider("Count", min_value=1, max_value=200, value=60, help="Specify the number of data points to generate.")
        response_time_range_input = st.slider("Response Time Range (ms)", min_value=50, max_value=300, value=(100, 250), help="Select the range of response times in milliseconds.")
        null_percentage_input = st.slider("Null Percentage", min_value=0.0, max_value=1.0, value=0.5, help="Select the percentage of null values in the generated data.")
        submit_button = st.form_submit_button(label='Generate Data')

        if submit_button:
            st.session_state.df = generate_random_data(
                date=HARD_CODED_DATE,
                start_time=start_time_input,
                end_time=end_time_input,
                count=count_input,
                response_time_range=response_time_range_input,
                null_percentage=null_percentage_input
            )

def percentile_form() -> None:
    freq_input = st.selectbox("Period (bin)", ['1min', '5min', '15min'], key='freq_input', help="Select the frequency for aggregating the data.")
    percentile_input = st.slider("Percentile", min_value=0.0, max_value=1.0, value=0.95, key='percentile_input', help="Select the percentile for calculating the aggregated summary data.")
    if not st.session_state.df.empty:
        st.session_state.percentile_df = calculate_percentile(st.session_state.df, freq_input, percentile_input)

def summary_by_period_form() -> None:
    period_length_input = st.selectbox("Period Length", ['1min', '5min', '15min'], key='period_length_input', help="Select the period length for aggregating the summary data.")
    if not st.session_state.df.empty:
        st.session_state.summary_by_period_df = aggregate_data(st.session_state.df, period_length_input)

def alarm_state_form() -> None:
    threshold_input = st.number_input("Threshold (ms)", min_value=50, max_value=300, value=150, key='threshold_input', help="Specify the threshold value for evaluating the alarm state.")
    datapoints_to_alarm_input = st.number_input("Datapoints to Alarm", min_value=1, value=3, key='datapoints_to_alarm_input', help="Specify the number of data points required to trigger an alarm.")
    evaluation_range_input = st.number_input("Evaluation Range", min_value=1, value=5, key='evaluation_range_input', help="Specify the range of data points to evaluate for alarm state.")
    aggregation_function_input = st.selectbox(
        "Aggregation Function",
        ['p50', 'p95', 'p99', 'max', 'min', 'average'],
        key='aggregation_function_input',
        help="Select the aggregation function for visualizing the data and computing alarms."
    )
    alarm_condition_input = st.selectbox(
        "Alarm Condition",
        ['>', '>=', '<', '<='],
        key='alarm_condition_input',
        help="Select the condition for evaluating the alarm state."
    )
    if not st.session_state.summary_by_period_df.empty:
        st.session_state.alarm_state_df = evaluate_alarm_state(
            summary_df=st.session_state.summary_by_period_df,
            threshold=threshold_input,
            datapoints_to_alarm=datapoints_to_alarm_input,
            evaluation_range=evaluation_range_input,
            aggregation_function=aggregation_function_input,
            alarm_condition=alarm_condition_input
        )

def display_dataframe(title: str, df: pd.DataFrame) -> None:
    st.write(title)
    st.dataframe(df)

def plot_time_series(df: pd.DataFrame, threshold: int, alarm_condition: str, evaluation_range: int) -> None:
    timestamps = df['Timestamp']
    response_times = df[st.session_state.aggregation_function_input]

    segments = []
    current_segment = {'timestamps': [], 'values': []}

    for timestamp, value in zip(timestamps, response_times):
        if pd.isna(value):
            if current_segment['timestamps']:
                segments.append(current_segment)
                current_segment = {'timestamps': [], 'values': []}
        else:
            current_segment['timestamps'].append(timestamp)
            current_segment['values'].append(value)

    if current_segment['timestamps']:
        segments.append(current_segment)

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Response Time (ms)', color=color)

    for segment in segments:
        ax1.plot(segment['timestamps'], segment['values'], color=color, linewidth=0.5)
        ax1.scatter(segment['timestamps'], segment['values'], color=color, s=10)

    line_style = '--' if alarm_condition in ['<', '>'] else '-'
    ax1.axhline(y=threshold, color='r', linestyle=line_style, linewidth=0.8, label='Threshold')
    ax1.tick_params(axis='y', labelcolor=color)

    if alarm_condition in ['<=', '<']:
        ax1.fill_between(timestamps, 0, threshold, color='pink', alpha=0.3)
    else:
        ax1.fill_between(timestamps, threshold, response_times.max(), color='pink', alpha=0.3)

    period_indices = range(len(df))
    ax2 = ax1.twiny()
    ax2.set_xticks(period_indices)
    ax2.set_xticklabels(period_indices, fontsize=8)
    ax2.set_xlabel('Time Periods', fontsize=8)
    ax2.xaxis.set_tick_params(width=0.5)

    for idx in period_indices:
        if idx % evaluation_range == 0:
            ax1.axvline(x=df['Timestamp'].iloc[idx], color='green', linestyle='-', alpha=0.3)
            max_value = max(filter(lambda x: x is not None, df[st.session_state.aggregation_function_input]))
            ax1.text(df['Timestamp'].iloc[idx], max_value * 0.95, f"[{idx // evaluation_range}]", rotation=90, verticalalignment='bottom', color='grey', alpha=0.7, fontsize=8)
        else:
            ax1.axvline(x=df['Timestamp'].iloc[idx], color='grey', linestyle='--', alpha=0.3)

    ax1.annotate('Alarm threshold', xy=(0.98, threshold), xycoords=('axes fraction', 'data'), ha='right', va='bottom', fontsize=8, color='red', backgroundcolor='none')

    fig.tight_layout()
    st.pyplot(fig)

def display_alarm_state_evaluation(df: pd.DataFrame) -> None:
    st.write("Alarm State Evaluation")
    st.dataframe(df)

def display_key_tables() -> None:
    st.write("### Key")

    # Symbols
    st.write("#### Symbols")
    symbol_data = {
        "Symbol": ["X", "-", "0"],
        "Meaning": [
            "Breaching data point: This data point exceeds the threshold.",
            "Missing data point: This data point is missing or not reported.",
            "Non-breaching data point: This data point is within the threshold."
        ]
    }
    symbol_df = pd.DataFrame(symbol_data)
    st.table(symbol_df)

    # Columns
    st.write("#### Columns")
    column_data = {
        "Column": ["MISSING", "IGNORE", "BREACHING", "NOT BREACHING"],
        "Meaning": [
            "Action to take when all data points are missing. Possible values: INSUFFICIENT_DATA, Retain current state, ALARM, OK.",
            "Action to take when data points are missing but ignored. Possible values: Retain current state, ALARM, OK.",
            "Action to take when missing data points are treated as breaching. Possible values: ALARM, OK.",
            "Action to take when missing data points are treated as not breaching. Possible values: ALARM, OK."
        ]
    }
    column_df = pd.DataFrame(column_data)
    st.table(column_df)

    # States
    st.write("#### States")
    state_data = {
        "State": ["ALARM", "OK", "Retain current state", "INSUFFICIENT_DATA"],
        "Description": [
            "Alarm state is triggered.",
            "Everything is within the threshold.",
            "The current alarm state is maintained.",
            "Not enough data to make a determination."
        ]
    }
    state_df = pd.DataFrame(state_data)
    st.table(state_df)

if __name__ == "__main__":
    main()



# File: ./utils.py
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
    percentile_df: pd.DataFrame = df.groupby(pd.Grouper(key='Timestamp', freq=freq))["ResponseTime(ms)"]\
                                    .quantile(percentile).reset_index(name=f"p{int(percentile * 100)}_ResponseTime(ms)")
    percentile_df.replace(to_replace=np.nan, value=None, inplace=True)
    return percentile_df

def aggregate_data(
    df: pd.DataFrame,
    period_length: str
) -> pd.DataFrame:
    aggregation_funcs = {
        'p50': lambda x: np.percentile(x.dropna(), 50),
        'p95': lambda x: np.percentile(x.dropna(), 95),
        'p99': lambda x: np.percentile(x.dropna(), 99),
        'max': lambda x: np.max(x.dropna()),
        'min': lambda x: np.min(x.dropna()),
        'average': lambda x: np.mean(x.dropna())
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
            if dp is None:
                data_point_repr += '-'
            elif check_condition(dp, threshold, alarm_condition):
                data_point_repr += 'X'
            else:
                data_point_repr += '0'

        if len(chunk) < evaluation_range:
            data_point_repr += '-' * (evaluation_range - len(chunk))

        if data_point_repr.count('-') > (evaluation_range - datapoints_to_alarm):
            num_dp_that_must_be_filled = datapoints_to_alarm - sum([data_point_repr.count('0'), data_point_repr.count('X')])

        data_table_dict["DataPoints"].append(data_point_repr)
        data_table_dict["# of data points that must be filled"].append(num_dp_that_must_be_filled)

        if num_dp_that_must_be_filled > 0:
            data_table_dict["MISSING"].append("INSUFFICIENT_DATA" if data_point_repr.count('-') == evaluation_range else "Retain current state")
            data_table_dict["IGNORE"].append("Retain current state")
            data_table_dict["BREACHING"].append("ALARM")
            data_table_dict["NOT BREACHING"].append("OK")
        else:
            data_table_dict["MISSING"].append("OK")
            data_table_dict["IGNORE"].append("Retain current state")
            data_table_dict["BREACHING"].append("ALARM" if 'X' * datapoints_to_alarm in data_point_repr else "OK")
            data_table_dict["NOT BREACHING"].append("ALARM" if '0' * datapoints_to_alarm not in data_point_repr else "OK")

    return pd.DataFrame(data_table_dict)



