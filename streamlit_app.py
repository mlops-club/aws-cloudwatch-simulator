import streamlit as st
import pandas as pd
from datetime import time, date
from utils import generate_random_data, evaluate_alarm_state, aggregate_data
from textwrap import dedent
from matplotlib import pyplot as plt

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
        st.scatter_chart(st.session_state.df.set_index("Timestamp"))

    # Section 2 - Calculate Aggregations
    st.header("Section 2 - Calculate Aggregations")
    aggregation_form()

    if not st.session_state.aggregated_df.empty:
        display_dataframe("Aggregated Summary Data (Storage)", st.session_state.aggregated_df)
        aggregation_function_input__storage = st.selectbox(
            "Aggregation Function (Storage)",
            ['p50', 'p95', 'p99', 'max', 'min', 'average'],
            key='aggregation_function_input__storage',
            help="Select the aggregation function for visualizing the data."
        )
        st.line_chart(st.session_state.aggregated_df.set_index("Timestamp")[st.session_state.aggregation_function_input__storage])

    # Section 3 - Summary Data Aggregated by Period
    st.header("Section 3 - Summary Data Aggregated by Period")
    summary_by_period_form()

    if not st.session_state.summary_by_period_df.empty:
        display_dataframe("Summary Data Aggregated by Period (for Alarm)", st.session_state.summary_by_period_df)
        aggregation_function_input__alarm = st.selectbox(
            "Aggregation Function (Alarm)",
            ['p50', 'p95', 'p99', 'max', 'min', 'average'],
            key='aggregation_function_input__alarm',
            help="Select the aggregation function for visualizing the data."
        )
        st.line_chart(st.session_state.summary_by_period_df.set_index("Timestamp")[st.session_state.aggregation_function_input__alarm])

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
    if 'aggregated_df' not in st.session_state:
        st.session_state.aggregated_df = pd.DataFrame()
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

def aggregation_form() -> None:
    freq_input = st.selectbox("Period (bin)", ['1min', '5min', '15min'], key='freq_input', help="Select the frequency for aggregating the data.")
    if not st.session_state.df.empty:
        st.session_state.aggregated_df = aggregate_data(st.session_state.df, freq_input)

def summary_by_period_form() -> None:
    period_length_input = st.selectbox("Period Length", ['1min', '5min', '15min'], key='period_length_input', help="Select the period length for aggregating the summary data.")
    if not st.session_state.df.empty:
        st.session_state.summary_by_period_df = aggregate_data(st.session_state.df, period_length_input)

def alarm_state_form() -> None:
    threshold_input = st.slider("Threshold (ms)", min_value=50, max_value=300, value=150, key='threshold_input', help="Specify the threshold value for evaluating the alarm state.")
    datapoints_to_alarm_input = st.number_input("Datapoints to Alarm", min_value=1, value=3, key='datapoints_to_alarm_input', help="Specify the number of data points required to trigger an alarm.")
    evaluation_range_input = st.number_input("Evaluation Range", min_value=1, value=5, key='evaluation_range_input', help="Specify the range of data points to evaluate for alarm state.")
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
            aggregation_function=st.session_state.aggregation_function_input__alarm,
            alarm_condition=alarm_condition_input
        )

def display_dataframe(title: str, df: pd.DataFrame) -> None:
    st.write(title)
    st.dataframe(df)

def plot_time_series(df: pd.DataFrame, threshold: int, alarm_condition: str, evaluation_range: int) -> None:
    timestamps = df['Timestamp']
    response_times = df[st.session_state.aggregation_function_input__alarm]

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
            max_value = max(filter(lambda x: x is not None, df[st.session_state.aggregation_function_input__alarm]))
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
        "Symbol": ["🔴", "⚫️", "🟢"],
        "Meaning": [
            "Breaching data point: This data point exceeds the threshold.",
            "Missing data point: This data point is missing or not reported.",
            "Non-breaching data point: This data point is within the threshold."
        ]
    }
    symbol_df = pd.DataFrame(symbol_data)
    st.table(symbol_df)

    # Columns
    st.write(dedent("""    #### Columns: Strategies for handling missing data points [docs](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/AlarmThatSendsEmail.html#alarms-and-missing-data)
             
    Sometimes, no metric events may have been reported during a given time period. In this case,
    you must decide how you will treat missing data points. Ignore it? Or consider it a failure.
             
    Here are the 4 supported strategies in AWS:
    """))

    column_data = {
        "Strategy": ["missing", "ignore", "breaching", "notBreaching"],
        "Explanation": [
            "If all data points in the alarm evaluation range are missing, the alarm transitions to INSUFFICIENT_DATA. Possible values: INSUFFICIENT_DATA, Retain current state, ALARM, OK.",
            "The current alarm state is maintained. Possible values: Retain current state, ALARM, OK.",
            "Missing data points are treated as \"bad\" and breaching the threshold. Possible values: ALARM, OK.",
            "Missing data points are treated as \"good\" and within the threshold. Possible values: ALARM, OK."
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
