import streamlit as st
import pandas as pd
from datetime import time, date
from utils import generate_random_data, evaluate_alarm_state, aggregate_data, re_aggregate_data, downsample
from textwrap import dedent
from matplotlib import pyplot as plt

# Constants
TODAYS_DATE = date.today()

def main():
    st.title("AWS CloudWatch Simulator")
    st.markdown(dedent("""\
    Monitoring and alerting can be confusing to learn. There is some theory you need to understand first.
                       
    This app is an interative tutorial to help you understand how to record metrics describing the performance
    of an app, and build alerts off of them using AWS CloudWatch.
                       
    Lets get started! 🎉
    """))

    # Initialize session state
    initialize_session_state()

    # Section 1 - Generate random data
    st.header("1 - Generate a series of measurements")
    st.markdown(dedent("""\
    Suppose we have a REST API with a ✨very popular✨ `GET /greeting?name=...` endpoint.
                       
    Each time someone calls the endpoint, we can record how long it takes to respond, aka the ***response latency***.
                       
    Use this form to generate a random dataset of response times.
    """))

    generate_data_form()

    if not st.session_state.df.empty:
        st.markdown("### Recorded request latencies")
        display_dataframe("Raw timeseries events", st.session_state.df)
        st.scatter_chart(st.session_state.df.set_index("timestamp"))

        st.markdown(dedent("""\
        #### 🚚 ➡ ☁️
        We can ship these metrics to a time series database such as AWS CloudWatch in a few ways.              
        """))
        st.warning("In the CloudWatch Metrics database, data points are organized into [Namespaces, Metrics, and Dimensions](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/cloudwatch_concepts.html#Namespace). Think of a Metric as a dedicated table in a database for a single timeseries, e.g. response latency measurements.", icon="💡")
        st.markdown(dedent("""\
        #### Option 1 - AWS SDK (good)
                           
        Our application could use the AWS SDK to upload the data points using the AWS CloudWatch endpoints, e.g.
                                 
        ```python
        import boto3
        cloudwatch = boto3.client('cloudwatch')
        cloudwatch.put_metric_data(
            Namespace='MyApp',
            MetricData=[
                {
                    'MetricName': 'Latency',
                    'timestamp': '2021-08-01T12:00:00',
                    'Value': 102,
                    'Unit': 'Milliseconds'
                },
                ... # more metrics data points, recorded at different times
            ]
        )      
        ```
                           
        It is more cost effective to send data points in a batch, but they can be sent individually as well.
                           
        ---
                           
        #### Option 2 - Structured Logs (better)

        Our application could write metrics to stdout in AWS's [Embedded Metric Format (EMF)](https://www.youtube.com/watch?v=HdopVzW6pX0) (structured JSON) and sent to CloudWatch Logs.
                           
        CloudWatch logs automatically extracts metrics from EMF-formatted logs and sends them to CloudWatch Metrics.
                           
        That is great because it is 
                           
        1. 💰 **cheaper**: you are not charged for calls to CloudWatch's PutMetric endpoint, and 
        2. ⚡️ **faster**: logging to stdout is WAY faster than making a network call--especially a 2-way, synchonous HTTP call. And a side process can batch and send our logs without our app having to slow down or worry about that.
                           
        ---
                           
        ### Option 3 - Built-in Metrics (best)
                           
        Some common metrics, such as API Gateway response latency or Lambda runtime can actually be recorded 
        in CloudWatch Metrics automatically. No code required!
                           
        This is ideal, but not all metrics are automatically captured, such as application-specific metrics like "how many OpenAI tokens have we used?"
                           
        ---
        """))

    if not st.session_state.df.empty:

        # Section 2 - Calculate Aggregations
        st.header("2 - AWS aggregates the metrics")
        st.markdown(dedent("""\
        This step represents our metrics data after AWS CloudWatch processes and stores it.

        Storing raw metrics data can be expensive 💰 (see [CloudWatch Metrics pricing](https://aws.amazon.com/cloudwatch/pricing/)). If your app has high traffic, or bad code, you could send 100s, 1,000s, or 1,000,000s+ of measurement 
        data points per second to AWS CloudWatch.
                        
        This metrics data is meant to be analyzed with queries that power visualizations and alerts--which requires compute--which costs more money the more metrics data you have stored.
                        
        AWS CloudWatch generally aggregates data into a ***resolution*** of 5 minute intervals. 
                        
        In other words, CloudWatch bins data, genrally into ***periods*** of 5 minutes
        and only stores aggregate statistics for each period. This decreases the amount of data stored and queried by orders of magnitude. ✅
                        
        You can pay more for AWS to aggregate data at a "higher" (or "finer") resolution, e.g. 1-minute or even 1-second periods. 
        """))
        st.info("Use this form to aggregate the raw data points into periods of different lengths and plot some of [the many statistics that CloudWatch computes](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/Statistics-definitions.html) over aggregated periods.", icon="📌")
        aggregation_form()

        if not st.session_state.aggregated_df.empty:
            display_dataframe("Aggregated Statistics over Periods", st.session_state.aggregated_df)
            aggregation_function_input__storage = st.selectbox(
                "Aggregation Statistic (just for exploration; does not affect downstream steps)",
                ['p50', 'p95', 'p99', 'max', 'min', 'average'],
                key='aggregation_function_input__storage',
                help="Select the aggregation function for visualizing the data."
            )
            st.line_chart(st.session_state.aggregated_df.set_index("timestamp")[aggregation_function_input__storage])

        # Section 3 - Summary Data Aggregated by Period
        st.header("3 - Optionally aggregate metrics further for Alarms and Dashboards")
        st.markdown(dedent("""\
        You can plot metrics in a CloudWatch dashboard. 
                        
        When doing this, you can choose to aggregate the data further or run additional queries on it to analyze it and answer particular questions.
                        
        We will skip discussing dashboards and focus on ***alerts*** (or ***alarms*** in CloudWatch terms).
                        
        Suppose we want an alert that triggers if our endpoint starts to take longer than usual to respond.
                        
        CloudWatch's concept of alarms can alert you when a metric, such as response latency, "breaches" a certain *threshold* for a certain *number of periods*.
        """))
        st.info("Use this form to bin the data into periods (optionally of shorter length than the previous step).\n\nThis will set the period length used to create an alarm in the next step.", icon="📌")
        summary_by_period_form()

        if not st.session_state.summary_by_period_df.empty:
            display_dataframe("Summary Data Aggregated by Period (for Alarm)", st.session_state.summary_by_period_df)
            aggregation_function_input__alarm = st.selectbox(
                "Aggregation Statistic (used for alarm evaluation in next step)",
                ['p50', 'p95', 'p99', 'max', 'min', 'average'],
                key='aggregation_function_input__alarm',
                help="Select the aggregation function for visualizing the data."
            )
            st.line_chart(st.session_state.summary_by_period_df.set_index("timestamp")[aggregation_function_input__alarm])

        # Section 4 - Evaluate Alarm State
        st.header("4 - Configure and evaluate an alarm")
        
        # define what "breaching" means (threshold and condition) and evaluate the data
        alarm_state_form()
        plot_time_series(st.session_state.summary_by_period_df, st.session_state.threshold_input, st.session_state.alarm_condition_input, st.session_state.evaluation_range_input)
        datapoints_to_alarm_input = st.number_input("Datapoints to Alarm", min_value=1, value=3, key='datapoints_to_alarm_input', help="Specify the number of data points with in the overall evaluation range that must be breaching in order to trigger an alarm.")
        evaluate_breaching_data_points()
        st.write("%d out of %d data points must be breaching to trigger an alarm." % (st.session_state.datapoints_to_alarm_input, st.session_state.evaluation_range_input))    
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
        start_time_input = st.time_input("Start Time", time(12, 0), help="No generated data points will have earlier timestamps than this.")
        end_time_input = st.time_input("End Time", time(12, 30), help="No generated data points will have later timestamps than this.")
        count_input = st.slider("Number of requests", min_value=1, max_value=200, value=20, help="Specify the number of data points to generate.")
        response_time_range_input = st.slider("Response Time Range (ms)", min_value=50, max_value=300, value=(140, 180), help="Select the range of response times in milliseconds. The generated response latencies will be in this range.")
        null_percentage_input = st.slider("Percentage of null data points", min_value=0.0, max_value=1.0, value=0., help="Select the percentage of null values in the generated data. We will use this to simulate 'missing data'--or time periods where no requests were recorded.\n\nCloudWatch does not actually have a concept of data points with null values.")
        submit_button = st.form_submit_button(label='Generate Data')

        if submit_button:
            st.session_state.df = generate_random_data(
                date=TODAYS_DATE,
                start_time=start_time_input,
                end_time=end_time_input,
                count=count_input,
                response_time_range=response_time_range_input,
                null_percentage=null_percentage_input
            )

def aggregation_form() -> None:
    freq_input = st.selectbox("Storage resolution for metric", ['1min', '2min', '3min', '5min', '10min', '15min'], key='freq_input', help="Select the frequency for aggregating the data.")
    if not st.session_state.df.empty:
        st.session_state.aggregated_df = aggregate_data(st.session_state.df, freq_input)

def summary_by_period_form() -> None:
    period_length_input = st.selectbox("Period Length", ['1min', '2min', '3min', '5min', '10min', '15min'], key='period_length_input', help="Select the period length for aggregating the summary data.")
    if not st.session_state.aggregated_df.empty:
        agg_period = int(st.session_state.freq_input.replace('min', ''))
        new_period = int(period_length_input.replace('min', ''))

        if new_period < agg_period:
            st.warning(f"The data from Step 2 was downsampled from a {agg_period}-minute resolution to a {new_period}-minute resolution.\n\nRepresentative values for each finer-resolution period were interpolated.", icon="📌")
        elif new_period > agg_period:
            st.warning(f"The data from Step 2 was re-aggregated to a lower resolution (longer period) of {new_period} minutes.\n\nThe resulting values for min, max, and average reflect the values of the collected metrics, but p50, p95, and p99 are merely estimates.", icon="📌")

        if new_period < agg_period:
            st.session_state.summary_by_period_df = downsample(st.session_state.aggregated_df, new_period)
        else:
            st.session_state.summary_by_period_df = re_aggregate_data(st.session_state.aggregated_df, period_length_input)

def alarm_state_form() -> None:
    threshold_input = st.slider("Threshold (ms)", min_value=50, max_value=300, value=160, key='threshold_input', help="Specify the threshold value for evaluating the alarm state.")
    alarm_condition_input = st.selectbox(
        "Alarm Condition",
        ['>', '>=', '<', '<='],
        key='alarm_condition_input',
        help="Select the condition for evaluating the alarm state."
    )
    
    evaluation_range_input = st.number_input("Evaluation Range (# periods btw green bars)", min_value=1, value=5, key='evaluation_range_input', help="Specify the number of consecutive data points to evaluate for alarm state.")

def evaluate_breaching_data_points() -> None:
    if not st.session_state.summary_by_period_df.empty:
        st.session_state.alarm_state_df = evaluate_alarm_state(
            summary_df=st.session_state.summary_by_period_df,
            threshold=st.session_state.threshold_input,
            datapoints_to_alarm=st.session_state.datapoints_to_alarm_input,
            evaluation_range=st.session_state.evaluation_range_input,
            aggregation_function=st.session_state.aggregation_function_input__alarm,
            alarm_condition=st.session_state.alarm_condition_input
        )    

def display_dataframe(title: str, df: pd.DataFrame) -> None:
    st.write(title)
    st.dataframe(df)

def plot_time_series(df: pd.DataFrame, threshold: int, alarm_condition: str, evaluation_range: int) -> None:
    timestamps = df['timestamp']
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
    ax1.set_xlabel('timestamp')
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
            ax1.axvline(x=df['timestamp'].iloc[idx], color='green', linestyle='-', alpha=0.3)
            max_value = max(filter(lambda x: x is not None, df[st.session_state.aggregation_function_input__alarm]))
            ax1.text(df['timestamp'].iloc[idx], max_value * 0.95, f"[{idx // evaluation_range}]", rotation=90, verticalalignment='bottom', color='grey', alpha=0.7, fontsize=8)
        else:
            ax1.axvline(x=df['timestamp'].iloc[idx], color='grey', linestyle='--', alpha=0.3)

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
            "Breaching data point: This data point breaches the threshold and alarm condition (<, <=, >=, >)",
            "Missing data point: This data point is missing or not reported",
            "Non-breaching data point: This data point is does not breach the threshold and alarm condition (<, <=, >=, >)"
        ]
    }
    symbol_df = pd.DataFrame(symbol_data)
    st.table(symbol_df)

    # Columns
    st.write(dedent("""\
    #### Columns: [The 4 Strategies](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/AlarmThatSendsEmail.html#alarms-and-missing-data) for handling missing data points
             
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
