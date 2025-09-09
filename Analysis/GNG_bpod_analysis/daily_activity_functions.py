import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import ast


def daily_activity_single_animal(project_data, index):
    """
    Plot daily activity for a single animal showing trial counts over time of day (15-min bins).
    """
    if project_data is None or project_data.empty:
        st.info("No data loaded.")
        return
    
    if "StartTime" not in project_data.columns:
        st.info("No StartTime data available for activity analysis.")
        return
    
    # Get the specific session data
    session_data = project_data.iloc[index]
    start_times = session_data["StartTime"]
    
    if pd.isna(start_times) or not start_times:
        st.info("No start time data available for this session.")
        return
    
    # Parse start times from string format
    try:
        if isinstance(start_times, str):
            start_times_list = ast.literal_eval(start_times)
        else:
            start_times_list = start_times
        
        # Convert to datetime objects
        times = []
        for time_str in start_times_list:
            try:
                # Parse time string like '11:49:57.236554'
                time_obj = pd.to_datetime(time_str, format='%H:%M:%S.%f').time()
                times.append(time_obj)
            except:
                try:
                    # Fallback for different format
                    time_obj = pd.to_datetime(time_str).time()
                    times.append(time_obj)
                except:
                    continue
        
        if not times:
            st.info("Could not parse start times.")
            return
            
    except Exception as e:
        st.error(f"Error parsing start times: {e}")
        return
    
    # Create 15-minute bins throughout the day
    bin_size_minutes = 30
    bins = []
    bin_labels = []
    
    for hour in range(24):
        for minute in range(0, 60, bin_size_minutes):
            start_time = pd.Timestamp.combine(pd.Timestamp.today().date(), 
                                            pd.Timestamp(f"{hour:02d}:{minute:02d}:00").time())
            end_time = start_time + pd.Timedelta(minutes=bin_size_minutes)
            bins.append((start_time.time(), end_time.time()))
            bin_labels.append(f"{hour:02d}:{minute:02d}")
    
    # Count trials in each bin
    bin_counts = [0] * len(bins)
    
    for time_obj in times:
        for i, (bin_start, bin_end) in enumerate(bins):
            if bin_start <= time_obj < bin_end:
                bin_counts[i] += 1
                break
    
    # Create the plot
    fig = go.Figure()
    
    # Convert bin labels to datetime for proper x-axis
    x_values = [pd.Timestamp(f"2024-01-01 {label}:00") for label in bin_labels]
    
    fig.add_trace(go.Scatter(
        x=x_values,
        y=bin_counts,
        mode='lines+markers',
        name="Trial Count",
        line=dict(color='steelblue', width=2),
        marker=dict(size=4, color='steelblue')
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Daily Activity Pattern - {session_data['MouseName']} ({session_data['SessionDate']})",
        xaxis_title="Time of Day",
        yaxis_title="Number of Trials (15-min bins)",
        xaxis=dict(
            tickformat="%H:%M",
            tickmode='array',
            tickvals=x_values[::4],  # Show every 4th tick (hourly)
            ticktext=[label for i, label in enumerate(bin_labels) if i % 4 == 0]
        ),
        height=500,
        width=900,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def daily_activity_multi_animal(project_data):
    """
    Plot daily activity for multiple animals on a selected date, showing trial counts over time of day.
    """
    if project_data is None or project_data.empty:
        st.info("No data loaded.")
        return
    
    if "StartTime" not in project_data.columns:
        st.info("No StartTime data available for activity analysis.")
        return
    
    # Get unique dates
    dates = sorted(project_data["SessionDate"].astype(str).unique())
    if len(dates) == 0:
        st.info("No dates found in data.")
        return
    
    selected_date = st.selectbox("Select a date", options=dates, 
                                index=max(0, len(dates) - 1), 
                                key="daily_activity_date")
    
    # Filter data for selected date
    date_data = project_data[project_data["SessionDate"].astype(str) == str(selected_date)]
    
    if date_data.empty:
        st.info(f"No data found for date {selected_date}")
        return
    
    # Get unique mice for this date
    mice = sorted(date_data["MouseName"].unique())
    if len(mice) == 0:
        st.info("No animals found for selected date.")
        return
    
    # Create 15-minute bins throughout the day
    bin_size_minutes = 15
    bins = []
    bin_labels = []
    
    for hour in range(24):
        for minute in range(0, 60, bin_size_minutes):
            start_time = pd.Timestamp.combine(pd.Timestamp.today().date(), 
                                            pd.Timestamp(f"{hour:02d}:{minute:02d}:00").time())
            end_time = start_time + pd.Timedelta(minutes=bin_size_minutes)
            bins.append((start_time.time(), end_time.time()))
            bin_labels.append(f"{hour:02d}:{minute:02d}")
    
    fig = go.Figure()
    
    # Convert bin labels to datetime for proper x-axis
    x_values = [pd.Timestamp(f"2024-01-01 {label}:00") for label in bin_labels]
    
    colors = ['steelblue', 'darkorange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
    
    for i, mouse in enumerate(mice):
        mouse_data = date_data[date_data["MouseName"] == mouse]
        if len(mouse_data) == 0:
            continue
            
        # Get start times for this mouse
        start_times = mouse_data.iloc[0]["StartTime"]
        
        if pd.isna(start_times) or not start_times:
            continue
        
        # Parse start times
        try:
            if isinstance(start_times, str):
                start_times_list = ast.literal_eval(start_times)
            else:
                start_times_list = start_times
            
            # Convert to datetime objects
            times = []
            for time_str in start_times_list:
                try:
                    time_obj = pd.to_datetime(time_str, format='%H:%M:%S.%f').time()
                    times.append(time_obj)
                except:
                    try:
                        time_obj = pd.to_datetime(time_str).time()
                        times.append(time_obj)
                    except:
                        continue
            
            if not times:
                continue
                
        except Exception:
            continue
        
        # Count trials in each bin
        bin_counts = [0] * len(bins)
        
        for time_obj in times:
            for j, (bin_start, bin_end) in enumerate(bins):
                if bin_start <= time_obj < bin_end:
                    bin_counts[j] += 1
                    break
        
        # Add trace for this mouse
        fig.add_trace(go.Scatter(
            x=x_values,
            y=bin_counts,
            mode='lines+markers',
            name=str(mouse),
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=4, color=colors[i % len(colors)])
        ))
    
    if len(fig.data) == 0:
        st.info("No activity data found for any animals on selected date.")
        return
    
    # Update layout
    fig.update_layout(
        title=f"Daily Activity Pattern by Animal — {selected_date} (15-min bins)",
        xaxis_title="Time of Day",
        yaxis_title="Number of Trials",
        xaxis=dict(
            tickformat="%H:%M",
            tickmode='array',
            tickvals=x_values[::4],  # Show every 4th tick (hourly)
            ticktext=[label for i, label in enumerate(bin_labels) if i % 4 == 0]
        ),
        height=500,
        width=900,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
