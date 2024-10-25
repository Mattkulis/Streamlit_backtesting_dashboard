import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import numpy as np

# Streamlit app
st.set_page_config(layout="wide", page_title="Trading Strategy Backtest Dashboard")

st.title("Trading Strategy Backtest Dashboard")

# File path
file_path = r"C:\Users\User\Desktop\pyton\MSTR_2019_to_Present_(10-24-2024).xlsx"

def load_and_prepare_data(file_path):
    """
    Load data from Excel and prepare it for analysis
    """
    try:
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            return pd.DataFrame()
        
        data = pd.read_excel(file_path)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['9ema'] = data['close'].ewm(span=9, adjust=False).mean()
        data['is_red'] = data['open'] > data['close']
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Time period selection
def get_date_range():
    today = datetime.now()
    periods = {
        "Day": today - timedelta(days=1),
        "Week": today - timedelta(weeks=1),
        "Month": today - timedelta(days=30),
        "Year to Date": datetime(today.year, 1, 1),
        "Year (Trailing 12 Months)": today - timedelta(days=365),
        "2 Years": today - timedelta(days=730),
        "3 Years": today - timedelta(days=1095),
        "All Available Data": None,
        "Custom": "custom"
    }
    
    selected_period = st.sidebar.selectbox("Select Time Period", list(periods.keys()))
    
    if selected_period == "Custom":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start Date", today - timedelta(days=30))
        with col2:
            end_date = st.date_input("End Date", today)
        # Convert dates to datetime
        start_date = datetime.combine(start_date, datetime.min.time())
        end_date = datetime.combine(end_date, datetime.max.time())
    elif selected_period == "All Available Data":
        return None, None
    else:
        end_date = datetime.combine(today.date(), datetime.max.time())
        start_date = datetime.combine(periods[selected_period].date(), datetime.min.time())
    
    return start_date, end_date

# Load and filter data
@st.cache_data
def process_data(file_path, start_date=None, end_date=None):
    data = load_and_prepare_data(file_path)
    if data.empty:
        return pd.DataFrame()
    
    if start_date and end_date:
        data = data[(data['timestamp'] >= start_date) & 
                    (data['timestamp'] <= end_date)]
    
    return data

# Backtest strategy (using the previous backtest logic)
def backtest_strategy(df):
    """
    Run the backtest on the prepared data
    """
    positions = []
    current_position = 0
    consecutive_red = 0
    
    for i in range(len(df)):
        if i < 6:
            positions.append(0)
            continue
            
        current_candle = df.iloc[i]
        
        # Update consecutive red candles count
        if current_candle['is_red']:
            consecutive_red += 1
        else:
            consecutive_red = 0
            
        # Exit conditions
        if current_position == 1:
            if consecutive_red >= 3:
                current_position = 0
            elif consecutive_red >= 2 and current_candle['close'] < current_candle['9ema']:
                current_position = 0
        
        # Entry conditions
        elif current_position == 0:
            if (not current_candle['is_red'] and
                check_prior_6_opens(df, i) and
                current_candle['close'] > current_candle['9ema'] and
                check_volume_condition(df, i)):
                current_position = 1
        
        positions.append(current_position)
    
    df['position'] = positions
    return df

# Helper functions (from previous code)
def check_volume_condition(df, current_idx):
    if current_idx < 6:
        return False
    current_volume = df.iloc[current_idx]['volume']
    previous_6_candles = df.iloc[current_idx-6:current_idx]
    red_candles_volume = previous_6_candles[previous_6_candles['is_red']]['volume']
    return current_volume > red_candles_volume.max() if len(red_candles_volume) > 0 else True

def check_prior_6_opens(df, current_idx):
    if current_idx < 6:
        return False
    current_close = df.iloc[current_idx]['close']
    previous_6_opens = df.iloc[current_idx-6:current_idx]['open']
    return all(current_close > prev_open for prev_open in previous_6_opens)

# Main dashboard
start_date, end_date = get_date_range()
data = process_data(file_path, start_date, end_date)

if not data.empty:
    # Run backtest
    results = backtest_strategy(data)
    results['returns'] = results['close'].pct_change()
    results['strategy_returns'] = results['returns'] * results['position'].shift(1)
    results['cumulative_returns'] = (1 + results['strategy_returns']).cumprod()
    
    # Calculate metrics
    total_trades = len(results[results['position'] != results['position'].shift(1)]) - 1
    winning_trades = len(results[(results['strategy_returns'] > 0) & (results['position'].shift(1) == 1)])
    losing_trades = total_trades - winning_trades
    total_return = results['cumulative_returns'].iloc[-1] - 1
    
    # Display metrics
    st.subheader("Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Return", f"{total_return:.2%}")
    with col2:
        st.metric("Total Trades", total_trades)
    with col3:
        st.metric("Win Rate", f"{(winning_trades/total_trades*100):.1f}%" if total_trades > 0 else "0%")
    with col4:
        st.metric("Avg Return per Trade", f"{(total_return/total_trades*100):.2f}%" if total_trades > 0 else "0%")
    
    # Performance by day of week and hour charts
    st.subheader("Performance Analysis")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Day of week performance
        results['day_of_week'] = results['timestamp'].dt.day_name()
        day_performance = results.groupby('day_of_week')['strategy_returns'].sum().reset_index()
        fig_dow = px.bar(day_performance, x='strategy_returns', y='day_of_week', 
                        orientation='h', title="Performance by Day of Week",
                        color='strategy_returns', color_continuous_scale='RdYlGn')
        st.plotly_chart(fig_dow, use_container_width=True)
    
    with col2:
        # Hour performance
        results['hour'] = results['timestamp'].dt.hour
        hour_performance = results.groupby('hour')['strategy_returns'].sum().reset_index()
        fig_hour = px.bar(hour_performance, x='strategy_returns', y='hour',
                         orientation='h', title="Performance by Hour",
                         color='strategy_returns', color_continuous_scale='RdYlGn')
        st.plotly_chart(fig_hour, use_container_width=True)
    
    with col3:
        # Win/Loss distribution
        win_loss_data = pd.DataFrame({
            'Category': ['Wins', 'Losses'],
            'Count': [winning_trades, losing_trades]
        })
        fig_pie = px.pie(win_loss_data, values='Count', names='Category',
                        title='Win/Loss Distribution',
                        color='Category', color_discrete_map={'Wins': 'green', 'Losses': 'red'})
        fig_pie.update_traces(textposition='inside', textinfo='percent+label', textfont_size=13)
        fig_pie.update_layout(width=400, height=320)  # 20% smaller
        st.plotly_chart(fig_pie)
    
    with col4:
        # Cumulative returns
        fig_cumulative = px.line(results, x='timestamp', y='cumulative_returns',
                                title="Cumulative Returns")
        st.plotly_chart(fig_cumulative, use_container_width=True)

else:
    st.warning("Unable to load data. Please check the file path and ensure the Excel file is accessible.")