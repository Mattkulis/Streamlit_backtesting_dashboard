import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import numpy as np

st.set_page_config(layout="wide", page_title="Trading Strategy Backtest Dashboard")
st.title("Trading Strategy Backtest Dashboard")

# File path
file_path = r"C:\Users\User\Desktop\pyton\MSTR_2019_to_Present_(10-24-2024).xlsx"

def load_and_prepare_data(file_path):
    try:
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            return pd.DataFrame()
        
        data = pd.read_excel(file_path)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['9ema'] = data['close'].ewm(span=9, adjust=False).mean()
        data['is_red'] = data['open'] > data['close']
        # Calculate average price for execution
        data['execution_price'] = (data['open'] + data['high'] + data['low'] + data['close']) / 4
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

def calculate_ratios(returns_series, risk_free_rate=0.02):
    """Calculate Sortino and Calmar ratios"""
    excess_returns = returns_series - (risk_free_rate / 252)  # Daily risk-free rate
    
    # Sortino Ratio
    negative_returns = returns_series[returns_series < 0]
    downside_std = np.sqrt(np.mean(negative_returns**2))
    sortino_ratio = (np.mean(excess_returns) * 252) / (downside_std * np.sqrt(252)) if downside_std != 0 else 0
    
    # Calmar Ratio
    max_drawdown = calculate_max_drawdown(returns_series)
    calmar_ratio = (np.mean(returns_series) * 252) / abs(max_drawdown) if max_drawdown != 0 else 0
    
    return sortino_ratio, calmar_ratio

def calculate_max_drawdown(returns_series):
    """Calculate maximum drawdown"""
    cum_returns = (1 + returns_series).cumprod()
    rolling_max = cum_returns.expanding(min_periods=1).max()
    drawdowns = cum_returns / rolling_max - 1
    return drawdowns.min()

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

def backtest_strategy(df):
    positions = []
    current_shares = 0
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
            
        # Position sizing rules
        if current_shares > 0:
            # Exit conditions
            if consecutive_red >= 3 or (consecutive_red >= 2 and current_candle['close'] < current_candle['9ema']):
                current_shares = 0
            # Add to position if momentum continues
            elif not current_candle['is_red'] and current_shares < 300 and current_candle['close'] > current_candle['9ema']:
                current_shares += 100
        
        # Entry conditions
        elif current_shares == 0:
            if (not current_candle['is_red'] and
                check_prior_6_opens(df, i) and
                current_candle['close'] > current_candle['9ema'] and
                check_volume_condition(df, i)):
                current_shares = 100
        
        positions.append(current_shares)
    
    df['position'] = positions
    return df

def calculate_trade_metrics(results):
    """Calculate detailed trade metrics"""
    trade_changes = results[results['position'] != results['position'].shift(1)].copy()
    trade_changes['trade_type'] = np.where(trade_changes['position'] > trade_changes['position'].shift(1), 'entry', 'exit')
    
    trades = []
    current_entry = None
    current_shares = 0
    
    for idx, row in trade_changes.iterrows():
        if row['trade_type'] == 'entry':
            current_entry = row
            current_shares = row['position']
        elif row['trade_type'] == 'exit' and current_entry is not None:
            pnl = (row['execution_price'] - current_entry['execution_price']) * current_shares
            hold_time = (row['timestamp'] - current_entry['timestamp']).total_seconds() / 3600  # in hours
            trades.append({
                'entry_time': current_entry['timestamp'],
                'exit_time': row['timestamp'],
                'hold_time': hold_time,
                'pnl': pnl,
                'shares': current_shares
            })
    
    if not trades:
        return pd.DataFrame()
    
    trades_df = pd.DataFrame(trades)
    return trades_df

# Main dashboard execution
def main():
    start_date, end_date = get_date_range()
    
    # Load initial data
    data = load_and_prepare_data(file_path)
    
    if not data.empty:
        # Filter data based on selected date range
        if start_date and end_date:
            data = data[(data['timestamp'] >= start_date) & 
                       (data['timestamp'] <= end_date)]
        
        # Run backtest
        results = backtest_strategy(data)
        results['returns'] = (results['execution_price'].pct_change() * 
                            results['position'].shift(1) / 100)  # Divide by 100 to account for shares
        results['cumulative_returns'] = (1 + results['returns']).cumprod()
        
        trades_df = calculate_trade_metrics(results)
        
        if not trades_df.empty:
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]
            
            sortino_ratio, calmar_ratio = calculate_ratios(results['returns'])
            
            st.subheader("Detailed Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", f"${trades_df['pnl'].sum():,.2f}")
                st.metric("Average Trade P&L", f"${trades_df['pnl'].mean():,.2f}")
            with col2:
                st.metric("Largest Win", f"${winning_trades['pnl'].max():,.2f}" if not winning_trades.empty else "$0")
                st.metric("Largest Loss", f"${losing_trades['pnl'].min():,.2f}" if not losing_trades.empty else "$0")
            with col3:
                st.metric("Win Rate", f"{(len(winning_trades)/len(trades_df)*100):.1f}%")
                st.metric("Average Win Hold Time", f"{winning_trades['hold_time'].mean():.1f}h" if not winning_trades.empty else "0h")
            with col4:
                st.metric("Sortino Ratio", f"{sortino_ratio:.2f}")
                st.metric("Calmar Ratio", f"{calmar_ratio:.2f}")
            
            # Performance visualizations
            st.subheader("Performance Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                # Cumulative returns chart
                fig_cumulative = px.line(results, x='timestamp', y='cumulative_returns',
                                       title="Cumulative Returns")
                st.plotly_chart(fig_cumulative, use_container_width=True)
            
            with col2:
                # Win/Loss distribution
                win_loss_data = pd.DataFrame({
                    'Category': ['Wins', 'Losses'],
                    'Count': [len(winning_trades), len(losing_trades)]
                })
                fig_pie = px.pie(win_loss_data, values='Count', names='Category',
                                title='Win/Loss Distribution',
                                color='Category', 
                                color_discrete_map={'Wins': 'green', 'Losses': 'red'})
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Display detailed trade history
            st.subheader("Trade History")
            trade_history = trades_df.copy()
            trade_history['entry_time'] = trade_history['entry_time'].dt.strftime('%Y-%m-%d %H:%M')
            trade_history['exit_time'] = trade_history['exit_time'].dt.strftime('%Y-%m-%d %H:%M')
            trade_history['hold_time'] = trade_history['hold_time'].round(1)
            trade_history['pnl'] = trade_history['pnl'].round(2)
            st.dataframe(trade_history.sort_values('entry_time', ascending=False))
    else:
        st.warning("Unable to load data. Please check the file path and ensure the Excel file is accessible.")

if __name__ == "__main__":
    main()