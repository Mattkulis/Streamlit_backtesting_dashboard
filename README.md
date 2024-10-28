Ensure dependancies and python env downloaded & run v6.7 
#The images are ytd, trailing 12mo, & 2year lookback periods for reference of strategy performance.

![image](https://github.com/user-attachments/assets/9c6d2b91-eace-4b32-89ff-8d09345b6cd8)



Entry Criteria:

Current candle must be green (close > open)
Close must be greater than 9 EMA
Current candle's close must be higher than the opens of the previous 6 candles
Current volume must be greater than the maximum volume of any red candles in the previous 6 candles

Exit Criteria:

Either 3 consecutive red candles
OR 2 consecutive red candles AND close below 9 EMA

Let me implement your requested changes:
Enhanced Trading Strategy Backtest DashboardClick to open code
Key changes and improvements made:

Fixed execution price calculation of execuion using the average: (open + high + low + close) / 4
Implemented position sizing:

Initial position is 100 shares
Can add 100 shares up to 300 total if momentum continues
Exits entire position when exit conditions are met
