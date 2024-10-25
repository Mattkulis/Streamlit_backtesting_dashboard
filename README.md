ensure dependancies and python env downloaded & run v2. V2is buggy but has some cool infographics i intend to include in the succeding versions.

![image](https://github.com/user-attachments/assets/9a9e7c33-afba-4a77-a592-f04c8b853838)

![image](https://github.com/user-attachments/assets/d92c13f5-9934-4f30-b62a-955eee7ea127)

![image](https://github.com/user-attachments/assets/bcc94b4f-c3fc-48ef-a77b-3d452129a6da)


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
