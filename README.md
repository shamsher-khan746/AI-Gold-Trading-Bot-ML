# 🤖 AI Gold Trading Bot (XAU/USD) with Machine Learning

A continuous, automated algorithmic trading system built using Python and Machine Learning. The bot operates on a strict 45-minute candle-close loop to analyze live market structures, manage risk, and execute trades without human emotional bias.

## 📊 Backtesting Performance Metrics (1.5 Years)
* **💰 Starting Balance:** $100.00
* **💰 Final Balance:** $438.30
* **💵 Net Profit/Loss:** +$338.30 (Over 338% ROI)
* **📊 Total Trades Executed:** 81
* **✅ Wins:** 46 
* **❌ Losses:** 35
* **🎯 Win Rate:** 56.8%

## 🛠️ Core Features
* **Real-Time Prediction:** Triggers market execution only when model prediction crosses a strict confidence threshold.
* **Risk Management:** Calculates dynamic Stop Loss (SL) and Take Profit (TP) levels per cycle based on current market price.
* **Continuous Monitoring:** Uses an automated loop (`time.sleep`) to continuously evaluate setups every 45 minutes.

*Disclaimer: Developed for educational and data science portfolio benchmarking purposes.*
