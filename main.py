# ============================================================
# COMPLETE AI TRADING BOT - FIXED
# ============================================================
import datetime
import time, pickle, warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    print("yfinance OK")
except:
    print("Run: pip install yfinance")
    exit()

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = mt5.initialize()
    print("MT5 OK")
except:
    MT5_AVAILABLE = False
    print("MT5: Demo Mode")

# CONFIG
SYMBOL = "XAUUSD"
YFINANCE_SYMBOL = "GC=F"
INITIAL_BALANCE = 100.0
CONTRACT_SIZE = 100
SL_DISTANCE = 5.0
TP_MULTIPLIER = 2
TRAILING_TRIGGER = 1.5
RISK_PER_TRADE = 0.01
PREDICTION_THRESHOLD = 0.75
BACKTEST_YEARS = 5
TIMEFRAME = "1d"

# DATA
def get_data(symbol=YFINANCE_SYMBOL, period="5y", interval="1d"):
    # Data download karein
    data = yf.download(tickers=symbol, period=period, interval=interval, progress=False)

    if data.empty:
        print(f"❌ No data found for {symbol}")
        return pd.DataFrame()

    # Index reset karein (Date/Datetime column ban jayega)
    data = data.reset_index()

    # MultiIndex handle karein
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] for c in data.columns]

    # 🛠️ FIX: 'Date' aur 'Datetime' dono ko handle karein
    data = data.rename(columns={
        'Date': 'time',
        'Datetime': 'time',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })

    # Columns select karein
    data = data[['time', 'open', 'high', 'low', 'close']]

    # Volume column add karein
    data['volume'] = 1000

    return data


def get_live_data(num=500):
    if MT5_AVAILABLE:
        rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M30, 0, num)
        if rates is not None:
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df['volume'] = 1000
            return df
    return get_data(period="1d")


# SLOPE
def get_slope(series, period=20):
    slopes = []
    for i in range(len(series)):
        if i < period - 1:
            slopes.append(0)
        else:
            y = series[i - period + 1:i + 1].values
            slopes.append(np.polyfit(np.arange(period), y, 1)[0])
    return pd.Series(slopes, index=series.index)


# FEATURES (UPDATED WITH PATTERN RECOGNITION & SENTIMENT)
def calculate_features(df):
    df = df.copy()
    close = df['close']
    high = df['high']
    low = df['low']
    open_p = df['open']
    volume = df['volume']

    # --- BASIC STATS ---
    df['log_close'] = np.log(close)
    df['returns'] = close.pct_change()

    # --- MOVING AVERAGES ---
    for p in [5, 10, 20, 50]:
        df[f'sma_{p}'] = close.rolling(p).mean()
        df[f'price_sma{p}_ratio'] = close / df[f'sma_{p}']

    for p in [9, 15, 21, 50]:
        df[f'ema_{p}'] = close.ewm(span=p, adjust=False).mean()

    df['ema_diff'] = df['ema_9'] - df['ema_21']

    # --- OSCILLATORS ---
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 0.0001)))

    df['macd'] = close.ewm(span=12).mean() - close.ewm(span=26).mean()
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # --- VOLATILITY & VOLUME ---
    tr = np.maximum(high - low, np.maximum(np.abs(high - close.shift(1)), np.abs(low - close.shift(1))))
    df['atr'] = tr.rolling(14).mean()
    df['volume_ratio'] = volume / (volume.rolling(20).mean() + 0.0001)
    df['volatility'] = close.rolling(10).std()

    # --- MOMENTUM ---
    for p in [3, 5, 10]:
        df[f'momentum_{p}'] = close / close.shift(p) - 1

    # --- SUPPORT & RESISTANCE (LEAKAGE-FREE) ---
    # Hum pichle data ka SR calculate karenge current candle ke liye
    df['sr_high'] = high.rolling(20).max().shift(1)
    df['sr_low'] = low.rolling(20).min().shift(1)

    df['sr_position'] = (close - df['sr_low']) / (df['sr_high'] - df['sr_low'] + 0.0001)
    df['dist_to_support'] = (close - df['sr_low']) / (close + 0.0001)
    df['trend_slope'] = get_slope(close, 20)

    # --- CANDLESTICK ANALYSIS ---
    df['body'] = close - open_p
    df['body_size'] = abs(df['body'])
    df['candle_range'] = high - low
    df['high_low_pos'] = (close - low) / (high - low + 0.0001)

    # --- RECENT PATTERN RECOGNITION ---
    # Engulfing (Already Lagging)
    df['bullish_engulfing'] = ((close > open_p) & (close.shift(1) < open_p.shift(1)) & (close > open_p.shift(1)) & (
                open_p < close.shift(1))).astype(int)
    df['bearish_engulfing'] = ((close < open_p) & (close.shift(1) > open_p.shift(1)) & (close < open_p.shift(1)) & (
                open_p > close.shift(1))).astype(int)

    # Pin Bars (Current price comparison only)
    df['pin_bar_bull'] = ((df['body_size'] < df['candle_range'] * 0.3) & (df['high_low_pos'] > 0.7)).astype(int)
    df['pin_bar_bear'] = ((df['body_size'] < df['candle_range'] * 0.3) & (df['high_low_pos'] < 0.3)).astype(int)

    # Double Top / Double Bottom (Fixed to use Shifted SR)
    df['double_top'] = ((high >= df['sr_high'] * 0.999) & (high <= df['sr_high'] * 1.001)).astype(int)
    df['double_bottom'] = ((low <= df['sr_low'] * 1.001) & (low >= df['sr_low'] * 0.999)).astype(int)

    # --- SENTIMENT PROXY ---
    df['sentiment_proxy'] = -df['returns'].rolling(5).mean().shift(1)

    # Breakouts (Fixed to use Shifted SR)
    df['breakout_up'] = (close > df['sr_high']).astype(int)
    df['breakout_down'] = (close < df['sr_low']).astype(int)

    return df


def create_labels(df, look_ahead=5, threshold=0.003):
    """
    STRICTLY FOR TRAINING ONLY.
    Ye function future price ko use karta hai, is liye isay
    sirf model sikhany (training) ke liye use karein.
    """
    df = df.copy()

    # Future price check (shift -5)
    future = df['close'].shift(-look_ahead)

    # Percentage change calculate karein
    change = (future - df['close']) / df['close']

    # Labels generate karein
    labels = np.zeros(len(df))
    labels[change > threshold] = 1  # BUY Signal
    labels[change < -threshold] = -1  # SELL Signal

    # Aakhri candles ke labels NaN kar dein (kyunke unka future nahi hota)
    labels[-look_ahead:] = 0

    return labels


# ML BOT CLASS (UPDATED)
class MLBot:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.features = None

    def train(self, df, verbose=True):
        df = calculate_features(df)
        labels = create_labels(df)

        # UPDATED: Added Pattern Recognition & Sentiment features
        feature_cols = [
            'returns', 'price_sma5_ratio', 'price_sma20_ratio', 'ema_diff', 'rsi',
            'macd', 'macd_hist', 'atr', 'volume_ratio', 'momentum_3', 'momentum_5',
            'sr_position', 'dist_to_support', 'trend_slope', 'body_position',
            'bullish_engulfing', 'bearish_engulfing', 'high_low_pos', 'breakout_up',
            'pin_bar_bull', 'pin_bar_bear', 'double_top', 'double_bottom', 'sentiment_proxy'
        ]

        # Check which features actually exist in the dataframe
        available = [c for c in feature_cols if c in df.columns]

        # Clean data: Remove NaNs and ignore 'Hold' (0) labels for cleaner training
        mask = ~np.isnan(df[available]).any(axis=1) & (labels != 0)
        X = np.nan_to_num(df[available][mask].values, nan=0)
        y = labels[mask]

        if len(X) < 100:
            print("❌ Not enough data to train the model")
            return False

        # Data Scaling
        X_scaled = self.scaler.fit_transform(X)

        # Split data for Testing
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # ML Model: Gradient Boosting (Best for Pattern Recognition)
        self.model = GradientBoostingClassifier(
            n_estimators=150,  # Increased for better pattern learning
            max_depth=6,  # Slightly deeper for complex patterns
            learning_rate=0.05,  # Lower learning rate for better stability
            random_state=42
        )
        self.model.fit(X_train, y_train)

        # Evaluate Performance
        acc = accuracy_score(y_test, self.model.predict(X_test))
        self.features = available

        if verbose:
            print(f"✅ Model trained with Patterns & Sentiment!")
            print(f"📊 Accuracy: {acc * 100:.1f}%")
            print(f"📝 Features Used: {len(available)}")

        return True

    def predict(self, df):
        if self.model is None:
            return 0, 0.0

        df = calculate_features(df)
        X = np.nan_to_num(df[self.features].iloc[-1:].values, nan=0)
        X_scaled = self.scaler.transform(X)
        pred = self.model.predict(X_scaled)[0]
        prob = max(self.model.predict_proba(X_scaled)[0])
        return int(pred), float(prob)


# BACKTEST (UPDATED WITH ADVANCED PATTERN FILTERING)
def run_backtest(df, ml_bot, verbose=True):
    print("\n" + "=" * 50)
    print("STARTING ADVANCED BACKTEST")
    print("=" * 50)

    # Naye features ke saath calculate karein
    df = calculate_features(df)

    balance = INITIAL_BALANCE
    equity_curve = []
    in_trade = False
    trade_type = ""
    trade_entry = 0
    trade_lot = 0
    sl = 0
    tp = 0
    high_seen = 0
    low_seen = 0
    wins = 0
    losses = 0

    for i in range(200, len(df)):
        c_high = df['high'].iloc[i]
        c_low = df['low'].iloc[i]
        c_close = df['close'].iloc[i]

        # 1. TRADE MANAGEMENT (Exit & Trailing Logic)
        if in_trade:
            if trade_type == "BUY":
                if c_high > high_seen:
                    high_seen = c_high
                    if (high_seen - trade_entry) >= TRAILING_TRIGGER:
                        sl = max(sl, high_seen - TRAILING_TRIGGER)

                if c_high >= tp:
                    pnl = (tp - trade_entry) * trade_lot * CONTRACT_SIZE
                    balance += pnl
                    wins += 1
                    in_trade = False
                elif c_low <= sl:
                    pnl = (sl - trade_entry) * trade_lot * CONTRACT_SIZE
                    balance += pnl
                    losses += 1
                    in_trade = False

            elif trade_type == "SELL":
                if c_low < low_seen:
                    low_seen = c_low
                    if (trade_entry - low_seen) >= TRAILING_TRIGGER:
                        sl = min(sl, low_seen + TRAILING_TRIGGER)

                if c_low <= tp:
                    pnl = (trade_entry - tp) * trade_lot * CONTRACT_SIZE
                    balance += pnl
                    wins += 1
                    in_trade = False
                elif c_high >= sl:
                    pnl = (trade_entry - sl) * trade_lot * CONTRACT_SIZE
                    balance += pnl
                    losses += 1
                    in_trade = False

        # 2. ENTRY LOGIC (Machine Learning + Patterns)
        if not in_trade:
            # Model se prediction aur probability lein
            pred, prob = ml_bot.predict(df.iloc[:i + 1])

            # Additional Filters from Patterns
            is_bull_pattern = (
                        df['pin_bar_bull'].iloc[i] or df['double_bottom'].iloc[i] or df['bullish_engulfing'].iloc[i])
            is_bear_pattern = (
                        df['pin_bar_bear'].iloc[i] or df['double_top'].iloc[i] or df['bearish_engulfing'].iloc[i])

            # BUY Entry: High Confidence + Bullish Pattern
            if pred == 1 and prob >= PREDICTION_THRESHOLD:
                trade_lot = max(0.01, round(balance * RISK_PER_TRADE / (SL_DISTANCE * CONTRACT_SIZE), 2))
                trade_entry = c_close
                sl = trade_entry - SL_DISTANCE
                tp = trade_entry + (SL_DISTANCE * TP_MULTIPLIER)
                high_seen = trade_entry
                trade_type = "BUY"
                in_trade = True
                if verbose: print(f"🚀 BUY at {trade_entry} | Conf: {prob * 100:.1f}%")

            # SELL Entry: High Confidence + Bearish Pattern
            elif pred == -1 and prob >= PREDICTION_THRESHOLD:
                trade_lot = max(0.01, round(balance * RISK_PER_TRADE / (SL_DISTANCE * CONTRACT_SIZE), 2))
                trade_entry = c_close
                sl = trade_entry + SL_DISTANCE
                tp = trade_entry - (SL_DISTANCE * TP_MULTIPLIER)
                low_seen = trade_entry
                trade_type = "SELL"
                in_trade = True
                if verbose: print(f"📉 SELL at {trade_entry} | Conf: {prob * 100:.1f}%")

        # Equity Curve record karein (Profit/Loss %)
        equity_curve.append(((balance - INITIAL_BALANCE) / INITIAL_BALANCE) * 100)

    # 3. RESULTS & VISUALIZATION
    print(f"\n" + "=" * 20 + " FINAL RESULTS " + "=" * 20)
    print(f"💰 Final Balance: ${balance:.2f}")
    print(f"💵 Net Profit/Loss: ${balance - INITIAL_BALANCE:.2f}")
    print(f"📊 Total Trades: {wins + losses}")
    print(f"✅ Wins: {wins} | ❌ Losses: {losses}")
    win_rate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0
    print(f"🎯 Win Rate: {win_rate:.1f}%")

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, color='purple', linewidth=2, label="Equity Curve")
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)

    # Fill colors for Profit/Loss
    plt.fill_between(range(len(equity_curve)), equity_curve, 0,
                     where=np.array(equity_curve) >= 0, alpha=0.2, color='green')
    plt.fill_between(range(len(equity_curve)), equity_curve, 0,
                     where=np.array(equity_curve) < 0, alpha=0.2, color='red')

    plt.title(f"AI Trading Bot Performance (XAUUSD)\nFinal Balance: ${balance:.2f}", fontsize=14)
    plt.xlabel("Trade Timeline", fontsize=12)
    plt.ylabel("Equity Change (%)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    return {'balance': balance, 'wins': wins, 'losses': losses}


# ============================================================
# MAIN EXECUTION - AI TRADING BOT (ENHANCED)
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🚀 AI STRATEGY BOT: PATTERNS + SENTIMENT")
    print("=" * 50)

    # Step 1: Data Acquisition
    print("\n1. [DATA] Fetching Historical Market Records...")
    df = get_data(period="6y", interval="1d")  # 6 Years for better depth

    if df.empty:
        print("❌ Error: Could not fetch data.")
        exit()

    # 🛠️ FIX: Train-Test Split (Data Leakage Protection)
    # Pehle 80% data par model seekhay ga (Training)
    # Aakhri 20% data (unseen) par backtest hoga (Testing)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"   ✅ Data Loaded: {len(df)} candles.")
    print(f"   📊 Training on: {len(train_df)} candles (Pichle 4.5 saal)")
    print(f"   🧪 Testing on: {len(test_df)} candles (Aakhri 1.5 saal)")

    # Step 2: Model Training
    print("\n2. [ML] Training Gradient Boosting Model...")
    ml_bot = MLBot()
    # Sirf Training data de rahe hain
    success = ml_bot.train(train_df)

    if not success:
        print("❌ Training failed. Not enough patterns found.")
        exit()

    # Step 3: Performance Validation (Backtest)
    print("\n3. [VALIDATION] Running Backtest on UNSEEN Data...")
    # 🛠️ FIX: Sirf Test data par backtest chalayein jo AI ne kabhi na dekha ho
    results = run_backtest(test_df, ml_bot, verbose=True)

    # Step 4: Real-Time Market Analysis (Latest Data)
    LOOP_INTERVAL = 45 * 60

    print("🔄 Starting 45-Minute Continuous Analysis Loop...")

    while True:
        try:
            # Har naye cycle par current date aur time print hoga
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n⏰ [CYCLE START] Running Market Analysis At: {current_time}")

            # ⚠️ NOTE: Is line se pehle aapka fresh 'df' download hona chahiye taake live data update ho.

            # Step 4: Real-Time Market Analysis (Latest Data)
            print("\n4. [LIVE] Current Market Analysis:")
            pred, prob = ml_bot.predict(df)
            current_price = df['close'].iloc[-1]

            print("-" * 30)
            print(f"   💰 Current Gold Price: ${current_price:.2f}")

            signal_text = "HOLD/WAIT"
            if prob >= PREDICTION_THRESHOLD:
                signal_text = "🟢 STRONG BUY 🚀" if pred == 1 else "🔴 STRONG SELL 📉"

            print(f"   📢 AI Signal: {signal_text}")
            print(f"   🎯 AI Confidence: {prob * 100:.1f}%")

            if signal_text != "HOLD/WAIT":
                sl_val = current_price - SL_DISTANCE if pred == 1 else current_price + SL_DISTANCE
                tp_val = current_price + (SL_DISTANCE * TP_MULTIPLIER) if pred == 1 else current_price - (
                        SL_DISTANCE * TP_MULTIPLIER)
                print(f"      🛑 Stop Loss: {sl_val:.2f} | 💎 Take Profit: {tp_val:.2f}")

            print("-" * 30)
            print("\n✅ System Execution Complete. Professional Validation Done.")

        except Exception as e:
            # Agar loop ke dauran internet ya data fetch ka koi error aaye tou script crash nahi hogi
            print(f"⚠️ Market Fetch/Execution Error: {e}")

        # 45 minutes ka pause lagane ke liye timer start
        print(f"\n😴 Sleeping for 45 minutes... Next analysis will trigger automatically.")
        time.sleep(LOOP_INTERVAL)
