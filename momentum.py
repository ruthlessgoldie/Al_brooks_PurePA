# Import required libraries
import pandas as pd  # Data manipulation and analysis library
import numpy as np  # Library for efficient numerical computation
from binance.client import Client  # Binance API client library
from concurrent.futures import ThreadPoolExecutor  # Library for parallel processing

# Define a class to analyze momentum signals
class EnhancedMomentumAnalyzer:
    def __init__(self, data, batch_size=1000):
        """
        Initialize the analyzer with input data and batch size.
        
        Parameters:
        data (pandas.DataFrame): Input data for analysis
        batch_size (int): Batch size for parallel processing
        """
        self.data = data.copy()  # Create a copy of the input data
        self.batch_size = batch_size  # Set the batch size
        self.signals = pd.DataFrame(index=self.data.index)  # Initialize signal columns

        # Initialize signal columns
        for col in ['large_candle', 'micro_gap_up', 'micro_gap_down', 'clean_candle', 'reversal_candle', 'two_legged_pullback']:
            self.signals[col] = False  # Set initial values to False

    def _calculate_common_metrics(self):
        """
        Calculate commonly used metrics for momentum analysis.
        """
        # Calculate trend indicators
        self.data['is_green'] = (self.data['Close'] > self.data['Open']).astype(bool)  # Green candle metric
        self.data['is_red'] = (self.data['Close'] < self.data['Open']).astype(bool)  # Red candle metric
        self.data['body_size'] = np.abs(self.data['Close'] - self.data['Open'])  # Body size metric
        self.data['upper_wick'] = self.data['High'] - self.data[['Open', 'Close']].max(axis=1)  # Upper wick metric
        self.data['lower_wick'] = self.data[['Open', 'Close']].min(axis=1) - self.data['Low']  # Lower wick metric

        # Calculate trend indicators
        self.data['sma_20'] = self.data['Close'].rolling(window=20).mean()  # Simple moving average (SMA) for 20 periods
        self.data['sma_50'] = self.data['Close'].rolling(window=50).mean()  # SMA for 50 periods
        self.data['trend'] = (self.data['Close'] > self.data['sma_20']) & (self.data['sma_20'] > self.data['sma_50'])  # Trend indicator
        self.data['downtrend'] = (self.data['Close'] < self.data['sma_20']) & (self.data['sma_20'] < self.data['sma_50'])  # Downtrend indicator

        # Calculate micro_gap metric
        self.data['micro_gap'] = self.data['Close'] - self.data['Open']

    def _calculate_batch_signals(self, start_idx, end_idx):
        """
        Calculate signals for a batch of data.
        
        Parameters:
        start_idx (int): Start index of the batch
        end_idx (int): End index of the batch
        
        Returns:
        pd.DataFrame: Signal DataFrame for the batch
        """
        # Get the batch data
        batch_data = self.data.iloc[start_idx:end_idx].copy()  # Create a copy of the batch data
        
        # Initialize signal DataFrame
        signals = pd.DataFrame(index=batch_data.index)  # Create a new signal DataFrame with the same index as the batch data

        # Two-legged pullback signal
        pullback_length = 2  # Pullback length for two-legged pullback
        is_green_series = batch_data['is_green']  # Green candle series
        is_red_series = batch_data['is_red']  # Red candle series

        # Bullish two-legged pullback
        uptrend = batch_data['trend']  # Trend indicator
        consecutive_greens = is_green_series.rolling(window=pullback_length + 1).sum()  # Consecutive green candles
        consecutive_reds = is_red_series.rolling(window=pullback_length).sum()  # Consecutive red candles

        signals['two_legged_pullback_bull'] = (
            uptrend & 
            (consecutive_greens.shift(pullback_length) >= pullback_length) & 
            (consecutive_reds == pullback_length) & 
            is_green_series.shift(-1)
        )  # Bullish two-legged pullback signal

        # Bearish two-legged pullback
        downtrend = batch_data['downtrend']  # Downtrend indicator
        consecutive_reds_bear = is_red_series.rolling(window=pullback_length + 1).sum()  # Consecutive red candles
        consecutive_greens_bear = is_green_series.rolling(window=pullback_length).sum()  # Consecutive green candles

        signals['two_legged_pullback_bear'] = (
            downtrend & 
            (consecutive_reds_bear.shift(pullback_length) >= pullback_length) & 
            (consecutive_greens_bear == pullback_length) & 
            is_red_series.shift(-1)
        )  # Bearish two-legged pullback signal

        # Combine bullish and bearish signals
        signals['two_legged_pullback'] = signals['two_legged_pullback_bull'] | signals['two_legged_pullback_bear']

        # Large candle signal
        signals['large_candle'] = (batch_data['High'] - batch_data['Low']) > 0.1  # Large candle metric

        # Micro gap up signal
        signals['micro_gap_up'] = (batch_data['Close'] - batch_data['Open']) > 0.01  # Micro gap up metric

        # Micro gap down signal
        signals['micro_gap_down'] = (batch_data['Open'] - batch_data['Close']) > 0.01  # Micro gap down metric

        # Clean candle signal
        signals['clean_candle'] = (batch_data['High'] - batch_data['Low']) > 0.05  # Clean candle metric

        # Reversal candle signal
        signals['reversal_candle'] = (batch_data['Close'] - batch_data['Open']) > 0.05  # Reversal candle metric

        return signals.astype(bool)  # Return the signal DataFrame as a boolean Series

    def detect_signals(self):
        """
        Detect momentum signals for the entire dataset.
        """
        with ThreadPoolExecutor() as executor:
            futures = []

            for start_idx in range(0, len(self.data), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(self.data))
                futures.append(
                    executor.submit(self._calculate_batch_signals, start_idx, end_idx)
                )

            for future in futures:
                batch_signals = future.result()
                self.signals.update(batch_signals)

    def calculate_signal_performance(self, forward_periods=[3, 5, 10]):
        """
        Calculate the performance of momentum signals.
        
        Parameters:
        forward_periods (list): List of periods to calculate forward returns
        Returns:
        dict: Dictionary of signal performance scores for each period
        """
        performance_by_period = {}

        for period in forward_periods:
            self.signals[f'forward_return_{period}'] = (
                self.data['Close'].shift(-period) - self.data['Close']
            ) / self.data['Close'] * 100

            period_scores = {}

            for signal_type in ['large_candle', 'micro_gap_up', 'micro_gap_down', 'clean_candle', 'reversal_candle', 'two_legged_pullback']:
                mask = self.signals[signal_type]
                if mask.any():
                    success_rate = (self.signals[mask][f'forward_return_{period}'] > 0).mean() * 100
                    period_scores[signal_type] = success_rate
                else:
                    period_scores[signal_type] = 0.0

            performance_by_period[period] = period_scores

        return performance_by_period

    def analyze(self):
        """
        Analyze the momentum signals for the entire dataset.
        """
        self._calculate_common_metrics()
        self.detect_signals()
        performance_scores = self.calculate_signal_performance()
        return performance_scores

def get_data_from_binance(symbol, interval, lookback, batch_size=3000):
    """
    Get data from Binance API.
    
    Parameters:
    symbol (str): Symbol to retrieve data for
    interval (str): Interval to retrieve data in
    lookback (int): Lookback period for the data
    batch_size (int): Batch size for parallel processing
    
    Returns:
    pandas.DataFrame: Dataframe containing the retrieved data
    """
    try:
        client = Client("", "")
        interval_map = {
            '1m': Client.KLINE_INTERVAL_1MINUTE,
            '3m': Client.KLINE_INTERVAL_3MINUTE,
            '5m': Client.KLINE_INTERVAL_5MINUTE,
            '15m': Client.KLINE_INTERVAL_15MINUTE,
            '30m': Client.KLINE_INTERVAL_30MINUTE,
            '1h': Client.KLINE_INTERVAL_1HOUR,
            '2h': Client.KLINE_INTERVAL_2HOUR,
            '4h': Client.KLINE_INTERVAL_4HOUR,
            '6h': Client.KLINE_INTERVAL_6HOUR,
            '8h': Client.KLINE_INTERVAL_8HOUR,
            '12h': Client.KLINE_INTERVAL_12HOUR,
            '1d': Client.KLINE_INTERVAL_1DAY,
        }

        all_klines = []
        remaining = lookback

        while remaining > 0:
            current_batch = min(batch_size, remaining)
            batch_klines = client.get_klines(
                symbol=symbol,
                interval=interval_map[interval],
                limit=current_batch
            )
            all_klines.extend(batch_klines)
            remaining -= current_batch

        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        float_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[float_columns] = df[float_columns].astype('float32')

        return df

    except Exception as e:
        print(f"Veri çekme hatası: {e}")
        return None

def main(symbol, interval, lookback):
    """
    Main function to run the analysis.
    
    Parameters:
    symbol (str): Symbol to retrieve data for
    interval (str): Interval to retrieve data in
    lookback (int): Lookback period for the data
    
    Returns:
    None
    """
    print(f"Analiz başlatiliyor: {symbol} - {interval}")

    ohlc_data = get_data_from_binance(symbol, interval, lookback)
    if ohlc_data is None or ohlc_data.empty:
        print("Veri çekilemedi veya işlem yapamilyamadı.")
        return

    print(f"İşlenen mum sayısı: {len(ohlc_data)}")

    analyzer = EnhancedMomentumAnalyzer(ohlc_data)
    performance_scores = analyzer.analyze()

    print("\nPerformans Skorları (Farklı Periyotlar İçin):")
    for period, scores in performance_scores.items():
        print(f"\n{period} mum sonraki:")
        for signal, score in scores.items():
            print(f"{signal}: {score:.2f}%")

if __name__ == "__main__":
    main(symbol="BTCUSDT", interval="5m", lookback=3000)
