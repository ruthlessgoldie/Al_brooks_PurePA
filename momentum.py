import pandas as pd
import numpy as np
from binance.client import Client
from concurrent.futures import ThreadPoolExecutor
import warnings

class OptimizedMomentumAnalyzer:
    def __init__(self, data, batch_size=3000):
        self.data = data.copy()
        self.batch_size = batch_size
        self.signals = pd.DataFrame(index=self.data.index)
        self.performance_scores = {}
        
        # Initialize signal columns
        for col in ['large_candle', 'micro_gap_up', 'micro_gap_down', 'clean_candle', 'reversal_candle']:
            self.signals[col] = False
        
        self._calculate_common_metrics()
    
    def _calculate_common_metrics(self):
        """Calculate commonly used metrics"""
        self.data['is_green'] = (self.data['Close'] > self.data['Open']).astype(bool)
        self.data['is_red'] = (self.data['Close'] < self.data['Open']).astype(bool)
        self.data['body_size'] = np.abs(self.data['Close'] - self.data['Open'])
        self.data['upper_wick'] = self.data['High'] - self.data[['Open', 'Close']].max(axis=1)
        self.data['lower_wick'] = self.data[['Open', 'Close']].min(axis=1) - self.data['Low']
        self.data['avg_body_size'] = self.data['body_size'].rolling(window=3, min_periods=1).mean()
    
    def _calculate_batch_signals(self, start_idx, end_idx):
        """Calculate signals for a batch"""
        batch_data = self.data.iloc[start_idx:end_idx].copy()
        signals = pd.DataFrame(index=batch_data.index)
        
        # Large candle
        signals['large_candle'] = (batch_data['body_size'] > batch_data['avg_body_size'] * 1.1)
        
        # Micro gap up
        prev_is_green = batch_data['is_green'].shift(1).fillna(False)
        signals['micro_gap_up'] = (
            (batch_data['Open'] > batch_data['Close'].shift(1)) &
            batch_data['is_green'] &
            prev_is_green
        )
        
        # Micro gap down
        prev_is_red = batch_data['is_red'].shift(1).fillna(False)
        signals['micro_gap_down'] = (
            (batch_data['Open'] < batch_data['Close'].shift(1)) &
            batch_data['is_red'] &
            prev_is_red
        )
        
        # Clean candle
        with np.errstate(divide='ignore', invalid='ignore'):
            wick_ratio = np.where(
                batch_data['is_green'],
                batch_data['upper_wick'] / batch_data['body_size'],
                batch_data['lower_wick'] / batch_data['body_size']
            )
            wick_ratio = np.nan_to_num(wick_ratio, nan=np.inf)
        
        signals['clean_candle'] = (wick_ratio <= 0.05)
        
        # Reversal candle
        with np.errstate(divide='ignore', invalid='ignore'):
            body_size_ratio = batch_data['body_size'] / batch_data['body_size'].shift(1)
            body_size_ratio = np.nan_to_num(body_size_ratio, nan=0)
        
        direction_change = (batch_data['is_green'] != batch_data['is_green'].shift(1)).fillna(False)
        price_overlap = np.where(
            batch_data['is_green'],
            batch_data['Close'] > batch_data['Close'].shift(1),
            batch_data['Close'] < batch_data['Close'].shift(1)
        )
        
        signals['reversal_candle'] = (
            direction_change &
            price_overlap &
            (body_size_ratio >= 0.6)
        )
        
        return signals.astype(bool)
    
    def detect_signals(self):
        """Detect all signals in parallel"""
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
    
    def calculate_signal_performance(self, forward_period=3):
        """Calculate performance metrics for signals"""
        self.signals['forward_return'] = (
            self.data['Close'].shift(-forward_period) - self.data['Close']
        )
        
        self.signals['micro_gap'] = (
            self.signals['micro_gap_up'] | self.signals['micro_gap_down']
        )
        
        for signal_type in ['large_candle', 'micro_gap', 'clean_candle', 'reversal_candle']:
            mask = self.signals[signal_type]
            if mask.any():
                if signal_type == 'micro_gap':
                    success = (
                        (self.signals['micro_gap_up'] & (self.signals['forward_return'] > 0)) |
                        (self.signals['micro_gap_down'] & (self.signals['forward_return'] < 0))
                    )
                    success_rate = success[mask].mean() * 100
                else:
                    success_rate = (self.signals[mask]['forward_return'] > 0).mean() * 100
                
                self.performance_scores[signal_type] = success_rate
            else:
                self.performance_scores[signal_type] = 0.0
    
    def analyze(self):
        """Run the complete analysis"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.detect_signals()
            self.calculate_signal_performance()
        return self.performance_scores

def get_data_from_binance(symbol, interval, lookback, batch_size=3000):
    """Fetch data from Binance"""
    try:
        client = Client("", "")
        interval_map = {
            '5m': Client.KLINE_INTERVAL_5MINUTE,
            '15m': Client.KLINE_INTERVAL_15MINUTE,
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

def main(symbol,interval,lookback):
    """Main function"""
    symbol = symbol
    interval = interval
    lookback = lookback
    batch_size = 3000
    
    print(f"Analiz başlatılıyor: {symbol} - {interval}")
    
    ohlc_data = get_data_from_binance(symbol, interval, lookback, batch_size)
    if ohlc_data is None or ohlc_data.empty:
        print("Veri çekilemedi veya işlem yapılamadı.")
        return
    
    print(f"İşlenen mum sayısı: {len(ohlc_data)}")
    
    analyzer = OptimizedMomentumAnalyzer(ohlc_data, batch_size=batch_size)
    performance_scores = analyzer.analyze()
    
    print("\nPerformans Skorları:")
    for signal, score in performance_scores.items():
        print(f"{signal}: {score:.2f}%")

if __name__ == "__main__":
    main('BTCUSDT','5m',3000)
