import pandas as pd
import numpy as np
import yfinance as yf

class MomentumAnalyzer:
    def __init__(self, data):
        self.data = data.copy()  # Make a copy of input data
        self.signals = pd.DataFrame(index=self.data.index)
        self.results = {}
        
    def calculate_ema_trend(self, ema_period=20, trend_length=3):
        """EMA trend gücünü hesaplar"""
        # EMA hesaplama
        self.data.loc[:, 'ema20'] = self.data['Close'].ewm(span=ema_period, adjust=False).mean()
        
        # Fiyatın EMA üzerinde olup olmadığını kontrol et
        self.signals.loc[:, 'above_ema'] = self.data['Close'] > self.data['ema20']
        
        # Trend başlangıç noktalarını bul
        self.signals.loc[:, 'trend_start'] = (
            (self.signals['above_ema'] != self.signals['above_ema'].shift(1)) & 
            (self.signals['above_ema'] == True)
        )
        
        # Trend başarı ve uzunluk kolonlarını başlat
        self.signals.loc[:, 'trend_success'] = False
        self.signals.loc[:, 'trend_length'] = 0
        
        # Her trend başlangıcı için ileri doğru kontrol
        for i in range(len(self.data) - trend_length):
            if self.signals['trend_start'].iloc[i]:
                success = True
                length = 0
                for j in range(1, min(trend_length + 1, len(self.data) - i)):
                    if not self.signals['above_ema'].iloc[i + j]:
                        success = False
                        break
                    length += 1
                # loc kullanarak değer ata
                self.signals.loc[self.signals.index[i], 'trend_success'] = success
                self.signals.loc[self.signals.index[i], 'trend_length'] = length

    def detect_large_candles(self, lookback_period=3, threshold_multiplier=1.1):
        """Büyük mumları tespit eder"""
        self.data.loc[:, 'body_size'] = abs(self.data['Close'] - self.data['Open'])
        self.data.loc[:, 'avg_body_size'] = self.data['body_size'].rolling(window=lookback_period).mean()
        self.signals.loc[:, 'large_candle'] = (self.data['body_size'] > 
                                        self.data['avg_body_size'] * threshold_multiplier)
        self.signals.loc[:, 'candle_direction'] = np.where(
            self.data['Close'] > self.data['Open'], 1, -1
        )

    def detect_micro_gaps(self):
        """Micro gap'leri tespit eder"""
        self.data.loc[:, 'prev_close'] = self.data['Close'].shift(1)
        self.data.loc[:, 'is_bullish'] = (self.data['Close'] > self.data['Open']).astype(int)
        self.data.loc[:, 'prev_bullish'] = self.data['is_bullish'].shift(1)
        
        self.data.dropna(inplace=True)
        
        self.signals.loc[:, 'micro_gap_up'] = (
            (self.data['is_bullish'] == 1) & 
            (self.data['prev_bullish'] == 1) & 
            (self.data['Open'] > self.data['prev_close'])
        ).astype(bool)
        
        self.signals.loc[:, 'micro_gap_down'] = (
            (self.data['is_bullish'] == 0) & 
            (self.data['prev_bullish'] == 0) & 
            (self.data['Open'] < self.data['prev_close'])
        ).astype(bool)

    def detect_clean_candles(self, wick_threshold=0.03):
        """Trend yönünde fitilsiz/az fitilli mumları tespit eder"""
        body_size = abs(self.data['Close'] - self.data['Open'])
        
        upper_wick = np.where(
            self.data['Close'] > self.data['Open'],
            self.data['High'] - self.data['Close'],
            0
        )
        upper_wick_ratio = np.where(body_size != 0, upper_wick / body_size, 0)
        
        lower_wick = np.where(
            self.data['Close'] < self.data['Open'],
            self.data['Close'] - self.data['Low'],
            0
        )
        lower_wick_ratio = np.where(body_size != 0, lower_wick / body_size, 0)
        
        self.signals.loc[:, 'clean_bullish'] = (
            (self.data['Close'] > self.data['Open']) & 
            (upper_wick_ratio <= wick_threshold)
        )
        
        self.signals.loc[:, 'clean_bearish'] = (
            (self.data['Close'] < self.data['Open']) & 
            (lower_wick_ratio <= wick_threshold)
        )

    def calculate_momentum_signals(self, forward_period=2):
        """Momentum sinyallerini hesaplar"""
        self.signals.loc[:, 'forward_return'] = self.data['Close'].shift(-forward_period) - self.data['Close']
        self.signals.loc[:, 'momentum_success'] = (
            self.signals['forward_return'] * self.signals['candle_direction'] > 0
        )
        
        self.signals.loc[:, 'gap_success'] = (
            (self.signals['micro_gap_up'] & (self.signals['forward_return'] > 0)) |
            (self.signals['micro_gap_down'] & (self.signals['forward_return'] < 0))
        )
        
        self.signals.loc[:, 'clean_success'] = (
            (self.signals['clean_bullish'] & (self.signals['forward_return'] > 0)) |
            (self.signals['clean_bearish'] & (self.signals['forward_return'] < 0))
        )

    def analyze_performance(self):
        """Tüm momentum sinyallerinin performans analizini yapar"""
        valid_signals = self.signals[self.signals['large_candle']].copy()
        total_signals = len(valid_signals)
        successful_signals = valid_signals['momentum_success'].sum()
        
        gap_signals = len(self.signals[self.signals['micro_gap_up'] | self.signals['micro_gap_down']])
        gap_success = self.signals['gap_success'].sum()
        
        clean_signals = len(self.signals[self.signals['clean_bullish'] | self.signals['clean_bearish']])
        clean_success = self.signals['clean_success'].sum()
        
        trend_signals = len(self.signals[self.signals['trend_start']])
        trend_success = self.signals['trend_success'].sum()
        avg_trend_length = self.signals[self.signals['trend_start']]['trend_length'].mean()
        
        self.results = {
            'Toplam Büyük Mum': total_signals,
            'Başarılı Büyük Mum': successful_signals,
            'Büyük Mum Başarı Oranı (%)': (successful_signals / total_signals * 100) if total_signals > 0 else 0,
            'Micro Gap Sinyal Sayısı': gap_signals,
            'Micro Gap Başarı Sayısı': gap_success,
            'Micro Gap Başarı Oranı (%)': (gap_success / gap_signals * 100) if gap_signals > 0 else 0,
            'Tıraşlı Mum Sayısı': clean_signals,
            'Tıraşlı Mum Başarı': clean_success,
            'Tıraşlı Mum Başarı Oranı (%)': (clean_success / clean_signals * 100) if clean_signals > 0 else 0,
            'EMA Trend Başlangıç Sayısı': trend_signals,
            'Başarılı EMA Trend Sayısı': trend_success,
            'EMA Trend Başarı Oranı (%)': (trend_success / trend_signals * 100) if trend_signals > 0 else 0,
            'Ortalama Trend Uzunluğu': avg_trend_length if not pd.isna(avg_trend_length) else 0
        }
        
        return self.results

    def analyze_combined_signals(self):
        """Tüm sinyallerin kombinasyonlarını analiz eder"""
        # Kombinasyon durumlarını oluştur
        self.signals.loc[:, 'ema_large_candle'] = (
            self.signals['trend_start'] & 
            self.signals['large_candle']
        )
        
        self.signals.loc[:, 'ema_clean_candle'] = (
            self.signals['trend_start'] & 
            (self.signals['clean_bullish'] | self.signals['clean_bearish'])
        )
        
        self.signals.loc[:, 'ema_micro_gap'] = (
            self.signals['trend_start'] & 
            (self.signals['micro_gap_up'] | self.signals['micro_gap_down'])
        )
        
        self.signals.loc[:, 'strong_trend_signal'] = (
            self.signals['trend_start'] & 
            self.signals['large_candle'] & 
            (self.signals['clean_bullish'] | self.signals['clean_bearish'])
        )
        
        # Her kombinasyonun başarı oranını hesapla
        results = {}
        
        # EMA + Büyük Mum
        ema_large = self.signals[self.signals['ema_large_candle']]
        if len(ema_large) > 0:
            success_rate = (ema_large['trend_success'].sum() / len(ema_large)) * 100
            avg_length = ema_large['trend_length'].mean()
            results['EMA + Büyük Mum'] = {
                'Sinyal Sayısı': len(ema_large),
                'Başarı Oranı': success_rate,
                'Ortalama Trend Uzunluğu': avg_length
            }
        
        # EMA + Clean Candle
        ema_clean = self.signals[self.signals['ema_clean_candle']]
        if len(ema_clean) > 0:
            success_rate = (ema_clean['trend_success'].sum() / len(ema_clean)) * 100
            avg_length = ema_clean['trend_length'].mean()
            results['EMA + Clean Candle'] = {
                'Sinyal Sayısı': len(ema_clean),
                'Başarı Oranı': success_rate,
                'Ortalama Trend Uzunluğu': avg_length
            }
        
        # EMA + Micro Gap
        ema_gap = self.signals[self.signals['ema_micro_gap']]
        if len(ema_gap) > 0:
            success_rate = (ema_gap['trend_success'].sum() / len(ema_gap)) * 100
            avg_length = ema_gap['trend_length'].mean()
            results['EMA + Micro Gap'] = {
                'Sinyal Sayısı': len(ema_gap),
                'Başarı Oranı': success_rate,
                'Ortalama Trend Uzunluğu': avg_length
            }
        
        # Güçlü Trend Sinyali (Hepsi Bir Arada)
        strong_signals = self.signals[self.signals['strong_trend_signal']]
        if len(strong_signals) > 0:
            success_rate = (strong_signals['trend_success'].sum() / len(strong_signals)) * 100
            avg_length = strong_signals['trend_length'].mean()
            results['Güçlü Trend Sinyali'] = {
                'Sinyal Sayısı': len(strong_signals),
                'Başarı Oranı': success_rate,
                'Ortalama Trend Uzunluğu': avg_length
            }
        
        return results

def get_data_from_yfinance(symbol, period, interval):
    """yfinance'den veri çeker"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        return df
    except Exception as e:
        print(f"Veri çekme hatası: {e}")
        return None

def run_analysis(symbol, period, interval):
    """Analizi çalıştırır ve sonuçları gösterir"""
    ohlc_data = get_data_from_yfinance(symbol, period, interval)
    if ohlc_data is None or ohlc_data.empty:
        print("Veri çekilemedi veya işlem yapılamadı.")
        return None, None
    
    print(f"İşlenen mum sayısı: {len(ohlc_data)}")
    
    analyzer = MomentumAnalyzer(ohlc_data)
    
    analyzer.calculate_ema_trend()
    analyzer.detect_large_candles()
    analyzer.detect_micro_gaps()
    analyzer.detect_clean_candles()
    analyzer.calculate_momentum_signals()
    results = analyzer.analyze_performance()
    combined_results = analyzer.analyze_combined_signals()
    
    
    print("\nSonuçlar:")
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.2f}")
        else:
            print(f"{metric}: {value}")
    
    print("\nKombinasyon Sonuçları:")
    for combo, metrics in combined_results.items():
        print(f"\n{combo}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.2f}")
            else:
                print(f"{metric}: {value}")
    
    return ohlc_data, analyzer.signals

if __name__ == "__main__":
    run_analysis(symbol='BTC-USD', interval='5m', period='1mo')