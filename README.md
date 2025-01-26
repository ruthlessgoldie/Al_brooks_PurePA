# Binance Momentum Signal Analyzer v4

## Overview
This Python script performs advanced momentum signal analysis on cryptocurrency trading data retrieved from the Binance API. It detects various trading signals and evaluates their potential performance across different time periods.

## Features
- Retrieves historical price data from Binance
- Analyzes multiple momentum signals:
  - Large candle signals
  - Micro gap up/down signals
  - Clean candle signals
  - Reversal candle signals
  - Two-legged pullback signals
- Parallel processing for efficient signal detection
- Performance scoring for different forward periods

## Prerequisites
- Python 3.7+
- Required libraries:
  - pandas
  - numpy
  - python-binance
  - concurrent.futures

## Installation
1. Clone the repository
2. Install required dependencies:
```bash
pip install pandas numpy python-binance
```

## Configuration
- Replace empty Binance API credentials in `get_data_from_binance()` method with your actual API key and secret
- Customize the `main()` function parameters:
  - `symbol`: Trading pair (e.g., "BTCUSDT")
  - `interval`: Candle interval (1m, 3m, 5m, 15m, 30m, 1h, etc.)
  - `lookback`: Number of historical candles to analyze

## Usage
```python
main(symbol="BTCUSDT", interval="5m", lookback=3000)
```

## Signal Types
- **Large Candle**: Candles with significant price range
- **Micro Gap Up/Down**: Small price jumps between candles
- **Clean Candle**: Candles with clear price movement
- **Reversal Candle**: Potential trend reversal indicators
- **Two-Legged Pullback**: Short-term trend retracements

## Performance Metrics
The script calculates success rates for each signal type across different forward periods (default: 3, 5, 10 candles), showing percentage of signals resulting in positive returns.

## Limitations
- Requires Binance API access
- Performance metrics are historical and not guaranteed future performance
- Requires careful risk management in trading

## Contributing
Contributions, issues, and feature requests are welcome. Feel free to check the issues page.

## License
MIT
