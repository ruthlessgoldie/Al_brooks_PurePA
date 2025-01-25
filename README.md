# Momentum Analyzer

## Overview
This Python script provides a comprehensive momentum trading signal analysis tool for financial markets. It uses technical indicators and candlestick pattern analysis to detect potential trading opportunities.

## Features
- EMA (Exponential Moving Average) trend detection
- Large candle identification
- Micro gap detection
- Clean candle pattern recognition
- Performance analysis of trading signals

## Prerequisites
- Python 3.7+
- Required libraries:
  - pandas
  - numpy
  - yfinance

## Installation
```bash
pip install pandas numpy yfinance
```

## Usage
```python
from momentum import run_analysis

# Example: Analyze Bitcoin 5-minute data for the last month
run_analysis(symbol='BTC-USD', interval='5m', period='1mo')
```

## Key Analysis Methods
- `calculate_ema_trend()`: Calculates EMA trend strength
- `detect_large_candles()`: Identifies significant price movement candles
- `detect_micro_gaps()`: Detects small price gaps
- `detect_clean_candles()`: Finds candles with minimal wicks
- `analyze_performance()`: Generates comprehensive trading signal performance metrics

## Output
The script provides detailed performance metrics including:
- Signal success rates
- Number of signals
- Average trend length
- Performance of combined signal strategies

## Customization
Modify parameters in method calls to adjust:
- EMA periods
- Candle size thresholds
- Signal detection criteria

## License
[Insert your chosen license]

## Disclaimer
This tool is for educational purposes. Always conduct thorough research and risk management in trading.
