# Optimized Momentum Analyzer V3

## Overview
Advanced trading signal analysis tool with parallel processing for efficient cryptocurrency market analysis.

## Features
- Parallel signal detection
- Multiple signal types:
  - Large candle detection
  - Micro gap tracking
  - Clean candle identification
  - Reversal candle recognition
- Performance scoring for each signal type
- Binance API integration

## Prerequisites
- Python 3.8+
- Libraries:
  ```
  pandas
  numpy
  python-binance
  ```

## Installation
```bash
pip install pandas numpy python-binance
```

## Configuration
1. Add Binance API credentials in `get_data_from_binance()`
2. Customize signal detection parameters

## Usage Example
```python
from optimized_momentum_analyzer import main

# Analyze Bitcoin 5-minute data
main(symbol='BTCUSDT', interval='5m', lookback=3000)
```

## Signal Types
- Large Candle: Significant price movement
- Micro Gap: Small price discontinuities
- Clean Candle: Minimal wick patterns
- Reversal Candle: Potential trend change indicators

## Performance Metrics
Outputs success rates for each signal type as percentages.

## Customization
- Adjust `batch_size` for processing speed
- Modify `forward_period` in performance calculation

## Disclaimer
For educational purposes. Trading involves financial risk.

## License
MIT
