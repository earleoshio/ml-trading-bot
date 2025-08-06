# Custom AutoML System for Cryptocurrency Trading Bot

This repository contains a comprehensive Custom AutoML system specifically designed for cryptocurrency trading patterns. The system provides end-to-end automation of the machine learning pipeline, from feature engineering to model deployment and monitoring.

## 🚀 Features

### Core AutoML Components

#### 1. **Automated Feature Engineering**
- **Technical Indicators**: 40+ indicators including RSI, MACD, Bollinger Bands, ATR, Stochastic, etc.
- **Market Microstructure**: Order book analysis, liquidity measures, trade size distribution
- **Sentiment Analysis**: Social media and news sentiment with crypto-specific slang handling
- **Pattern Recognition**: Candlestick patterns, chart patterns, support/resistance levels
- **Feature Selection**: Advanced selection using mutual information, LASSO, and stability analysis

#### 2. **Dynamic Model Selection & Ensemble**
- **Model Registry**: XGBoost, RandomForest, SVM, LSTM, Transformer, MLP models
- **Hyperparameter Optimization**: Optuna-based optimization with multi-objective support
- **Ensemble Management**: Dynamic weighting, stacking, and regime-aware model selection
- **Meta-Learning**: Automated model recommendation based on data characteristics

#### 3. **Market Regime Detection**
- **Volatility Regimes**: GARCH-based detection, volatility clustering analysis
- **Trend Detection**: Multi-method trend analysis with support/resistance identification
- **Regime Classification**: Automatic model switching based on detected market conditions
- **Performance Tracking**: Regime-specific performance monitoring

#### 4. **Automated Strategy Generation**
- **Signal Calibration**: Multi-signal combination and strength normalization
- **Position Sizing**: Kelly criterion, volatility-adjusted sizing
- **Risk Management**: Stop-loss optimization, correlation-based position limits
- **Multi-timeframe Analysis**: Strategy synthesis across different time horizons

#### 5. **Performance Monitoring & Auto-Retraining**
- **Real-time Tracking**: Comprehensive performance metrics monitoring
- **Concept Drift Detection**: Statistical and model-based drift detection
- **Automatic Retraining**: Trigger-based model updates and retraining
- **A/B Testing Framework**: Model performance comparison and selection

## 📁 Project Structure

```
src/automl/
├── feature_engineering/
│   ├── __init__.py
│   ├── technical_indicators.py      # Technical analysis indicators
│   ├── market_microstructure.py     # Order book and liquidity analysis
│   ├── sentiment_features.py        # Social media and news sentiment
│   ├── pattern_recognition.py       # Chart and candlestick patterns
│   └── feature_selector.py          # Automated feature selection
├── model_selection/
│   ├── __init__.py
│   ├── model_registry.py           # Model management and lifecycle
│   ├── hyperparameter_optimizer.py # Optuna-based optimization
│   ├── ensemble_manager.py         # Dynamic ensemble management
│   └── meta_learner.py            # Meta-learning and model recommendation
├── regime_detection/
│   ├── __init__.py
│   ├── volatility_regimes.py      # Volatility regime detection
│   ├── trend_detection.py         # Trend analysis and detection
│   └── regime_classifier.py       # Market regime classification
├── strategy_generation/           # Strategy generation components
├── monitoring/                    # Performance monitoring and drift detection
└── automl_pipeline.py            # Main orchestration pipeline

config/
├── automl_config.yaml            # Main configuration file

examples/
├── basic_example.py              # Basic usage example
└── advanced_example.py          # Advanced usage patterns

tests/
├── test_basic.py                 # Basic unit tests
└── test_integration.py          # Integration tests
```

## 🛠 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/earleoshio/ml-trading-bot.git
cd ml-trading-bot
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install TA-Lib (required for technical indicators):**
```bash
# On Ubuntu/Debian:
sudo apt-get install libta-lib-dev
pip install TA-Lib

# On macOS:
brew install ta-lib
pip install TA-Lib

# On Windows:
# Download appropriate wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib‑0.4.xx‑cpxx‑cpxx‑win_amd64.whl
```

4. **Install the package:**
```bash
pip install -e .
```

## 🚀 Quick Start

### Basic Usage

```python
from automl import AutoMLPipeline
import pandas as pd

# Initialize the pipeline
pipeline = AutoMLPipeline()

# Load your cryptocurrency price data (OHLCV format)
price_data = pd.read_csv('your_price_data.csv')

# Train the pipeline
pipeline.fit(price_data)

# Make predictions
predictions = pipeline.predict(new_price_data)

# Evaluate performance
metrics = pipeline.evaluate(test_data)
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
```

### Advanced Usage with Custom Configuration

```python
from automl import AutoMLPipeline

# Initialize with custom config
pipeline = AutoMLPipeline(config_path='config/custom_config.yaml')

# Initialize components
pipeline.initialize_components()

# Add external data sources
external_data = {
    'social_media': social_sentiment_data,
    'news': news_sentiment_data,
    'order_book': order_book_data
}

# Train with external data
pipeline.fit(price_data, external_data=external_data)

# Get pipeline status
status = pipeline.get_pipeline_status()
print(f"Active models: {len(status['pipeline_state']['current_models'])}")
print(f"Current regime: {status['pipeline_state']['active_regimes']['current_regime']}")
```

## 📊 Configuration

The system is highly configurable through YAML files. Key configuration sections include:

### Feature Engineering
```yaml
feature_engineering:
  technical_indicators:
    enabled: true
    sma_periods: [5, 10, 20, 50, 100, 200]
    rsi_periods: [14, 21, 30]
  
  sentiment_features:
    enabled: true
    sources:
      twitter_weight: 0.4
      reddit_weight: 0.3
      news_weight: 0.3
```

### Model Selection
```yaml
model_selection:
  hyperparameter_optimization:
    n_trials: 100
    timeout: 3600
    primary_metric: "sharpe_ratio"
  
  ensemble_management:
    combination_methods:
      default_method: "adaptive_weighted"
    weight_update:
      update_frequency: 24  # hours
```

## 📈 Features in Detail

### Technical Indicators
- **Trend**: SMA, EMA, MACD, ADX, Parabolic SAR
- **Momentum**: RSI, Stochastic, CCI, Williams %R, ROC
- **Volatility**: Bollinger Bands, ATR, Keltner Channels
- **Volume**: OBV, VWAP, Money Flow Index, PVT
- **Crypto-specific**: Funding rates, perpetual basis, whale detection

### Pattern Recognition
- **Candlestick Patterns**: Doji, Hammer, Engulfing, Morning/Evening Star
- **Chart Patterns**: Triangles, Flags, Head & Shoulders, Double Tops/Bottoms
- **Support/Resistance**: Dynamic level identification with strength calculation
- **Trend Lines**: Automated trend line fitting with statistical validation

### Regime Detection
- **Volatility Regimes**: Low, medium, high, extreme volatility classification
- **Trend Regimes**: Uptrend, downtrend, sideways market detection
- **Method Ensemble**: Threshold-based, GARCH-based, clustering-based detection

### Model Selection
- **Available Models**: XGBoost, RandomForest, SVM, LSTM, Transformer, MLP
- **Hyperparameter Optimization**: Multi-objective optimization with trading metrics
- **Meta-Learning**: Data characteristic-based model recommendation
- **Ensemble Methods**: Simple average, weighted average, stacking, voting

## 🧪 Testing

Run the test suite to verify installation:

```bash
# Run basic tests
python tests/test_basic.py

# Run example (demonstrates functionality)
python examples/basic_example.py
```

## 📊 Performance Monitoring

The system includes comprehensive monitoring:

- **Real-time Metrics**: Returns, Sharpe ratio, maximum drawdown, win rate
- **Concept Drift Detection**: Statistical tests and performance degradation monitoring
- **Auto-retraining**: Configurable triggers for model updates
- **Regime Tracking**: Performance attribution by market regime

## 🔧 Extending the System

### Adding New Models
```python
from automl.model_selection import BaseModel

class CustomModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__('CustomModel', 'regression')
        # Implementation
    
    def fit(self, X, y):
        # Training logic
        pass
    
    def predict(self, X):
        # Prediction logic
        pass
```

### Adding New Features
```python
from automl.feature_engineering import TechnicalIndicators

class CustomFeatures(TechnicalIndicators):
    def generate_custom_features(self, data):
        # Custom feature engineering logic
        return enhanced_data
```

## 📚 Documentation

- **Configuration Guide**: See `config/automl_config.yaml` for all options
- **API Documentation**: Generated from docstrings in source code
- **Examples**: Check `examples/` directory for usage patterns

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. The authors are not responsible for any financial losses incurred from using this system. Always conduct thorough backtesting and risk assessment before deploying any trading strategy.

## 🙏 Acknowledgments

- **TA-Lib**: Technical Analysis Library for technical indicators
- **Optuna**: Automatic hyperparameter optimization framework
- **scikit-learn**: Machine learning library for feature selection and modeling
- **XGBoost**: Gradient boosting framework for high-performance modeling

---

**Built with ❤️ for the cryptocurrency trading community**
