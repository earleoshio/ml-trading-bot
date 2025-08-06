"""
Basic Example of AutoML Pipeline Usage

This example demonstrates how to use the Custom AutoML system
for cryptocurrency trading.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from automl import AutoMLPipeline


def generate_sample_crypto_data(n_periods: int = 1000) -> pd.DataFrame:
    """
    Generate sample cryptocurrency price data for demonstration.
    
    Args:
        n_periods: Number of time periods to generate
        
    Returns:
        DataFrame with OHLCV data
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Start with base price
    base_price = 50000.0
    prices = [base_price]
    
    # Generate price series with volatility clustering and trends
    for i in range(n_periods - 1):
        # Add some trend and mean reversion
        trend = 0.0001 * np.sin(i / 100)  # Cyclical trend
        mean_reversion = -0.01 * (prices[-1] / base_price - 1)  # Mean reversion
        
        # Volatility clustering effect
        vol_base = 0.02
        vol_clustering = 0.01 * np.abs(np.random.normal(0, 1))
        volatility = vol_base + vol_clustering
        
        # Price change
        price_change = trend + mean_reversion + np.random.normal(0, volatility)
        new_price = prices[-1] * (1 + price_change)
        
        # Ensure price stays positive
        new_price = max(new_price, 1000)
        prices.append(new_price)
    
    # Convert to numpy array
    close_prices = np.array(prices)
    
    # Generate OHLCV data
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(hours=n_periods),
        periods=n_periods,
        freq='H'
    )
    
    # Generate other OHLC values
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.01, n_periods)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.01, n_periods)))
    
    # Ensure OHLC relationships are valid
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    
    # Adjust high/low to be valid
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))
    
    # Generate volume (inversely correlated with price changes)
    price_changes = np.abs(np.diff(close_prices, prepend=close_prices[0]))
    volume = 1000000 + np.random.exponential(500000, n_periods) * (1 + price_changes)
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    })
    
    data.set_index('timestamp', inplace=True)
    
    return data


def basic_automl_example():
    """Basic example of using the AutoML pipeline."""
    
    print("=== Custom AutoML for Crypto Trading - Basic Example ===\n")
    
    # Step 1: Generate sample data
    print("1. Generating sample cryptocurrency data...")
    price_data = generate_sample_crypto_data(n_periods=500)
    
    print(f"   Generated {len(price_data)} hours of price data")
    print(f"   Price range: ${price_data['close'].min():.2f} - ${price_data['close'].max():.2f}")
    print(f"   Data period: {price_data.index[0]} to {price_data.index[-1]}\n")
    
    # Step 2: Initialize AutoML Pipeline
    print("2. Initializing AutoML Pipeline...")
    
    # Use a basic config for this example
    pipeline = AutoMLPipeline()
    
    print("   Pipeline initialized successfully\n")
    
    # Step 3: Split data into train/test
    print("3. Splitting data into training and test sets...")
    
    split_idx = int(len(price_data) * 0.8)
    train_data = price_data.iloc[:split_idx]
    test_data = price_data.iloc[split_idx:]
    
    print(f"   Training data: {len(train_data)} periods")
    print(f"   Test data: {len(test_data)} periods\n")
    
    # Step 4: Train the pipeline
    print("4. Training AutoML pipeline (this may take a few minutes)...")
    
    try:
        pipeline.fit(train_data)
        print("   Pipeline training completed successfully!\n")
    except Exception as e:
        print(f"   Training failed: {e}")
        print("   This is expected in a demonstration without all dependencies installed.\n")
        return
    
    # Step 5: Make predictions
    print("5. Making predictions on test data...")
    
    try:
        predictions = pipeline.predict(test_data)
        print(f"   Generated {len(predictions)} predictions")
        
        # Simple performance check
        if len(predictions) > 0:
            print(f"   Prediction range: {np.min(predictions):.6f} to {np.max(predictions):.6f}")
        
    except Exception as e:
        print(f"   Prediction failed: {e}")
        print("   This is expected in a demonstration environment.\n")
        return
    
    # Step 6: Evaluate performance
    print("6. Evaluating pipeline performance...")
    
    try:
        metrics = pipeline.evaluate(test_data)
        
        print("   Performance metrics:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"   - {metric}: {value:.6f}")
        
    except Exception as e:
        print(f"   Evaluation failed: {e}")
        print("   This is expected in a demonstration environment.\n")
    
    # Step 7: Get pipeline status
    print("7. Pipeline status and configuration:")
    
    status = pipeline.get_pipeline_status()
    
    print(f"   - Initialized: {status['pipeline_state']['initialized']}")
    print(f"   - Trained: {status['pipeline_state']['trained']}")
    print(f"   - Components active: {sum(status['component_status'].values())}/{len(status['component_status'])}")
    print(f"   - Current models: {len(status['pipeline_state'].get('current_models', []))}")
    
    if status['pipeline_state'].get('active_regimes'):
        current_regime = status['pipeline_state']['active_regimes'].get('current_regime', 'unknown')
        print(f"   - Current market regime: {current_regime}")
    
    print("\n=== Example completed! ===")


def feature_engineering_example():
    """Example focusing on feature engineering capabilities."""
    
    print("\n=== Feature Engineering Example ===\n")
    
    # Generate sample data
    price_data = generate_sample_crypto_data(n_periods=200)
    
    # Initialize individual components for demonstration
    from automl.feature_engineering import TechnicalIndicators, PatternRecognition
    
    print("1. Generating technical indicators...")
    tech_indicators = TechnicalIndicators()
    
    try:
        features_with_indicators = tech_indicators.generate_features(price_data)
        original_cols = len(price_data.columns)
        new_cols = len(features_with_indicators.columns)
        
        print(f"   Added {new_cols - original_cols} technical indicator features")
        print(f"   Sample features: {list(features_with_indicators.columns[-5:])}")
    except Exception as e:
        print(f"   Technical indicators generation failed: {e}")
        print("   This is expected without TA-Lib installed.")
    
    print("\n2. Generating pattern recognition features...")
    pattern_recognition = PatternRecognition()
    
    try:
        features_with_patterns = pattern_recognition.generate_features(price_data)
        original_cols = len(price_data.columns)
        new_cols = len(features_with_patterns.columns)
        
        print(f"   Added {new_cols - original_cols} pattern recognition features")
        print(f"   Sample features: {list(features_with_patterns.columns[-5:])}")
    except Exception as e:
        print(f"   Pattern recognition failed: {e}")
        print("   This is expected without TA-Lib and scipy dependencies.")
    
    print("\n=== Feature Engineering Example completed! ===")


def regime_detection_example():
    """Example focusing on regime detection capabilities."""
    
    print("\n=== Regime Detection Example ===\n")
    
    # Generate sample data with different volatility periods
    print("1. Generating sample data with volatility regimes...")
    
    # Create data with varying volatility
    np.random.seed(42)
    n_periods = 300
    timestamps = pd.date_range(start=datetime.now() - timedelta(hours=n_periods), periods=n_periods, freq='H')
    
    # Generate price series with regime changes
    prices = [50000.0]
    volatilities = []
    
    for i in range(n_periods - 1):
        # Create volatility regimes
        if i < 100:
            vol = 0.01  # Low volatility regime
        elif i < 200:
            vol = 0.04  # High volatility regime
        else:
            vol = 0.02  # Medium volatility regime
        
        volatilities.append(vol)
        
        price_change = np.random.normal(0, vol)
        new_price = prices[-1] * (1 + price_change)
        prices.append(max(new_price, 1000))
    
    # Create DataFrame
    close_prices = np.array(prices)
    price_data = pd.DataFrame({
        'timestamp': timestamps,
        'close': close_prices,
        'high': close_prices * 1.01,
        'low': close_prices * 0.99,
        'open': np.roll(close_prices, 1),
        'volume': np.random.exponential(1000000, n_periods)
    }).set_index('timestamp')
    
    print(f"   Generated {n_periods} periods with 3 volatility regimes")
    
    # Initialize regime detection
    from automl.regime_detection import VolatilityRegimeDetector
    
    print("\n2. Detecting volatility regimes...")
    
    try:
        vol_detector = VolatilityRegimeDetector()
        detected_regimes = vol_detector.detect_regimes(price_data)
        
        regime_counts = detected_regimes.value_counts()
        print("   Detected regimes:")
        for regime, count in regime_counts.items():
            print(f"   - {regime}: {count} periods ({count/len(detected_regimes)*100:.1f}%)")
        
        print(f"   Current regime: {detected_regimes.iloc[-1]}")
        
    except Exception as e:
        print(f"   Regime detection failed: {e}")
        print("   This is expected without all required dependencies.")
    
    print("\n=== Regime Detection Example completed! ===")


if __name__ == "__main__":
    # Run all examples
    try:
        basic_automl_example()
        feature_engineering_example()
        regime_detection_example()
    except KeyboardInterrupt:
        print("\n\nExample interrupted by user.")
    except Exception as e:
        print(f"\n\nExample failed with error: {e}")
        print("This is expected in a demonstration environment without all dependencies.")
    
    print("\n" + "="*60)
    print("Thank you for exploring the Custom AutoML system!")
    print("="*60)