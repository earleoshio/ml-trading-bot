"""
Sentiment Features Module

Implements sentiment analysis features for cryptocurrency trading,
including social media sentiment, news sentiment, and market psychology indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from loguru import logger
import re
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentFeatures:
    """
    Sentiment analysis features for cryptocurrency trading.
    
    Analyzes sentiment from various sources including:
    - Social media (Twitter, Reddit, Telegram)
    - News articles and press releases
    - Market psychology indicators
    - Fear & Greed index proxies
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize sentiment analysis system.
        
        Args:
            config: Configuration dictionary with sentiment analysis parameters
        """
        self.config = config or self._default_config()
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
    def _default_config(self) -> Dict:
        """Default configuration for sentiment analysis."""
        return {
            'sources': {
                'twitter_weight': 0.4,
                'reddit_weight': 0.3,
                'news_weight': 0.3
            },
            'processing': {
                'text_cleaning': True,
                'remove_stopwords': True,
                'handle_crypto_slang': True,
                'min_text_length': 10
            },
            'indicators': {
                'sentiment_periods': [1, 4, 12, 24, 168],  # hours
                'volatility_sentiment_correlation_period': 24,
                'trend_sentiment_correlation_period': 168
            },
            'psychology': {
                'fear_greed_components': ['volatility', 'momentum', 'volume', 'dominance'],
                'contrarian_threshold': 0.8,  # Extreme sentiment threshold
                'sentiment_divergence_threshold': 0.3
            },
            'crypto_keywords': {
                'bullish': ['moon', 'lambo', 'hodl', 'diamond hands', 'to the moon', 'bullish', 'pump'],
                'bearish': ['crash', 'dump', 'rekt', 'paper hands', 'bearish', 'short'],
                'neutral': ['consolidation', 'sideways', 'accumulation']
            }
        }
    
    def generate_features(self, 
                         price_data: pd.DataFrame,
                         social_media_data: Optional[pd.DataFrame] = None,
                         news_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate sentiment features from various data sources.
        
        Args:
            price_data: OHLCV price data with timestamp
            social_media_data: Social media posts with text and timestamp
            news_data: News articles with text and timestamp
            
        Returns:
            DataFrame with sentiment features added
        """
        logger.info("Generating sentiment features...")
        
        result_df = price_data.copy()
        
        # Add market psychology indicators derived from price action
        result_df = self._add_market_psychology_features(result_df)
        
        # Add social media sentiment if available
        if social_media_data is not None:
            result_df = self._add_social_media_sentiment(result_df, social_media_data)
        
        # Add news sentiment if available
        if news_data is not None:
            result_df = self._add_news_sentiment(result_df, news_data)
        
        # Add composite sentiment indicators
        result_df = self._add_composite_sentiment_indicators(result_df)
        
        # Add sentiment-price relationships
        result_df = self._add_sentiment_price_features(result_df)
        
        logger.info(f"Generated {len(result_df.columns) - len(price_data.columns)} sentiment features")
        return result_df
    
    def _add_market_psychology_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add market psychology indicators derived from price action."""
        result = data.copy()
        
        # Fear & Greed Index components
        
        # Volatility component (inverted - high volatility = fear)
        returns = np.log(data['close'] / data['close'].shift(1))
        volatility = returns.rolling(24).std()
        result['fear_greed_volatility'] = 1 - (volatility / volatility.rolling(168).max())
        
        # Momentum component
        momentum_short = data['close'].pct_change(24)
        momentum_long = data['close'].pct_change(168)
        result['fear_greed_momentum'] = np.tanh(momentum_short / 0.1)  # Normalize
        
        # Volume component (unusual volume = fear or greed)
        volume_ma = data['volume'].rolling(168).mean()
        volume_ratio = data['volume'] / volume_ma
        result['fear_greed_volume'] = np.tanh((volume_ratio - 1) / 2)
        
        # Market dominance proxy (using volume concentration)
        volume_std = data['volume'].rolling(24).std()
        volume_mean = data['volume'].rolling(24).mean()
        result['market_dominance_proxy'] = volume_std / (volume_mean + 1e-8)
        
        # Composite Fear & Greed Index
        components = ['fear_greed_volatility', 'fear_greed_momentum', 'fear_greed_volume']
        result['fear_greed_index'] = result[components].mean(axis=1)
        
        # Market psychology states
        result['extreme_fear'] = (result['fear_greed_index'] < 0.2).astype(int)
        result['extreme_greed'] = (result['fear_greed_index'] > 0.8).astype(int)
        result['neutral_sentiment'] = ((result['fear_greed_index'] >= 0.4) & 
                                      (result['fear_greed_index'] <= 0.6)).astype(int)
        
        return result
    
    def _add_social_media_sentiment(self, price_data: pd.DataFrame, 
                                   social_data: pd.DataFrame) -> pd.DataFrame:
        """Add social media sentiment features."""
        result = price_data.copy()
        
        # Placeholder implementation - would need actual social media data processing
        # In practice, this would aggregate sentiment scores by time periods
        
        for period in self.config['indicators']['sentiment_periods']:
            result[f'social_sentiment_{period}h'] = 0.0  # Placeholder
            result[f'social_volume_{period}h'] = 0.0    # Post count
            result[f'social_engagement_{period}h'] = 0.0 # Likes, retweets, etc.
        
        # Sentiment momentum
        result['social_sentiment_momentum'] = 0.0  # Change in sentiment
        result['social_volume_momentum'] = 0.0     # Change in post volume
        
        # Platform-specific sentiment
        result['twitter_sentiment'] = 0.0
        result['reddit_sentiment'] = 0.0
        result['telegram_sentiment'] = 0.0
        
        # Sentiment dispersion (agreement/disagreement)
        result['sentiment_dispersion'] = 0.0
        result['sentiment_polarization'] = 0.0
        
        return result
    
    def _add_news_sentiment(self, price_data: pd.DataFrame, 
                           news_data: pd.DataFrame) -> pd.DataFrame:
        """Add news sentiment features."""
        result = price_data.copy()
        
        # Placeholder implementation - would need actual news data processing
        
        for period in self.config['indicators']['sentiment_periods']:
            result[f'news_sentiment_{period}h'] = 0.0      # Average sentiment
            result[f'news_count_{period}h'] = 0.0          # Number of articles
            result[f'news_importance_{period}h'] = 0.0     # Weighted by source importance
        
        # News sentiment momentum
        result['news_sentiment_momentum'] = 0.0
        result['news_volume_momentum'] = 0.0
        
        # Source-specific sentiment
        result['mainstream_news_sentiment'] = 0.0
        result['crypto_news_sentiment'] = 0.0
        result['regulatory_news_sentiment'] = 0.0
        
        return result
    
    def _add_composite_sentiment_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add composite sentiment indicators."""
        result = data.copy()
        
        # Weighted composite sentiment
        weights = self.config['sources']
        
        # This would normally combine actual sentiment scores
        result['composite_sentiment'] = 0.0  # Placeholder
        result['composite_sentiment_ma'] = 0.0  # Moving average
        result['composite_sentiment_std'] = 0.0  # Volatility
        
        # Sentiment trend analysis
        result['sentiment_trend'] = 0.0  # Linear trend in sentiment
        result['sentiment_acceleration'] = 0.0  # Change in trend
        
        # Sentiment cycles (weekly, daily patterns)
        if 'timestamp' in data.columns:
            data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
            data['day_of_week'] = pd.to_datetime(data['timestamp']).dt.dayofweek
            
            # Hourly sentiment patterns
            result['hourly_sentiment_effect'] = 0.0  # Placeholder
            result['daily_sentiment_effect'] = 0.0   # Placeholder
        
        return result
    
    def _add_sentiment_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment-price relationship features."""
        result = data.copy()
        
        # Sentiment-price correlation
        returns = np.log(data['close'] / data['close'].shift(1))
        
        # This would normally use actual sentiment data
        correlation_period = self.config['indicators']['volatility_sentiment_correlation_period']
        result['sentiment_price_correlation'] = 0.0  # Placeholder
        
        # Sentiment divergence from price
        result['sentiment_price_divergence'] = 0.0  # Placeholder
        
        # Contrarian indicators
        result['contrarian_signal'] = 0.0  # When sentiment is extreme vs price action
        
        # Sentiment momentum vs price momentum
        result['sentiment_momentum_divergence'] = 0.0  # Placeholder
        
        return result
    
    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text using multiple methods.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores from different analyzers
        """
        if not text or len(text) < self.config['processing']['min_text_length']:
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        # Clean text
        cleaned_text = self._clean_text(text)
        
        # VADER sentiment
        vader_scores = self.vader_analyzer.polarity_scores(cleaned_text)
        
        # TextBlob sentiment
        blob = TextBlob(cleaned_text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        # Crypto-specific sentiment
        crypto_sentiment = self._analyze_crypto_sentiment(cleaned_text)
        
        return {
            'compound': vader_scores['compound'],
            'positive': vader_scores['pos'],
            'negative': vader_scores['neg'],
            'neutral': vader_scores['neu'],
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity,
            'crypto_bullish': crypto_sentiment['bullish'],
            'crypto_bearish': crypto_sentiment['bearish']
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text for sentiment analysis."""
        if not self.config['processing']['text_cleaning']:
            return text
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags symbols (but keep the text)
        text = re.sub(r'[@#]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Handle crypto-specific slang if enabled
        if self.config['processing']['handle_crypto_slang']:
            text = self._handle_crypto_slang(text)
        
        return text
    
    def _handle_crypto_slang(self, text: str) -> str:
        """Replace crypto slang with standard terms for better sentiment analysis."""
        slang_replacements = {
            'hodl': 'hold',
            'rekt': 'destroyed',
            'moon': 'surge up',
            'lambo': 'wealth',
            'diamond hands': 'strong hold',
            'paper hands': 'weak sell',
            'fud': 'fear uncertainty doubt',
            'fomo': 'fear of missing out',
            'dyor': 'do your own research',
            'wagmi': 'we are going to make it',
            'ngmi': 'not going to make it',
            'ath': 'all time high',
            'btfd': 'buy the dip'
        }
        
        for slang, replacement in slang_replacements.items():
            text = text.replace(slang, replacement)
        
        return text
    
    def _analyze_crypto_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze crypto-specific sentiment using keyword matching."""
        bullish_keywords = self.config['crypto_keywords']['bullish']
        bearish_keywords = self.config['crypto_keywords']['bearish']
        
        text_words = text.lower().split()
        
        bullish_count = sum(1 for word in text_words if any(keyword in word for keyword in bullish_keywords))
        bearish_count = sum(1 for word in text_words if any(keyword in word for keyword in bearish_keywords))
        
        total_sentiment_words = bullish_count + bearish_count
        
        if total_sentiment_words == 0:
            return {'bullish': 0.0, 'bearish': 0.0}
        
        return {
            'bullish': bullish_count / total_sentiment_words,
            'bearish': bearish_count / total_sentiment_words
        }
    
    def aggregate_sentiment_data(self, 
                                sentiment_data: pd.DataFrame,
                                time_column: str = 'timestamp',
                                sentiment_column: str = 'sentiment') -> pd.DataFrame:
        """
        Aggregate sentiment data by time periods.
        
        Args:
            sentiment_data: DataFrame with individual sentiment scores
            time_column: Name of timestamp column
            sentiment_column: Name of sentiment score column
            
        Returns:
            DataFrame with aggregated sentiment metrics
        """
        if time_column not in sentiment_data.columns:
            logger.error(f"Time column '{time_column}' not found in sentiment data")
            return pd.DataFrame()
        
        sentiment_data[time_column] = pd.to_datetime(sentiment_data[time_column])
        sentiment_data = sentiment_data.set_index(time_column)
        
        aggregated = {}
        
        for period in self.config['indicators']['sentiment_periods']:
            period_str = f'{period}H'
            
            # Aggregate by period
            agg_data = sentiment_data.resample(period_str).agg({
                sentiment_column: ['mean', 'std', 'count', 'min', 'max']
            })
            
            # Flatten column names
            agg_data.columns = [f'sentiment_{period}h_{stat}' for stat in ['mean', 'std', 'count', 'min', 'max']]
            
            aggregated[period] = agg_data
        
        # Combine all periods
        result = pd.concat(aggregated.values(), axis=1)
        
        return result
    
    def detect_sentiment_anomalies(self, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in sentiment data that might predict price movements.
        
        Args:
            sentiment_data: DataFrame with sentiment time series
            
        Returns:
            DataFrame with anomaly indicators
        """
        result = sentiment_data.copy()
        
        # Z-score based anomalies
        for col in sentiment_data.columns:
            if 'sentiment' in col.lower():
                rolling_mean = sentiment_data[col].rolling(24).mean()
                rolling_std = sentiment_data[col].rolling(24).std()
                z_score = (sentiment_data[col] - rolling_mean) / (rolling_std + 1e-8)
                
                result[f'{col}_zscore'] = z_score
                result[f'{col}_anomaly'] = (np.abs(z_score) > 2.5).astype(int)
        
        # Sudden sentiment shifts
        for col in sentiment_data.columns:
            if 'sentiment' in col.lower():
                sentiment_change = sentiment_data[col].diff()
                change_threshold = sentiment_change.rolling(168).std() * 2
                result[f'{col}_sudden_shift'] = (np.abs(sentiment_change) > change_threshold).astype(int)
        
        return result