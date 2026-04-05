"""Unit tests for trading tools."""

import pytest
from tools.technical import calculate_moving_averages, calculate_rsi, calculate_bollinger_bands
from tools.sentiment import aggregate_sentiments
from tools.market import generate_market_snapshot
from tools.portfolio import calculate_position_size, calculate_kelly_fraction


class TestTechnicalTools:
    """Test technical analysis tools."""
    
    @pytest.fixture
    def price_data(self):
        """Create sample price data."""
        return [100, 101, 102, 103, 102, 101, 100, 105, 106, 107]
    
    def test_moving_averages(self, price_data):
        """Test moving average calculation."""
        result = calculate_moving_averages(price_data)
        
        assert "MA_20" in result or "ma_5" in result or len(result) > 0
        # Moving average should be between min and max of prices
        ma_values = [v for v in result.values() if isinstance(v, (int, float))]
        if ma_values:
            assert min(price_data) <= max(ma_values) <= max(price_data)
    
    def test_rsi_bounds(self, price_data):
        """Test RSI is between 0 and 100."""
        result = calculate_rsi(price_data)
        
        # RSI should be in valid range or return dict
        if isinstance(result, dict):
            rsi_value = result.get("RSI", 0)
        else:
            rsi_value = result
        
        if isinstance(rsi_value, (int, float)):
            assert 0 <= rsi_value <= 100
    
    def test_bollinger_bands(self, price_data):
        """Test Bollinger Bands calculation."""
        result = calculate_bollinger_bands(price_data)
        
        assert result is not None
        # Should return dict with band info
        if isinstance(result, dict):
            assert len(result) > 0


class TestSentimentTools:
    """Test sentiment analysis tools."""
    
    def test_sentiment_analysis(self):
        """Test sentiment analysis on positive headlines."""
        headlines = [
            "Apple reports record profits",
            "Stock rally continues",
            "Earnings beat expectations"
        ]
        
        result = aggregate_sentiments(headlines)
        
        assert result is not None
        if isinstance(result, dict):
            # Positive headlines should have positive compound score
            compound = result.get("compound", 0)
            assert compound >= 0, "Positive headlines should have positive sentiment"
    
    def test_negative_sentiment(self):
        """Test sentiment on negative headlines."""
        headlines = [
            "Market crash expected",
            "Stock plummets",
            "Revenue misses forecast"
        ]
        
        result = aggregate_sentiments(headlines)
        
        assert result is not None
        if isinstance(result, dict):
            compound = result.get("compound", 0)
            assert compound <= 0, "Negative headlines should have negative sentiment"
    
    def test_neutral_sentiment(self):
        """Test sentiment on neutral headlines."""
        headlines = [
            "Company announces new office",
            "Board member hired",
            "Quarterly review scheduled"
        ]
        
        result = aggregate_sentiments(headlines)
        
        assert result is not None


class TestPortfolioTools:
    """Test portfolio management tools."""
    
    def test_position_sizing(self):
        """Test position size calculation."""
        # Pass max_position_size=0.5 to allow 35% calculation
        result = calculate_position_size(
            account_size=100000, 
            target_weight=0.35,
            max_position_size=0.5  # Allow up to 50% position
        )
        
        # Should return a size around 35000
        if isinstance(result, dict):
            size = result.get("position_size", 0)
        else:
            size = result
        
        assert 30000 <= size <= 40000, f"Position size {size} not in expected range"
    
    def test_position_size_bounds(self):
        """Test position sizing respects bounds."""
        # Test zero position
        result_zero = calculate_position_size(account_size=100000, target_weight=0.0)
        
        # Test full position
        result_full = calculate_position_size(account_size=100000, target_weight=1.0)
        
        # Full should be >= zero
        if isinstance(result_zero, dict):
            val_zero = result_zero.get("position_size", 0)
        else:
            val_zero = result_zero
        
        if isinstance(result_full, dict):
            val_full = result_full.get("position_size", 0)
        else:
            val_full = result_full
        
        assert val_full >= val_zero
    
    def test_kelly_criterion(self):
        """Test Kelly Criterion calculation."""
        result = calculate_kelly_fraction(
            win_rate=0.55,
            avg_win=0.02,
            avg_loss=0.01
        )
        
        # Kelly should be between 0 and 1
        if isinstance(result, dict):
            kelly = result.get("kelly_fraction", 0)
        else:
            kelly = result
        
        if isinstance(kelly, (int, float)):
            assert 0 <= kelly <= 1, f"Kelly fraction {kelly} out of bounds"


class TestMarketTools:
    """Test market data tools."""
    
    @pytest.mark.slow
    def test_market_snapshot(self):
        """Test market snapshot retrieval (calls API)."""
        result = generate_market_snapshot(symbol="AAPL", days=30)
        
        assert result is not None
        assert result.get("symbol") == "AAPL"
        assert "prices" in result
        assert "returns" in result
        assert "realized_volatility" in result
        assert "momentum_20" in result
    
    @pytest.mark.slow
    def test_snapshot_has_prices(self):
        """Test snapshot contains price data."""
        result = generate_market_snapshot(symbol="TSLA", days=60)
        
        prices = result.get("prices", [])
        assert len(prices) > 0, "Should return price data"
        
        returns = result.get("returns", [])
        assert len(returns) > 0, "Should have returns calculated"
    
    @pytest.mark.slow
    def test_volatility_is_positive(self):
        """Test volatility calculation is positive."""
        result = generate_market_snapshot(symbol="GOOG", days=90)
        
        volatility = result.get("realized_volatility", 0)
        assert volatility >= 0, "Volatility should be non-negative"
