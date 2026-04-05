"""
Portfolio allocation and position sizing tools.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Position:
    """Represents a portfolio position."""
    symbol: str
    target_weight: float
    confidence: float
    action: str  # 'buy', 'short', 'hold'
    risk_level: str


def calculate_position_size(
    account_size: float,
    target_weight: float,
    max_position_size: float = 0.05,
) -> dict:
    """
    Calculate position size for a given target weight.
    
    Args:
        account_size: total portfolio value
        target_weight: target weight as percentage
        max_position_size: maximum allowed position size
        
    Returns:
        dict with position sizing details
    """
    notional_value = account_size * target_weight
    
    # Apply max position size constraint
    max_allowed = account_size * max_position_size
    constrained_notional = min(notional_value, max_allowed)
    constrained_weight = constrained_notional / account_size if account_size > 0 else 0
    
    return {
        "account_size": account_size,
        "target_weight": float(target_weight),
        "target_notional": float(notional_value),
        "max_allowed": float(max_allowed),
        "constrained_weight": float(constrained_weight),
        "constrained_notional": float(constrained_notional),
        "position_size": float(constrained_notional),  # ← Key that tests expect
        "position_size_pct": float((constrained_notional / account_size * 100) if account_size > 0 else 0),
    }


def calculate_kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
) -> float:
    """
    Calculate Kelly Criterion for position sizing.
    
    Args:
        win_rate: probability of winning (0-1)
        avg_win: average profit on winning trade
        avg_loss: average loss on losing trade
        
    Returns:
        optimal fraction of portfolio to risk
    """
    if avg_loss <= 0:
        return 0.0
    
    kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
    
    # Apply safety factor (usually 0.25x - 0.5x Kelly)
    safe_kelly = kelly * 0.25
    
    # Cap between 0 and 0.25 for safety
    return max(0.0, min(0.25, safe_kelly))


def calculate_risk_reward_ratio(
    entry_price: float,
    stop_loss: float,
    take_profit: float,
) -> dict:
    """
    Calculate risk/reward ratio for a trade.
    
    Args:
        entry_price: entry price for the trade
        stop_loss: stop loss price
        take_profit: take profit price
        
    Returns:
        dict with risk/reward metrics
    """
    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)
    
    ratio = reward / risk if risk > 0 else 0
    
    return {
        "entry_price": float(entry_price),
        "stop_loss": float(stop_loss),
        "take_profit": float(take_profit),
        "risk_points": float(risk),
        "reward_points": float(reward),
        "risk_reward_ratio": float(ratio),
        "risk_pct": float((risk / entry_price * 100) if entry_price > 0 else 0),
        "reward_pct": float((reward / entry_price * 100) if entry_price > 0 else 0),
    }


def optimize_portfolio_weights(
    positions: List[Dict],
    account_size: float,
    constraints: Dict = None,
) -> dict:
    """
    Optimize portfolio weights based on confidence and risk levels.
    
    Args:
        positions: list of position dicts with symbol, target_weight, confidence, risk_level
        account_size: total portfolio value
        constraints: dict with min/max weights per risk level
        
    Returns:
        dict with optimized allocation
    """
    if constraints is None:
        constraints = {
            "low": {"max": 0.10},
            "medium": {"max": 0.08},
            "high": {"max": 0.05},
        }
    
    optimized_positions = []
    total_weight = 0
    
    # Weight positions by confidence
    for pos in positions:
        risk_level = pos.get("risk_level", "medium")
        max_weight = constraints.get(risk_level, {}).get("max", 0.05)
        
        # Adjust weight by confidence
        adjusted_weight = pos.get("target_weight", 0) * pos.get("confidence", 0.5)
        constrained_weight = min(adjusted_weight, max_weight)
        
        optimized_positions.append({
            "symbol": pos.get("symbol"),
            "action": pos.get("action"),
            "confidence": pos.get("confidence"),
            "risk_level": risk_level,
            "target_weight": pos.get("target_weight"),
            "adjusted_weight": float(adjusted_weight),
            "constrained_weight": float(constrained_weight),
            "notional_value": float(account_size * constrained_weight),
        })
        
        total_weight += constrained_weight
    
    # Normalize if total exceeds 1.0
    if total_weight > 1.0:
        for pos in optimized_positions:
            pos["constrained_weight"] /= total_weight
            pos["notional_value"] = account_size * pos["constrained_weight"]
        total_weight = 1.0
    
    # Cash allocation
    cash_weight = 1.0 - total_weight
    
    return {
        "positions": optimized_positions,
        "total_allocation": float(total_weight),
        "cash_weight": float(cash_weight),
        "cash_value": float(account_size * cash_weight),
        "fully_invested": cash_weight < 0.05,
    }


def calculate_portfolio_metrics(positions: List[Dict], prices: Dict[str, float]) -> dict:
    """
    Calculate portfolio-level metrics.
    
    Args:
        positions: list of position dicts with symbol, notional_value, etc.
        prices: dict mapping symbols to current prices
        
    Returns:
        dict with portfolio metrics
    """
    total_notional = 0
    long_notional = 0
    short_notional = 0
    
    for pos in positions:
        notional = pos.get("notional_value", 0)
        action = pos.get("action", "hold")
        
        total_notional += abs(notional)
        if action == "buy":
            long_notional += notional
        elif action == "short":
            short_notional += notional
    
    gross_leverage = (long_notional + short_notional) / total_notional if total_notional > 0 else 0
    net_leverage = (long_notional - short_notional) / total_notional if total_notional > 0 else 0
    
    return {
        "total_notional": float(total_notional),
        "long_notional": float(long_notional),
        "short_notional": float(short_notional),
        "gross_leverage": float(gross_leverage),
        "net_leverage": float(net_leverage),
        "position_count": len(positions),
    }
