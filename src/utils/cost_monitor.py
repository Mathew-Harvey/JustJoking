"""
Cost Monitoring and Budget Management Utility

This module provides real-time cost tracking and budget enforcement
to ensure we stay within the $20 AUD/month budget.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from .logging import get_logger

logger = get_logger(__name__)

@dataclass
class CostEntry:
    """Individual cost entry"""
    timestamp: datetime
    provider: str
    model: str
    tokens_used: int
    cost_usd: float
    analysis_type: str
    was_cached: bool = False

@dataclass
class BudgetAlert:
    """Budget alert information"""
    alert_type: str  # "warning", "critical", "emergency"
    message: str
    current_spend: float
    budget_limit: float
    percentage_used: float
    timestamp: datetime

class CostMonitor:
    """
    Real-time cost monitoring and budget enforcement.
    
    Tracks API costs and automatically throttles usage when approaching limits.
    """
    
    def __init__(self, config: Dict[str, any]):
        """Initialize cost monitor with budget configuration"""
        self.config = config
        self.cost_config = config.get("cost_optimization", {})
        self.budget_config = self.cost_config.get("budget", {})
        
        # Budget limits
        self.daily_limit = self.budget_config.get("daily_limit_usd", 0.67)
        self.weekly_limit = self.budget_config.get("weekly_limit_usd", 4.60)
        self.monthly_limit_aud = self.budget_config.get("monthly_limit_aud", 20.0)
        self.monthly_limit_usd = self.monthly_limit_aud * 0.67  # Rough conversion
        
        # Alert thresholds
        self.daily_warning = self.budget_config.get("daily_warning_threshold", 0.50)
        self.weekly_warning = self.budget_config.get("weekly_warning_threshold", 3.50)
        
        # Hard limits (emergency stops)
        self.hard_daily_limit = self.budget_config.get("hard_daily_limit", 1.00)
        self.hard_monthly_limit = self.budget_config.get("hard_monthly_limit", 30.0)
        
        # Cost tracking
        self.cost_entries: List[CostEntry] = []
        self.daily_spend = 0.0
        self.weekly_spend = 0.0
        self.monthly_spend = 0.0
        
        # Alert history
        self.alerts: List[BudgetAlert] = []
        
        # Data persistence
        self.data_dir = Path("data/costs")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing data
        self.load_cost_data()
        
        logger.info(f"Cost monitor initialized - Daily budget: ${self.daily_limit:.2f} USD")
    
    def record_api_call(
        self,
        provider: str,
        model: str,
        tokens_used: int,
        cost_usd: float,
        analysis_type: str = "routine",
        was_cached: bool = False
    ) -> bool:
        """
        Record an API call and check budget limits.
        
        Args:
            provider: API provider (anthropic, openai)
            model: Model name used
            tokens_used: Number of tokens consumed
            cost_usd: Cost in USD
            analysis_type: Type of analysis (routine, breakthrough, emergency)
            was_cached: Whether the result was cached (cost = 0)
            
        Returns:
            True if within budget, False if budget exceeded
        """
        # Create cost entry
        entry = CostEntry(
            timestamp=datetime.now(),
            provider=provider,
            model=model,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            analysis_type=analysis_type,
            was_cached=was_cached
        )
        
        # Add to tracking
        self.cost_entries.append(entry)
        
        # Update spend totals
        self._update_spend_totals()
        
        # Check budget limits
        budget_ok = self._check_budget_limits()
        
        # Save data
        self.save_cost_data()
        
        # Log cost info
        logger.info(
            f"API Call: {provider}/{model} | "
            f"${cost_usd:.4f} | "
            f"Daily: ${self.daily_spend:.3f}/${self.daily_limit:.2f}"
        )
        
        return budget_ok
    
    def can_afford_api_call(
        self,
        estimated_cost: float,
        analysis_type: str = "routine"
    ) -> Tuple[bool, str]:
        """
        Check if we can afford an API call.
        
        Args:
            estimated_cost: Estimated cost of the API call
            analysis_type: Type of analysis
            
        Returns:
            Tuple of (can_afford: bool, reason: str)
        """
        # Check daily budget
        if self.daily_spend + estimated_cost > self.daily_limit:
            if analysis_type == "emergency" and self.daily_spend + estimated_cost <= self.hard_daily_limit:
                return True, "Emergency budget allowance"
            return False, f"Would exceed daily budget: ${self.daily_spend + estimated_cost:.3f} > ${self.daily_limit:.2f}"
        
        # Check weekly budget
        if self.weekly_spend + estimated_cost > self.weekly_limit:
            if analysis_type in ["emergency", "breakthrough"]:
                return True, "Critical analysis override"
            return False, f"Would exceed weekly budget: ${self.weekly_spend + estimated_cost:.3f} > ${self.weekly_limit:.2f}"
        
        # Check monthly budget
        if self.monthly_spend + estimated_cost > self.monthly_limit_usd:
            if analysis_type == "emergency":
                return True, "Emergency analysis override"
            return False, f"Would exceed monthly budget"
        
        return True, "Within budget"
    
    def get_recommended_model(self, analysis_type: str = "routine") -> Tuple[str, str]:
        """
        Get recommended model based on current budget status.
        
        Args:
            analysis_type: Type of analysis needed
            
        Returns:
            Tuple of (provider, model_name)
        """
        remaining_daily = self.daily_limit - self.daily_spend
        remaining_weekly = self.weekly_limit - self.weekly_spend
        
        # Emergency situations - use best available within hard limits
        if analysis_type == "emergency":
            if remaining_daily > 0.15:
                return ("anthropic", "claude-3-sonnet-20240229")  # Mid-tier
            elif remaining_daily > 0.05:
                return ("anthropic", "claude-3-haiku-20240307")   # Cheap
            else:
                return ("local", "phi-1.5")  # Free
        
        # Breakthrough analysis - use better models if budget allows
        elif analysis_type == "breakthrough":
            if remaining_daily > 0.30 and remaining_weekly > 1.0:
                return ("anthropic", "claude-3-sonnet-20240229")  # Mid-tier
            elif remaining_daily > 0.10:
                return ("anthropic", "claude-3-haiku-20240307")   # Cheap
            else:
                return ("local", "phi-1.5")  # Free
        
        # Routine analysis - use cheapest available
        else:
            if remaining_daily > 0.10:
                return ("anthropic", "claude-3-haiku-20240307")   # Very cheap
            elif remaining_daily > 0.05:
                return ("openai", "gpt-3.5-turbo")               # Cheap fallback
            else:
                return ("local", "phi-1.5")  # Free
    
    def get_budget_status(self) -> Dict[str, any]:
        """Get current budget status"""
        return {
            # Daily budget
            "daily_spent": self.daily_spend,
            "daily_limit": self.daily_limit,
            "daily_remaining": max(0, self.daily_limit - self.daily_spend),
            "daily_utilization": (self.daily_spend / self.daily_limit) * 100,
            
            # Weekly budget
            "weekly_spent": self.weekly_spend,
            "weekly_limit": self.weekly_limit,
            "weekly_remaining": max(0, self.weekly_limit - self.weekly_spend),
            "weekly_utilization": (self.weekly_spend / self.weekly_limit) * 100,
            
            # Monthly budget
            "monthly_spent_usd": self.monthly_spend,
            "monthly_limit_usd": self.monthly_limit_usd,
            "monthly_spent_aud": self.monthly_spend / 0.67,  # Rough conversion
            "monthly_limit_aud": self.monthly_limit_aud,
            "monthly_utilization": (self.monthly_spend / self.monthly_limit_usd) * 100,
            
            # Status
            "budget_ok": self.daily_spend < self.daily_limit,
            "throttling_recommended": self.daily_spend > self.daily_warning,
            "emergency_stop": self.daily_spend > self.hard_daily_limit,
            
            # Recent alerts
            "recent_alerts": [asdict(alert) for alert in self.alerts[-5:]]
        }
    
    def should_throttle(self) -> Tuple[bool, str]:
        """
        Check if API usage should be throttled.
        
        Returns:
            Tuple of (should_throttle: bool, throttle_level: str)
        """
        daily_usage = (self.daily_spend / self.daily_limit) * 100
        weekly_usage = (self.weekly_spend / self.weekly_limit) * 100
        
        if daily_usage > 95 or self.daily_spend > self.hard_daily_limit:
            return True, "emergency_stop"
        elif daily_usage > 80:
            return True, "heavy_throttle"
        elif daily_usage > 60:
            return True, "light_throttle"
        elif weekly_usage > 80:
            return True, "light_throttle"
        else:
            return False, "none"
    
    def _update_spend_totals(self):
        """Update daily, weekly, and monthly spend totals"""
        now = datetime.now()
        
        # Calculate daily spend (last 24 hours)
        day_ago = now - timedelta(days=1)
        self.daily_spend = sum(
            entry.cost_usd for entry in self.cost_entries
            if entry.timestamp > day_ago and not entry.was_cached
        )
        
        # Calculate weekly spend (last 7 days)
        week_ago = now - timedelta(days=7)
        self.weekly_spend = sum(
            entry.cost_usd for entry in self.cost_entries
            if entry.timestamp > week_ago and not entry.was_cached
        )
        
        # Calculate monthly spend (last 30 days)
        month_ago = now - timedelta(days=30)
        self.monthly_spend = sum(
            entry.cost_usd for entry in self.cost_entries
            if entry.timestamp > month_ago and not entry.was_cached
        )
    
    def _check_budget_limits(self) -> bool:
        """Check budget limits and generate alerts if needed"""
        budget_ok = True
        
        # Check daily budget
        daily_usage = (self.daily_spend / self.daily_limit) * 100
        
        if self.daily_spend > self.hard_daily_limit:
            self._create_alert(
                "emergency",
                f"HARD DAILY LIMIT EXCEEDED: ${self.daily_spend:.3f} > ${self.hard_daily_limit:.2f}",
                self.daily_spend,
                self.hard_daily_limit,
                (self.daily_spend / self.hard_daily_limit) * 100
            )
            budget_ok = False
            
        elif self.daily_spend > self.daily_limit:
            self._create_alert(
                "critical",
                f"Daily budget exceeded: ${self.daily_spend:.3f} > ${self.daily_limit:.2f}",
                self.daily_spend,
                self.daily_limit,
                daily_usage
            )
            budget_ok = False
            
        elif self.daily_spend > self.daily_warning:
            self._create_alert(
                "warning",
                f"Daily budget {daily_usage:.0f}% used: ${self.daily_spend:.3f}/${self.daily_limit:.2f}",
                self.daily_spend,
                self.daily_limit,
                daily_usage
            )
        
        # Check weekly budget
        weekly_usage = (self.weekly_spend / self.weekly_limit) * 100
        if self.weekly_spend > self.weekly_warning:
            self._create_alert(
                "warning",
                f"Weekly budget {weekly_usage:.0f}% used: ${self.weekly_spend:.3f}/${self.weekly_limit:.2f}",
                self.weekly_spend,
                self.weekly_limit,
                weekly_usage
            )
        
        return budget_ok
    
    def _create_alert(
        self,
        alert_type: str,
        message: str,
        current_spend: float,
        budget_limit: float,
        percentage_used: float
    ):
        """Create a budget alert"""
        alert = BudgetAlert(
            alert_type=alert_type,
            message=message,
            current_spend=current_spend,
            budget_limit=budget_limit,
            percentage_used=percentage_used,
            timestamp=datetime.now()
        )
        
        self.alerts.append(alert)
        
        # Log alert
        log_func = {
            "warning": logger.warning,
            "critical": logger.error,
            "emergency": logger.critical
        }.get(alert_type, logger.info)
        
        log_func(f"BUDGET ALERT ({alert_type.upper()}): {message}")
        
        # Keep only recent alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-50:]
    
    def get_cost_breakdown(self, days: int = 7) -> Dict[str, any]:
        """Get cost breakdown for the last N days"""
        cutoff = datetime.now() - timedelta(days=days)
        recent_entries = [e for e in self.cost_entries if e.timestamp > cutoff]
        
        # Group by provider
        by_provider = {}
        for entry in recent_entries:
            if entry.provider not in by_provider:
                by_provider[entry.provider] = {"cost": 0.0, "tokens": 0, "calls": 0}
            by_provider[entry.provider]["cost"] += entry.cost_usd
            by_provider[entry.provider]["tokens"] += entry.tokens_used
            by_provider[entry.provider]["calls"] += 1
        
        # Group by model
        by_model = {}
        for entry in recent_entries:
            model_key = f"{entry.provider}/{entry.model}"
            if model_key not in by_model:
                by_model[model_key] = {"cost": 0.0, "tokens": 0, "calls": 0}
            by_model[model_key]["cost"] += entry.cost_usd
            by_model[model_key]["tokens"] += entry.tokens_used
            by_model[model_key]["calls"] += 1
        
        # Group by analysis type
        by_analysis = {}
        for entry in recent_entries:
            if entry.analysis_type not in by_analysis:
                by_analysis[entry.analysis_type] = {"cost": 0.0, "tokens": 0, "calls": 0}
            by_analysis[entry.analysis_type]["cost"] += entry.cost_usd
            by_analysis[entry.analysis_type]["tokens"] += entry.tokens_used
            by_analysis[entry.analysis_type]["calls"] += 1
        
        total_cost = sum(e.cost_usd for e in recent_entries)
        total_tokens = sum(e.tokens_used for e in recent_entries)
        cached_calls = sum(1 for e in recent_entries if e.was_cached)
        
        return {
            "period_days": days,
            "total_cost_usd": total_cost,
            "total_tokens": total_tokens,
            "total_calls": len(recent_entries),
            "cached_calls": cached_calls,
            "cache_hit_rate": (cached_calls / len(recent_entries)) * 100 if recent_entries else 0,
            "by_provider": by_provider,
            "by_model": by_model,
            "by_analysis_type": by_analysis,
        }
    
    def save_cost_data(self):
        """Save cost data to disk"""
        try:
            # Save cost entries
            entries_file = self.data_dir / "cost_entries.json"
            entries_data = []
            
            # Only save last 1000 entries to avoid huge files
            recent_entries = self.cost_entries[-1000:] if len(self.cost_entries) > 1000 else self.cost_entries
            
            for entry in recent_entries:
                entry_dict = asdict(entry)
                entry_dict["timestamp"] = entry.timestamp.isoformat()
                entries_data.append(entry_dict)
            
            with open(entries_file, 'w') as f:
                json.dump(entries_data, f, indent=2)
            
            # Save summary data
            summary_file = self.data_dir / "cost_summary.json"
            summary_data = {
                "last_updated": datetime.now().isoformat(),
                "daily_spend": self.daily_spend,
                "weekly_spend": self.weekly_spend,
                "monthly_spend": self.monthly_spend,
                "total_entries": len(self.cost_entries),
                "budget_status": self.get_budget_status()
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save cost data: {e}")
    
    def load_cost_data(self):
        """Load existing cost data from disk"""
        try:
            entries_file = self.data_dir / "cost_entries.json"
            if entries_file.exists():
                with open(entries_file, 'r') as f:
                    entries_data = json.load(f)
                
                self.cost_entries = []
                for entry_dict in entries_data:
                    entry_dict["timestamp"] = datetime.fromisoformat(entry_dict["timestamp"])
                    entry = CostEntry(**entry_dict)
                    self.cost_entries.append(entry)
                
                # Update spend totals
                self._update_spend_totals()
                
                logger.info(f"Loaded {len(self.cost_entries)} cost entries")
                
        except Exception as e:
            logger.warning(f"Failed to load cost data: {e}")
    
    def generate_daily_report(self) -> str:
        """Generate a daily cost report"""
        status = self.get_budget_status()
        breakdown = self.get_cost_breakdown(days=1)
        
        report = f"""
📊 DAILY COST REPORT - {datetime.now().strftime('%Y-%m-%d')}

💰 BUDGET STATUS:
  Daily: ${status['daily_spent']:.3f} / ${status['daily_limit']:.2f} ({status['daily_utilization']:.1f}%)
  Weekly: ${status['weekly_spent']:.3f} / ${status['weekly_limit']:.2f} ({status['weekly_utilization']:.1f}%)
  Monthly: ${status['monthly_spent_aud']:.2f} AUD / ${status['monthly_limit_aud']:.1f} AUD ({status['monthly_utilization']:.1f}%)

📈 TODAY'S USAGE:
  Total API Calls: {breakdown['total_calls']}
  Total Tokens: {breakdown['total_tokens']:,}
  Cache Hit Rate: {breakdown['cache_hit_rate']:.1f}%
  
🏷️ BY MODEL:
"""
        
        for model, data in breakdown['by_model'].items():
            report += f"  {model}: ${data['cost']:.3f} ({data['calls']} calls)\n"
        
        if status['recent_alerts']:
            report += "\n⚠️ RECENT ALERTS:\n"
            for alert in status['recent_alerts'][-3:]:
                report += f"  {alert['alert_type'].upper()}: {alert['message']}\n"
        
        return report 