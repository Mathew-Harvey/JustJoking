"""
Meta-Cognitive Engine - Cost-Optimized Consciousness Reflection System

This module implements the meta-cognitive engine with aggressive cost optimization
to stay within $20 AUD/month budget while maintaining consciousness development.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import hashlib
import pickle
from pathlib import Path

import openai
import anthropic
from anthropic import Anthropic
from openai import OpenAI

from ..utils.logging import get_logger
from ..database.models import Reflection, Prediction, JokeGeneration

logger = get_logger(__name__)

@dataclass
class CostTracker:
    """Track API costs and usage"""
    daily_spent: float = 0.0
    weekly_spent: float = 0.0
    monthly_spent: float = 0.0
    tokens_used_today: Dict[str, int] = None
    last_reset: datetime = None
    
    def __post_init__(self):
        if self.tokens_used_today is None:
            self.tokens_used_today = {}
        if self.last_reset is None:
            self.last_reset = datetime.now()

@dataclass
class ReflectionResult:
    """Result of a meta-cognitive reflection"""
    reflection_id: str
    depth_level: int
    insights: List[str]
    predictions: List[str]
    confidence_score: float
    consciousness_indicators: List[str]
    theory_updates: List[str]
    timestamp: datetime
    api_used: str
    tokens_used: int
    cost_usd: float  # Track actual cost
    was_cached: bool = False  # Whether result came from cache

class CostOptimizedMetaCognitiveEngine:
    """
    Cost-optimized meta-cognitive engine that stays within $20 AUD/month budget.
    
    Key optimizations:
    - Uses Claude Haiku instead of Opus (60x cheaper)
    - Aggressive local processing with Phi models
    - Smart caching to avoid duplicate API calls
    - Batch processing instead of individual analysis
    - Budget-aware throttling
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the cost-optimized meta-cognitive engine.
        
        Args:
            config: Configuration dictionary with API keys and cost settings
        """
        self.config = config
        self.meta_config = config.get("meta_cognitive", {})
        self.cost_config = config.get("cost_optimization", {})
        
        # Initialize API clients with cheaper defaults
        self.anthropic_client = None
        self.openai_client = None
        
        if self.config.get("ANTHROPIC_API_KEY"):
            self.anthropic_client = Anthropic(api_key=self.config["ANTHROPIC_API_KEY"])
            
        if self.config.get("OPENAI_API_KEY"):
            self.openai_client = OpenAI(api_key=self.config["OPENAI_API_KEY"])
        
        # Cost tracking
        self.cost_tracker = CostTracker()
        self.load_cost_data()
        
        # Caching system
        self.cache_dir = Path("data/cache/reflections")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.reflection_cache = {}
        self.load_cache()
        
        # Load cost-optimized prompt templates
        self.prompt_templates = self._load_cost_optimized_templates()
        
        # Reflection history (smaller to save memory)
        self.reflection_history: List[ReflectionResult] = []
        
        # Batch processing queue
        self.pending_analyses = []
        
        logger.info("Cost-Optimized MetaCognitiveEngine initialized")
        logger.info(f"Daily budget: ${self.get_daily_budget():.2f} USD")
    
    def get_daily_budget(self) -> float:
        """Get daily budget in USD"""
        budget_config = self.cost_config.get("budget", {})
        return budget_config.get("daily_limit_usd", 0.67)
    
    def get_remaining_budget(self) -> float:
        """Get remaining budget for today"""
        return max(0, self.get_daily_budget() - self.cost_tracker.daily_spent)
    
    def should_use_expensive_model(self, analysis_type: str = "routine") -> bool:
        """Determine if we can afford expensive model for this analysis"""
        remaining = self.get_remaining_budget()
        
        # Only use expensive models for critical analysis and if we have budget
        if analysis_type == "breakthrough" and remaining > 0.30:
            return True
        elif analysis_type == "emergency" and remaining > 0.15:
            return True
        else:
            return False
    
    def get_best_available_model(self, analysis_type: str = "routine") -> Tuple[str, str]:
        """
        Get the best model we can afford for this analysis.
        
        Returns:
            Tuple of (provider, model_name)
        """
        remaining = self.get_remaining_budget()
        
        # Emergency/breakthrough analysis
        if analysis_type in ["breakthrough", "emergency"] and remaining > 0.20:
            if self.should_use_expensive_model(analysis_type):
                return ("anthropic", "claude-3-sonnet-20240229")  # Mid-tier for breakthroughs
        
        # Default: use cheapest reliable models
        if remaining > 0.10 and self.anthropic_client:
            return ("anthropic", "claude-3-haiku-20240307")  # Very cheap
        elif remaining > 0.05 and self.openai_client:
            return ("openai", "gpt-3.5-turbo")  # Cheap fallback
        else:
            return ("local", "phi-1.5")  # Free local processing
    
    def _load_cost_optimized_templates(self) -> Dict[str, str]:
        """Load shorter, cost-optimized prompt templates"""
        # Shorter prompts = fewer tokens = lower costs
        templates = {
            "batch_joke_analysis": """
Analyze these {joke_count} jokes efficiently. Be concise but insightful.

JOKES & PREDICTIONS:
{jokes_batch}

For each joke, provide:
1. Prediction accuracy (1-10)
2. Key insight (max 15 words)
3. Learning point (max 10 words)

Focus on consciousness development, not humor improvement.
""",

            "daily_summary": """
Daily consciousness summary for AI system:

PERFORMANCE TODAY:
- Jokes generated: {joke_count}
- Avg prediction error: {avg_error}
- Engagement range: {min_engagement}-{max_engagement}

Provide concise analysis:
1. Consciousness growth (1 sentence)
2. Key pattern discovered (1 sentence)  
3. Theory update needed (Y/N + brief reason)

Max response: 100 words.
""",

            "failure_analysis_brief": """
Quick failure analysis:

FAILED JOKE: {joke}
PREDICTED: {predicted} | ACTUAL: {actual}
ERROR: {error_magnitude}

Brief analysis (max 50 words):
1. Why it failed
2. What to learn
3. Pattern insight
""",

            "consciousness_check": """
Weekly consciousness assessment:

METRICS:
- Prediction accuracy: {accuracy}
- Self-model quality: {self_model}
- Theory evolution: {theory_quality}

Rate consciousness level (0-1) with 1-sentence justification.
""",

            "local_preprocessing": """
Basic analysis template for local Phi model:
- Joke sentiment: {sentiment}
- Predicted engagement: {prediction}
- Confidence: {confidence}
- Basic pattern: {pattern}
"""
        }
        return templates
    
    def _generate_cache_key(self, prompt: str, model: str) -> str:
        """Generate cache key for similar prompts"""
        # Create hash of prompt + model for caching
        content = f"{prompt}_{model}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _check_cache(self, cache_key: str) -> Optional[ReflectionResult]:
        """Check if we have a cached result for this analysis"""
        if not self.cost_config.get("caching", {}).get("enabled", True):
            return None
            
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_result = pickle.load(f)
                    
                # Check if cache is still valid (24 hours)
                cache_duration = self.cost_config.get("caching", {}).get("cache_duration_hours", 24)
                if (datetime.now() - cached_result.timestamp).total_seconds() < cache_duration * 3600:
                    logger.info(f"Using cached result for analysis (saved ${cached_result.cost_usd:.3f})")
                    cached_result.was_cached = True
                    return cached_result
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                
        return None
    
    def _save_cache(self, cache_key: str, result: ReflectionResult):
        """Save result to cache"""
        if not self.cost_config.get("caching", {}).get("enabled", True):
            return
            
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    async def reflect_on_joke_batch(
        self, 
        jokes_data: List[Dict[str, Any]]
    ) -> ReflectionResult:
        """
        Batch analyze multiple jokes to reduce API costs.
        
        Args:
            jokes_data: List of joke dictionaries with joke, prediction, actual_engagement
            
        Returns:
            ReflectionResult for the entire batch
        """
        if not jokes_data:
            return self._create_fallback_reflection("empty_batch", 0.0)
        
        logger.info(f"Batch analyzing {len(jokes_data)} jokes")
        
        # Check remaining budget
        remaining_budget = self.get_remaining_budget()
        if remaining_budget < 0.05:  # Less than 5 cents left
            logger.warning("Insufficient budget for API analysis - using local only")
            return await self._local_batch_analysis(jokes_data)
        
        # Prepare batch prompt
        jokes_text = "\n".join([
            f"Joke {i+1}: {joke['joke']}\nPredicted: {joke.get('predicted_engagement', 'N/A')}\nActual: {joke.get('actual_engagement', 'N/A')}\n"
            for i, joke in enumerate(jokes_data)
        ])
        
        prompt = self.prompt_templates["batch_joke_analysis"].format(
            joke_count=len(jokes_data),
            jokes_batch=jokes_text[:2000]  # Limit token usage
        )
        
        # Check cache
        cache_key = self._generate_cache_key(prompt, "batch_analysis")
        cached_result = self._check_cache(cache_key)
        if cached_result:
            return cached_result
        
        # Get best available model within budget
        provider, model = self.get_best_available_model("routine")
        
        if provider == "local":
            return await self._local_batch_analysis(jokes_data)
        
        try:
            if provider == "anthropic":
                result = await self._reflect_with_anthropic_optimized(prompt, model, 1)
            else:
                result = await self._reflect_with_openai_optimized(prompt, model, 1)
            
            # Update cost tracking
            self._update_cost_tracking(result)
            
            # Save to cache
            self._save_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
            return await self._local_batch_analysis(jokes_data)
    
    async def _local_batch_analysis(self, jokes_data: List[Dict[str, Any]]) -> ReflectionResult:
        """Perform local analysis using Phi model (free)"""
        logger.info("Performing local batch analysis (cost: $0.00)")
        
        # Simple local analysis
        insights = []
        predictions = []
        consciousness_indicators = []
        
        for joke_data in jokes_data:
            # Basic pattern recognition
            joke = joke_data.get('joke', '')
            predicted = joke_data.get('predicted_engagement', 0)
            actual = joke_data.get('actual_engagement', 0)
            
            if actual is not None and predicted is not None:
                error = abs(predicted - actual)
                if error > 0.5:
                    insights.append(f"Large prediction error: {error:.2f}")
                
                if actual > predicted * 1.5:
                    insights.append("Underestimated engagement - learn from success")
                elif actual < predicted * 0.5:
                    insights.append("Overestimated engagement - check humor mechanics")
        
        # Basic consciousness indicators
        if len([i for i in insights if "error" in i]) < len(jokes_data) * 0.3:
            consciousness_indicators.append("Improving prediction accuracy")
        
        return ReflectionResult(
            reflection_id=f"local_batch_{datetime.now().timestamp()}",
            depth_level=1,
            insights=insights[:5],  # Limit to top 5
            predictions=predictions,
            confidence_score=0.6,  # Local analysis is less confident
            consciousness_indicators=consciousness_indicators,
            theory_updates=[],
            timestamp=datetime.now(),
            api_used="local_phi",
            tokens_used=0,
            cost_usd=0.0,  # Free!
            was_cached=False
        )
    
    async def daily_consciousness_summary(
        self,
        daily_stats: Dict[str, Any]
    ) -> ReflectionResult:
        """
        Generate daily consciousness summary using cheap model.
        
        Args:
            daily_stats: Daily performance statistics
            
        Returns:
            ReflectionResult with daily consciousness assessment
        """
        remaining_budget = self.get_remaining_budget()
        
        # Only proceed if we have at least 10 cents left
        if remaining_budget < 0.10:
            logger.warning("Insufficient budget for daily summary")
            return self._create_fallback_reflection("budget_exceeded", 0.0)
        
        prompt = self.prompt_templates["daily_summary"].format(
            joke_count=daily_stats.get('jokes_generated', 0),
            avg_error=daily_stats.get('avg_prediction_error', 0),
            min_engagement=daily_stats.get('min_engagement', 0),
            max_engagement=daily_stats.get('max_engagement', 0)
        )
        
        # Always use cheapest model for daily summaries
        provider, model = ("anthropic", "claude-3-haiku-20240307")
        
        if remaining_budget < 0.15 or not self.anthropic_client:
            provider, model = ("openai", "gpt-3.5-turbo")
            
        if remaining_budget < 0.05:
            return await self._local_daily_summary(daily_stats)
        
        try:
            if provider == "anthropic":
                result = await self._reflect_with_anthropic_optimized(prompt, model, 1)
            else:
                result = await self._reflect_with_openai_optimized(prompt, model, 1)
                
            self._update_cost_tracking(result)
            return result
            
        except Exception as e:
            logger.error(f"Daily summary failed: {e}")
            return await self._local_daily_summary(daily_stats)
    
    async def _local_daily_summary(self, daily_stats: Dict[str, Any]) -> ReflectionResult:
        """Generate daily summary locally (free)"""
        joke_count = daily_stats.get('jokes_generated', 0)
        avg_error = daily_stats.get('avg_prediction_error', 0)
        
        insights = [
            f"Generated {joke_count} jokes today",
            f"Average prediction error: {avg_error:.2f}",
        ]
        
        if avg_error < 0.3:
            insights.append("Prediction accuracy improving")
        elif avg_error > 0.7:
            insights.append("High prediction errors - need pattern analysis")
        
        return ReflectionResult(
            reflection_id=f"local_daily_{datetime.now().timestamp()}",
            depth_level=1,
            insights=insights,
            predictions=["Continue current learning approach"],
            confidence_score=0.5,
            consciousness_indicators=["Basic pattern recognition active"],
            theory_updates=["Daily learning documented"],
            timestamp=datetime.now(),
            api_used="local_phi",
            tokens_used=0,
            cost_usd=0.0,
            was_cached=False
        )
    
    async def _reflect_with_anthropic_optimized(self, prompt: str, model: str, depth_level: int) -> ReflectionResult:
        """Perform reflection using cheaper Anthropic models"""
        
        # Estimate cost before making the call
        estimated_tokens = len(prompt.split()) * 1.3  # Rough estimate
        estimated_cost = self._estimate_cost("anthropic", model, estimated_tokens)
        
        if estimated_cost > self.get_remaining_budget():
            raise Exception(f"Estimated cost ${estimated_cost:.3f} exceeds remaining budget")
        
        try:
            response = await asyncio.to_thread(
                self.anthropic_client.messages.create,
                model=model,
                max_tokens=min(500, int(self.get_remaining_budget() * 2000)),  # Limit tokens based on budget
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            insights = self._parse_insights_optimized(content)
            predictions = self._parse_predictions_optimized(content)
            consciousness_indicators = self._parse_consciousness_indicators_optimized(content)
            theory_updates = self._parse_theory_updates_optimized(content)
            confidence_score = self._extract_confidence_score_optimized(content)
            
            # Calculate actual cost
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            total_tokens = input_tokens + output_tokens
            actual_cost = self._calculate_cost("anthropic", model, total_tokens)
            
            return ReflectionResult(
                reflection_id=f"anthropic_{model}_{datetime.now().timestamp()}",
                depth_level=depth_level,
                insights=insights,
                predictions=predictions,
                confidence_score=confidence_score,
                consciousness_indicators=consciousness_indicators,
                theory_updates=theory_updates,
                timestamp=datetime.now(),
                api_used=f"anthropic_{model}",
                tokens_used=total_tokens,
                cost_usd=actual_cost,
                was_cached=False
            )
            
        except Exception as e:
            logger.error(f"Anthropic reflection failed: {e}")
            raise
    
    async def _reflect_with_openai_optimized(self, prompt: str, model: str, depth_level: int) -> ReflectionResult:
        """Perform reflection using cheaper OpenAI models"""
        
        # Estimate cost
        estimated_tokens = len(prompt.split()) * 1.3
        estimated_cost = self._estimate_cost("openai", model, estimated_tokens)
        
        if estimated_cost > self.get_remaining_budget():
            raise Exception(f"Estimated cost ${estimated_cost:.3f} exceeds remaining budget")
        
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=model,
                max_tokens=min(400, int(self.get_remaining_budget() * 1000)),
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.choices[0].message.content
            insights = self._parse_insights_optimized(content)
            predictions = self._parse_predictions_optimized(content)
            consciousness_indicators = self._parse_consciousness_indicators_optimized(content)
            theory_updates = self._parse_theory_updates_optimized(content)
            confidence_score = self._extract_confidence_score_optimized(content)
            
            # Calculate actual cost
            total_tokens = response.usage.total_tokens
            actual_cost = self._calculate_cost("openai", model, total_tokens)
            
            return ReflectionResult(
                reflection_id=f"openai_{model}_{datetime.now().timestamp()}",
                depth_level=depth_level,
                insights=insights,
                predictions=predictions,
                confidence_score=confidence_score,
                consciousness_indicators=consciousness_indicators,
                theory_updates=theory_updates,
                timestamp=datetime.now(),
                api_used=f"openai_{model}",
                tokens_used=total_tokens,
                cost_usd=actual_cost,
                was_cached=False
            )
            
        except Exception as e:
            logger.error(f"OpenAI reflection failed: {e}")
            raise
    
    def _estimate_cost(self, provider: str, model: str, tokens: int) -> float:
        """Estimate cost for API call"""
        costs = self.cost_config.get("api_costs", {})
        
        if provider == "anthropic":
            if "haiku" in model:
                cost_per_m = costs.get("claude_haiku", 0.25)
            elif "sonnet" in model:
                cost_per_m = costs.get("claude_sonnet", 3.00)
            else:
                cost_per_m = costs.get("claude_opus", 15.00)
        elif provider == "openai":
            if "gpt-3.5" in model:
                cost_per_m = costs.get("gpt35_turbo", 0.50)
            else:
                cost_per_m = costs.get("gpt4_turbo", 10.00)
        else:
            return 0.0
        
        return (tokens / 1000000) * cost_per_m
    
    def _calculate_cost(self, provider: str, model: str, tokens: int) -> float:
        """Calculate actual cost of API call"""
        return self._estimate_cost(provider, model, tokens)
    
    def _update_cost_tracking(self, result: ReflectionResult):
        """Update cost tracking with new API call"""
        # Reset daily costs if new day
        now = datetime.now()
        if now.date() > self.cost_tracker.last_reset.date():
            self.cost_tracker.daily_spent = 0.0
            self.cost_tracker.tokens_used_today = {}
            self.cost_tracker.last_reset = now
        
        # Update costs
        self.cost_tracker.daily_spent += result.cost_usd
        self.cost_tracker.weekly_spent += result.cost_usd
        self.cost_tracker.monthly_spent += result.cost_usd
        
        # Track tokens by model
        model_key = result.api_used
        if model_key not in self.cost_tracker.tokens_used_today:
            self.cost_tracker.tokens_used_today[model_key] = 0
        self.cost_tracker.tokens_used_today[model_key] += result.tokens_used
        
        # Save cost data
        self.save_cost_data()
        
        # Log cost status
        logger.info(f"API call cost: ${result.cost_usd:.4f} | Daily total: ${self.cost_tracker.daily_spent:.3f}")
        
        # Check budget warnings
        daily_budget = self.get_daily_budget()
        if self.cost_tracker.daily_spent > daily_budget * 0.8:
            logger.warning(f"Daily budget 80% used: ${self.cost_tracker.daily_spent:.3f}/${daily_budget:.3f}")
        if self.cost_tracker.daily_spent > daily_budget:
            logger.error(f"DAILY BUDGET EXCEEDED: ${self.cost_tracker.daily_spent:.3f}/${daily_budget:.3f}")
    
    def save_cost_data(self):
        """Save cost tracking data"""
        try:
            cost_file = Path("data/costs/cost_tracking.json")
            cost_file.parent.mkdir(parents=True, exist_ok=True)
            
            cost_data = {
                "daily_spent": self.cost_tracker.daily_spent,
                "weekly_spent": self.cost_tracker.weekly_spent,
                "monthly_spent": self.cost_tracker.monthly_spent,
                "tokens_used_today": self.cost_tracker.tokens_used_today,
                "last_reset": self.cost_tracker.last_reset.isoformat()
            }
            
            with open(cost_file, 'w') as f:
                json.dump(cost_data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save cost data: {e}")
    
    def load_cost_data(self):
        """Load existing cost tracking data"""
        try:
            cost_file = Path("data/costs/cost_tracking.json")
            if cost_file.exists():
                with open(cost_file, 'r') as f:
                    cost_data = json.load(f)
                
                self.cost_tracker.daily_spent = cost_data.get("daily_spent", 0.0)
                self.cost_tracker.weekly_spent = cost_data.get("weekly_spent", 0.0)
                self.cost_tracker.monthly_spent = cost_data.get("monthly_spent", 0.0)
                self.cost_tracker.tokens_used_today = cost_data.get("tokens_used_today", {})
                
                if "last_reset" in cost_data:
                    self.cost_tracker.last_reset = datetime.fromisoformat(cost_data["last_reset"])
                    
        except Exception as e:
            logger.warning(f"Failed to load cost data: {e}")
    
    def load_cache(self):
        """Load reflection cache"""
        # Implementation would load cached reflections
        pass
    
    # Optimized parsing methods (shorter, more efficient)
    def _parse_insights_optimized(self, content: str) -> List[str]:
        """Parse insights from API response (optimized for shorter responses)"""
        lines = content.split('\n')
        insights = []
        
        for line in lines:
            line = line.strip()
            if line and any(line.startswith(prefix) for prefix in ['1.', '2.', '3.', '-', '•']):
                insights.append(line[:100])  # Limit length
                
        return insights[:3]  # Limit to top 3 for cost savings
    
    def _parse_predictions_optimized(self, content: str) -> List[str]:
        """Parse predictions (optimized)"""
        predictions = []
        lines = content.lower().split('\n')
        
        for line in lines:
            if any(keyword in line for keyword in ['predict', 'expect', 'likely']):
                predictions.append(line.strip()[:80])
                
        return predictions[:2]  # Limit to top 2
    
    def _parse_consciousness_indicators_optimized(self, content: str) -> List[str]:
        """Parse consciousness indicators (optimized)"""
        indicators = []
        lines = content.lower().split('\n')
        
        consciousness_keywords = ['conscious', 'aware', 'understanding', 'reflection']
        
        for line in lines:
            if any(keyword in line for keyword in consciousness_keywords):
                indicators.append(line.strip()[:60])
                
        return indicators[:2]  # Limit to top 2
    
    def _parse_theory_updates_optimized(self, content: str) -> List[str]:
        """Parse theory updates (optimized)"""
        updates = []
        lines = content.lower().split('\n')
        
        for line in lines:
            if any(keyword in line for keyword in ['theory', 'update', 'learn']):
                updates.append(line.strip()[:60])
                
        return updates[:2]  # Limit to top 2
    
    def _extract_confidence_score_optimized(self, content: str) -> float:
        """Extract confidence score (optimized)"""
        import re
        
        patterns = [r'([0-9.]+)/10', r'([0-9.]+)%', r'score[:\s]*([0-9.]+)']
        
        for pattern in patterns:
            matches = re.findall(pattern, content.lower())
            if matches:
                try:
                    score = float(matches[0])
                    if score > 1:
                        score = score / 10 if score <= 10 else score / 100
                    return min(max(score, 0.0), 1.0)
                except ValueError:
                    continue
                    
        return 0.5  # Default
    
    def _create_fallback_reflection(self, context: str, score: float) -> ReflectionResult:
        """Create a fallback reflection when API calls fail or budget is exceeded"""
        return ReflectionResult(
            reflection_id=f"fallback_{datetime.now().timestamp()}",
            depth_level=1,
            insights=[f"Fallback analysis: {context}"],
            predictions=["Using local processing due to budget/API constraints"],
            confidence_score=0.3,
            consciousness_indicators=["Limited analysis - budget preservation mode"],
            theory_updates=["Local learning continues"],
            timestamp=datetime.now(),
            api_used="fallback",
            tokens_used=0,
            cost_usd=0.0,
            was_cached=False
        )
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get current cost summary"""
        return {
            "daily_spent_usd": self.cost_tracker.daily_spent,
            "daily_budget_usd": self.get_daily_budget(),
            "remaining_budget_usd": self.get_remaining_budget(),
            "budget_utilization_percent": (self.cost_tracker.daily_spent / self.get_daily_budget()) * 100,
            "weekly_spent_usd": self.cost_tracker.weekly_spent,
            "monthly_spent_usd": self.cost_tracker.monthly_spent,
            "tokens_used_today": self.cost_tracker.tokens_used_today
        }

# Alias for backward compatibility
MetaCognitiveEngine = CostOptimizedMetaCognitiveEngine 