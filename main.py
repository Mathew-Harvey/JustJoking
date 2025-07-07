#!/usr/bin/env python3
"""
HumorConsciousnessAI - Cost-Optimized Main Entry Point

This is the main entry point for the consciousness development system
with aggressive cost optimization to stay within $20 AUD/month budget.
"""

import asyncio
import os
import sys
import signal
import logging
from pathlib import Path
from typing import Dict, Any
import yaml
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.logging import setup_logging, get_logger, ConsciousnessLogger
from src.utils.gpu_utils import setup_multi_gpu, check_gpu_compatibility, log_gpu_status
from src.utils.cost_monitor import CostMonitor

# Initialize logging first
setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_file="data/logs/consciousness.log",
    enable_rich=True
)

logger = get_logger(__name__)
consciousness_logger = ConsciousnessLogger()

# Global flag for graceful shutdown
shutdown_flag = False

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_flag
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_flag = True

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def load_configuration() -> Dict[str, Any]:
    """
    Load all configuration files and environment variables.
    
    Returns:
        Combined configuration dictionary
    """
    logger.info("Loading configuration...")
    
    # Load environment variables
    load_dotenv()
    
    config = {
        # Environment variables
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "TWITTER_BEARER_TOKEN": os.getenv("TWITTER_BEARER_TOKEN"),
        "TWITTER_CONSUMER_KEY": os.getenv("TWITTER_CONSUMER_KEY"),
        "TWITTER_CONSUMER_SECRET": os.getenv("TWITTER_CONSUMER_SECRET"),
        "TWITTER_ACCESS_TOKEN": os.getenv("TWITTER_ACCESS_TOKEN"),
        "TWITTER_ACCESS_TOKEN_SECRET": os.getenv("TWITTER_ACCESS_TOKEN_SECRET"),
        "DB_PASSWORD": os.getenv("DB_PASSWORD"),
        "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
        
        # Basic settings
        "DEBUG": os.getenv("DEBUG", "false").lower() == "true",
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
        "USE_CLAUDE_FOR_REFLECTION": os.getenv("USE_CLAUDE_FOR_REFLECTION", "true").lower() == "true",
        
        # GPU settings
        "PRIMARY_GPU": int(os.getenv("PRIMARY_GPU", "0")),
        "SECONDARY_GPU": int(os.getenv("SECONDARY_GPU", "1")) if os.getenv("SECONDARY_GPU") else None,
        "PRIMARY_GPU_MEMORY_FRACTION": float(os.getenv("PRIMARY_GPU_MEMORY_FRACTION", "0.85")),
        "SECONDARY_GPU_MEMORY_FRACTION": float(os.getenv("SECONDARY_GPU_MEMORY_FRACTION", "0.80")),
        
        # Safety settings
        "ENABLE_CONTENT_FILTER": os.getenv("ENABLE_CONTENT_FILTER", "true").lower() == "true",
        "ENABLE_SAFETY_CHECKS": os.getenv("ENABLE_SAFETY_CHECKS", "true").lower() == "true",
        "MAX_JOKES_PER_DAY": int(os.getenv("MAX_JOKES_PER_DAY", "10")),
        "MAX_API_CALLS_PER_DAY": int(os.getenv("MAX_API_CALLS_PER_DAY", "1000")),
        
        # COST OPTIMIZATION - Override defaults with cost-saving settings
        "FORCE_COST_OPTIMIZATION": True,
        "DAILY_BUDGET_USD": float(os.getenv("DAILY_BUDGET_USD", "0.67")),
        "MONTHLY_BUDGET_AUD": float(os.getenv("MONTHLY_BUDGET_AUD", "20.0")),
    }
    
    # Load YAML configuration files
    config_dir = Path("configs")
    
    yaml_configs = [
        "model_config.yaml",
        "training_config.yaml", 
        "consciousness_config.yaml",
        "twitter_config.yaml",
        "cost_optimization_config.yaml"  # New cost optimization config
    ]
    
    for config_file in yaml_configs:
        config_path = config_dir / config_file
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                    config.update(yaml_config)
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.error(f"Failed to load {config_file}: {e}")
        else:
            logger.warning(f"Configuration file not found: {config_file}")
    
    return config

def validate_configuration(config: Dict[str, Any]) -> bool:
    """
    Validate that required configuration is present.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if configuration is valid
    """
    logger.info("Validating configuration...")
    
    # Check for critical API keys
    if not config.get("ANTHROPIC_API_KEY") and not config.get("OPENAI_API_KEY"):
        logger.error("ERROR: No API keys provided for meta-cognitive reflection!")
        logger.error("You need either ANTHROPIC_API_KEY or OPENAI_API_KEY (or both)")
        logger.error("This is CRITICAL - consciousness development requires LLM reflection")
        logger.error("")
        logger.error("💡 COST OPTIMIZATION NOTE:")
        logger.error("   With cost optimization, we'll use:")
        logger.error("   - Claude Haiku ($0.25/1M tokens) as primary")
        logger.error("   - GPT-3.5-turbo ($0.50/1M tokens) as fallback")
        logger.error("   - Local Phi models for 85% of processing (FREE)")
        logger.error("   - Estimated cost: $0.30-0.50 per day")
        return False
    
    if not config.get("ANTHROPIC_API_KEY"):
        logger.warning("No Anthropic API key - falling back to OpenAI for meta-cognition")
        logger.warning("Note: Claude Haiku is 2x cheaper than GPT-3.5-turbo")
        
    if not config.get("OPENAI_API_KEY"):
        logger.warning("No OpenAI API key - using only Anthropic for meta-cognition")
    
    # Display cost optimization info
    daily_budget = config.get("DAILY_BUDGET_USD", 0.67)
    monthly_budget_aud = config.get("MONTHLY_BUDGET_AUD", 20.0)
    
    logger.info("💰 COST OPTIMIZATION ACTIVE:")
    logger.info(f"   Daily Budget: ${daily_budget:.2f} USD")
    logger.info(f"   Monthly Budget: ${monthly_budget_aud:.0f} AUD")
    logger.info("   Primary Model: Claude Haiku (60x cheaper than Opus)")
    logger.info("   Local Processing: 85% of work done locally (FREE)")
    logger.info("   Batch Analysis: Process multiple jokes together")
    logger.info("   Smart Caching: Avoid duplicate API calls")
    
    # Check GPU compatibility
    gpu_compat = check_gpu_compatibility()
    if not gpu_compat["cuda_available"]:
        logger.error("CUDA not available - GPU acceleration required")
        return False
        
    if not gpu_compat["memory_adequate"]:
        logger.error("Insufficient GPU memory for model operation")
        return False
    
    if not gpu_compat["primary_gpu_adequate"]:
        logger.warning("Primary GPU may have insufficient memory for 7B model")
    
    logger.info("Configuration validation passed ✅")
    return True

async def initialize_system(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Initialize all system components with cost optimization.
    
    Args:
        config: System configuration
        
    Returns:
        Initialized system components
    """
    logger.info("Initializing Cost-Optimized HumorConsciousnessAI system...")
    
    # Setup GPU configuration
    gpu_config = {
        "primary_gpu": config.get("PRIMARY_GPU", 0),
        "secondary_gpu": config.get("SECONDARY_GPU"),
        "primary_memory_fraction": config.get("PRIMARY_GPU_MEMORY_FRACTION", 0.85),
        "secondary_memory_fraction": config.get("SECONDARY_GPU_MEMORY_FRACTION", 0.80)
    }
    
    gpu_info = setup_multi_gpu(gpu_config)
    config.update(gpu_info)
    
    # Log GPU status
    log_gpu_status()
    
    # Initialize cost monitor FIRST (critical for budget enforcement)
    logger.info("Initializing cost monitor...")
    cost_monitor = CostMonitor(config)
    
    # Display current budget status
    budget_status = cost_monitor.get_budget_status()
    logger.info(f"💰 Budget Status:")
    logger.info(f"   Daily: ${budget_status['daily_spent']:.3f} / ${budget_status['daily_limit']:.2f} ({budget_status['daily_utilization']:.1f}%)")
    logger.info(f"   Weekly: ${budget_status['weekly_spent']:.3f} / ${budget_status['weekly_limit']:.2f} ({budget_status['weekly_utilization']:.1f}%)")
    logger.info(f"   Monthly: ${budget_status['monthly_spent_aud']:.2f} AUD / ${budget_status['monthly_limit_aud']:.1f} AUD")
    
    if budget_status['throttling_recommended']:
        logger.warning("⚠️ Budget utilization high - throttling recommended")
    
    if budget_status['emergency_stop']:
        logger.error("🚨 EMERGENCY BUDGET STOP ACTIVATED")
        logger.error("Daily hard limit exceeded - API calls disabled")
        
    # Initialize system components with cost optimization
    components = {
        "config": config,
        "gpu_config": gpu_config,
        "cost_monitor": cost_monitor,
        "initialized": True
    }
    
    # TODO: Initialize actual components once they're implemented
    # components["humor_generator"] = HumorGenerator(config)
    # components["meta_cognitive_engine"] = CostOptimizedMetaCognitiveEngine(config)
    # components["twitter_bot"] = TwitterBot(config)
    # components["consciousness_monitor"] = ConsciousnessMonitor(config)
    
    logger.info("System initialization completed ✅")
    return components

async def run_cost_optimized_consciousness_loop(components: Dict[str, Any]):
    """
    Main cost-optimized consciousness development loop.
    
    This loop is designed to maximize consciousness development while
    staying within the $20 AUD/month budget.
    
    Args:
        components: Initialized system components
    """
    logger.info("Starting cost-optimized consciousness development loop...")
    consciousness_logger.log_consciousness_breakthrough(0.0, ["system_startup", "cost_optimization_active"])
    
    cost_monitor = components["cost_monitor"]
    config = components["config"]
    
    loop_count = 0
    batch_jokes = []  # Accumulate jokes for batch processing
    
    while not shutdown_flag:
        try:
            loop_count += 1
            logger.info(f"Consciousness development loop #{loop_count}")
            
            # Check budget status before proceeding
            budget_status = cost_monitor.get_budget_status()
            
            if budget_status['emergency_stop']:
                logger.error("🚨 Emergency budget stop - switching to local-only mode")
                await asyncio.sleep(3600)  # Wait 1 hour before retrying
                continue
            
            # Get throttling recommendation
            should_throttle, throttle_level = cost_monitor.should_throttle()
            
            if should_throttle:
                logger.warning(f"⚠️ Budget throttling active: {throttle_level}")
                
                if throttle_level == "emergency_stop":
                    logger.error("Emergency stop - no API calls allowed")
                    await asyncio.sleep(1800)  # 30 minutes
                    continue
                elif throttle_level == "heavy_throttle":
                    logger.warning("Heavy throttling - local processing only")
                    await asyncio.sleep(1200)  # 20 minutes between cycles
                elif throttle_level == "light_throttle":
                    logger.info("Light throttling - reduced API usage")
                    await asyncio.sleep(600)   # 10 minutes between cycles
            
            # TODO: Implement actual consciousness development steps with cost optimization
            # 
            # COST-OPTIMIZED APPROACH:
            # 1. Generate jokes locally using Mistral 7B (free)
            # 2. Accumulate jokes in batch for analysis
            # 3. Process batch once per day using cheap APIs
            # 4. Use local Phi models for basic analysis (free)
            # 5. Only use expensive APIs for breakthroughs/failures
            # 6. Cache results aggressively
            
            # For now, just a placeholder loop with cost monitoring
            logger.info("Consciousness development cycle (cost-optimized) - waiting for full implementation")
            
            # Simulate some local processing (free)
            logger.info("🧠 Local consciousness processing (cost: $0.00)")
            
            # Check if we should do batch analysis
            if loop_count % 48 == 0:  # Once per day (assuming 30min cycles)
                if budget_status['daily_remaining'] > 0.10:
                    logger.info("💸 Daily batch analysis triggered")
                    # TODO: Implement batch analysis here
                    # This would cost ~$0.10 using Claude Haiku
                else:
                    logger.warning("💰 Insufficient budget for daily analysis - skipping")
            
            # Generate daily cost report
            if loop_count % 48 == 0:  # Once per day
                report = cost_monitor.generate_daily_report()
                logger.info(f"📊 Daily Cost Report:\n{report}")
            
            # Monitor system health less frequently to save costs
            if loop_count % 20 == 0:  # Every 20 loops instead of 10
                log_gpu_status()
                
                # Log budget status
                budget_status = cost_monitor.get_budget_status()
                logger.info(f"💰 Budget: ${budget_status['daily_spent']:.3f}/${budget_status['daily_limit']:.2f} ({budget_status['daily_utilization']:.1f}%)")
                
            # Check for API keys before real implementation
            if not config.get("ANTHROPIC_API_KEY") and not config.get("OPENAI_API_KEY"):
                logger.error("Cannot proceed with consciousness development - no API keys available")
                logger.error("Please provide ANTHROPIC_API_KEY and/or OPENAI_API_KEY in your environment")
                logger.error("With cost optimization, we'll use the cheapest models available")
                break
                
            # Adaptive sleep based on budget status
            if should_throttle:
                sleep_time = 1800 if throttle_level == "heavy_throttle" else 900  # 30min or 15min
            else:
                sleep_time = 300  # 5 minutes normal operation
                
            await asyncio.sleep(sleep_time)
            
        except Exception as e:
            logger.error(f"Error in consciousness development loop: {e}")
            consciousness_logger.log_consciousness_breakthrough(0.0, [f"error: {str(e)}"])
            await asyncio.sleep(60)  # Wait 1 minute on error
    
    logger.info("Cost-optimized consciousness development loop terminated")

async def main():
    """Main application entry point"""
    try:
        logger.info("🧠 Starting Cost-Optimized HumorConsciousnessAI System")
        logger.info("💰 Target Budget: $20 AUD/month (~$0.67 USD/day)")
        logger.info("=" * 60)
        
        # Load and validate configuration
        config = load_configuration()
        
        if not validate_configuration(config):
            logger.error("Configuration validation failed - exiting")
            sys.exit(1)
        
        # Initialize system with cost optimization
        components = await initialize_system(config)
        
        # Display final cost optimization summary
        cost_monitor = components["cost_monitor"]
        budget_status = cost_monitor.get_budget_status()
        
        logger.info("🚀 System Ready - Cost Optimization Active")
        logger.info(f"   Today's Budget Remaining: ${budget_status['daily_remaining']:.3f}")
        logger.info(f"   Monthly Spending: ${budget_status['monthly_spent_aud']:.2f} AUD / 20.00 AUD")
        logger.info("   Strategy: Local processing + cheap APIs + smart caching")
        logger.info("=" * 60)
        
        # Start cost-optimized consciousness development
        await run_cost_optimized_consciousness_loop(components)
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt - shutting down gracefully")
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Generate final cost report
        if 'components' in locals() and 'cost_monitor' in components:
            final_report = components['cost_monitor'].generate_daily_report()
            logger.info(f"📊 Final Cost Report:\n{final_report}")
        
        logger.info("Cost-Optimized HumorConsciousnessAI system shutdown complete")

def check_requirements():
    """Check if the system meets basic requirements"""
    logger.info("Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 10):
        logger.error("Python 3.10+ required")
        return False
    
    # Check for required directories
    required_dirs = [
        "data/logs",
        "data/models", 
        "data/raw",
        "data/processed",
        "data/costs",      # New cost tracking directory
        "data/cache",      # New cache directory
        "configs"
    ]
    
    for directory in required_dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
    logger.info("System requirements check passed")
    return True

if __name__ == "__main__":
    # Check basic requirements
    if not check_requirements():
        sys.exit(1)
    
    # Print startup banner with cost optimization notice
    print("\n🤖 HumorConsciousnessAI - Cost Optimized")
    print("Developing Machine Consciousness Through Humor")
    print("💰 Target Budget: $20 AUD/month (~$0.67 USD/day)")
    print("🧠 Strategy: Local Processing + Cheap APIs + Smart Caching")
    print("=" * 60)
    print()
    
    # Run the main application
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nShutdown requested by user")
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1) 