"""
GPU utilities for multi-GPU setup and memory management
"""

import os
import logging
from typing import Dict, List, Optional, Tuple
import json

try:
    import torch
    import torch.cuda
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

from .logging import get_logger

logger = get_logger(__name__)

def setup_multi_gpu(gpu_config: Dict[str, any]) -> Dict[str, any]:
    """
    Setup multi-GPU configuration for RTX 4070 Ti Super + GTX 1080.
    
    Args:
        gpu_config: Configuration dictionary with GPU settings
        
    Returns:
        Validated GPU configuration
    """
    if not HAS_TORCH:
        logger.error("PyTorch not available - GPU setup failed")
        return {"cuda_available": False}
    
    if not torch.cuda.is_available():
        logger.error("CUDA not available")
        return {"cuda_available": False}
    
    gpu_count = torch.cuda.device_count()
    logger.info(f"Detected {gpu_count} CUDA devices")
    
    if gpu_count < 2:
        logger.warning("Less than 2 GPUs detected - falling back to single GPU")
        gpu_config["secondary_gpu"] = None
    
    # Set CUDA_VISIBLE_DEVICES if specified
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        logger.info(f"CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    # Configure memory fractions
    primary_gpu = gpu_config.get("primary_gpu", 0)
    secondary_gpu = gpu_config.get("secondary_gpu", 1)
    
    # Set memory fractions if specified
    primary_fraction = gpu_config.get("primary_memory_fraction", 0.85)
    secondary_fraction = gpu_config.get("secondary_memory_fraction", 0.80)
    
    try:
        if primary_gpu < gpu_count:
            torch.cuda.set_per_process_memory_fraction(primary_fraction, primary_gpu)
            logger.info(f"Set GPU {primary_gpu} memory fraction to {primary_fraction}")
            
        if secondary_gpu and secondary_gpu < gpu_count:
            torch.cuda.set_per_process_memory_fraction(secondary_fraction, secondary_gpu)
            logger.info(f"Set GPU {secondary_gpu} memory fraction to {secondary_fraction}")
            
    except Exception as e:
        logger.warning(f"Failed to set memory fractions: {e}")
    
    # Log GPU information
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1e9
        logger.info(f"GPU {i}: {props.name} - {memory_gb:.1f}GB")
    
    return {
        "cuda_available": True,
        "gpu_count": gpu_count,
        "primary_gpu": primary_gpu,
        "secondary_gpu": secondary_gpu if secondary_gpu and secondary_gpu < gpu_count else None,
        "primary_memory_fraction": primary_fraction,
        "secondary_memory_fraction": secondary_fraction
    }

def get_gpu_info() -> List[Dict[str, any]]:
    """
    Get detailed information about available GPUs.
    
    Returns:
        List of GPU information dictionaries
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        return []
    
    gpu_info = []
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        
        # Get memory info
        torch.cuda.set_device(i)
        memory_allocated = torch.cuda.memory_allocated(i)
        memory_reserved = torch.cuda.memory_reserved(i)
        memory_total = props.total_memory
        
        gpu_data = {
            "index": i,
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_gb": memory_total / 1e9,
            "allocated_memory_gb": memory_allocated / 1e9,
            "reserved_memory_gb": memory_reserved / 1e9,
            "free_memory_gb": (memory_total - memory_reserved) / 1e9,
            "multiprocessor_count": props.multi_processor_count,
            "max_threads_per_block": props.max_threads_per_block
        }
        
        # Add NVML info if available
        if HAS_NVML:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert to watts
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                gpu_data.update({
                    "temperature_c": temperature,
                    "power_usage_w": power_usage,
                    "gpu_utilization_percent": utilization.gpu,
                    "memory_utilization_percent": utilization.memory
                })
            except Exception as e:
                logger.debug(f"Failed to get NVML info for GPU {i}: {e}")
        
        gpu_info.append(gpu_data)
    
    return gpu_info

def optimize_memory(device: str = "cuda") -> None:
    """
    Optimize GPU memory usage.
    
    Args:
        device: CUDA device to optimize
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        return
    
    try:
        # Clear cache
        torch.cuda.empty_cache()
        
        # Set memory growth if possible
        if hasattr(torch.cuda, 'set_memory_growth'):
            torch.cuda.set_memory_growth(device, True)
            
        logger.info(f"Memory optimization applied to {device}")
        
    except Exception as e:
        logger.warning(f"Failed to optimize memory: {e}")

def check_gpu_compatibility() -> Dict[str, bool]:
    """
    Check GPU compatibility for the project requirements.
    
    Returns:
        Compatibility check results
    """
    results = {
        "cuda_available": False,
        "sufficient_gpus": False,
        "primary_gpu_adequate": False,
        "secondary_gpu_adequate": False,
        "memory_adequate": False
    }
    
    if not HAS_TORCH or not torch.cuda.is_available():
        return results
    
    results["cuda_available"] = True
    
    gpu_count = torch.cuda.device_count()
    results["sufficient_gpus"] = gpu_count >= 1  # At least 1 GPU required
    
    if gpu_count >= 1:
        # Check primary GPU (should be RTX 4070 Ti Super or similar)
        props_0 = torch.cuda.get_device_properties(0)
        memory_gb_0 = props_0.total_memory / 1e9
        
        # RTX 4070 Ti Super has 16GB, minimum 8GB for 7B model
        results["primary_gpu_adequate"] = memory_gb_0 >= 8.0
        
        if gpu_count >= 2:
            # Check secondary GPU (should be GTX 1080 or better)
            props_1 = torch.cuda.get_device_properties(1)
            memory_gb_1 = props_1.total_memory / 1e9
            
            # GTX 1080 has 8-12GB, minimum 6GB for local analysis
            results["secondary_gpu_adequate"] = memory_gb_1 >= 6.0
        else:
            results["secondary_gpu_adequate"] = True  # Not required
    
    # Overall memory check
    total_memory = sum(
        torch.cuda.get_device_properties(i).total_memory / 1e9
        for i in range(gpu_count)
    )
    results["memory_adequate"] = total_memory >= 12.0  # Minimum for basic operation
    
    return results

def monitor_gpu_usage() -> Dict[str, any]:
    """
    Monitor current GPU usage.
    
    Returns:
        Current GPU usage statistics
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    usage_stats = {
        "timestamp": torch.cuda.Event(enable_timing=True),
        "devices": []
    }
    
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        
        device_stats = {
            "device_id": i,
            "memory_allocated_mb": torch.cuda.memory_allocated(i) / 1e6,
            "memory_reserved_mb": torch.cuda.memory_reserved(i) / 1e6,
            "memory_total_mb": torch.cuda.get_device_properties(i).total_memory / 1e6
        }
        
        # Calculate utilization
        total_memory = device_stats["memory_total_mb"]
        used_memory = device_stats["memory_reserved_mb"]
        device_stats["memory_utilization_percent"] = (used_memory / total_memory) * 100
        
        usage_stats["devices"].append(device_stats)
    
    return usage_stats

def log_gpu_status():
    """Log current GPU status for monitoring"""
    if not HAS_TORCH or not torch.cuda.is_available():
        logger.warning("CUDA not available for status logging")
        return
    
    logger.info("=== GPU Status ===")
    
    gpu_info = get_gpu_info()
    for gpu in gpu_info:
        logger.info(
            f"GPU {gpu['index']}: {gpu['name']} | "
            f"Memory: {gpu['allocated_memory_gb']:.1f}GB / {gpu['total_memory_gb']:.1f}GB | "
            f"Free: {gpu['free_memory_gb']:.1f}GB"
        )
        
        if 'temperature_c' in gpu:
            logger.info(
                f"  Temperature: {gpu['temperature_c']}°C | "
                f"Power: {gpu['power_usage_w']:.1f}W | "
                f"Util: {gpu['gpu_utilization_percent']}%"
            )

def allocate_model_to_gpu(model_size: str, preferred_gpu: int = 0) -> int:
    """
    Determine best GPU for a model based on size and available memory.
    
    Args:
        model_size: Size description (e.g., "7B", "1.3B")
        preferred_gpu: Preferred GPU index
        
    Returns:
        Best GPU index for the model
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    
    # Estimate memory requirements (rough estimates)
    memory_requirements = {
        "7B": 14.0,    # 7B model with 4-bit quantization
        "3B": 6.0,     # 3B model 
        "1.3B": 3.0,   # 1.3B model
        "1B": 2.5,     # 1B model
    }
    
    required_memory = memory_requirements.get(model_size, 8.0)  # Default 8GB
    
    gpu_info = get_gpu_info()
    
    # Check preferred GPU first
    if preferred_gpu < len(gpu_info):
        if gpu_info[preferred_gpu]["free_memory_gb"] >= required_memory:
            logger.info(f"Allocating {model_size} model to preferred GPU {preferred_gpu}")
            return preferred_gpu
    
    # Find best alternative
    for i, gpu in enumerate(gpu_info):
        if gpu["free_memory_gb"] >= required_memory:
            logger.info(f"Allocating {model_size} model to GPU {i} (sufficient memory)")
            return i
    
    # Fallback to GPU with most memory
    best_gpu = max(range(len(gpu_info)), key=lambda i: gpu_info[i]["free_memory_gb"])
    logger.warning(
        f"Insufficient memory for {model_size} model, using GPU {best_gpu} "
        f"with {gpu_info[best_gpu]['free_memory_gb']:.1f}GB"
    )
    return best_gpu 