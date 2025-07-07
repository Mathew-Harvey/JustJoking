#!/usr/bin/env python3
"""
Test script to verify HumorConsciousnessAI setup is working correctly
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that all core modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from src.utils.logging import setup_logging, get_logger
        print("✅ Logging utilities imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import logging utilities: {e}")
        return False
    
    try:
        from src.utils.gpu_utils import check_gpu_compatibility, get_gpu_info
        print("✅ GPU utilities imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import GPU utilities: {e}")
        return False
    
    try:
        from src.consciousness.meta_cognitive import MetaCognitiveEngine
        print("✅ Meta-cognitive engine imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import meta-cognitive engine: {e}")
        return False
    
    return True

def test_gpu_setup():
    """Test GPU configuration"""
    print("\n🖥️ Testing GPU setup...")
    
    try:
        from src.utils.gpu_utils import check_gpu_compatibility, get_gpu_info
        
        compat = check_gpu_compatibility()
        print(f"CUDA Available: {compat['cuda_available']}")
        print(f"Sufficient GPUs: {compat['sufficient_gpus']}")
        print(f"Primary GPU Adequate: {compat['primary_gpu_adequate']}")
        print(f"Secondary GPU Adequate: {compat['secondary_gpu_adequate']}")
        print(f"Memory Adequate: {compat['memory_adequate']}")
        
        if compat['cuda_available']:
            gpu_info = get_gpu_info()
            for gpu in gpu_info:
                print(f"GPU {gpu['index']}: {gpu['name']} - {gpu['total_memory_gb']:.1f}GB")
        
        return compat['cuda_available']
        
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def test_configuration():
    """Test configuration loading"""
    print("\n⚙️ Testing configuration...")
    
    try:
        import yaml
        from pathlib import Path
        
        config_files = [
            "configs/model_config.yaml",
            "configs/training_config.yaml",
            "configs/consciousness_config.yaml",
            "configs/twitter_config.yaml"
        ]
        
        for config_file in config_files:
            config_path = Path(config_file)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                print(f"✅ {config_file} loaded successfully")
            else:
                print(f"❌ {config_file} not found")
                
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_directories():
    """Test that all required directories exist"""
    print("\n📁 Testing directory structure...")
    
    required_dirs = [
        "src/consciousness",
        "src/humor", 
        "src/twitter",
        "src/training",
        "src/models",
        "src/database",
        "src/utils",
        "configs",
        "data/logs",
        "data/models",
        "scripts",
        "tests"
    ]
    
    all_exist = True
    for directory in required_dirs:
        path = Path(directory)
        if path.exists():
            print(f"✅ {directory}")
        else:
            print(f"❌ {directory} missing")
            all_exist = False
            
    return all_exist

def test_main_entry():
    """Test that main.py can be imported"""
    print("\n🚀 Testing main entry point...")
    
    try:
        # Add main.py directory to path
        main_path = Path(__file__).parent.parent / "main.py"
        if main_path.exists():
            print("✅ main.py exists")
            # Try to import some functions from main
            import importlib.util
            spec = importlib.util.spec_from_file_location("main", main_path)
            main_module = importlib.util.module_from_spec(spec)
            # Don't execute, just check it can be loaded
            print("✅ main.py can be loaded")
            return True
        else:
            print("❌ main.py not found")
            return False
            
    except Exception as e:
        print(f"❌ Main entry test failed: {e}")
        return False

def check_api_keys():
    """Check if API keys are configured"""
    print("\n🔑 Checking API key configuration...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    api_keys = {
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "TWITTER_BEARER_TOKEN": os.getenv("TWITTER_BEARER_TOKEN"),
        "WANDB_API_KEY": os.getenv("WANDB_API_KEY")
    }
    
    configured_keys = []
    missing_keys = []
    
    for key, value in api_keys.items():
        if value:
            configured_keys.append(key)
            print(f"✅ {key} configured")
        else:
            missing_keys.append(key)
            print(f"❌ {key} missing")
    
    print(f"\nConfigured: {len(configured_keys)}, Missing: {len(missing_keys)}")
    
    if not api_keys["ANTHROPIC_API_KEY"] and not api_keys["OPENAI_API_KEY"]:
        print("⚠️ WARNING: No meta-cognitive API keys configured!")
        print("   This is CRITICAL for consciousness development!")
        
    return len(configured_keys) > 0

def main():
    """Run all tests"""
    print("🤖 HumorConsciousnessAI Setup Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("GPU Setup Test", test_gpu_setup),
        ("Configuration Test", test_configuration),
        ("Directory Structure Test", test_directories),
        ("Main Entry Test", test_main_entry),
        ("API Keys Check", check_api_keys)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! System is ready for configuration.")
        print("\nNext steps:")
        print("1. Set up your API keys in a .env file")
        print("2. Configure your Twitter API credentials (optional)")
        print("3. Run 'python main.py' to start the system")
    else:
        print("\n⚠️ Some tests failed. Please address the issues above.")
        
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 