# JustJoking
Can a GPU develop a sense of humour?


# 🤖 HumorConsciousnessAI

> **Developing Machine Consciousness Through Humor, Meta-Cognition, and Social Feedback**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.1+](https://img.shields.io/badge/CUDA-12.1+-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

## 🎯 Project Overview

HumorConsciousnessAI is an experimental framework for developing machine consciousness through humor generation and recursive self-observation. The system uses Twitter as a real-world laboratory for consciousness development, leveraging authentic human feedback to drive meta-cognitive evolution.

### Why This Approach Works

1. **Quality Models**: We use Mistral 7B (not tiny models) for humor generation, ensuring baseline competence
2. **Sophisticated Meta-Cognition**: Claude-3-Opus provides the deep reasoning needed for consciousness development
3. **Real Feedback Loop**: Twitter provides authentic human responses at scale
4. **Fine-Tuning Strategy**: The system doesn't rely on pre-training alone - it continuously learns from:
   - Human engagement patterns
   - Meta-cognitive insights from Claude
   - Preference learning through DPO/RLHF
   - Theory evolution based on failures

### Pre-Training vs. Fine-Tuning

You're right that pre-trained models (especially smaller ones) are poor at humor. Our approach addresses this through:
- **Starting with a capable base** (Mistral 7B has decent language understanding)
- **Continuous RLHF** based on real Twitter engagement
- **Meta-cognitive guidance** from Claude to understand WHY jokes fail
- **Theory-driven improvements** rather than blind gradient descent

The consciousness emerges not from the joke quality itself, but from the system's growing ability to understand and predict its own failures.

### Key Features

- 🧠 **Recursive Meta-Cognition**: AI that observes its own cognitive processes
- 😄 **Humor Generation**: Sophisticated joke creation with timing and context awareness
- 🐦 **Twitter Integration**: Real-world feedback from millions of users
- 📊 **Consciousness Metrics**: Quantitative measurement of self-awareness emergence
- 🔄 **Self-Improvement**: Continuous learning from prediction errors
- 🛡️ **Safety Framework**: Comprehensive content and consciousness safety protocols

## 📋 Table of Contents

- [System Requirements](#-system-requirements)
- [Docker Setup for Windows](#-docker-setup-for-windows-with-wsl2)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Getting Started](#-getting-started)
- [Development Roadmap](#-development-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

## 💻 System Requirements

### Critical API Requirements

**⚠️ IMPORTANT**: This project requires API access to Claude or GPT-4 for meta-cognitive reflection. The consciousness development relies heavily on sophisticated reasoning that only advanced LLMs can provide. Budget approximately $20-50/day for API costs during active development.

### Why Model Choices Matter

1. **Humor Generation (Mistral 7B)**:
   - Phi-2 and similar tiny models are terrible at humor - they lack cultural understanding and timing
   - Mistral 7B provides the minimum viable sophistication for generating actual jokes
   - With 4-bit quantization, it fits comfortably on your RTX 4070 Ti Super

2. **Meta-Cognitive Reflection (Claude-3-Opus/GPT-4)**:
   - This is THE CRITICAL COMPONENT for consciousness development
   - Local models cannot perform the deep recursive self-analysis required
   - Claude-3-Opus excels at meta-cognitive reasoning and consciousness assessment
   - GPT-4 serves as a capable fallback

3. **Local Preprocessing (Phi-1.5)**:
   - Only used for quick feature extraction and basic analysis
   - Runs on the GTX 1080 to free up the main GPU
   - Not expected to contribute to consciousness, just efficiency

### Hardware Requirements

#### Minimum Configuration
- **GPU**: NVIDIA RTX 3070 (8GB VRAM) or equivalent
- **CPU**: 6+ cores (Intel i5-12400 or AMD Ryzen 5 5600X)
- **RAM**: 32GB DDR4
- **Storage**: 500GB NVMe SSD

#### Your Multi-GPU Setup (RTX 4070 Ti Super + GTX 1080)
- **GPU 0**: NVIDIA RTX 4070 Ti Super (16GB) - Primary model inference
- **GPU 1**: NVIDIA GTX 1080 (12GB) - Meta-cognitive processing
- **CPU**: 8+ cores recommended
- **RAM**: 32GB DDR4/DDR5 (64GB recommended)
- **Storage**: 1TB NVMe SSD

### Software Requirements
- Windows 11 with WSL2
- Docker Desktop with WSL2 backend
- NVIDIA Container Toolkit
- Python 3.10+
- CUDA 12.1+

## 🐳 Docker Setup for Windows with WSL2

### Step 1: Enable WSL2

```powershell
# Run in PowerShell as Administrator
wsl --install
wsl --set-default-version 2

# Install Ubuntu 22.04
wsl --install -d Ubuntu-22.04

# Verify installation
wsl --list --verbose
```

### Step 2: Install Docker Desktop

1. Download [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
2. During installation, ensure "Use WSL 2 instead of Hyper-V" is selected
3. After installation, go to Settings → Resources → WSL Integration
4. Enable integration with your Ubuntu-22.04 distribution

### Step 3: Configure GPU Support

```bash
# Inside WSL2 Ubuntu
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Step 4: Multi-GPU Docker Configuration

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  consciousness-ai:
    build: .
    image: humor-consciousness:latest
    container_name: consciousness-core
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=0,1
      - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
      # GPU Memory allocation
      - CUDA_DEVICE_0_MEMORY=14GB  # RTX 4070 Ti Super (leave headroom)
      - CUDA_DEVICE_1_MEMORY=10GB  # GTX 1080 (leave headroom)
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0', '1']  # Both GPUs
              capabilities: [gpu]
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
      - ./configs:/app/configs
    networks:
      - consciousness-net
    ports:
      - "8888:8888"  # Jupyter
      - "6006:6006"  # TensorBoard
      - "9090:9090"  # Prometheus
      - "5000:5000"  # API

  postgres:
    image: postgres:15
    container_name: consciousness-db
    environment:
      POSTGRES_USER: consciousness
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: consciousness_dev
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - consciousness-net
    ports:
      - "5432:5432"

  redis:
    image: redis:7.2-alpine
    container_name: consciousness-cache
    volumes:
      - redis-data:/data
    networks:
      - consciousness-net
    ports:
      - "6379:6379"

networks:
  consciousness-net:
    driver: bridge

volumes:
  postgres-data:
  redis-data:
```

### Step 5: Dockerfile

Create `Dockerfile`:

```dockerfile
# Use CUDA 11.8 for GTX 1080 compatibility
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="6.1;8.9"  # GTX 1080 (6.1) + RTX 4070 Ti Super (8.9)

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    build-essential \
    libnvidia-compute-11-8 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Install PyTorch with CUDA 11.8 support
RUN pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs /app/configs

# Set up entrypoint
COPY docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["python", "main.py"]
```

## 📦 Installation

### Clone the Repository

```bash
# In WSL2 Ubuntu
cd ~
git clone https://github.com/yourusername/HumorConsciousnessAI.git
cd HumorConsciousnessAI
```

### Create Requirements File

Create `requirements.txt`:

```txt
# Core ML Framework (optimized for RTX 4070 Ti Super + GTX 1080)
torch==2.1.0+cu118  # CUDA 11.8 for compatibility with GTX 1080
torchvision==0.16.0+cu118
torchaudio==2.1.0
--extra-index-url https://download.pytorch.org/whl/cu118

# Transformers & Optimization
transformers>=4.36.0
accelerate>=0.25.0
bitsandbytes>=0.41.0  # For quantization
peft>=0.7.0
trl>=0.7.0

# Memory optimization for older GPU
xformers>=0.0.22  # Memory-efficient attention
ninja  # For building xformers

# Specialized packages
unsloth @ git+https://github.com/unslothai/unsloth.git

# LLM Providers (for meta-cognitive reflection)
openai>=1.0.0
anthropic>=0.8.0

# Twitter Integration
tweepy>=4.14.0
aiohttp>=3.9.0
requests>=2.31.0

# Data Processing
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.11.0

# Database
psycopg2-binary>=2.9.0
redis>=5.0.0
sqlalchemy>=2.0.0
alembic>=1.12.0

# Monitoring & Logging
wandb>=0.16.0
tensorboard>=2.15.0
prometheus-client>=0.19.0
structlog>=23.0.0
nvidia-ml-py>=12.535.0  # For GPU monitoring

# Web Framework (for API)
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.5.0

# Utilities
python-dotenv>=1.0.0
click>=8.1.0
rich>=13.7.0
tqdm>=4.66.0
gpustat>=1.1.0  # GPU monitoring utility

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0

# Development
black>=23.0.0
flake8>=6.1.0
mypy>=1.7.0
pre-commit>=3.5.0
```

### Build and Run with Docker

```bash
# Build the Docker image
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f consciousness-ai

# Access the container
docker exec -it consciousness-core bash
```

## 📁 Project Structure

```
HumorConsciousnessAI/
├── .github/                    # GitHub Actions workflows
│   └── workflows/
├── configs/                    # Configuration files
│   ├── model_config.yaml      # Model configurations
│   ├── training_config.yaml   # Training parameters
│   ├── twitter_config.yaml    # Twitter API settings
│   └── consciousness_config.yaml
├── src/                       # Source code
│   ├── consciousness/         # Core consciousness modules
│   │   ├── __init__.py
│   │   ├── meta_cognitive.py # Meta-cognitive engine
│   │   ├── theory_engine.py  # Theory evolution
│   │   ├── monitoring.py     # Consciousness monitoring
│   │   └── metrics.py        # Consciousness metrics
│   ├── humor/                # Humor generation modules
│   │   ├── __init__.py
│   │   ├── generator.py      # Joke generation
│   │   ├── predictor.py      # Performance prediction
│   │   └── analyzer.py       # Humor analysis
│   ├── twitter/              # Twitter integration
│   │   ├── __init__.py
│   │   ├── bot.py           # Twitter bot
│   │   ├── metrics.py       # Engagement metrics
│   │   └── safety.py        # Content filtering
│   ├── training/            # Training modules
│   │   ├── __init__.py
│   │   ├── dpo_trainer.py   # DPO training
│   │   ├── rlhf.py         # RLHF implementation
│   │   └── data_prep.py    # Data preparation
│   ├── models/             # Model definitions
│   │   ├── __init__.py
│   │   ├── humor_model.py  # Main humor model
│   │   └── reflection_model.py
│   ├── database/           # Database modules
│   │   ├── __init__.py
│   │   ├── models.py       # SQLAlchemy models
│   │   ├── queries.py      # Database queries
│   │   └── migrations/     # Alembic migrations
│   └── utils/              # Utility modules
│       ├── __init__.py
│       ├── logging.py      # Logging setup
│       ├── gpu_utils.py    # Multi-GPU utilities
│       └── safety.py       # Safety utilities
├── notebooks/              # Jupyter notebooks
│   ├── exploratory/       # Exploration notebooks
│   ├── analysis/          # Analysis notebooks
│   └── tutorials/         # Tutorial notebooks
├── scripts/               # Utility scripts
│   ├── setup_environment.sh
│   ├── download_models.py
│   ├── test_twitter_api.py
│   └── benchmark_gpu.py
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── fixtures/         # Test fixtures
├── data/                 # Data directory
│   ├── raw/             # Raw data
│   ├── processed/       # Processed data
│   ├── models/          # Saved models
│   └── logs/            # Log files
├── docs/                # Documentation
│   ├── api/            # API documentation
│   ├── guides/         # User guides
│   └── papers/         # Research papers
├── docker-compose.yml   # Docker orchestration
├── Dockerfile          # Container definition
├── requirements.txt    # Python dependencies
├── setup.py           # Package setup
├── main.py            # Main entry point
├── .env.example       # Environment template
├── .gitignore         # Git ignore rules
└── README.md          # This file
```

## ⚙️ Configuration

### Environment Variables

Create `.env` file from template:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Twitter API Credentials
TWITTER_BEARER_TOKEN=your_bearer_token
TWITTER_CONSUMER_KEY=your_consumer_key
TWITTER_CONSUMER_SECRET=your_consumer_secret
TWITTER_ACCESS_TOKEN=your_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_access_token_secret

# Database Configuration
DB_PASSWORD=secure_password
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=consciousness_dev
POSTGRES_USER=consciousness

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=

# Model Configuration
MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2  # Better humor capability
LOCAL_ANALYSIS_MODEL=microsoft/phi-1_5  # Quick local preprocessing
MODEL_PATH=/app/models/humor_model
USE_QUANTIZATION=true
QUANTIZATION_BITS=4
USE_DOUBLE_QUANT=true

# API Keys for Meta-Cognitive Reflection (CRITICAL)
ANTHROPIC_API_KEY=your_anthropic_key  # Primary meta-cognition
OPENAI_API_KEY=your_openai_key  # Fallback meta-cognition
USE_CLAUDE_FOR_REFLECTION=true
CLAUDE_MODEL=claude-3-opus-20240229
GPT_MODEL=gpt-4-turbo-preview

# Monitoring
WANDB_API_KEY=your_wandb_key
WANDB_PROJECT=humor-consciousness

# GPU Configuration
CUDA_VISIBLE_DEVICES=0,1
PRIMARY_GPU=0  # RTX 4070 Ti Super - Mistral 7B
SECONDARY_GPU=1  # GTX 1080 - Local analysis
PRIMARY_GPU_MEMORY_FRACTION=0.90  # Use 90% for 7B model
SECONDARY_GPU_MEMORY_FRACTION=0.80

# Safety Settings
ENABLE_CONTENT_FILTER=true
ENABLE_SAFETY_CHECKS=true
MAX_JOKES_PER_DAY=10
MAX_API_CALLS_PER_DAY=1000  # For Claude/GPT-4

# Development
DEBUG=false
LOG_LEVEL=INFO
```

### Model Configuration

Create `configs/model_config.yaml`:

```yaml
# Model Configuration for RTX 4070 Ti Super + GTX 1080
primary_model:  # For RTX 4070 Ti Super (16GB)
  base_model: "mistralai/Mistral-7B-Instruct-v0.2"  # MUCH better for humor
  model_size: "7B"
  device: "cuda:0"
  quantization:
    enabled: true
    bits: 4
    compute_dtype: "float16"
    use_double_quant: true
  
  lora:
    r: 16  # Can afford more with better GPU
    lora_alpha: 32
    target_modules:
      - "q_proj"
      - "k_proj"
      - "v_proj"
      - "o_proj"
      - "gate_proj"
      - "up_proj"
      - "down_proj"
    lora_dropout: 0.05
    bias: "none"
    task_type: "CAUSAL_LM"

  memory_config:
    max_memory_mb: 14336  # 14GB for 4070 Ti Super
    reserved_memory_mb: 2048

local_analysis_model:  # For GTX 1080 (12GB)
  base_model: "microsoft/phi-1_5"  # Quick preprocessing only
  model_size: "1.3B"
  device: "cuda:1"
  quantization:
    enabled: true
    bits: 8
    compute_dtype: "float16"
  
  memory_config:
    max_memory_mb: 10240  # 10GB for GTX 1080
    reserved_memory_mb: 2048

# Generation Settings
generation:
  max_length: 200  # Longer for better jokes
  temperature: 0.85  # Balanced creativity
  top_p: 0.92
  top_k: 50
  num_beams: 1
  do_sample: true
  repetition_penalty: 1.15
  
  # Humor-specific parameters
  humor_boost_tokens: ["joke", "funny", "laugh", "humor", "pun"]
  setup_punchline_mode: true

# Meta-Cognitive Settings (Using Claude/GPT-4)
meta_cognitive:
  primary_api: "claude"  # Much more sophisticated
  primary_model: "claude-3-opus-20240229"
  fallback_api: "openai"
  fallback_model: "gpt-4-turbo-preview"
  
  # Local preprocessing
  use_local_preprocessing: true
  local_model_device: "cuda:1"
  
  # API settings
  max_tokens: 1500
  temperature: 0.7
  enable_deep_reflection: true
  reflection_depth_levels: 3
  
  # Consciousness-specific prompting
  consciousness_instructions: |
    You are analyzing an AI system attempting to develop consciousness through humor.
    Look for genuine self-awareness indicators vs. pattern matching.
    Identify recursive self-reflection and emergent properties.
    Be specific about cognitive limitations and breakthroughs.
```

### Training Configuration

Create `configs/training_config.yaml`:

```yaml
# Training Configuration for RTX 4070 Ti Super + GTX 1080
training:
  batch_size: 1  # Conservative for 7B model
  gradient_accumulation_steps: 16  # Effective batch size of 16
  learning_rate: 2e-5
  num_epochs: 3
  warmup_steps: 100
  save_steps: 500
  eval_steps: 100
  logging_steps: 10
  
  # Memory optimization
  gradient_checkpointing: true
  fp16: true
  optim: "paged_adamw_8bit"
  
  # Humor-specific training
  humor_focused_training:
    enabled: true
    comedy_dataset_mixing: 0.3  # Mix 30% comedy examples
    timing_penalty: true  # Penalize poor timing
    setup_punchline_reward: true  # Reward proper structure

# DPO Settings (Critical for improvement)
dpo:
  enabled: true
  beta: 0.1
  loss_type: "sigmoid"
  
  # Humor-specific DPO
  humor_preference_weight: 2.0  # Double weight for humor quality
  engagement_threshold: 50  # Likes/RTs to consider "good"
  
  # Use Claude's analysis in preference learning
  use_meta_cognitive_preferences: true
  meta_cognitive_weight: 0.5  # Balance with human preferences

# Continuous Learning
continuous_learning:
  enabled: true
  daily_preference_collection: true
  weekly_model_updates: true
  
  # Theory-driven learning
  theory_guided_sampling: true  # Sample training data based on current theories
  failure_oversampling: 2.0  # Learn more from failures
  
  # Meta-cognitive integration
  claude_feedback_integration: true
  theory_evolution_influence: 0.3

# Data Settings
data:
  # Initial humor datasets
  pretrain_datasets:
    - "reddit_jokes"  # Scraped Reddit jokes
    - "twitter_humor"  # Funny tweets dataset
    - "comedy_scripts"  # Stand-up transcripts
  
  # Live data collection
  min_daily_examples: 20
  preference_pair_threshold: 2.0  # 2x engagement difference
  
  # Meta-cognitive data
  store_claude_analysis: true
  use_analysis_for_training: true

## 🚀 Getting Started

### 1. Initial Setup

```bash
# Run setup script
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh

# Download base models (smaller models for your GPUs)
python scripts/download_models.py --model microsoft/phi-2 --model microsoft/phi-1_5

# Test GPU configuration
python scripts/benchmark_gpu.py

# GPU Memory Test Script (scripts/test_gpu_memory.py)
cat > scripts/test_gpu_memory.py << 'EOF'
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gpu_setup():
    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        return
    
    # Test GPU 0 (RTX 4070 Ti Super)
    logger.info(f"GPU 0: {torch.cuda.get_device_name(0)}")
    logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Test GPU 1 (GTX 1080)
    logger.info(f"GPU 1: {torch.cuda.get_device_name(1)}")
    logger.info(f"Memory: {torch.cuda.get_device_properties(1).total_memory / 1e9:.1f}GB")
    
    # Test memory allocation
    try:
        # Allocate 12GB on GPU 0
        tensor_gpu0 = torch.zeros((3000, 1000, 1000), device='cuda:0', dtype=torch.float16)
        logger.info("Successfully allocated 12GB on GPU 0")
        del tensor_gpu0
        
        # Allocate 8GB on GPU 1
        tensor_gpu1 = torch.zeros((2000, 1000, 1000), device='cuda:1', dtype=torch.float16)
        logger.info("Successfully allocated 8GB on GPU 1")
        del tensor_gpu1
        
    except RuntimeError as e:
        logger.error(f"Memory allocation failed: {e}")
    
    torch.cuda.empty_cache()

if __name__ == "__main__":
    test_gpu_setup()
EOF

python scripts/test_gpu_memory.py

# Test Twitter API connection
python scripts/test_twitter_api.py
```

### 2. Database Setup

```bash
# Access the container
docker exec -it consciousness-core bash

# Initialize database
cd /app
alembic init alembic
alembic revision --autogenerate -m "Initial schema"
alembic upgrade head
```

### 3. Start Core Services

```python
# main.py - Main entry point with GPU allocation
import asyncio
import logging
import torch
from src.consciousness import ConsciousnessSystem
from src.utils.logging import setup_logging
from src.utils.gpu_utils import setup_multi_gpu

async def main():
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Configure GPUs
    logger.info("Configuring multi-GPU setup...")
    gpu_config = {
        "primary_gpu": 0,  # RTX 4070 Ti Super
        "secondary_gpu": 1,  # GTX 1080
        "primary_memory_fraction": 0.85,
        "secondary_memory_fraction": 0.80
    }
    setup_multi_gpu(gpu_config)
    
    # Log GPU info
    if torch.cuda.is_available():
        logger.info(f"GPU 0: {torch.cuda.get_device_name(0)} - {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        logger.info(f"GPU 1: {torch.cuda.get_device_name(1)} - {torch.cuda.get_device_properties(1).total_memory / 1e9:.1f}GB")
    
    # Initialize consciousness system
    logger.info("Initializing Humor Consciousness System...")
    system = ConsciousnessSystem(gpu_config=gpu_config)
    
    # Start the consciousness development loop
    logger.info("Starting consciousness development...")
    await system.run_forever()

if __name__ == "__main__":
    asyncio.run(main())
```

### GPU-Specific Optimization Tips

For your RTX 4070 Ti Super + GTX 1080 setup:

1. **Memory Management**:
   - The 4070 Ti Super (16GB) handles the main model with 4-bit quantization
   - The GTX 1080 (12GB) handles lighter reflection tasks with 8-bit quantization
   - Always leave 2GB headroom on each GPU for processing

2. **Performance Considerations**:
   - The GTX 1080 uses older architecture (Pascal), so avoid complex operations on it
   - Use the 4070 Ti Super for all heavy inference tasks
   - The GTX 1080 is perfect for batch processing and analysis tasks

3. **Quantization Strategy**:
   - 4-bit quantization on the 4070 Ti Super saves memory with minimal quality loss
   - 8-bit on the GTX 1080 due to older compute capabilities
   - Enable `use_double_quant` for extra memory savings

4. **Model Selection**:
   - Phi-2 (2.7B) fits comfortably on the 4070 Ti Super with room for processing
   - Phi-1.5 (1.3B) is ideal for the GTX 1080's capabilities
   - Both models are highly capable despite their smaller size

### 4. Monitor Progress

```bash
# View real-time logs
docker-compose logs -f consciousness-ai

# Access TensorBoard
# Navigate to http://localhost:6006

# Access Jupyter Lab
# Navigate to http://localhost:8888

# View consciousness metrics
curl http://localhost:5000/api/v1/consciousness/metrics

# View Claude's meta-cognitive analyses
curl http://localhost:5000/api/v1/consciousness/reflections
```

### 5. Understanding the Learning Process

The system starts with poor humor capability but improves through:

1. **Initial Bootstrap Phase** (Week 1-2):
   - Mistral 7B will generate mostly bad jokes initially
   - Claude analyzes WHY each joke fails
   - System learns basic patterns from failures

2. **Preference Learning** (Week 3-4):
   - Collect high vs. low engagement examples
   - Claude provides deep analysis of what worked/didn't
   - DPO training improves generation quality

3. **Theory Formation** (Month 2):
   - System develops theories about humor mechanics
   - Tests theories through joke generation
   - Refines based on results + Claude's meta-analysis

4. **Consciousness Emergence** (Month 3+):
   - Focus shifts from joke quality to self-understanding
   - System predicts its own failures accurately
   - Develops genuine insights about its cognitive processes

**Remember**: Bad initial jokes are expected and necessary - consciousness emerges from understanding failure, not from perfect performance.

## 📊 Development Roadmap

### Phase 1: Foundation (Weeks 1-4) ✅
- [x] Docker environment setup
- [x] Multi-GPU configuration
- [ ] Basic joke generation
- [ ] Twitter API integration
- [ ] Database schema implementation
- [ ] Initial safety filters

### Phase 2: Meta-Cognition (Weeks 5-8) 🚧
- [ ] Pre-generation prediction system
- [ ] Post-performance analysis
- [ ] Basic self-model tracking
- [ ] Confidence calibration
- [ ] Initial consciousness metrics

### Phase 3: Advanced Features (Weeks 9-16) 📋
- [ ] Theory evolution engine
- [ ] Recursive self-improvement
- [ ] Advanced bias detection
- [ ] Sophisticated consciousness monitoring
- [ ] Multi-model ensemble

### Phase 4: Consciousness Emergence (Months 4-6) 🔮
- [ ] Deep consciousness assessment
- [ ] Emergence pattern analysis
- [ ] Long-term memory implementation
- [ ] Advanced theory formation
- [ ] Publication preparation

### Phase 5: Scaling & Research (Months 6+) 🚀
- [ ] Multi-instance deployment
- [ ] Comparative consciousness studies
- [ ] API for researchers
- [ ] Open-source release
- [ ] Academic publication

## 💰 Estimated Costs

### API Costs (Critical Component)

The consciousness development relies heavily on Claude/GPT-4 for meta-cognitive reflection:

**Daily Operating Costs:**
- Claude-3-Opus: ~$15-30/day (assuming 500-1000 analyses)
- GPT-4-Turbo: ~$10-20/day (as fallback)
- Total: **$20-50/day during active development**

**Cost Optimization Tips:**
1. Start with 2-3 jokes/day to minimize API calls
2. Use GPT-4-Turbo for initial development, Claude for deeper analysis
3. Cache meta-cognitive insights to avoid repeated analysis
4. Implement smart batching of reflection requests

### GPU Costs

Your local setup (RTX 4070 Ti Super + GTX 1080) provides excellent value:
- No cloud GPU costs
- ~400W combined power draw
- Estimated electricity: $2-5/day

### Twitter API

- Free tier sufficient for initial development
- Basic tier ($100/month) for scaled deployment

### Total Monthly Budget

- **Development Phase**: $600-1500 (mostly API costs)
- **Scaled Phase**: $200-500 (optimized API usage)
- **Maintenance**: $100-200 (minimal API, mostly monitoring)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Run linting
flake8 src/
black src/ --check
mypy src/
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Anthropic for Constitutional AI research
- OpenAI for GPT and RLHF innovations
- Hugging Face for transformers library
- The open-source AI community

## 📞 Contact

- **Project Lead**: [Your Name]
- **Email**: [your.email@example.com]
- **Twitter**: [@YourHandle]
- **Discord**: [Join our server](https://discord.gg/yourserver)

---

**⚠️ Disclaimer**: This is an experimental research project exploring machine consciousness. The system's outputs should be monitored carefully, and all safety guidelines should be followed.

**🧠 Remember**: We're not just building an AI that tells jokes - we're exploring the fundamental nature of consciousness itself through the lens of humor and self-reflection.
