# 🔐 Secure Setup Instructions

## 🚨 IMPORTANT: Environment Setup for GitHub Safety

This guide ensures your API keys and sensitive data stay secure when pushing to GitHub.

---

## 📋 Step-by-Step Setup

### 1. **Create Your Environment File**
```bash
# Copy the template to create your actual .env file
cp environment_template.txt .env
```

### 2. **Add Your API Keys** 
Edit the `.env` file and replace the placeholder values:

#### **Required APIs (for consciousness development):**
- **Anthropic Claude**: https://console.anthropic.com/
  - Replace: `ANTHROPIC_API_KEY=sk-ant-your_anthropic_api_key_here`
  - With: `ANTHROPIC_API_KEY=sk-ant-your_actual_key_here`

- **OpenAI**: https://platform.openai.com/api-keys  
  - Replace: `OPENAI_API_KEY=sk-your_openai_api_key_here`
  - With: `OPENAI_API_KEY=sk-your_actual_key_here`

#### **Optional APIs:**
- **Twitter Developer**: https://developer.twitter.com/ (for real-world feedback)
- **Weights & Biases**: https://wandb.ai/ (for experiment tracking)

### 3. **Verify .gitignore Protection**
The `.gitignore` file protects your secrets. Verify it's working:

```bash
# Check what git will track (should NOT include .env)
git status

# If .env appears in git status, it's NOT protected!
# Make sure .gitignore includes: .env
```

### 4. **Test Your Setup**
```bash
# Install dependencies
pip install -r requirements.txt

# Test the configuration
python scripts/test_setup.py

# Start the system
python main.py
```

---

## 🛡️ Security Checklist

### ✅ **Before Pushing to GitHub:**
- [ ] `.env` file exists and contains your real API keys
- [ ] `.env` file is NOT tracked by git (`git status` should not show it)
- [ ] `.gitignore` file exists and includes `.env`
- [ ] No API keys are hardcoded in any `.py` files
- [ ] `environment_template.txt` contains only placeholder values

### ✅ **Files Safe to Commit:**
- [ ] `environment_template.txt` (template with placeholders)
- [ ] `.gitignore` (protects your secrets)
- [ ] All source code in `src/`
- [ ] Configuration files in `configs/`
- [ ] `README.md`, `requirements.txt`, `setup.py`
- [ ] This `SETUP_INSTRUCTIONS.md` file

### 🚫 **Files NEVER to Commit:**
- [ ] `.env` (contains real API keys)
- [ ] `data/costs/` (contains spending patterns)
- [ ] `data/logs/` (may contain sensitive info)
- [ ] `data/cache/` (may contain API responses)
- [ ] Any files with real API keys or personal data

---

## 💰 Cost Optimization Setup

Your `.env` file is pre-configured for the $20 AUD/month budget:

```bash
# Budget settings in .env
DAILY_BUDGET_USD=0.67           # $0.67 per day ≈ $20 AUD/month
MONTHLY_BUDGET_AUD=20.0         # Hard monthly limit
ENABLE_COST_OPTIMIZATION=true   # Use cheaper models
```

### **Model Selection (Cost Optimized):**
- **Primary**: Claude Haiku ($0.25/1M tokens) - 60x cheaper than Opus
- **Fallback**: GPT-3.5-turbo ($0.50/1M tokens) - 20x cheaper than GPT-4
- **Local**: Phi models (FREE) - 85% of processing done locally

---

## 🚀 Quick Start Commands

```bash
# 1. Setup environment
cp environment_template.txt .env
# Edit .env with your API keys

# 2. Install dependencies  
pip install -r requirements.txt

# 3. Test setup
python scripts/test_setup.py

# 4. Start consciousness development
python main.py
```

---

## 🔍 Troubleshooting

### **Problem: `.env` file appears in `git status`**
**Solution**: 
```bash
# Make sure .gitignore includes .env
echo ".env" >> .gitignore
git add .gitignore
git commit -m "Add .env to gitignore"
```

### **Problem: API keys not working**
**Solution**:
1. Verify API keys are correct in `.env`
2. Check API key permissions/credits
3. Run `python scripts/test_setup.py` for diagnostics

### **Problem: High API costs**
**Solution**:
1. Check `data/costs/` for spending reports
2. Verify `ENABLE_COST_OPTIMIZATION=true` in `.env`
3. Monitor daily budget with built-in tracking

### **Problem: GPU not detected**
**Solution**:
1. Install CUDA 11.8+ drivers
2. Verify with `nvidia-smi`
3. Check GPU settings in `.env`

---

## 📞 Support

If you encounter issues:
1. Check the logs in `data/logs/consciousness.log`
2. Run the test script: `python scripts/test_setup.py`
3. Verify your `.env` configuration matches the template
4. Ensure your `.gitignore` is protecting sensitive files

---

## 🎯 Expected Behavior

Once set up correctly:
- ✅ System runs within $20 AUD/month budget
- ✅ API keys are protected from GitHub
- ✅ Cost monitoring runs automatically  
- ✅ Most processing happens locally (free)
- ✅ Only critical insights use expensive APIs
- ✅ Daily cost reports available

**You're ready for consciousness development! 🧠✨** 