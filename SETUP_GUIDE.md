# 🚀 Complete Setup Guide for Enhanced Medical Recommendation System

## 📋 **Prerequisites**
- Python 3.8+
- OpenAI API Key
- All dependencies installed

## 🔑 **Step 1: Get OpenAI API Key**

1. **Visit**: [OpenAI Platform](https://platform.openai.com/api-keys)
2. **Sign in** with your OpenAI account
3. **Click "Create API Key"**
4. **Copy the key** (starts with `sk-...`)

## ⚙️ **Step 2: Set Up API Key**

### **Method 1: Environment Variable (Recommended)**
```bash
# Windows Command Prompt
set OPENAI_API_KEY=your_actual_api_key_here

# Windows PowerShell
$env:OPENAI_API_KEY="your_actual_api_key_here"

# Linux/Mac
export OPENAI_API_KEY="your_actual_api_key_here"
```

### **Method 2: .env File**
Create a file named `.env` in your project directory:
```
OPENAI_API_KEY=your_actual_api_key_here
```

### **Method 3: Automated Setup**
```bash
python setup_openai.py
```

## 🧪 **Step 3: Test the Integration**

### **Test 1: Basic Connection**
```bash
python test_openai.py
```

### **Test 2: Full System Test**
```bash
python test_openai.py
```

## 🚀 **Step 4: Run the System**

### **Option A: Enhanced System (with OpenAI)**
```bash
python enhanced_main.py
```

### **Option B: Simple System (without OpenAI)**
```bash
python simple_main.py
```

## 🌐 **Step 5: Access the Web Interface**

1. **Open your browser**
2. **Go to**: `http://localhost:5000`
3. **Enter symptoms** (e.g., "fever, headache, fatigue")
4. **Get enhanced recommendations**

## 🔍 **Troubleshooting**

### **Issue 1: API Key Not Found**
```
Warning: OPENAI_API_KEY not found. LLM features will be limited.
```
**Solution**: Set your API key using one of the methods above.

### **Issue 2: Import Errors**
```
ModuleNotFoundError: No module named 'openai'
```
**Solution**: 
```bash
pip install openai
```

### **Issue 3: Connection Failed**
```
❌ OpenAI API connection failed
```
**Solution**: 
- Check your API key is correct
- Ensure you have internet connection
- Verify your OpenAI account has API access

### **Issue 4: Enhanced System Errors**
```
Error initializing enhanced system
```
**Solution**: 
- Run the simple system first: `python simple_main.py`
- Check all dependencies are installed
- Verify your API key works with the test script

## 📊 **System Status Check**

### **Check if Enhanced Features are Available**
Visit: `http://localhost:5000/api/system_status`

### **Expected Response (with OpenAI)**
```json
{
  "enhanced_system_available": true,
  "framework_components": {
    "llm_extraction": true,
    "knowledge_graph": true,
    "gat_fusion": true,
    "preventive_knowledge": true
  }
}
```

### **Expected Response (without OpenAI)**
```json
{
  "enhanced_system_available": false,
  "framework_components": {
    "llm_extraction": false,
    "knowledge_graph": false,
    "gat_fusion": false,
    "preventive_knowledge": false
  }
}
```

## 🎯 **Feature Comparison**

| Feature | Simple System | Enhanced System |
|---------|---------------|-----------------|
| Disease Prediction | ✅ | ✅ |
| Basic Recommendations | ✅ | ✅ |
| LLM Knowledge Extraction | ❌ | ✅ |
| Knowledge Graph | ❌ | ✅ |
| Graph Attention Network | ❌ | ✅ |
| Preventive Knowledge | ❌ | ✅ |
| Confidence Scoring | ❌ | ✅ |
| Multi-source Recommendations | ❌ | ✅ |

## 🚀 **Quick Start Commands**

```bash
# 1. Set API key
set OPENAI_API_KEY=your_api_key_here

# 2. Test integration
python test_openai.py

# 3. Run enhanced system
python enhanced_main.py

# 4. Open browser
# Go to: http://localhost:5000
```

## 📞 **Need Help?**

1. **Check the logs** for error messages
2. **Run test scripts** to identify issues
3. **Verify API key** is set correctly
4. **Check dependencies** are installed
5. **Try simple system** first to ensure basic functionality

## 🎉 **Success Indicators**

✅ **Simple System Working**: Basic predictions work  
✅ **API Key Set**: No "OPENAI_API_KEY not found" warnings  
✅ **OpenAI Connection**: Test script shows "connection successful"  
✅ **Enhanced System**: Shows "Enhanced framework available: True"  
✅ **Web Interface**: Accessible at http://localhost:5000  
✅ **Predictions**: Get enhanced recommendations with confidence scores  

---

**Note**: The system gracefully degrades to basic functionality if OpenAI API is not available, so you can always use the simple system as a fallback.
