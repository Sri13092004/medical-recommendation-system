# Enhanced Medical Recommendation System

A hybrid AI-powered medical recommendation system that combines:
- OpenAI GPT for medical reasoning
- Knowledge Graph for symptom-disease relationships
- Random Forest classifier for baseline predictions
- Commonsense symptom mapping

## Features

- **Hybrid Reasoning**: Combines LLM, Knowledge Graph, and traditional ML
- **Commonsense Mapping**: Maps lay terms to medical symptoms
- **Detailed Explanations**: Provides comprehensive medical reasoning
- **Confidence Scoring**: Multi-source confidence calculation
- **Real-time Predictions**: Fast symptom-to-disease mapping

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set OpenAI API key:**
   ```bash
   # Windows PowerShell
   $env:OPENAI_API_KEY = 'your-api-key-here'
   
   # Linux/Mac
   export OPENAI_API_KEY='your-api-key-here'
   ```

3. **Run the application:**
   ```bash
   python enhanced_main.py
   ```

4. **Open browser:**
   ```
   http://localhost:5000
   ```

## Usage

Enter symptoms separated by commas (e.g., "headache, fever, nausea") and get:
- Disease prediction with detailed medical reasoning
- Confidence scores from multiple sources
- Precautions, medications, diet, and workout recommendations

## Files Structure

- `enhanced_main.py` - Main Flask application
- `llm_extractor.py` - OpenAI integration and symptom normalization
- `knowledge_graph.py` - Knowledge graph construction and queries
- `enhanced_recommendation_system.py` - Complete framework implementation
- `templates/index.html` - Web interface
- `kaggle_dataset/` - Medical datasets

## Deployment

Ready for deployment on:
- Railway
- Render
- Heroku
- DigitalOcean App Platform

## API Key

Get your OpenAI API key from: https://platform.openai.com/api-keys
