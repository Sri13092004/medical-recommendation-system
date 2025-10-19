#!/usr/bin/env python3
"""
Test script for OpenAI API integration
"""

import os
import sys
from openai import OpenAI

def test_openai_connection():
    """Test OpenAI API connection"""
    print("🧪 Testing OpenAI API connection...")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found!")
        print("Set it with: $env:OPENAI_API_KEY='your_api_key_here'")
        return False
    
    print(f"✅ API key found: {api_key[:10]}...")
    
    try:
        client = OpenAI()
        # Test with a simple medical prompt
        prompt = """
        You are a medical expert. Given a disease, identify its primary symptoms.
        Disease: Diabetes

        Return a JSON list of symptoms with confidence scores:
        [{"symptom": "symptom_name", "confidence": 0.95, "severity": "mild|moderate|severe"}]
        """

        print("🔄 Sending test request to OpenAI...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a medical expert providing accurate healthcare information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )

        print("✅ OpenAI API connection successful!")
        print(f"Response: {response.choices[0].message.content[:200]}...")
        return True

    except Exception as e:
        print(f"❌ OpenAI API test failed: {e}")
        return False

def test_enhanced_system():
    """Test the enhanced system with OpenAI"""
    print("\n🚀 Testing enhanced system with OpenAI...")
    
    try:
        from enhanced_recommendation_system import EnhancedMedicalRecommendationSystem
        
        # Initialize enhanced system
        print("Initializing enhanced system...")
        system = EnhancedMedicalRecommendationSystem()
        
        # Test with sample symptoms
        test_symptoms = ["high fever", "headache", "fatigue"]
        print(f"Testing with symptoms: {test_symptoms}")
        
        # Get enhanced recommendation
        recommendation = system.predict_disease_enhanced(test_symptoms)
        
        print(f"✅ Enhanced prediction successful!")
        print(f"Predicted Disease: {recommendation.disease}")
        print(f"Confidence: {recommendation.confidence:.2f}")
        print(f"Medications: {len(recommendation.medications)} found")
        print(f"Lifestyle recommendations: {len(recommendation.lifestyle_recommendations)} found")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced system test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 OpenAI Integration Test")
    print("=" * 35)
    
    # Test 1: Basic OpenAI connection
    openai_ok = test_openai_connection()
    
    if openai_ok:
        # Test 2: Enhanced system
        enhanced_ok = test_enhanced_system()
        
        if enhanced_ok:
            print("\n🎉 All tests passed! Your OpenAI integration is working!")
            print("\nNext steps:")
            print("1. Run: python enhanced_main.py")
            print("2. Open: http://localhost:5000")
            print("3. Test the enhanced features!")
        else:
            print("\n⚠️  OpenAI works but enhanced system has issues.")
    else:
        print("\n❌ OpenAI API test failed.")
        print("Please check your API key and try again.")

if __name__ == "__main__":
    main()
