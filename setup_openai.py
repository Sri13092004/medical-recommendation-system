#!/usr/bin/env python3
"""
Setup script for OpenAI API configuration
This script helps users configure the OpenAI API for the enhanced medical recommendation system.
"""

import os
import sys
import subprocess
from openai import OpenAI

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
        print("✅ OpenAI package installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing OpenAI: {e}")
        return False

def setup_openai_api():
    """Setup OpenAI API key"""
    print("\n🔑 Setting up OpenAI API...")
    print("To use the enhanced features, you need an OpenAI API key.")
    print("Get your API key from: https://platform.openai.com/api-keys")
    
    api_key = input("\nEnter your OpenAI API key (or press Enter to skip): ").strip()
    
    if api_key:
        # Set environment variable for current session
        os.environ['OPENAI_API_KEY'] = api_key
        
        # Create .env file for persistent storage
        with open('.env', 'w') as f:
            f.write(f"OPENAI_API_KEY={api_key}\n")
        
        print("✅ OpenAI API key configured!")
        print("💡 The API key has been saved to .env file for future use.")
        return True
    else:
        print("⚠️  Skipping API key setup. Enhanced features will be limited.")
        return False

def test_openai_connection():
    """Test OpenAI API connection"""
    print("\n🧪 Testing OpenAI API connection...")
    try:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ No API key found. Please run setup again.")
            return False

        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello, are you working?"}],
            max_tokens=50,
            temperature=0.2,
        )

        print("✅ OpenAI API connection successful!")
        print(f"Test response: {response.choices[0].message.content[:100]}...")
        return True

    except Exception as e:
        print(f"❌ OpenAI API connection failed: {e}")
        return False

def create_env_file():
    """Create .env file template"""
    env_content = """# Enhanced Medical Recommendation System Environment Variables
# Get your OpenAI API key from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Model configuration
GAT_HIDDEN_DIM=64
GAT_N_HEADS=4
GAT_N_LAYERS=2
"""
    
    with open('.env.template', 'w') as f:
        f.write(env_content)
    
    print("📄 Created .env.template file for reference")

def main():
    """Main setup function"""
    print("🚀 Enhanced Medical Recommendation System Setup (OpenAI)")
    print("=" * 60)
    
    # Step 1: Install requirements
    if not install_requirements():
        print("❌ Setup failed at requirements installation")
        return False
    
    # Step 2: Setup OpenAI API
    openai_configured = setup_openai_api()
    
    # Step 3: Test connection if API key provided
    if openai_configured:
        test_openai_connection()
    
    # Step 4: Create environment template
    create_env_file()
    
    print("\n🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Run: python enhanced_main.py")
    print("2. Open: http://localhost:5000")
    print("3. Test the enhanced features!")
    
    if not openai_configured:
        print("\n⚠️  Note: To use all enhanced features, configure your OpenAI API key:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print("   or edit the .env file")
    
    return True

if __name__ == "__main__":
    main()
