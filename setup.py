#!/usr/bin/env python3
"""
Setup script for the Mental Health Q&A Chat Application
"""

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.8+ is required.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install required Python packages."""
    print("\n📦 Installing dependencies...")
    
    # Upgrade pip first
    if not run_command("pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        return False
    
    return True

def create_env_file():
    """Create a .env file template."""
    print("\n🔧 Creating environment file template...")
    
    env_content = """# Mental Health Chat Application Environment Variables
# Copy this file to .env and update with your actual values

# Claude 3.5 Proxy URL (required for AI responses)
LLM_PROXY_URL=https://your-claude-proxy.example.com/generate

# Google Perspective API Key (optional, for content moderation)
PERSPECTIVE_API_KEY=your-perspective-api-key-here
"""
    
    try:
        with open(".env.template", "w") as f:
            f.write(env_content)
        print("✅ Created .env.template file")
        print("📝 Please copy .env.template to .env and update with your actual values")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env.template: {e}")
        return False

def run_tests():
    """Run the test script to verify installation."""
    print("\n🧪 Running tests...")
    return run_command("python test_app.py", "Running application tests")

def main():
    """Main setup function."""
    print("🚀 Setting up Mental Health Q&A Chat Application\n")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Failed to install dependencies. Please check the error messages above.")
        sys.exit(1)
    
    # Create environment file template
    create_env_file()
    
    # Run tests
    if not run_tests():
        print("\n❌ Some tests failed. Please check the error messages above.")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Copy .env.template to .env and update with your API keys")
    print("2. Run the application: streamlit run app.py")
    print("3. Open your browser to http://localhost:8501")
    print("\n🐳 Alternative: Use Docker with 'docker-compose up --build'")

if __name__ == "__main__":
    main() 