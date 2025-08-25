#!/bin/bash

# FastAPI Chatbot Setup Script

echo "🚀 Setting up FastAPI Chatbot with Text Embedding..."

# Check if Python is installed
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null
then
    echo "❌ Python is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Use python3 if available, otherwise python
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    PYTHON_CMD=python
fi

echo "🐍 Using Python: $PYTHON_CMD"

# Install dependencies
echo "📦 Installing Python dependencies..."
$PYTHON_CMD -m pip install -r requirements.txt

# Check if .env file exists
if [ ! -f .env ]; then
    echo "🔧 Creating .env file from template..."
    cp env.template .env
    echo ""
    echo "⚠️  IMPORTANT: Edit .env file and add your OpenAI API key:"
    echo "   OPENAI_API_KEY=your_actual_api_key_here"
    echo ""
else
    echo "✅ .env file already exists"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your OpenAI API key (if not done already)"
echo "2. Start the server: cd src && python start_server.py"
echo "3. In another terminal, start the client: cd src && python client.py"
echo ""
echo "📚 Check README.md for detailed instructions and examples"