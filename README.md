# Mental Health Q&A Chat Application

A compassionate AI-powered chat application designed to provide mental health support and guidance using semantic search and natural language processing.

## Features

- 🤖 **AI-Powered Responses**: Uses Claude 3.5 for empathetic, supportive responses
- 🔍 **Semantic Search**: Finds relevant Q&A pairs using sentence embeddings
- 💬 **Interactive Chat Interface**: Clean, intuitive Streamlit-based UI
- 🛡️ **Content Moderation**: Built-in safety features for sensitive content
- 📱 **Responsive Design**: Works on desktop and mobile devices
- 🔄 **Real-time Streaming**: Responses appear as they're generated

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (optional):
   ```bash
   export LLM_PROXY_URL="https://your-claude-proxy.example.com/generate"
   export PERSPECTIVE_API_KEY="your-perspective-api-key"
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## How It Works

### Architecture

1. **Data Loading**: The app loads a CSV file containing mental health Q&A pairs
2. **Embedding Generation**: Uses SentenceTransformers to create vector embeddings for all questions
3. **FAISS Index**: Builds an efficient similarity search index
4. **User Interaction**: 
   - User types a question/concern
   - System finds similar Q&A pairs using semantic search
   - User can select relevant context from sidebar
   - AI generates personalized response using selected context

### Key Components

- **`app.py`**: Main Streamlit application
- **`Dataset.csv`**: Sample mental health Q&A dataset
- **`requirements.txt`**: Python dependencies

## Configuration

### Environment Variables

- `LLM_PROXY_URL`: URL for your Claude 3.5 proxy endpoint
- `PERSPECTIVE_API_KEY`: Google Perspective API key for content moderation

### Customizing the Dataset

Replace `Dataset.csv` with your own mental health Q&A data. The file should have two columns:
- `Context`: The question or concern
- `Response`: The supportive response or advice

## Usage

1. **Start a conversation**: Type your mental health concern in the chat input
2. **Review similar Q&A**: Check the sidebar for relevant context
3. **Select context**: Use checkboxes to choose which Q&A pairs to include
4. **Generate advice**: Click "Generate Advice" to get a personalized response
5. **Continue chatting**: The conversation history is maintained throughout the session

## Safety Features

- **Content Moderation**: Built-in filtering for inappropriate content
- **Professional Disclaimer**: Clear indication that this is peer support, not professional therapy
- **Crisis Support**: Automatic detection and appropriate response to crisis situations
- **Privacy**: No data is stored permanently - all conversations are session-based

## Deployment

### Local Development
```bash
streamlit run app.py
```

### Docker Deployment
```bash
docker build -t mental-health-chat .
docker run -p 8501:8501 mental-health-chat
```

### Cloud Deployment
The app can be deployed on:
- Streamlit Cloud
- Heroku
- Google Cloud Run
- AWS Elastic Beanstalk

## Customization

### Adding New Features

1. **Custom Embedding Models**: Modify the `SentenceTransformer` model in `load_data_and_index()`
2. **Additional Safety Checks**: Enhance the `moderate_text()` function
3. **UI Improvements**: Customize the Streamlit interface
4. **Response Templates**: Modify the prompt engineering in the generate advice section

### Extending the Dataset

Add more Q&A pairs to `Dataset.csv` to improve the system's knowledge base. Consider including:
- Different types of mental health concerns
- Various coping strategies
- Crisis intervention resources
- Professional referral information

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
2. **Memory Issues**: The FAISS index loads into memory - consider using a smaller dataset for limited resources
3. **API Connection**: Verify your LLM proxy URL is accessible and properly configured
4. **Dataset Format**: Ensure your CSV file has the correct column names (`Context`, `Response`)

### Performance Optimization

- Use a smaller embedding model for faster processing
- Implement caching for frequently accessed data
- Consider using a production-grade vector database for large datasets

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational and research purposes. Please ensure compliance with local regulations regarding mental health applications.

## Disclaimer

This application is designed for peer support and educational purposes only. It is not a substitute for professional mental health care. If you're experiencing a mental health crisis, please contact emergency services or a mental health professional immediately.

## Support

For questions or issues, please open an issue in the repository or contact the development team.

---

Built with ❤️ using Streamlit, Sentence-Transformers, FAISS & Claude 3.5 