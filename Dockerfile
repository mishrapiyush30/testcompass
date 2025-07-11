FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create Streamlit config directory and file
RUN mkdir -p /root/.streamlit && \
    printf "[server]\nheadless = true\nport = 8000\nenableCORS = false\nenableXsrfProtection = false\n" > /root/.streamlit/config.toml

# Expose port (Render expects 8000)
EXPOSE 8000

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"] 