FROM mcr.microsoft.com/playwright/python:v1.48.0-noble

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Default execution command
CMD ["python", "news_machine.py"]