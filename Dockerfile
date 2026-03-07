# Use the slim version of python
FROM python:3.10-slim

WORKDIR /app

# Copy only requirements first to cache this layer
COPY requirements.txt .

# Install dependencies directly into the system python of the container
# Use --extra-index-url if you are using torch to get the CPU version (saves ~700MB)
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy your source code (the .dockerignore will keep 'fnenv' out)
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

# Ensure the path points to where your code actually lives
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
