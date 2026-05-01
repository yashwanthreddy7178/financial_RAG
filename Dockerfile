# Step 1: Use a lightweight Python base image
FROM python:3.10-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Install necessary system dependencies
# gcc and build-essential are often required to compile certain Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy ONLY the requirements file first
# We do this before copying the rest of the code to leverage Docker layer caching.
# If requirements.txt doesn't change, Docker skips this slow step on future builds!
COPY requirements.txt .

# Step 5: Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Copy the rest of our application code into the container
COPY . .

# Step 7: Pre-download the FlashRank model!
# CRITICAL AWS FIX: If we don't do this here, the container will hang for 30 seconds
# downloading the model from HuggingFace every single time AWS spins up a new instance.
# By running this Python command during the build, the downloaded model becomes baked
# directly into the Docker image itself.
RUN python -c "from flashrank import Ranker; Ranker(model_name='ms-marco-MiniLM-L-12-v2', cache_dir='flashrank_cache')"

# Step 8: Expose the port that Uvicorn will run on
EXPOSE 8000

# Step 9: Define the command that runs when the container starts
# We use Uvicorn with 4 workers to handle multiple concurrent user requests in production
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
