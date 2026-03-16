# 1. Start from a lightweight Python base image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Efficient layering: Copy only the requirements file first
# This allows Docker to cache the installed dependencies as long as requirements.txt doesn't change
COPY requirements.txt .

# 4. Install dependencies
# Using --no-cache-dir keeps the final image size as small as possible
RUN pip install --default-timeout=2000 --retries=10 --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application code (like gan_train.py) into the container
COPY . .

# 6. Define the default command to run when the container starts
CMD ["python", "gan_train.py"]