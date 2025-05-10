FROM python:3.11-slim-buster

# Install Node.js and deps first (Kaniko needs all deps ready before COPY if build context is clean)
RUN apt-get update && apt-get install -y curl gnupg ca-certificates && \
    curl -fsSL https://deb.nodesource.com/setup_16.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy backend code and install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the entire app (including frontend)
COPY . .

# Install frontend dependencies
WORKDIR /app/frontend
RUN npm install

# Back to root app directory
WORKDIR /app

# Expose ports
EXPOSE 8000 3000

# Copy and make script executable
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
