FROM python:3.11-slim-buster

# Install Node.js
RUN apt-get update && apt-get install -y curl gnupg && \
    curl -fsSL https://deb.nodesource.com/setup_16.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean

# Set workdir
WORKDIR /app

# Copy backend code and install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy everything else (app + frontend)
COPY . .

# Install frontend deps and build
WORKDIR /app/frontend
RUN npm install

# Back to root app directory
WORKDIR /app

# Expose both ports
EXPOSE 8000 3000

# Copy launcher script
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]