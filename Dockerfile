FROM python:3.11-slim-buster

# 1) Force APT over IPv4 so it won't hang on IPv6
RUN echo 'Acquire::ForceIPv4 "true";' \
     > /etc/apt/apt.conf.d/99force-ipv4

# Install Node.js + dependencies
RUN apt-get update && apt-get install -y curl gnupg ca-certificates && \
    curl -fsSL https://deb.nodesource.com/setup_16.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy all source code
COPY . .

# Expose frontend and backend ports
EXPOSE 8000 3000

# Copy and prepare start script
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
