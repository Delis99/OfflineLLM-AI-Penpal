#!/bin/bash
# deploy-website.sh
# Run this on your VPS to deploy the frontend
# Usage: bash deploy-website.sh

echo "Deploying AI Penpal website..."

# Install nginx, Flask, and Google token verification support
apt install -y nginx
pip install flask flask-cors google-auth --break-system-packages

# Create web directory
mkdir -p /var/www/offlinellm

# Copy files (run this from your Mac first):
# scp index.html web_api.py root@72.62.161.121:/root/
# Then on the server:
cp /root/index.html /var/www/offlinellm/index.html

# Setup nginx
cp /root/nginx.conf /etc/nginx/sites-available/offlinellm.me
ln -sf /etc/nginx/sites-available/offlinellm.me /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx

# Setup Flask API as systemd service
cat > /etc/systemd/system/ai-penpal-api.service << 'EOF'
[Unit]
Description=AI Penpal Web API
After=network.target

[Service]
WorkingDirectory=/root/ai-penpal
EnvironmentFile=-/etc/ai-penpal-api.env
ExecStart=/usr/bin/python3 web_api.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl enable ai-penpal-api
systemctl start ai-penpal-api

echo "Done! Visit http://offlinellm.me"
echo "Before first login, create /etc/ai-penpal-api.env with GOOGLE_CLIENT_ID and FLASK_SECRET_KEY."
echo ""
echo "Services running:"
systemctl status ai-penpal --no-pager -l
systemctl status ai-penpal-api --no-pager -l
