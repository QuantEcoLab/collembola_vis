#!/bin/bash
# Deployment script for Collembola Detection Pipeline at advandeb.com/collembola

set -e  # Exit on error

echo "=== Collembola Detection Pipeline Deployment ==="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
REPO_DIR="/home/adeb/dev/collembola_vis"
SERVICE_NAME="collembola.service"
NGINX_CONF="nginx-collembola.conf"

cd "$REPO_DIR"

# Step 1: Pull latest code (if in git)
echo -e "${YELLOW}[1/7] Checking for updates...${NC}"
if [ -d .git ]; then
    git pull
    echo -e "${GREEN}✓ Code updated${NC}"
else
    echo "Not a git repository, skipping pull"
fi
echo ""

# Step 2: Update Python dependencies
echo -e "${YELLOW}[2/7] Updating Python dependencies...${NC}"
if [ -f "$HOME/miniforge3/envs/collembola/bin/pip" ]; then
    "$HOME/miniforge3/envs/collembola/bin/pip" install -r requirements.txt --quiet
    echo -e "${GREEN}✓ Python dependencies updated${NC}"
else
    echo -e "${RED}✗ Conda environment not found at $HOME/miniforge3/envs/collembola${NC}"
    exit 1
fi
echo ""

# Step 3: Build frontend
echo -e "${YELLOW}[3/7] Building frontend...${NC}"
cd frontend
if [ ! -d node_modules ]; then
    echo "Installing npm dependencies..."
    npm install
fi
npm run build
echo -e "${GREEN}✓ Frontend built to frontend/dist/${NC}"
cd ..
echo ""

# Step 4: Update nginx configuration
echo -e "${YELLOW}[4/7] Updating nginx configuration...${NC}"
if [ -f "$NGINX_CONF" ]; then
    sudo cp "$NGINX_CONF" /etc/nginx/sites-available/collembola
    
    # Create symlink if it doesn't exist
    if [ ! -L /etc/nginx/sites-enabled/collembola ]; then
        sudo ln -s /etc/nginx/sites-available/collembola /etc/nginx/sites-enabled/
    fi
    
    # Test nginx configuration
    sudo nginx -t
    echo -e "${GREEN}✓ Nginx configuration updated${NC}"
else
    echo -e "${RED}✗ nginx-collembola.conf not found${NC}"
    exit 1
fi
echo ""

# Step 5: Update systemd service
echo -e "${YELLOW}[5/7] Updating systemd service...${NC}"
if [ -f "$SERVICE_NAME" ]; then
    sudo cp "$SERVICE_NAME" /etc/systemd/system/
    sudo systemctl daemon-reload
    echo -e "${GREEN}✓ Systemd service updated${NC}"
else
    echo -e "${RED}✗ collembola.service not found${NC}"
    exit 1
fi
echo ""

# Step 6: Restart services
echo -e "${YELLOW}[6/7] Restarting services...${NC}"
sudo systemctl restart collembola
sleep 2
sudo systemctl reload nginx
echo -e "${GREEN}✓ Services restarted${NC}"
echo ""

# Step 7: Verify deployment
echo -e "${YELLOW}[7/7] Verifying deployment...${NC}"
if systemctl is-active --quiet collembola; then
    echo -e "${GREEN}✓ Backend service is running${NC}"
else
    echo -e "${RED}✗ Backend service failed to start${NC}"
    sudo systemctl status collembola
    exit 1
fi

if sudo nginx -t 2>/dev/null; then
    echo -e "${GREEN}✓ Nginx configuration is valid${NC}"
else
    echo -e "${RED}✗ Nginx configuration error${NC}"
    exit 1
fi

# Test backend health endpoint
sleep 2
if curl -s http://localhost:9000/collembola/api/health | grep -q "ok"; then
    echo -e "${GREEN}✓ Backend API is responding${NC}"
else
    echo -e "${YELLOW}⚠ Backend API health check failed (may need more time to start)${NC}"
fi
echo ""

echo -e "${GREEN}=== Deployment Complete ===${NC}"
echo ""
echo "Application deployed at: https://advandeb.com/collembola"
echo ""
echo "Useful commands:"
echo "  - Check backend logs:  sudo journalctl -u collembola -f"
echo "  - Check nginx logs:    sudo tail -f /var/log/nginx/error.log"
echo "  - Restart backend:     sudo systemctl restart collembola"
echo "  - Reload nginx:        sudo systemctl reload nginx"
echo ""
