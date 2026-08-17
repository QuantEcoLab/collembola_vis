#!/bin/bash
# Deployment script for Collembola Detection Pipeline with Cloudflare Tunnel
# Deploys to: advandeb.com/collembola

set -e  # Exit on error

echo "=== Collembola Detection Pipeline Deployment (Cloudflare) ==="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
REPO_DIR="/home/adeb/dev/collembola_vis"
SERVICE_NAME="collembola.service"
NGINX_CONF="nginx-collembola-localhost.conf"

cd "$REPO_DIR"

# Step 1: Pull latest code (if in git)
echo -e "${YELLOW}[1/8] Checking for updates...${NC}"
if [ -d .git ]; then
    git pull
    echo -e "${GREEN}✓ Code updated${NC}"
else
    echo "Not a git repository, skipping pull"
fi
echo ""

# Step 2: Update Python dependencies
echo -e "${YELLOW}[2/8] Updating Python dependencies...${NC}"
if [ -f "$HOME/miniforge3/envs/collembola/bin/pip" ]; then
    "$HOME/miniforge3/envs/collembola/bin/pip" install -r requirements.txt --quiet
    echo -e "${GREEN}✓ Python dependencies updated${NC}"
else
    echo -e "${RED}✗ Conda environment not found at $HOME/miniforge3/envs/collembola${NC}"
    exit 1
fi
echo ""

# Step 3: Build frontend
echo -e "${YELLOW}[3/8] Building frontend...${NC}"
cd frontend
if [ ! -d node_modules ]; then
    echo "Installing npm dependencies..."
    npm install
fi
npm run build
npm run build:collembola-path
echo -e "${GREEN}✓ Frontend built to frontend/dist/ and frontend/dist-collembola/${NC}"
cd ..
echo ""

# Step 4: Fix permissions
echo -e "${YELLOW}[4/8] Setting permissions...${NC}"
chmod 755 /home/adeb
chmod -R 755 frontend/dist
chmod -R 755 frontend/dist-collembola
echo -e "${GREEN}✓ Permissions set${NC}"
echo ""

# Step 5: Update nginx configuration
echo -e "${YELLOW}[5/8] Updating nginx configuration...${NC}"
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
    echo -e "${RED}✗ $NGINX_CONF not found${NC}"
    exit 1
fi
echo ""

# Step 6: Update systemd service
echo -e "${YELLOW}[6/8] Updating systemd service...${NC}"
if [ -f "$SERVICE_NAME" ]; then
    sudo cp "$SERVICE_NAME" /etc/systemd/system/
    sudo systemctl daemon-reload
    echo -e "${GREEN}✓ Systemd service updated${NC}"
else
    echo -e "${RED}✗ $SERVICE_NAME not found${NC}"
    exit 1
fi
echo ""

# Step 7: Restart services
echo -e "${YELLOW}[7/8] Restarting services...${NC}"
sudo systemctl restart collembola
sleep 2
sudo systemctl reload nginx
echo -e "${GREEN}✓ Services restarted${NC}"

# Optional: Update Cloudflare tunnel if config changed
if [ -f cloudflared-config.yml ]; then
    echo "Updating Cloudflare tunnel config..."
    sudo cp cloudflared-config.yml /etc/cloudflared/config.yml
    sudo systemctl restart cloudflared
    echo -e "${GREEN}✓ Cloudflare tunnel restarted${NC}"
fi
echo ""

# Step 8: Verify deployment
echo -e "${YELLOW}[8/8] Verifying deployment...${NC}"
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

if systemctl is-active --quiet cloudflared; then
    echo -e "${GREEN}✓ Cloudflare tunnel is running${NC}"
else
    echo -e "${YELLOW}⚠ Cloudflare tunnel may have issues${NC}"
fi

# Test backend health endpoint
sleep 2
if curl -s http://localhost:9000/api/health | grep -q "ok"; then
    echo -e "${GREEN}✓ Backend API is responding${NC}"
else
    echo -e "${YELLOW}⚠ Backend API health check failed (may need more time to start)${NC}"
fi

# Test nginx
if curl -sI http://localhost:9100/collembola/ | grep -q "200 OK"; then
    echo -e "${GREEN}✓ Nginx is serving the app${NC}"
else
    echo -e "${YELLOW}⚠ Nginx response unexpected${NC}"
fi
echo ""

echo -e "${GREEN}=== Deployment Complete ===${NC}"
echo ""
echo "Application deployed at: https://advandeb.com/collembola"
echo "  (Allow 10-30 seconds for Cloudflare tunnel DNS propagation)"
echo ""
echo "Useful commands:"
echo "  - Check backend logs:        sudo journalctl -u collembola -f"
echo "  - Check nginx logs:          sudo tail -f /var/log/nginx/error.log"
echo "  - Check tunnel logs:         sudo journalctl -u cloudflared -f"
echo "  - Restart backend:           sudo systemctl restart collembola"
echo "  - Reload nginx:              sudo systemctl reload nginx"
echo "  - Restart cloudflare:        sudo systemctl restart cloudflared"
echo ""
echo "Local test URLs:"
echo "  - Backend:  http://localhost:9000/collembola/api/health"
echo "  - Nginx:    http://localhost:9100/collembola/"
echo ""
