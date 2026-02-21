# Deployment Guide: advandeb.com/collembola

This guide covers deploying the Collembola Detection Pipeline web application to advandeb.com using Cloudflare Tunnel.

## Architecture

- **Frontend**: React SPA built with Vite, served as static files by nginx on localhost:8100
- **Backend**: FastAPI application running via systemd service on port 8000
- **Reverse Proxy**: nginx on port 8100 handles subdirectory routing `/collembola/`
- **External Access**: Cloudflare Tunnel proxies advandeb.com/collembola/* to localhost:8100

## Prerequisites

1. **System packages** (already installed):
   ```bash
   nginx, python3-pip, nodejs, npm, cloudflared
   ```

2. **Conda environment** (already set up):
   ```bash
   conda activate collembola
   pip install -r requirements.txt
   ```

3. **Frontend dependencies**:
   ```bash
   cd frontend
   npm install
   cd ..
   ```

4. **Permissions** (required for nginx to serve files):
   ```bash
   chmod 755 /home/adeb
   chmod -R 755 /home/adeb/dev/collembola_vis/frontend/dist
   ```

## Current Deployment Status

✅ **DEPLOYED** - All services are running:
- Backend: http://localhost:8000/collembola/api/health
- Nginx: http://localhost:8100/collembola/
- Cloudflare Tunnel: https://advandeb.com/collembola

## Configuration Files

### 1. Frontend (vite.config.ts)
```typescript
base: '/collembola/'  // Subdirectory deployment
```

### 2. Backend (backend/main.py)
```python
root_path="/collembola"  // Handle subdirectory paths correctly
```

### 3. Nginx (nginx-collembola-localhost.conf)
- Listens on **127.0.0.1:8100** (localhost only)
- Routes `/collembola/*` paths
- Proxies API/WebSocket to backend on port 8000
- Serves static frontend files

### 4. Systemd Service (collembola.service)
- Runs backend on port 8000
- User: adeb
- Auto-restart on failure

### 5. Cloudflare Tunnel (/etc/cloudflared/config.yml)
```yaml
- hostname: advandeb.com
  service: http://localhost:8100
  path: /collembola/
```

## Quick Deployment Script

The automated deployment script handles everything:

```bash
cd /home/adeb/dev/collembola_vis
./deploy-cloudflare.sh
```

This script will:
1. Pull latest code (if in git)
2. Update Python dependencies
3. Build frontend with production optimizations
4. Update nginx configuration
5. Update systemd service
6. Restart all services
7. Verify deployment

## Manual Deployment Steps

### 1. Build Frontend
```bash
cd /home/adeb/dev/collembola_vis/frontend
npm install
npm run build
cd ..
```

### 2. Install Nginx Configuration
```bash
sudo cp nginx-collembola-localhost.conf /etc/nginx/sites-available/collembola
sudo ln -sf /etc/nginx/sites-available/collembola /etc/nginx/sites-enabled/collembola
sudo nginx -t
sudo systemctl reload nginx
```

### 3. Start Backend Service
```bash
sudo cp collembola.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable collembola
sudo systemctl restart collembola
```

### 4. Update Cloudflare Tunnel
```bash
sudo cp cloudflared-config.yml /etc/cloudflared/config.yml
sudo systemctl restart cloudflared
```

### 5. Verify Deployment
```bash
# Check all services
systemctl status collembola nginx cloudflared

# Test backend
curl http://localhost:8000/collembola/api/health

# Test nginx
curl http://localhost:8100/collembola/

# Test external access (wait 10-30 seconds for tunnel)
curl https://advandeb.com/collembola/
```

## Monitoring & Logs

### Check Service Status
```bash
systemctl status collembola
systemctl status nginx
systemctl status cloudflared
```

### View Logs
```bash
# Backend logs (real-time)
sudo journalctl -u collembola -f

# Nginx error logs
sudo tail -f /var/log/nginx/error.log

# Nginx access logs
sudo tail -f /var/log/nginx/access.log

# Cloudflare tunnel logs
sudo journalctl -u cloudflared -f
```

### Restart Services
```bash
# Restart backend only
sudo systemctl restart collembola

# Reload nginx (zero-downtime)
sudo systemctl reload nginx

# Restart cloudflare tunnel
sudo systemctl restart cloudflared
```

## Troubleshooting

### Backend Not Starting
```bash
sudo systemctl status collembola
sudo journalctl -u collembola -n 50

# Common issues:
# - Port 8000 already in use
# - Conda environment path incorrect
# - Missing Python dependencies
```

### Frontend 404 / Permission Errors
```bash
# Fix home directory permissions
chmod 755 /home/adeb
chmod -R 755 /home/adeb/dev/collembola_vis/frontend/dist

# Verify build exists
ls -la frontend/dist/

# Check nginx error log
sudo tail -f /var/log/nginx/error.log
```

### Nginx Config Issues
```bash
# Test configuration
sudo nginx -t

# View active configuration
sudo nginx -T | grep -A 30 "server.*8100"
```

### API Calls Failing
```bash
# Test backend directly
curl http://localhost:8000/collembola/api/health

# Test through nginx
curl http://localhost:8100/collembola/api/health

# Check proxy configuration
sudo nginx -T | grep -A 10 "/collembola/api"
```

### Cloudflare Tunnel Not Working
```bash
# Check tunnel status
sudo systemctl status cloudflared
sudo journalctl -u cloudflared -n 50

# Verify configuration
sudo cat /etc/cloudflared/config.yml

# Test local nginx first
curl http://localhost:8100/collembola/

# Restart tunnel
sudo systemctl restart cloudflared

# Wait 10-30 seconds for DNS propagation
sleep 30 && curl https://advandeb.com/collembola/
```

### Static Assets 404
```bash
# Verify base path
grep "base:" frontend/vite.config.ts

# Rebuild frontend
cd frontend && npm run build && cd ..
sudo systemctl reload nginx
```

## Updating the Application

```bash
cd /home/adeb/dev/collembola_vis

# Pull latest changes
git pull

# Update dependencies if needed
pip install -r requirements.txt
cd frontend && npm install && cd ..

# Rebuild and restart
cd frontend && npm run build && cd ..
sudo systemctl restart collembola
sudo systemctl reload nginx
```

## Performance & Security

### Backend Workers
Current setup uses 1 worker. For more users:
```ini
# In collembola.service
ExecStart=... --workers 4
```

### File Uploads
- Max upload size: 500MB (configured in nginx)
- Uploads go to: `data/uploads/`
- Outputs go to: `data/web_outputs/`

### Security Notes
1. ✅ Services only listen on localhost (protected by Cloudflare)
2. ✅ Cloudflare provides DDoS protection and SSL
3. ⚠️  No authentication - consider adding if needed
4. ✅ File permissions properly restricted

### Cloudflare Configuration
Your Cloudflare Tunnel automatically provides:
- HTTPS/SSL encryption
- DDoS protection  
- CDN caching for static assets
- Access control (if configured in Cloudflare dashboard)

## Backup & Maintenance

### Backup Important Data
```bash
cd /home/adeb/dev/collembola_vis

# Backup data
tar -czf collembola-data-$(date +%Y%m%d).tar.gz data/

# Backup models
tar -czf collembola-models-$(date +%Y%m%d).tar.gz models/
```

### Clean Old Outputs
```bash
# Remove outputs older than 30 days
find data/web_outputs -type f -mtime +30 -delete
find data/uploads -type f -mtime +30 -delete
```

### Monitor Disk Space
```bash
df -h /home/adeb
du -sh data/uploads data/web_outputs
```

## Environment Variables

Backend settings can be overridden with `COLLEMBOLA_` prefix:

```bash
# In collembola.service [Service] section:
Environment=COLLEMBOLA_DEFAULT_CONF=0.7
Environment=COLLEMBOLA_DEFAULT_DEVICE=0
```

See `backend/config.py` for all available settings.

## Development vs Production

### Development Mode
```bash
# Terminal 1: Backend with auto-reload
make backend

# Terminal 2: Frontend dev server
make frontend

# Access: http://localhost:5173
```

### Production Mode (Current)
- Frontend: Pre-built static files via nginx
- Backend: systemd service (no auto-reload)
- Nginx: Handles routing and reverse proxy
- Cloudflare: External SSL/CDN/access

## Port Summary

| Service | Port | Access |
|---------|------|--------|
| Backend (uvicorn) | 8000 | localhost only |
| Nginx (collembola) | 8100 | localhost only |
| Cloudflare Tunnel | - | Public (advandeb.com) |

## Access Points

- **Public**: https://advandeb.com/collembola
- **Direct nginx** (localhost): http://localhost:8100/collembola/
- **Direct backend** (localhost): http://localhost:8000/collembola/api/health
