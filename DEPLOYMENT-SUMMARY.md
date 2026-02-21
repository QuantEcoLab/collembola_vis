# Deployment Summary

## Successfully Deployed! ✅

**URL:** https://advandeb.com/collembola

## Quick Reference

### Services Status
```bash
systemctl status collembola   # Backend (port 8000)
systemctl status nginx         # Web server (port 8100)
systemctl status cloudflared   # Cloudflare tunnel
```

### Logs
```bash
sudo journalctl -u collembola -f     # Backend logs
sudo tail -f /var/log/nginx/error.log # Nginx errors
sudo journalctl -u cloudflared -f    # Tunnel logs
```

### Restart Services
```bash
sudo systemctl restart collembola  # Restart backend
sudo systemctl reload nginx        # Reload nginx (no downtime)
sudo systemctl restart cloudflared # Restart tunnel
```

### Update Deployment
```bash
cd /home/adeb/dev/collembola_vis
git pull
./deploy-cloudflare.sh
```

## Architecture

```
User Browser
    ↓ HTTPS
Cloudflare Edge
    ↓ Tunnel
advandeb.com → localhost:8100 (nginx)
    ↓ /collembola/api, /collembola/ws, /collembola/files
localhost:8000 (FastAPI backend)
```

## File Locations

- **Frontend build:** `frontend/dist/`
- **Backend code:** `backend/`
- **Nginx config:** `/etc/nginx/sites-available/collembola`
- **Systemd service:** `/etc/systemd/system/collembola.service`
- **Cloudflare config:** `/etc/cloudflared/config.yml`
- **Data directories:**
  - Uploads: `data/uploads/`
  - Outputs: `data/web_outputs/`
  - Calibration: `data/calibration/`

## Configuration Files

### Frontend (vite.config.ts)
```typescript
base: '/collembola/'  // Assets use /collembola/ prefix
```

### Frontend Router (main.tsx)
```typescript
<BrowserRouter basename="/collembola">
```

### Backend (backend/main.py)
```python
app = FastAPI(root_path="/collembola")  // API docs at /collembola/docs
```

### Nginx (nginx-collembola-localhost.conf)
- Listens on: `127.0.0.1:8100`
- Serves: `/collembola/*`
- Proxies: `/collembola/api`, `/collembola/ws`, `/collembola/files` → backend

### Cloudflare Tunnel (cloudflared-config.yml)
```yaml
- hostname: advandeb.com
  service: http://localhost:8100
```

## DNS Record

Created via CLI:
```bash
cloudflared tunnel route dns e2e3bfc1-235f-40fe-9b6c-99da3e9bf268 advandeb.com
```

Result: CNAME `advandeb.com` → `advandeb-tunnel` (proxied)

## Test Endpoints

```bash
# Health check
curl https://advandeb.com/collembola/api/health

# Frontend
curl https://advandeb.com/collembola/

# API docs
open https://advandeb.com/collembola/docs
```

## Troubleshooting

### Frontend not loading
```bash
# Rebuild frontend
cd frontend && npm run build && cd ..
# Fix permissions
chmod 755 /home/adeb
chmod -R 755 frontend/dist
# Reload nginx
sudo systemctl reload nginx
```

### API errors
```bash
# Check backend logs
sudo journalctl -u collembola -n 50
# Test backend directly
curl http://localhost:8000/collembola/api/health
# Restart backend
sudo systemctl restart collembola
```

### Tunnel issues
```bash
# Check tunnel status
cloudflared tunnel list
cloudflared tunnel info advandeb-tunnel
# Check logs
sudo journalctl -u cloudflared -n 50
# Restart tunnel
sudo systemctl restart cloudflared
```

## Development vs Production

### Development
```bash
make dev  # Starts both backend and frontend dev servers
# Access: http://localhost:5173
```

### Production (Current)
- Frontend: Pre-built static files
- Backend: Systemd service
- Access: https://advandeb.com/collembola

## Environment Variables

Override backend settings (in `/etc/systemd/system/collembola.service`):
```ini
Environment=COLLEMBOLA_DEFAULT_CONF=0.7
Environment=COLLEMBOLA_DEFAULT_DEVICE=0
Environment=COLLEMBOLA_UPLOADS_DIR=/custom/path
```

See `backend/config.py` for all available settings.

## Maintenance

### Backup
```bash
tar -czf collembola-backup-$(date +%Y%m%d).tar.gz \
  data/ models/ frontend/dist/
```

### Clean old files
```bash
find data/uploads -type f -mtime +30 -delete
find data/web_outputs -type f -mtime +30 -delete
```

### Monitor disk usage
```bash
du -sh data/*
df -h /home/adeb
```

## Security Notes

- ✅ Services only listen on localhost (not exposed to internet directly)
- ✅ Cloudflare provides DDoS protection and SSL
- ✅ Nginx proxies requests securely
- ⚠️  No authentication implemented (add if needed for production use)
- ✅ File permissions properly restricted (755 for nginx access)

## Support

For detailed information, see:
- `DEPLOYMENT.md` - Complete deployment guide
- `deploy-cloudflare.sh` - Automated deployment script
- `README.md` - Project overview and usage
