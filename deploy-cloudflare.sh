#!/bin/bash
# Deployment script for the Collembola Detection Pipeline.
#
# HOW PRODUCTION ACTUALLY SERVES (advandeb, /home/adeb/dev/collembola_vis):
#
#   Cloudflare tunnel e2e3bfc1
#   ├── collembola.advandeb.com -> localhost:9000  uvicorn directly, no nginx.
#   │                                              FastAPI serves the API at /api/*
#   │                                              and the SPA from frontend/dist
#   │                                              (vite base '/').
#   └── advandeb.com            -> localhost:8200  nginx site "advandeb"
#                                                  /collembola/ -> frontend/dist-collembola/
#                                                  (vite base '/collembola/')
#                                                  /collembola/api -> proxy 9000
#
# Because two base paths are live, BOTH frontend builds must be produced.
#
# This script deliberately does NOT touch:
#   - /etc/cloudflared/config.yml   The repo copy is not authoritative and has
#                                   drifted from the live one. Overwriting it
#                                   breaks cadapti, ollama and advandeb.com.
#   - /etc/nginx/sites-*            The live site is "advandeb", which this repo
#                                   does not own. nginx-collembola*.conf in this
#                                   repo describe a vestigial :8100 site that
#                                   proxies to a dead port; they are unused.
#   - /etc/systemd/system/          Only re-copy collembola.service by hand if
#                                   you actually changed it.
#
# Consequently this script needs NO sudo for the common case (frontend-only
# changes). Restarting the backend is a separate, explicit step - see the end.

set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

REPO_DIR="/home/adeb/dev/collembola_vis"
BACKEND_PORT=9000

cd "$REPO_DIR"

# ---------------------------------------------------------------------------
echo -e "${YELLOW}[1/5] Updating code...${NC}"
BEFORE=$(git rev-parse HEAD)
git pull --ff-only
AFTER=$(git rev-parse HEAD)
if [ "$BEFORE" = "$AFTER" ]; then
    echo "Already up to date at $(git log -1 --format='%h %s')"
else
    echo -e "${GREEN}✓ $BEFORE -> $AFTER${NC}"
fi
echo ""

# ---------------------------------------------------------------------------
echo -e "${YELLOW}[2/5] Python dependencies...${NC}"
PIP="$HOME/miniforge3/envs/collembola/bin/pip"
if [ ! -x "$PIP" ]; then
    echo -e "${RED}✗ Conda env missing at $HOME/miniforge3/envs/collembola${NC}"; exit 1
fi
if git diff --name-only "$BEFORE" "$AFTER" | grep -q '^requirements.txt$'; then
    "$PIP" install -r requirements.txt --quiet
    echo -e "${GREEN}✓ Dependencies updated${NC}"
else
    echo "requirements.txt unchanged, skipping"
fi
echo ""

# ---------------------------------------------------------------------------
# Build into temporary directories first. vite empties its output directory
# before writing, so building straight into dist/ means a failed build leaves
# production serving nothing. Build, verify, then swap.
echo -e "${YELLOW}[3/5] Building frontend (both base paths)...${NC}"
cd frontend
[ -d node_modules ] || npm ci

rm -rf dist.new dist-collembola.new
./node_modules/.bin/tsc -b
./node_modules/.bin/vite build --outDir dist.new --emptyOutDir
./node_modules/.bin/vite build --base /collembola/ --outDir dist-collembola.new --emptyOutDir
echo -e "${GREEN}✓ Built${NC}"
echo ""

# ---------------------------------------------------------------------------
echo -e "${YELLOW}[4/5] Verifying builds before swap...${NC}"
verify() {
    local dir="$1" expected="$2"
    local js css
    js=$(grep -o 'assets/[^"]*\.js'  "$dir/index.html" | head -1)
    css=$(grep -o 'assets/[^"]*\.css' "$dir/index.html" | head -1)
    [ -n "$js" ] && [ -f "$dir/$js" ]   || { echo -e "${RED}✗ $dir: missing JS bundle${NC}";  exit 1; }
    [ -n "$css" ] && [ -f "$dir/$css" ] || { echo -e "${RED}✗ $dir: missing CSS bundle${NC}"; exit 1; }
    grep -q "src=\"$expected" "$dir/index.html" \
        || { echo -e "${RED}✗ $dir: wrong asset base, expected $expected${NC}"; exit 1; }
    echo "  $dir OK ($js)"
}
verify dist.new            "/assets/"
verify dist-collembola.new "/collembola/assets/"
echo -e "${GREEN}✓ Both builds valid${NC}"
echo ""

# ---------------------------------------------------------------------------
echo -e "${YELLOW}[5/5] Swapping into place...${NC}"
rm -rf dist.bak dist-collembola.bak
[ -d dist ]            && mv dist dist.bak
[ -d dist-collembola ] && mv dist-collembola dist-collembola.bak
mv dist.new dist
mv dist-collembola.new dist-collembola
chmod -R 755 dist dist-collembola
echo -e "${GREEN}✓ Swapped (previous builds kept as dist.bak / dist-collembola.bak)${NC}"
cd ..
echo ""

# ---------------------------------------------------------------------------
echo -e "${YELLOW}Verifying live endpoints...${NC}"
fail=0
check() {
    local label="$1" url="$2" want="$3"
    if curl -s -m 10 "$url" | grep -q "$want"; then
        echo -e "  ${GREEN}✓${NC} $label"
    else
        echo -e "  ${RED}✗${NC} $label  ($url)"; fail=1
    fi
}
check "backend health (local :$BACKEND_PORT)" "http://localhost:$BACKEND_PORT/api/health"    '"ok"'
check "collembola.advandeb.com"               "https://collembola.advandeb.com/api/health"   '"ok"'
check "advandeb.com/collembola"               "https://advandeb.com/collembola/api/health"   '"ok"'
echo ""

if [ "$fail" -ne 0 ]; then
    echo -e "${RED}=== Verification FAILED ===${NC}"
    echo "Roll back with:"
    echo "  cd $REPO_DIR/frontend && rm -rf dist dist-collembola \\"
    echo "    && mv dist.bak dist && mv dist-collembola.bak dist-collembola"
    exit 1
fi

echo -e "${GREEN}=== Deployment Complete ===${NC}"
echo ""
echo "  https://collembola.advandeb.com/"
echo "  https://advandeb.com/collembola/"
echo ""
echo "Backend restart is NOT part of this script. It is only needed when"
echo "something under backend/ or requirements.txt changed:"
echo "  sudo systemctl restart collembola"
echo ""
echo "Logs:      sudo journalctl -u collembola -f"
echo "Rollback:  cd $REPO_DIR/frontend && rm -rf dist dist-collembola \\"
echo "             && mv dist.bak dist && mv dist-collembola.bak dist-collembola"
echo ""
