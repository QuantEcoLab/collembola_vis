#!/usr/bin/env python3
"""Build an immutable uniquely named local frontend, then atomically select it for preview."""

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT).decode().strip()
    if branch not in {'feature/integrate-exp03-no-os', 'feature/curated-detector-dataset', 'feature/exp03-centerline-and-curation'}:
        raise SystemExit('Wrong branch for exp03 preview build')
    frontend = ROOT / 'frontend'
    source = sorted(p for p in (frontend / 'src').rglob('*') if p.suffix in {'.ts', '.tsx', '.css'})
    source += [frontend / name for name in ['index.html', 'vite.config.ts', 'previewPlugin.ts', 'package-lock.json']]
    source_hash = hashlib.sha256(''.join(f'{p.relative_to(frontend)}:{sha(p)}\n' for p in source).encode()).hexdigest()
    now = datetime.now(timezone.utc)
    identifier = 'exp03-preview-' + now.strftime('%Y%m%dT%H%M%SZ') + '-' + source_hash[:8]
    built_at = now.isoformat(timespec='seconds').replace('+00:00', 'Z')
    local = ROOT / 'outputs/integration_exp03_local'
    dist = local / 'frontend_builds' / identifier
    dist.mkdir(parents=True, exist_ok=False)
    logs = local / 'logs'
    logs.mkdir(exist_ok=True)
    command = ['npm', 'exec', '--cache', '/tmp/opencode/npm-cache', '--yes', '--package=node@22', '--',
               'node', '/usr/bin/npm', 'run', 'build', '--', '--outDir', str(dist)]
    environment = dict(os.environ, VITE_BASE_PATH='/', VITE_LOCAL_PREVIEW='1', VITE_PREVIEW_BUILD_ID=identifier,
                       VITE_PREVIEW_BUILT_AT=built_at)
    with (logs / f'build-{identifier}.log').open('xb') as log:
        result = subprocess.run(command, cwd=frontend, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit(f'Build failed; previous active build preserved. See {logs / ("build-" + identifier + ".log")}')
    index = (dist / 'index.html').read_text()
    scripts = re.findall(r'<script[^>]+src="([^"]+)"', index)
    if len(scripts) != 1 or identifier not in scripts[0] or identifier not in index:
        raise SystemExit('Build identity/entrypoint missing; previous active build preserved')
    manifest = {'build_id': identifier, 'built_at_utc': built_at, 'branch': branch, 'source_sha256': source_hash,
                'dist': str(dist), 'entrypoint_js': scripts[0], 'entrypoint_sha256': sha(dist / scripts[0].lstrip('/')),
                'index_sha256': sha(dist / 'index.html'),
                'files': {p.relative_to(dist).as_posix(): sha(p) for p in dist.rglob('*') if p.is_file()},
                'command': command, 'preview_cache_cleanup': True}
    (dist / 'preview-build.json').write_text(json.dumps(manifest, indent=2))
    temporary = local / 'active_frontend.tmp'
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(local / 'active_frontend.json')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
