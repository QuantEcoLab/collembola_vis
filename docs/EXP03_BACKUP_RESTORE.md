# Restartable local exp03 backup

The publication backup is private local data, not a GitHub source artifact. It
contains source, photographs, annotations, curated detector versions, results,
consistent SQLite snapshots, detector/dedup models, the fixed exp03 checkpoint,
the exact Python environment and isolated dependencies, compiled frontend and
the historical input snapshot required by the split registry.

Versioned backups live under
`outputs/integration_exp03_local/backups/exp03_<UTC timestamp>/`.
`outputs/integration_exp03_local/latest_restart_backup.json` identifies the latest.
Keep the **whole backup folder**, including:

- `application-data-runtime.tar.zst` and its SHA256 checksum;
- `manifest.json` with every payload file hash and database table counts;
- `final-history.bundle` and `publication.json` for the published Git version;
- `tools/restore_exp03_backup.py` and `README_RESTORE.md`;
- the retained uncompressed `snapshot/` for direct inspection.

System requirements: compatible Linux x86_64, Python 3.13, git, tar, zstd and the
usual OpenCV/PyTorch shared system libraries. Node is not needed to serve the
included compiled frontend. Development dependencies can be reinstalled later
from `frontend/package-lock.json`; node_modules and disposable caches are omitted.
The local JWT signing key and environment credential files are deliberately not
archived. Sign in again after restoration; the launcher generates a new key.

## Restore without touching an existing installation

```bash
python3 /path/to/backup/tools/restore_exp03_backup.py \
  --backup /path/to/backup \
  --destination /absolute/path/that/does/not/exist
```

The tool refuses an existing destination. It verifies the compressed archive and
all payload file hashes before relocating any copied metadata. It restores the
feature branch from the Git bundle, remaps active app paths and symlinks, reseals
relocated curated-dataset manifests and updates the copied registry's file
inventory. Frozen research artifacts remain byte-identical historical records.
Only restored copies are modified; original data and the backup stay untouched.

From the restored `repo/` directory, start:

```bash
.venv-ruler-ft/bin/python -u -B scripts/run_local_exp03.py --port 8778
```

Open `http://127.0.0.1:8778/projects`. A separate port allows the restored copy to
be checked while the original application continues on port 8774. The snapshot
contains data already processed; starting it does not require retraining or
rerunning inference. `restore_report.json` records payload verification and the
paths relocated in that new installation.
