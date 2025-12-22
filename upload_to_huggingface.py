#!/usr/bin/env python3
"""
Upload model to Hugging Face Hub.

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login

Usage:
    python upload_to_huggingface.py --repo QuantEcoLab/collembola-yolo11n
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo

def main():
    parser = argparse.ArgumentParser(description='Upload model to Hugging Face Hub')
    parser.add_argument('--repo', required=True, help='Repository ID (e.g., QuantEcoLab/collembola-yolo11n)')
    parser.add_argument('--private', action='store_true', help='Make repository private')
    args = parser.parse_args()
    
    api = HfApi()
    
    # Create repository
    print(f"Creating repository: {args.repo}")
    try:
        create_repo(args.repo, repo_type='model', private=args.private, exist_ok=True)
        print("✓ Repository created/verified")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return
    
    # Files to upload
    files = [
        ('models/yolo11n_tiled_best.pt', 'yolo11n_tiled_best.pt'),
        ('HUGGINGFACE_README.md', 'README.md'),
        ('MODEL_CARD.md', 'MODEL_CARD.md'),
        ('PERFORMANCE.md', 'PERFORMANCE.md'),
        ('QUICKSTART.md', 'QUICKSTART.md'),
    ]
    
    # Upload files
    for local_path, remote_path in files:
        if not Path(local_path).exists():
            print(f"⚠ Skipping {local_path} (not found)")
            continue
        
        print(f"Uploading {local_path} → {remote_path}...")
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=remote_path,
                repo_id=args.repo,
                repo_type='model'
            )
            print(f"✓ Uploaded {remote_path}")
        except Exception as e:
            print(f"✗ Failed to upload {local_path}: {e}")
    
    print()
    print("✓ Upload complete!")
    print(f"Model available at: https://huggingface.co/{args.repo}")
    print()
    print("Test download with:")
    print(f"  from huggingface_hub import hf_hub_download")
    print(f"  model_path = hf_hub_download(repo_id='{args.repo}', filename='yolo11n_tiled_best.pt')")

if __name__ == '__main__':
    main()
