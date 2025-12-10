"""
Quick test script for CV-based region proposal.
Compares CV approach vs SAM approach on speed and proposal count.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import time

# Add project to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from collembola_pipeline.proposal_cv import propose_regions_cv


def test_cv_proposal(image_path: Path, verbose: bool = True):
    """Test CV proposal on a single plate image"""
    
    print(f"Testing CV proposal on: {image_path.name}")
    print("=" * 80)
    
    # Load image
    print("\nLoading image...")
    img = np.array(Image.open(image_path))
    print(f"Image shape: {img.shape}")
    print(f"Image size: {img.shape[0] * img.shape[1] / 1e6:.1f} megapixels")
    
    # Run CV proposal
    print("\nRunning CV-based region proposal...")
    start = time.time()
    
    proposals = propose_regions_cv(
        img,
        min_area=200,
        max_area=20000,
        min_eccentricity=0.70,
        background_kernel=51,
        threshold_method='adaptive',
        verbose=verbose
    )
    
    elapsed = time.time() - start
    
    print(f"\n{'=' * 80}")
    print("RESULTS:")
    print(f"{'=' * 80}")
    print(f"Proposals found: {len(proposals)}")
    print(f"Processing time: {elapsed:.2f} seconds")
    print(f"Speed: {img.shape[0] * img.shape[1] / elapsed / 1e6:.1f} megapixels/sec")
    
    if proposals:
        print(f"\nTop 10 proposals (by confidence):")
        for i, p in enumerate(proposals[:10]):
            print(f"  {i+1}. bbox={p.bbox}, area={p.area}, ecc={p.eccentricity:.3f}, conf={p.confidence:.3f}")
        
        # Statistics
        areas = [p.area for p in proposals]
        eccs = [p.eccentricity for p in proposals]
        confs = [p.confidence for p in proposals]
        
        print(f"\nStatistics:")
        print(f"  Area range: {min(areas)} - {max(areas)} px")
        print(f"  Eccentricity range: {min(eccs):.3f} - {max(eccs):.3f}")
        print(f"  Confidence range: {min(confs):.3f} - {max(confs):.3f}")
    
    return proposals


if __name__ == '__main__':
    import argparse
    
    p = argparse.ArgumentParser(description="Test CV-based region proposal")
    p.add_argument('image', type=Path, nargs='?', 
                   default=Path('data/slike/K1_Fe2O3001 (1).jpg'),
                   help="Path to plate image")
    p.add_argument('--quiet', action='store_true', help="Suppress verbose output")
    
    args = p.parse_args()
    
    test_cv_proposal(args.image, verbose=not args.quiet)
