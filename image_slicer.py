"""
Image Slicing Module for Rectangular to Square Patch Conversion

This module provides tools to convert rectangular images (especially satellite/medical imagery)
into square patches (1200×1200 pixels) with controlled overlap (20-30%) suitable for 
machine learning applications.

Author: Implementation based on mathematical framework
Date: April 2026
"""

import numpy as np
import os
import csv
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import warnings


class PatchInfo:
    """
    Data class to hold patch coordinate and metadata information.
    """
    def __init__(self, patch_id: int, y_start: int, y_end: int, x_start: int = 0, x_end: int = 1200):
        self.patch_id = patch_id
        self.y_start = y_start
        self.y_end = y_end
        self.x_start = x_start
        self.x_end = x_end
        self.width = x_end - x_start
        self.height = y_end - y_start
        self.overlap_with_next = 0
        self.overlap_percentage = 0.0
    
    def to_dict(self) -> Dict:
        """Convert patch info to dictionary."""
        return {
            'patch_id': self.patch_id,
            'x_start': self.x_start,
            'x_end': self.x_end,
            'y_start': self.y_start,
            'y_end': self.y_end,
            'width': self.width,
            'height': self.height,
            'overlap_with_next': self.overlap_with_next,
            'overlap_percentage': f"{self.overlap_percentage:.2f}%"
        }


class ImageSlicer:
    """
    Convert rectangular images to square patches with controlled overlap.
    
    Mathematical Framework:
    - Fundamental equation: h₀ = n · 1200 - (n-1) · k
    - Where: h₀ = original height, n = number of patches, k = overlap in pixels
    
    Solving for n: n = ceil((h₀ + k) / (1200 - k))
    Solving for k: k = (1200·n - h₀) / (n - 1)
    
    Constraints:
    - Patch size: 1200 × 1200 pixels (constant)
    - Overlap range: 240-360 pixels (20-30% of patch size)
    """
    
    # Constants
    PATCH_SIZE = 1200
    OVERLAP_MIN_RATIO = 0.20  # 20% minimum
    OVERLAP_MAX_RATIO = 0.30  # 30% maximum
    OVERLAP_MIN_PIXELS = int(PATCH_SIZE * OVERLAP_MIN_RATIO)  # 240 pixels
    OVERLAP_MAX_PIXELS = int(PATCH_SIZE * OVERLAP_MAX_RATIO)  # 360 pixels
    
    def __init__(self, image_height: int, image_width: int = 1200, target_overlap_ratio: float = 0.25):
        """
        Initialize ImageSlicer with image dimensions.
        
        Args:
            image_height: Height of the original image in pixels
            image_width: Width of the original image in pixels (default: 1200)
            target_overlap_ratio: Desired overlap as fraction (default: 0.25 = 25%)
        
        Raises:
            ValueError: If image dimensions are invalid
        """
        if image_height <= 0 or image_width <= 0:
            raise ValueError("Image dimensions must be positive integers")
        
        if not (self.OVERLAP_MIN_RATIO <= target_overlap_ratio <= self.OVERLAP_MAX_RATIO):
            raise ValueError(f"Target overlap must be between {self.OVERLAP_MIN_RATIO:.0%} and {self.OVERLAP_MAX_RATIO:.0%}")
        
        self.image_height = image_height
        self.image_width = image_width
        self.target_overlap_ratio = target_overlap_ratio
        self.target_overlap_pixels = int(self.PATCH_SIZE * target_overlap_ratio)
        
        # Calculated values (computed lazily)
        self._num_patches = None
        self._actual_overlap = None
        self._actual_overlap_ratio = None
        self._patches = None
    
    def calculate_patches(self, verbose: bool = False) -> Tuple[int, int, float]:
        """
        Calculate optimal number of patches and overlap distance.
        
        Uses mathematical framework to find optimal patch count that results in
        overlap percentage closest to target (default 25%).
        
        Args:
            verbose: If True, print calculation details
        
        Returns:
            Tuple of (num_patches, overlap_pixels, overlap_ratio)
        
        Examples:
            >>> slicer = ImageSlicer(8000)
            >>> n, k, ratio = slicer.calculate_patches()
            >>> print(f"Patches: {n}, Overlap: {k}px ({ratio:.1%})")
            Patches: 9, Overlap: 350px (29.2%)
        """
        if self._num_patches is not None:
            return self._num_patches, self._actual_overlap, self._actual_overlap_ratio
        
        # Special case: image height <= patch size
        if self.image_height <= self.PATCH_SIZE:
            self._num_patches = 1
            self._actual_overlap = 0
            self._actual_overlap_ratio = 0.0
            return self._num_patches, self._actual_overlap, self._actual_overlap_ratio
        
        best_n = None
        best_k = None
        best_error = float('inf')
        
        # Try all possible k values in valid range
        for k in range(self.OVERLAP_MIN_PIXELS, self.OVERLAP_MAX_PIXELS + 1):
            # Calculate required number of patches
            # Formula: n = ceil((h₀ + k) / (1200 - k))
            n_float = (self.image_height + k) / (self.PATCH_SIZE - k)
            n = int(np.ceil(n_float))
            
            # Verify with reverse formula
            # k = (1200·n - h₀) / (n - 1)
            if n > 1:
                k_actual = (self.PATCH_SIZE * n - self.image_height) / (n - 1)
            else:
                k_actual = 0
            
            # Check if actual overlap is in valid range
            if self.OVERLAP_MIN_PIXELS <= k_actual <= self.OVERLAP_MAX_PIXELS:
                overlap_ratio = k_actual / self.PATCH_SIZE
                error = abs(overlap_ratio - self.target_overlap_ratio)
                
                if error < best_error:
                    best_error = error
                    best_n = n
                    best_k = k_actual
        
        if best_n is None:
            # Fallback: use midpoint overlap if no perfect match
            warnings.warn("Could not find perfect overlap match, using fallback calculation")
            k = (self.OVERLAP_MIN_PIXELS + self.OVERLAP_MAX_PIXELS) // 2
            best_n = int(np.ceil((self.image_height + k) / (self.PATCH_SIZE - k)))
            if best_n > 1:
                best_k = (self.PATCH_SIZE * best_n - self.image_height) / (best_n - 1)
            else:
                best_k = 0
        
        self._num_patches = best_n
        self._actual_overlap = int(best_k)
        self._actual_overlap_ratio = best_k / self.PATCH_SIZE
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Image Slicing Calculation Results")
            print(f"{'='*60}")
            print(f"Original Image Dimensions: {self.image_width} × {self.image_height} px")
            print(f"Aspect Ratio: 1:{self.image_height/self.image_width:.2f}")
            print(f"Target Overlap: {self.target_overlap_ratio:.0%}")
            print(f"\nResults:")
            print(f"  Number of Patches: {self._num_patches}")
            print(f"  Overlap Distance: {self._actual_overlap} pixels")
            print(f"  Actual Overlap: {self._actual_overlap_ratio:.2%}")
            print(f"  Coverage: 100%")
            print(f"  Redundancy: {(self._num_patches * self.PATCH_SIZE - self.image_height) / self.image_height:.1%}")
            print(f"{'='*60}\n")
        
        return self._num_patches, self._actual_overlap, self._actual_overlap_ratio
    
    def generate_patch_coordinates(self) -> List[PatchInfo]:
        """
        Generate (x, y) coordinates for all patches.
        
        Returns:
            List of PatchInfo objects containing coordinate information
        
        Examples:
            >>> slicer = ImageSlicer(8000)
            >>> patches = slicer.generate_patch_coordinates()
            >>> for p in patches:
            ...     print(f"Patch {p.patch_id}: y=[{p.y_start}, {p.y_end}]")
        """
        if self._patches is not None:
            return self._patches
        
        n, k, _ = self.calculate_patches()
        patches = []
        
        for i in range(1, n + 1):
            # Position formula for i-th patch:
            # y_start = (i-1) * (1200 - k)
            # y_end = y_start + 1200
            y_start = (i - 1) * (self.PATCH_SIZE - k)
            y_end = min(y_start + self.PATCH_SIZE, self.image_height)
            
            patch = PatchInfo(
                patch_id=i,
                y_start=y_start,
                y_end=y_end,
                x_start=0,
                x_end=self.image_width
            )
            
            # Set overlap info
            if i < n:
                patch.overlap_with_next = k
                patch.overlap_percentage = (k / self.PATCH_SIZE) * 100
            
            patches.append(patch)
        
        self._patches = patches
        return patches
    
    def verify_coverage(self) -> bool:
        """
        Verify that patches provide 100% coverage of original image.
        
        Returns:
            True if coverage is verified, raises AssertionError otherwise
        
        Raises:
            AssertionError: If coverage verification fails
        """
        patches = self.generate_patch_coordinates()
        n, k, _ = self.calculate_patches()
        
        # Verify mathematical relationship
        theoretical_coverage = n * self.PATCH_SIZE - (n - 1) * k
        
        assert theoretical_coverage == self.image_height, \
            f"Coverage mismatch: {theoretical_coverage} != {self.image_height}"
        
        # Verify all pixels are covered
        assert patches[-1].y_end >= self.image_height, \
            f"Last patch doesn't reach image height: {patches[-1].y_end} < {self.image_height}"
        
        print("✓ Coverage verification passed: 100% coverage confirmed")
        return True
    
    def get_statistics(self) -> Dict:
        """
        Get detailed statistics about the slicing configuration.
        
        Returns:
            Dictionary containing various metrics and statistics
        """
        n, k, overlap_ratio = self.calculate_patches()
        patches = self.generate_patch_coordinates()
        
        # Calculate redundancy
        total_patch_pixels = n * self.PATCH_SIZE * self.image_width
        original_pixels = self.image_height * self.image_width
        redundancy = (total_patch_pixels - original_pixels) / original_pixels
        
        return {
            'original_dimensions': f"{self.image_width} × {self.image_height}",
            'aspect_ratio': f"1:{self.image_height/self.image_width:.2f}",
            'num_patches': n,
            'patch_size': f"{self.PATCH_SIZE} × {self.PATCH_SIZE}",
            'overlap_pixels': int(k),
            'overlap_percentage': f"{overlap_ratio:.2%}",
            'coverage': "100%",
            'redundancy': f"{redundancy:.1%}",
            'total_patch_pixels': total_patch_pixels,
            'original_pixels': original_pixels,
            'patches': patches
        }
    
    def print_summary(self):
        """Print a nicely formatted summary of the slicing configuration."""
        stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("IMAGE SLICING SUMMARY")
        print("="*70)
        print(f"\nOriginal Image:")
        print(f"  Dimensions: {stats['original_dimensions']} pixels")
        print(f"  Aspect Ratio: {stats['aspect_ratio']}")
        print(f"\nSlicing Configuration:")
        print(f"  Number of Patches: {stats['num_patches']}")
        print(f"  Patch Size: {stats['patch_size']} pixels")
        print(f"  Overlap: {stats['overlap_pixels']} pixels ({stats['overlap_percentage']})")
        print(f"\nResults:")
        print(f"  Coverage: {stats['coverage']}")
        print(f"  Redundancy: {stats['redundancy']}")
        print(f"  Total Patch Pixels: {stats['total_patch_pixels']:,}")
        print(f"  Original Pixels: {stats['original_pixels']:,}")
        print("="*70 + "\n")


class PatchExtractor:
    """
    Extract and save patches from actual image files.
    
    Supports common image formats (JPEG, PNG, TIFF, etc.)
    """
    
    def __init__(self, image_path: str, output_dir: str = "./patches"):
        """
        Initialize PatchExtractor.
        
        Args:
            image_path: Path to input image
            output_dir: Directory where patches will be saved
        
        Raises:
            FileNotFoundError: If image file doesn't exist
        """
        self.image_path = Path(image_path)
        self.output_dir = Path(output_dir)
        
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Import PIL only when needed
        try:
            from PIL import Image
            self.Image = Image
        except ImportError:
            raise ImportError("PIL/Pillow is required for image extraction. Install with: pip install Pillow")
        
        # Load image info
        with self.Image.open(self.image_path) as img:
            self.image_width, self.image_height = img.size
            self.image_mode = img.mode
        
        # Initialize slicer
        self.slicer = ImageSlicer(self.image_height, self.image_width)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_patches(self, padding_mode: str = 'reflect') -> List[str]:
        """
        Extract all patches from the image.
        
        Args:
            padding_mode: How to handle last patch if it's shorter than 1200px
                         ('reflect', 'edge', 'constant' using numpy padding modes)
        
        Returns:
            List of saved patch file paths
        
        Examples:
            >>> extractor = PatchExtractor("satellite_image.tif")
            >>> patch_files = extractor.extract_patches()
            >>> print(f"Extracted {len(patch_files)} patches")
        """
        patches = self.slicer.generate_patch_coordinates()
        patch_files = []
        
        print(f"\nExtracting {len(patches)} patches from {self.image_path.name}...")
        
        with self.Image.open(self.image_path) as img:
            for patch in patches:
                # Extract patch region
                crop_box = (patch.x_start, patch.y_start, patch.x_end, patch.y_end)
                patch_img = img.crop(crop_box)
                
                # Convert to numpy array for potential padding
                patch_array = np.array(patch_img)
                
                # Handle patches that might be shorter than 1200px (especially last patch)
                if patch_array.shape[0] < self.PATCH_SIZE:
                    patch_array = np._pad_with_mode(
                        patch_array,
                        ((0, self.PATCH_SIZE - patch_array.shape[0]), (0, 0), (0, 0)) 
                        if len(patch_array.shape) == 3 else
                        ((0, self.PATCH_SIZE - patch_array.shape[0]), (0, 0)),
                        padding_mode
                    )
                    patch_img = self.Image.fromarray(patch_array)
                
                # Save patch
                patch_filename = self.output_dir / f"patch_{patch.patch_id:03d}.png"
                patch_img.save(patch_filename, quality=95)
                patch_files.append(str(patch_filename))
                
                print(f"  ✓ Patch {patch.patch_id}/{len(patches)}: {patch_filename.name}")
        
        print(f"\nExtraction complete! {len(patch_files)} patches saved to {self.output_dir}\n")
        return patch_files
    
    def save_metadata_csv(self, filename: str = "patch_metadata.csv") -> str:
        """
        Save patch coordinates and metadata to CSV file.
        
        Args:
            filename: Name of CSV file to create
        
        Returns:
            Path to saved CSV file
        """
        patches = self.slicer.generate_patch_coordinates()
        csv_path = self.output_dir / filename
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['patch_id', 'x_start', 'x_end', 'y_start', 'y_end', 
                         'width', 'height', 'overlap_with_next', 'overlap_percentage']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for patch in patches:
                writer.writerow(patch.to_dict())
        
        print(f"✓ Metadata saved to {csv_path}")
        return str(csv_path)


# Utility functions
def calculate_overlap_for_height(height: int, num_patches: Optional[int] = None) -> Dict:
    """
    Calculate optimal overlap for a given image height.
    
    Args:
        height: Image height in pixels
        num_patches: If specified, calculate overlap for this many patches.
                    Otherwise, find optimal patch count.
    
    Returns:
        Dictionary with calculation results
    
    Examples:
        >>> result = calculate_overlap_for_height(8000)
        >>> print(f"Patches: {result['num_patches']}, Overlap: {result['overlap_pixels']}px")
    """
    slicer = ImageSlicer(height)
    
    if num_patches:
        # Calculate overlap for specific patch count
        k = (ImageSlicer.PATCH_SIZE * num_patches - height) / (num_patches - 1)
        overlap_ratio = k / ImageSlicer.PATCH_SIZE
        
        if not (ImageSlicer.OVERLAP_MIN_PIXELS <= k <= ImageSlicer.OVERLAP_MAX_PIXELS):
            print(f"Warning: Calculated overlap {k:.0f}px is outside valid range "
                  f"[{ImageSlicer.OVERLAP_MIN_PIXELS}, {ImageSlicer.OVERLAP_MAX_PIXELS}]px")
        
        return {
            'height': height,
            'num_patches': num_patches,
            'overlap_pixels': int(k),
            'overlap_percentage': f"{overlap_ratio:.2%}",
            'coverage': 100,
            'is_valid': ImageSlicer.OVERLAP_MIN_PIXELS <= k <= ImageSlicer.OVERLAP_MAX_PIXELS
        }
    else:
        # Find optimal patch count
        n, k, ratio = slicer.calculate_patches()
        return {
            'height': height,
            'num_patches': n,
            'overlap_pixels': int(k),
            'overlap_percentage': f"{ratio:.2%}",
            'coverage': 100,
            'is_valid': True
        }


def suggest_image_dimensions(height: int, aspect_ratio: float = 1.0) -> Dict:
    """
    Suggest image dimensions that work well with the slicing algorithm.
    
    Args:
        height: Desired image height
        aspect_ratio: Width to height ratio for the full image
    
    Returns:
        Dictionary with suggested dimensions and slicing info
    """
    width = int(height * aspect_ratio)
    slicer = ImageSlicer(height, width)
    stats = slicer.get_statistics()
    
    return {
        'suggested_width': width,
        'suggested_height': height,
        'num_patches': stats['num_patches'],
        'overlap_pixels': stats['overlap_pixels'],
        'aspect_ratio': stats['aspect_ratio'],
        'redundancy': stats['redundancy']
    }


if __name__ == "__main__":
    # Example usage
    print("Image Slicing Module - Direct Execution Example")
    print("=" * 70)
    
    # Example 1: Basic slicing calculation
    print("\nExample 1: Calculate slicing for 8000px height image")
    slicer = ImageSlicer(8000)
    n, k, ratio = slicer.calculate_patches(verbose=True)
    slicer.print_summary()
    
    # Example 2: Verify coverage
    print("\nExample 2: Verify coverage")
    slicer.verify_coverage()
    
    # Example 3: Get statistics
    print("\nExample 3: Get detailed statistics")
    stats = slicer.get_statistics()
    print(f"Number of patches: {stats['num_patches']}")
    print(f"Overlap: {stats['overlap_pixels']}px ({stats['overlap_percentage']})")
    print(f"Redundancy: {stats['redundancy']}")
    
    # Example 4: Utility function
    print("\nExample 4: Calculate overlap for custom heights")
    for h in [5000, 8000, 10000]:
        result = calculate_overlap_for_height(h)
        print(f"Height {h}px → {result['num_patches']} patches, "
              f"overlap {result['overlap_pixels']}px ({result['overlap_percentage']})")
