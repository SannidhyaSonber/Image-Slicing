"""
Image Slicing Implementation - Examples and Demonstrations

This script demonstrates practical usage of the ImageSlicer class including:
- Basic slicing calculations
- Real image processing
- Batch processing
- Visualization
- Performance analysis

Run this script directly: python example_usage.py
"""

import sys
from pathlib import Path
import numpy as np
from typing import List
import time

# Import the ImageSlicer module
from image_slicer import ImageSlicer, PatchExtractor, calculate_overlap_for_height, suggest_image_dimensions


def example_1_basic_calculation():
    """Example 1: Basic slicing calculation for a rectangular image."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Slicing Calculation")
    print("="*80)
    
    # Create slicer for 8000 × 1200 image
    print("\nScenario: Satellite image 1200px wide × 8000px tall")
    slicer = ImageSlicer(image_height=8000, image_width=1200)
    
    # Calculate patches with verbose output
    n, k, overlap_ratio = slicer.calculate_patches(verbose=True)
    
    # Display patches
    print("\nPatch Coordinates:")
    print("-" * 80)
    patches = slicer.generate_patch_coordinates()
    
    print(f"{'Patch #':<10} {'Y Start':<12} {'Y End':<12} {'Height':<10} {'Overlap w/ Next':<15}")
    print("-" * 80)
    for patch in patches:
        overlap_str = f"{patch.overlap_with_next}px ({patch.overlap_percentage:.1f}%)" if patch.overlap_with_next > 0 else "N/A"
        print(f"{patch.patch_id:<10} {patch.y_start:<12} {patch.y_end:<12} "
              f"{patch.height:<10} {overlap_str:<15}")
    
    # Summary
    slicer.print_summary()


def example_2_different_heights():
    """Example 2: Compare slicing results for different image heights."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Slicing Different Image Heights")
    print("="*80)
    
    heights = [5000, 6000, 7000, 8000, 9000, 10000]
    
    print(f"\n{'Height (px)':<15} {'Patches':<12} {'Overlap (px)':<15} {'Overlap %':<12} {'Redundancy':<12}")
    print("-" * 80)
    
    for height in heights:
        slicer = ImageSlicer(height)
        stats = slicer.get_statistics()
        n, k, ratio = slicer.calculate_patches()
        
        # Calculate redundancy
        total_pixels = n * ImageSlicer.PATCH_SIZE * 1200
        original_pixels = height * 1200
        redundancy = (total_pixels - original_pixels) / original_pixels
        
        print(f"{height:<15} {n:<12} {int(k):<15} {ratio:<12.1%} {redundancy:<12.1%}")


def example_3_overlap_analysis():
    """Example 3: Analyze relationship between overlap and patch count."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Overlap vs Patch Count Analysis")
    print("="*80)
    
    height = 8000
    print(f"\nAnalyzing image height: {height}px")
    print(f"{'Overlap %':<15} {'Overlap (px)':<15} {'Patches':<12} {'Valid?':<10}")
    print("-" * 80)
    
    for overlap_ratio in [0.15, 0.20, 0.25, 0.30, 0.35]:
        slicer = ImageSlicer(height, target_overlap_ratio=overlap_ratio)
        n, k, actual_ratio = slicer.calculate_patches()
        
        is_valid = (ImageSlicer.OVERLAP_MIN_PIXELS <= k <= ImageSlicer.OVERLAP_MAX_PIXELS)
        valid_str = "✓ Yes" if is_valid else "✗ No"
        
        print(f"{overlap_ratio:<15.0%} {int(k):<15} {n:<12} {valid_str:<10}")


def example_4_aspect_ratios():
    """Example 4: Slicing images with different aspect ratios."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Different Aspect Ratios")
    print("="*80)
    
    test_cases = [
        (1200, 5000, "Narrow (1:4.17)"),
        (1200, 8000, "Standard (1:6.67)"),
        (1200, 10000, "Very tall (1:8.33)"),
        (2400, 8000, "Wider (1:3.33)"),
        (600, 8000, "Very narrow (1:13.33)")
    ]
    
    print(f"\n{'Description':<25} {'Dimensions':<20} {'Patches':<10} {'Overlap':<15}")
    print("-" * 80)
    
    for width, height, desc in test_cases:
        slicer = ImageSlicer(height, width)
        n, k, ratio = slicer.calculate_patches()
        dim_str = f"{width} × {height}"
        overlap_str = f"{int(k)}px ({ratio:.1%})"
        print(f"{desc:<25} {dim_str:<20} {n:<10} {overlap_str:<15}")


def example_5_mathematical_verification():
    """Example 5: Verify mathematical relationships."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Mathematical Verification")
    print("="*80)
    
    height = 8000
    slicer = ImageSlicer(height)
    n, k, _ = slicer.calculate_patches()
    
    print(f"\nImage height (h₀): {height} pixels")
    print(f"Number of patches (n): {n}")
    print(f"Overlap distance (k): {int(k)} pixels")
    
    # Verify the fundamental equation: h₀ = n·1200 - (n-1)·k
    calculated_height = n * ImageSlicer.PATCH_SIZE - (n - 1) * int(k)
    
    print(f"\nFundamental Equation Verification:")
    print(f"  h₀ = n · 1200 - (n-1) · k")
    print(f"  {height} = {n} · 1200 - ({n}-1) · {int(k)}")
    print(f"  {height} = {n * ImageSlicer.PATCH_SIZE} - {(n-1) * int(k)}")
    print(f"  {height} = {calculated_height}")
    print(f"  ✓ Equation verified!" if calculated_height == height else f"  ✗ Mismatch!")
    
    # Calculate coverage
    print(f"\nCoverage Analysis:")
    total_patch_pixels = n * ImageSlicer.PATCH_SIZE
    total_overlap_pixels = (n - 1) * int(k)
    unique_pixels = total_patch_pixels - total_overlap_pixels
    
    print(f"  Total patch pixels: {n} × {ImageSlicer.PATCH_SIZE} = {total_patch_pixels:,}")
    print(f"  Total overlap pixels: ({n}-1) × {int(k)} = {total_overlap_pixels:,}")
    print(f"  Unique pixels: {total_patch_pixels:,} - {total_overlap_pixels:,} = {unique_pixels:,}")
    print(f"  Coverage: {unique_pixels}/{height} = {unique_pixels/height:.1%}")
    
    # Calculate redundancy
    redundancy = total_overlap_pixels / height
    print(f"  Redundancy: {total_overlap_pixels:,} / {height:,} = {redundancy:.1%}")


def example_6_create_synthetic_image_and_slice():
    """Example 6: Create a synthetic image and slice it."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Synthetic Image Creation and Slicing")
    print("="*80)
    
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("⚠ PIL/Pillow not installed. Skipping synthetic image example.")
        print("   Install with: pip install Pillow")
        return
    
    # Create synthetic image
    width, height = 1200, 8000
    print(f"\nCreating synthetic image: {width} × {height} pixels")
    
    # Create image with gradient pattern
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create gradient bands to visualize patches
    band_height = height // 4
    colors = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0)     # Yellow
    ]
    
    for i, color in enumerate(colors):
        start = i * band_height
        end = min((i + 1) * band_height, height)
        img_array[start:end, :] = color
    
    # Add text labels
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)
    
    # Draw section labels
    for i in range(4):
        y = i * band_height + band_height // 2
        draw.text((50, y), f"Section {i+1}", fill=(255, 255, 255))
    
    # Save synthetic image
    synthetic_path = Path("/tmp/synthetic_image.png")
    img.save(synthetic_path)
    print(f"✓ Synthetic image saved: {synthetic_path}")
    
    # Now slice it
    print(f"\nSlicing the synthetic image...")
    slicer = ImageSlicer(height, width)
    n, k, ratio = slicer.calculate_patches(verbose=True)
    
    # Extract patches
    try:
        extractor = PatchExtractor(str(synthetic_path), output_dir="/tmp/patches")
        patch_files = extractor.extract_patches()
        print(f"\n✓ Successfully extracted {len(patch_files)} patches")
        
        # Save metadata
        csv_path = extractor.save_metadata_csv()
        print(f"✓ Metadata saved to: {csv_path}")
        
    except Exception as e:
        print(f"✗ Error extracting patches: {e}")


def example_7_performance_benchmark():
    """Example 7: Performance benchmarking for different image sizes."""
    print("\n" + "="*80)
    print("EXAMPLE 7: Performance Benchmarking")
    print("="*80)
    
    sizes = [1000, 5000, 10000, 50000, 100000]
    
    print(f"\n{'Height (px)':<15} {'Calc Time (ms)':<20} {'Patches':<12} {'Overlap':<15}")
    print("-" * 80)
    
    for size in sizes:
        start = time.time()
        slicer = ImageSlicer(size)
        n, k, ratio = slicer.calculate_patches()
        elapsed = (time.time() - start) * 1000  # Convert to milliseconds
        
        overlap_str = f"{int(k)}px ({ratio:.1%})"
        print(f"{size:<15} {elapsed:<20.3f} {n:<12} {overlap_str:<15}")


def example_8_batch_processing():
    """Example 8: Batch processing multiple images."""
    print("\n" + "="*80)
    print("EXAMPLE 8: Batch Processing Simulation")
    print("="*80)
    
    # Simulate processing multiple images
    images = [
        ("satellite_strip_1.tif", 1200, 7500),
        ("satellite_strip_2.tif", 1200, 8200),
        ("medical_scan.tif", 1200, 9000),
    ]
    
    print(f"\n{'Image Name':<30} {'Size':<20} {'Patches':<12} {'Overlap':<15}")
    print("-" * 80)
    
    total_patches = 0
    for name, width, height in images:
        slicer = ImageSlicer(height, width)
        n, k, ratio = slicer.calculate_patches()
        total_patches += n
        
        size_str = f"{width} × {height}"
        overlap_str = f"{int(k)}px ({ratio:.1%})"
        print(f"{name:<30} {size_str:<20} {n:<12} {overlap_str:<15}")
    
    print("-" * 80)
    print(f"{'TOTAL':<30} {'':<20} {total_patches:<12}")


def example_9_memory_efficiency():
    """Example 9: Analyze memory efficiency."""
    print("\n" + "="*80)
    print("EXAMPLE 9: Memory Efficiency Analysis")
    print("="*80)
    
    height = 8000
    slicer = ImageSlicer(height)
    stats = slicer.get_statistics()
    
    n, k, ratio = slicer.calculate_patches()
    
    # Assuming 3-channel image (RGB)
    bytes_per_pixel = 3
    
    original_size_mb = (1200 * height * bytes_per_pixel) / (1024 * 1024)
    patch_size_mb = (ImageSlicer.PATCH_SIZE * ImageSlicer.PATCH_SIZE * bytes_per_pixel) / (1024 * 1024)
    total_patches_mb = (n * patch_size_mb)
    
    print(f"\nMemory Analysis (RGB Image, 3 bytes/pixel):")
    print(f"  Original image: {original_size_mb:.2f} MB")
    print(f"  Single patch: {patch_size_mb:.2f} MB")
    print(f"  Total for all {n} patches: {total_patches_mb:.2f} MB")
    print(f"  Memory overhead due to overlap: {(total_patches_mb - original_size_mb):.2f} MB ({stats['redundancy']})")
    
    print(f"\nGPU Loading Strategy (assuming 2GB GPU memory):")
    gpu_memory_mb = 2048
    batch_size = max(1, int(gpu_memory_mb / patch_size_mb))
    num_batches = (n + batch_size - 1) // batch_size
    
    print(f"  Single patch size: {patch_size_mb:.2f} MB")
    print(f"  Maximum batch size: {batch_size} patches")
    print(f"  Number of batches needed: {num_batches}")
    print(f"  Memory per batch: {batch_size * patch_size_mb:.2f} MB")


def example_10_comparison_with_alternatives():
    """Example 10: Compare overlap-based slicing with alternative approaches."""
    print("\n" + "="*80)
    print("EXAMPLE 10: Comparison with Alternative Approaches")
    print("="*80)
    
    height = 8000
    width = 1200
    
    print(f"\nImage: {width} × {height} pixels (Aspect ratio: 1:{height/width:.2f})")
    print("\nApproach Comparison:")
    print("-" * 80)
    
    # Approach 1: Proposed method (overlap-based)
    slicer = ImageSlicer(height, width)
    n_proposed, k_proposed, ratio_proposed = slicer.calculate_patches()
    
    print(f"\n1. PROPOSED (Overlap-based):")
    print(f"   Patches: {n_proposed}")
    print(f"   Overlap: {int(k_proposed)}px ({ratio_proposed:.1%})")
    print(f"   Coverage: 100%")
    print(f"   Data Loss: 0%")
    print(f"   Redundancy: {(n_proposed * ImageSlicer.PATCH_SIZE - height) / height:.1%}")
    print(f"   ✓ Pros: Full coverage, contextual continuity, no distortion")
    print(f"   ✗ Cons: Some redundancy, requires overlap handling in training")
    
    # Approach 2: Resizing
    print(f"\n2. RESIZING to 1200×1200:")
    print(f"   Patches: 1")
    print(f"   Overlap: 0%")
    print(f"   Coverage: 100%")
    print(f"   Data Loss: None (but distorted)")
    print(f"   Redundancy: 0%")
    print(f"   Aspect Ratio Change: 1:{height/width:.2f} → 1:1")
    print(f"   ✓ Pros: Simple, no redundancy")
    print(f"   ✗ Cons: Severe distortion, information loss in details")
    
    # Approach 3: Center cropping
    crop_height = 1200
    num_crops = height // crop_height
    
    print(f"\n3. CENTER CROPPING to 1200×1200:")
    print(f"   Patches: {num_crops}")
    print(f"   Overlap: 0%")
    print(f"   Coverage: {(num_crops * crop_height / height):.0%}")
    print(f"   Data Loss: {(1 - num_crops * crop_height / height):.0%}")
    print(f"   Redundancy: 0%")
    print(f"   ✓ Pros: No redundancy, clean patches")
    print(f"   ✗ Cons: Data loss, boundary artifacts, information waste")
    
    # Approach 4: Padding
    print(f"\n4. PADDING to 1200×1200:")
    print(f"   Patches: 1")
    print(f"   Overlap: 0%")
    print(f"   Coverage: 100%")
    print(f"   Data Loss: 0%")
    print(f"   Artificial Padding: {((1200 * 1200 - height * width) / (1200 * 1200)):.0%}")
    print(f"   ✓ Pros: No data loss")
    print(f"   ✗ Cons: Artificial patterns, model bias, reduced information density")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + "IMAGE SLICING IMPLEMENTATION - COMPREHENSIVE EXAMPLES".center(78) + "║")
    print("║" + "Converting Rectangular Images to Square ML-Compatible Patches".center(78) + "║")
    print("╚" + "="*78 + "╝")
    
    examples = [
        ("Basic Calculation", example_1_basic_calculation),
        ("Different Heights", example_2_different_heights),
        ("Overlap Analysis", example_3_overlap_analysis),
        ("Aspect Ratios", example_4_aspect_ratios),
        ("Mathematical Verification", example_5_mathematical_verification),
        ("Synthetic Image", example_6_create_synthetic_image_and_slice),
        ("Performance Benchmark", example_7_performance_benchmark),
        ("Batch Processing", example_8_batch_processing),
        ("Memory Efficiency", example_9_memory_efficiency),
        ("Comparison with Alternatives", example_10_comparison_with_alternatives),
    ]
    
    print("\nAvailable Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print(f"\nRunning all {len(examples)} examples...\n")
    
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n✗ Error in {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY: Quick Reference")
    print("="*80)
    
    print("\nQuick Start Code:")
    print("""
    from image_slicer import ImageSlicer
    
    # Create slicer for 8000px tall image
    slicer = ImageSlicer(image_height=8000)
    
    # Calculate patches
    num_patches, overlap_pixels, overlap_ratio = slicer.calculate_patches(verbose=True)
    
    # Get all patch coordinates
    patches = slicer.generate_patch_coordinates()
    for patch in patches:
        print(f"Patch {patch.patch_id}: y={patch.y_start}-{patch.y_end}")
    
    # Verify coverage
    slicer.verify_coverage()
    
    # Get statistics
    stats = slicer.get_statistics()
    """)
    
    print("\n✓ All examples completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
