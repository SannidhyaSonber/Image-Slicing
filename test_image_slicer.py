"""
Unit Tests for Image Slicing Module

Test coverage includes:
- Mathematical equation verification
- Patch generation and coordinates
- Edge cases and error handling
- Coverage and redundancy calculations
- Performance characteristics

Run tests with: python -m pytest test_image_slicer.py -v
Or simply: python test_image_slicer.py
"""

import unittest
import numpy as np
from image_slicer import ImageSlicer, PatchInfo, calculate_overlap_for_height


class TestImageSlicerBasics(unittest.TestCase):
    """Test basic ImageSlicer functionality."""
    
    def test_initialization_valid(self):
        """Test valid initialization."""
        slicer = ImageSlicer(8000, 1200)
        self.assertEqual(slicer.image_height, 8000)
        self.assertEqual(slicer.image_width, 1200)
    
    def test_initialization_invalid_dimensions(self):
        """Test initialization with invalid dimensions."""
        with self.assertRaises(ValueError):
            ImageSlicer(-1000, 1200)
        
        with self.assertRaises(ValueError):
            ImageSlicer(8000, 0)
    
    def test_initialization_invalid_overlap_ratio(self):
        """Test initialization with invalid overlap ratio."""
        with self.assertRaises(ValueError):
            ImageSlicer(8000, 1200, target_overlap_ratio=0.1)  # Too low
        
        with self.assertRaises(ValueError):
            ImageSlicer(8000, 1200, target_overlap_ratio=0.4)  # Too high


class TestPatchCalculation(unittest.TestCase):
    """Test patch calculation mathematics."""
    
    def test_standard_case_8000(self):
        """Test standard case: 1200 × 8000 image."""
        slicer = ImageSlicer(8000, 1200)
        n, k, ratio = slicer.calculate_patches()
        
        # Expected: 9 patches with ~350px overlap (29.2%)
        self.assertEqual(n, 9)
        self.assertAlmostEqual(k, 350, delta=5)
        self.assertGreaterEqual(ratio, 0.20)
        self.assertLessEqual(ratio, 0.30)
    
    def test_small_image(self):
        """Test small image (less than 1200px)."""
        slicer = ImageSlicer(800, 1200)
        n, k, ratio = slicer.calculate_patches()
        
        # Should still return valid patch configuration
        self.assertGreaterEqual(n, 1)
        self.assertGreaterEqual(ratio, 0.0)
    
    def test_very_large_image(self):
        """Test very large image."""
        slicer = ImageSlicer(50000, 1200)
        n, k, ratio = slicer.calculate_patches()
        
        # Should handle large images
        self.assertGreater(n, 10)
        self.assertGreaterEqual(ratio, 0.20)
        self.assertLessEqual(ratio, 0.30)
    
    def test_different_overlap_targets(self):
        """Test different overlap targets."""
        height = 8000
        
        # Test 20% target (minimum)
        slicer_20 = ImageSlicer(height, target_overlap_ratio=0.20)
        n_20, k_20, ratio_20 = slicer_20.calculate_patches()
        
        # Test 25% target (midpoint)
        slicer_25 = ImageSlicer(height, target_overlap_ratio=0.25)
        n_25, k_25, ratio_25 = slicer_25.calculate_patches()
        
        # Test 30% target (maximum)
        slicer_30 = ImageSlicer(height, target_overlap_ratio=0.30)
        n_30, k_30, ratio_30 = slicer_30.calculate_patches()
        
        # All should be valid
        for ratio in [ratio_20, ratio_25, ratio_30]:
            self.assertGreaterEqual(ratio, 0.20)
            self.assertLessEqual(ratio, 0.30)


class TestMathematicalEquations(unittest.TestCase):
    """Test the fundamental mathematical equations."""
    
    def test_fundamental_equation(self):
        """Test: h₀ = n · 1200 - (n-1) · k"""
        test_cases = [
            (5000, 1200),
            (8000, 1200),
            (10000, 1200),
            (6500, 1200),
        ]
        
        for height, width in test_cases:
            slicer = ImageSlicer(height, width)
            n, k, _ = slicer.calculate_patches()
            
            # Verify: h₀ = n·1200 - (n-1)·k
            calculated_height = n * ImageSlicer.PATCH_SIZE - (n - 1) * int(k)
            self.assertEqual(calculated_height, height,
                           f"Equation failed for height {height}: {calculated_height} != {height}")
    
    def test_solve_for_n_formula(self):
        """Test: n = (h₀ + k) / (1200 - k)"""
        height = 8000
        slicer = ImageSlicer(height)
        
        # Try different k values
        for k in range(ImageSlicer.OVERLAP_MIN_PIXELS, ImageSlicer.OVERLAP_MAX_PIXELS + 50, 10):
            n_calculated = int(np.ceil((height + k) / (ImageSlicer.PATCH_SIZE - k)))
            
            # Verify by checking backward formula
            if n_calculated > 1:
                k_back = (ImageSlicer.PATCH_SIZE * n_calculated - height) / (n_calculated - 1)
                coverage = n_calculated * ImageSlicer.PATCH_SIZE - (n_calculated - 1) * k_back
                
                self.assertAlmostEqual(coverage, height, delta=1)
    
    def test_solve_for_k_formula(self):
        """Test: k = (1200·n - h₀) / (n - 1)"""
        height = 8000
        
        for n in range(2, 15):
            k = (ImageSlicer.PATCH_SIZE * n - height) / (n - 1)
            
            # Verify backward
            coverage = n * ImageSlicer.PATCH_SIZE - (n - 1) * k
            self.assertAlmostEqual(coverage, height, delta=0.1)


class TestPatchGeneration(unittest.TestCase):
    """Test patch coordinate generation."""
    
    def test_patch_count_matches_calculation(self):
        """Test that generated patches match calculated count."""
        slicer = ImageSlicer(8000)
        n, _, _ = slicer.calculate_patches()
        patches = slicer.generate_patch_coordinates()
        
        self.assertEqual(len(patches), n)
    
    def test_patch_boundaries(self):
        """Test that patch boundaries are correct."""
        slicer = ImageSlicer(8000)
        patches = slicer.generate_patch_coordinates()
        
        # First patch should start at 0
        self.assertEqual(patches[0].y_start, 0)
        
        # Last patch should cover full height
        self.assertGreaterEqual(patches[-1].y_end, 8000)
        
        # All patches should be 1200px high (except possibly last if padded)
        for patch in patches[:-1]:
            self.assertEqual(patch.height, ImageSlicer.PATCH_SIZE)
    
    def test_patch_continuity(self):
        """Test that patches form continuous coverage."""
        slicer = ImageSlicer(8000)
        patches = slicer.generate_patch_coordinates()
        
        for i in range(len(patches) - 1):
            # Next patch should start before current patch ends (overlap)
            self.assertLess(patches[i + 1].y_start, patches[i].y_end)
    
    def test_patch_coordinates_immutability(self):
        """Test that patch coordinates are consistent across calls."""
        slicer = ImageSlicer(8000)
        
        patches1 = slicer.generate_patch_coordinates()
        patches2 = slicer.generate_patch_coordinates()
        
        # Should return same object (caching)
        self.assertIs(patches1, patches2)
        
        # Coordinates should be identical
        for p1, p2 in zip(patches1, patches2):
            self.assertEqual(p1.patch_id, p2.patch_id)
            self.assertEqual(p1.y_start, p2.y_start)
            self.assertEqual(p1.y_end, p2.y_end)


class TestCoverage(unittest.TestCase):
    """Test coverage and completeness."""
    
    def test_100_percent_coverage(self):
        """Test that all pixels are covered."""
        test_heights = [1000, 5000, 8000, 10000]
        
        for height in test_heights:
            slicer = ImageSlicer(height)
            n, k, _ = slicer.calculate_patches()
            
            # Coverage = unique pixels / original pixels
            total_patch_pixels = n * ImageSlicer.PATCH_SIZE
            overlap_pixels = (n - 1) * int(k)
            unique_pixels = total_patch_pixels - overlap_pixels
            
            coverage = unique_pixels / height
            self.assertAlmostEqual(coverage, 1.0, places=5,
                                 msg=f"Coverage failed for height {height}: {coverage:.4f}")
    
    def test_coverage_verification(self):
        """Test the verify_coverage method."""
        slicer = ImageSlicer(8000)
        result = slicer.verify_coverage()
        self.assertTrue(result)
    
    def test_redundancy_within_limits(self):
        """Test that redundancy is reasonable."""
        test_cases = [
            (5000, 0.35),   # Expected max ~35%
            (8000, 0.35),   # Expected max ~35%
            (10000, 0.30),  # Expected max ~30%
        ]
        
        for height, max_redundancy in test_cases:
            slicer = ImageSlicer(height)
            n, k, _ = slicer.calculate_patches()
            
            total_patch_pixels = n * ImageSlicer.PATCH_SIZE
            original_pixels = height
            redundancy = (total_patch_pixels - original_pixels) / original_pixels
            
            self.assertLessEqual(redundancy, max_redundancy,
                               f"Redundancy {redundancy:.1%} exceeds limit {max_redundancy:.1%}")


class TestStatistics(unittest.TestCase):
    """Test statistics and reporting."""
    
    def test_get_statistics(self):
        """Test statistics dictionary."""
        slicer = ImageSlicer(8000)
        stats = slicer.get_statistics()
        
        # Check required keys
        required_keys = ['original_dimensions', 'num_patches', 'patch_size',
                        'overlap_pixels', 'overlap_percentage', 'coverage', 'redundancy']
        for key in required_keys:
            self.assertIn(key, stats)
        
        # Check values
        self.assertEqual(stats['num_patches'], 9)
        self.assertEqual(stats['coverage'], "100%")
    
    def test_patch_info_to_dict(self):
        """Test PatchInfo.to_dict() method."""
        patch = PatchInfo(1, 0, 1200)
        patch_dict = patch.to_dict()
        
        self.assertEqual(patch_dict['patch_id'], 1)
        self.assertEqual(patch_dict['y_start'], 0)
        self.assertEqual(patch_dict['y_end'], 1200)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_calculate_overlap_for_height(self):
        """Test calculate_overlap_for_height function."""
        result = calculate_overlap_for_height(8000)
        
        self.assertEqual(result['height'], 8000)
        self.assertEqual(result['num_patches'], 9)
        self.assertEqual(result['coverage'], 100)
        self.assertTrue(result['is_valid'])
    
    def test_calculate_overlap_for_specific_patches(self):
        """Test calculating overlap for specific patch count."""
        result = calculate_overlap_for_height(8000, num_patches=10)
        
        self.assertEqual(result['num_patches'], 10)
        # Should calculate overlap for 10 patches
        k = result['overlap_pixels']
        self.assertGreater(k, 0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_height_equals_patch_size(self):
        """Test when height equals patch size (1200px)."""
        slicer = ImageSlicer(1200, 1200)
        n, k, ratio = slicer.calculate_patches()
        
        self.assertEqual(n, 1)
        self.assertEqual(k, 0)
    
    def test_height_slightly_larger_than_patch_size(self):
        """Test when height is slightly larger than patch size."""
        slicer = ImageSlicer(1300, 1200)
        n, k, ratio = slicer.calculate_patches()
        
        self.assertGreaterEqual(n, 1)
    
    def test_zero_width(self):
        """Test that zero width raises error."""
        with self.assertRaises(ValueError):
            ImageSlicer(8000, 0)
    
    def test_negative_dimensions(self):
        """Test that negative dimensions raise error."""
        with self.assertRaises(ValueError):
            ImageSlicer(-1, 1200)


class TestPerformance(unittest.TestCase):
    """Test performance characteristics."""
    
    def test_calculation_speed(self):
        """Test that calculation is fast."""
        import time
        
        slicer = ImageSlicer(8000)
        
        start = time.time()
        for _ in range(1000):
            slicer.calculate_patches()
        elapsed = time.time() - start
        
        # Should be able to calculate 1000 times in < 1 second
        self.assertLess(elapsed, 1.0)
    
    def test_large_image_handling(self):
        """Test handling of very large images."""
        slicer = ImageSlicer(1000000, 1200)  # 1 million pixels tall
        n, k, ratio = slicer.calculate_patches()
        
        # Should still work and give reasonable results
        self.assertGreater(n, 1)
        self.assertGreaterEqual(ratio, 0.20)
        self.assertLessEqual(ratio, 0.30)


class TestDifferentAspectRatios(unittest.TestCase):
    """Test with different image aspect ratios."""
    
    def test_square_image(self):
        """Test with square image (1200 × 1200)."""
        slicer = ImageSlicer(1200, 1200)
        n, k, ratio = slicer.calculate_patches()
        
        self.assertEqual(n, 1)
    
    def test_wide_image(self):
        """Test with wide image (2400 × 8000)."""
        slicer = ImageSlicer(8000, 2400)
        n, k, ratio = slicer.calculate_patches()
        
        self.assertGreater(n, 1)
    
    def test_very_narrow_image(self):
        """Test with very narrow image (600 × 8000)."""
        slicer = ImageSlicer(8000, 600)
        n, k, ratio = slicer.calculate_patches()
        
        self.assertGreater(n, 1)


def run_tests():
    """Run all tests with verbose output."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestImageSlicerBasics))
    suite.addTests(loader.loadTestsFromTestCase(TestPatchCalculation))
    suite.addTests(loader.loadTestsFromTestCase(TestMathematicalEquations))
    suite.addTests(loader.loadTestsFromTestCase(TestPatchGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestCoverage))
    suite.addTests(loader.loadTestsFromTestCase(TestStatistics))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilityFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestDifferentAspectRatios))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed. See details above.")
    
    print("="*80 + "\n")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
