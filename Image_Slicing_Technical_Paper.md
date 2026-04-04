# Adaptive Patch Extraction with Controlled Overlap: A Method for Converting Rectangular Satellite/Medical Imagery to Square ML-Compatible Datasets

**Authors**: Sannidhya  
**Date**: April 2026  
**Category**: Technical Research Paper / Engineering Journal  
**Keywords**: Image Processing, Machine Learning, Data Preprocessing, Patch-based Learning, Geospatial Imagery

---

## Abstract

High-resolution satellite or medical imagery often exhibits extreme aspect ratios—typically 1:6 to 1:8 (width:height)—which present significant challenges for training modern deep learning models designed for square input dimensions. This paper proposes a mathematically rigorous approach to slice rectangular images into uniform square patches (1200×1200 pixels) with controlled overlap (20-30%) while preserving spatial information and ensuring complete coverage. We derive closed-form equations for optimal patch count and overlap distance, validate the approach with practical examples, and demonstrate its applicability to real-world geospatial data preprocessing pipelines.

---

## 1. Introduction

### 1.1 Motivation

Machine learning models for image segmentation—particularly convolutional neural networks (CNNs)—are typically trained on square or near-square input tiles. This design choice is fundamental to the architecture of most modern frameworks (U-Net, ResNet, SegNet, etc.). However, remote sensing and medical imaging applications frequently produce elongated rectangular images:

- **Satellite/Aerial Imagery**: High-resolution scans often capture narrow corridor-like regions (e.g., 1200 pixels wide × 8000 pixels tall)
- **Medical Imaging**: Pathology slides and microscopy scans can exhibit similar elongated dimensions
- **Document Scanning**: Architectural blueprints, maps, and technical drawings frequently have extreme aspect ratios

### 1.2 Problem Definition

The naive approach of resizing or cropping these images to square dimensions introduces several problems:

1. **Information Loss**: Aggressive downsampling discards fine spatial details
2. **Aspect Ratio Distortion**: Resizing alters the spatial relationships between objects
3. **Edge Effects**: Simple cropping wastes data at image boundaries
4. **Training Bias**: Incomplete coverage may bias the model toward certain regions

### 1.3 Contribution

This paper presents a principled solution that:
- **Preserves spatial fidelity**: Uses the full resolution without distortion
- **Maximizes data utilization**: Achieves 100% coverage with minimal redundancy
- **Ensures contextual continuity**: Controlled overlap provides contextual information for boundary regions
- **Provides closed-form solutions**: Mathematical framework eliminates manual parameter tuning

---

## 2. Methodology: Image Slicing with Controlled Overlap

### 2.1 Coordinate System Definition

We establish a pixel-based coordinate system with the origin at the **left-bottom corner (0, 0)**:

| Dimension | Symbol | Semantic Meaning | Scale |
|-----------|--------|-----------------|-------|
| **Width (X-axis)** | Pixel | Longitudinal position | 1 pixel = 10 geographic units |
| **Height (Y-axis)** | Scan | Latitudinal position | 1 scan = 10 geographic units |

This convention aligns with geospatial coordinate systems where:
- **Longitude** increases left-to-right (X)
- **Latitude** increases bottom-to-top (Y)

### 2.2 Target Patch Specification

All output patches are standardized to:

$$P_{target} = 1200 \times 1200 \text{ pixels}$$

This dimension is chosen as an optimal balance between:
- Computational tractability (fits in typical GPU memory)
- Spatial resolution (captures relevant features)
- Overlap feasibility (20-30% range is practical)

---

## 3. Mathematical Framework

### 3.1 Core Relationship

Consider an original image of dimensions **1200 × h₀** pixels, where h₀ is the original height. We wish to partition this into n square patches of dimension 1200×1200, each overlapping by k pixels in the vertical direction.

**Fundamental Equation:**

$$h_0 = n \cdot 1200 - (n-1) \cdot k$$

**Explanation:**
- $n \cdot 1200$: Total pixel coverage if patches were laid end-to-end
- $(n-1) \cdot k$: Total overlap to subtract (there are n-1 boundaries between n patches)
- The overlap is subtracted because it's counted twice in the first term

### 3.2 Solving for Number of Patches (n)

Rearranging the fundamental equation to solve for n:

$$h_0 = 1200n - kn + k$$

$$h_0 = n(1200 - k) + k$$

$$h_0 - k = n(1200 - k)$$

$$\boxed{n = \frac{h_0 + k}{1200 - k}}$$

This relationship allows us to:
1. **Choose an overlap percentage** (typically 20-30%)
2. **Convert to pixels**: $k = 0.20 \text{ to } 0.30 \times 1200 = 240 \text{ to } 360 \text{ pixels}$
3. **Calculate required number of patches**

### 3.3 Constraint: Ensuring Integer Patch Count

Since n must be an integer, we compute:

$$n = \left\lceil \frac{h_0 + k}{1200 - k} \right\rceil$$

### 3.4 Solving for Overlap (k) Given n

Conversely, if we know n (number of patches) and want to find the actual overlap:

$$h_0 = n(1200 - k) + k$$

$$h_0 = 1200n - kn + k$$

$$h_0 = 1200n - k(n - 1)$$

$$k(n-1) = 1200n - h_0$$

$$\boxed{k = \frac{1200n - h_0}{n - 1}}$$

**Verification**: The overlap must satisfy:
$$240 \leq k \leq 360 \text{ pixels (20-30% of 1200)}$$

### 3.5 Patch Position Formula

For the **i-th patch** (where i = 1, 2, ..., n), the vertical boundaries are:

**Start position:**
$$y_{start}^{(i)} = (i-1)(1200 - k)$$

**End position:**
$$y_{end}^{(i)} = y_{start}^{(i)} + 1200$$

**Verification**: $y_{end}^{(n)} \leq h_0$ (last patch doesn't exceed image height)

---

## 4. Algorithm and Implementation

### 4.1 Input Parameters

```
Input:
  - Original image dimensions: W × h₀ (width × height in pixels)
  - Desired patch dimension: P = 1200 (assumed constant)
  - Overlap constraint: 0.20 ≤ overlap_ratio ≤ 0.30
  - Overlap range in pixels: 240 ≤ k ≤ 360
```

### 4.2 Step-by-Step Algorithm

#### **Phase 1: Determine Number of Patches**

```
Algorithm PATCH_COUNT(h₀, k_min=240, k_max=360):
  candidates = []
  
  for k in range(k_min, k_max + 1):
    n = ceil((h₀ + k) / (1200 - k))
    
    // Verify overlap is valid
    k_actual = (1200 * n - h₀) / (n - 1)
    
    if k_min ≤ k_actual ≤ k_max:
      candidates.append((n, k_actual, overlap_ratio))
  
  // Choose candidate with overlap closest to 25% (midpoint)
  return argmin(candidates, |overlap_ratio - 0.25|)
```

#### **Phase 2: Calculate Patch Positions**

```
Algorithm PATCH_POSITIONS(h₀, n, k):
  patches = []
  
  for i in range(1, n + 1):
    y_start = (i - 1) * (1200 - k)
    y_end = min(y_start + 1200, h₀)
    
    patches.append({
      'patch_id': i,
      'y_start': y_start,
      'y_end': y_end,
      'height': y_end - y_start,
      'width': 1200
    })
  
  return patches
```

#### **Phase 3: Extract and Save Patches**

```
Algorithm EXTRACT_PATCHES(image, patches):
  for patch in patches:
    sub_image = image[
      patch.y_start : patch.y_end,
      0 : 1200
    ]
    
    // Handle last patch padding if needed
    if sub_image.height < 1200:
      sub_image = pad(sub_image, target_height=1200)
    
    save_patch(sub_image, patch.patch_id)
```

### 4.3 Complexity Analysis

| Metric | Value |
|--------|-------|
| **Time Complexity** | O(n) where n = number of patches |
| **Space Complexity** | O(1200 × 1200 × channels) per patch |
| **I/O Operations** | 1 read (original) + n writes (patches) |

---

## 5. Case Study: Practical Example

### 5.1 Scenario

**Input Image**: 1200 × 8000 pixels (satellite/geospatial data)

**Parameters**:
- Width: 1200 pixels
- Height: 8000 pixels
- Aspect ratio: 1:6.67
- Target: Square patches of 1200×1200 with 20-30% overlap

### 5.2 Computation

#### Step 1: Calculate Overlap Range

$$k_{min} = 0.20 \times 1200 = 240 \text{ pixels}$$
$$k_{max} = 0.30 \times 1200 = 360 \text{ pixels}$$

#### Step 2: Test k = 240 pixels

$$n = \left\lceil \frac{8000 + 240}{1200 - 240} \right\rceil = \left\lceil \frac{8240}{960} \right\rceil = \lceil 8.58 \rceil = 9$$

#### Step 3: Verify with n = 9

$$k_{actual} = \frac{1200 \times 9 - 8000}{9 - 1} = \frac{10800 - 8000}{8} = \frac{2800}{8} = 350 \text{ pixels}$$

$$\text{Overlap Ratio} = \frac{350}{1200} = 0.2917 = 29.17\%$$

**Verification**: ✓ $240 \leq 350 \leq 360$

#### Step 4: Generate Patch Boundaries

| Patch # | y_start | y_end | Height | Status |
|---------|---------|-------|--------|--------|
| 1 | 0 | 1200 | 1200 | ✓ Full |
| 2 | 850 | 2050 | 1200 | ✓ Full |
| 3 | 1700 | 2900 | 1200 | ✓ Full |
| 4 | 2550 | 3750 | 1200 | ✓ Full |
| 5 | 3400 | 4600 | 1200 | ✓ Full |
| 6 | 4250 | 5450 | 1200 | ✓ Full |
| 7 | 5100 | 6300 | 1200 | ✓ Full |
| 8 | 5950 | 7150 | 1200 | ✓ Full |
| 9 | 6800 | 8000 | 1200 | ✓ Full |

**Total Coverage**: $9 \times 1200 - 8 \times 350 = 10800 - 2800 = 8000$ pixels ✓

### 5.3 Visualization

```
Original Image (1200 × 8000):
┌─────────────────┐
│                 │ h₀ = 8000
│   Patch 1       │ 0-1200
│                 │
├─────────────────┤ 850-1200 (overlap 350)
│   Patch 2       │ 850-2050
│                 │
├─────────────────┤ 1700-2050 (overlap 350)
│   Patch 3       │ 1700-2900
│                 │
│       ...       │ (6 more patches)
│                 │
├─────────────────┤ 6800-7150 (overlap 350)
│   Patch 9       │ 6800-8000
│                 │
└─────────────────┘

Each patch: 1200 × 1200
Overlap: 350 pixels (29.17%)
```

---

## 6. Results and Validation

### 6.1 Coverage Analysis

**Metric**: Coverage ratio = $\frac{\text{Total unique pixels}}{\text{Total input pixels}}$

$$\text{Coverage Ratio} = \frac{h_0}{h_0} = 1.0 = 100\%$$

Every pixel from the original image is present in at least one patch.

### 6.2 Redundancy Analysis

**Metric**: Redundancy = $\frac{\text{Total pixels across all patches} - h_0}{h_0}$

$$\text{Redundancy} = \frac{9 \times 1200 - 8000}{8000} = \frac{2800}{8000} = 0.35 = 35\%$$

This means:
- 35% of pixels are duplicated (appear in multiple patches)
- This 35% corresponds exactly to the 8 overlap regions
- Trade-off: Moderate redundancy for contextual continuity

### 6.3 ML Training Benefits

| Aspect | Benefit |
|--------|---------|
| **Input Shape** | 1200×1200 (matches standard CNN architectures) |
| **Batch Processing** | 9 samples from 1 image (increased effective dataset size) |
| **Context** | 350-pixel overlap provides contextual information for boundaries |
| **Augmentation** | Overlapping regions allow data augmentation without loss |
| **GPU Efficiency** | Patch size fits efficiently in GPU memory |

---

## 7. Advanced Topics

### 7.1 Handling Edge Cases

#### **Small Images** (h₀ < 1200)

If the original height is less than 1200 pixels, use single patch with padding:

$$n = 1$$
$$k = 0$$
$$\text{Padding} = 1200 - h_0$$

#### **Variable Overlap Requirements**

For different applications:
- **Segmentation**: 20-30% overlap (recommended for object boundaries)
- **Classification**: 10-15% overlap (sufficient for feature extraction)
- **Dense Prediction**: 30-40% overlap (capture fine details)

### 7.2 Multi-Dimensional Slicing

For rectangular images with both dimensions unequal (W × H where W ≠ H and both > 1200):

Apply slicing in both X and Y dimensions:

$$n_x = \frac{W + k_x}{1200 - k_x}, \quad n_y = \frac{H + k_y}{1200 - k_y}$$

**Total patches**: $n = n_x \times n_y$

### 7.3 Dynamic Overlap Selection

For automatic overlap selection based on image characteristics:

```
Algorithm ADAPTIVE_OVERLAP(h₀, sensitivity=0.25):
  // Target overlap ratio (as fraction)
  target_overlap_ratio = sensitivity
  
  best_n = None
  best_error = ∞
  
  for n in range(2, ceil(h₀/1200) + 2):
    k_required = (1200*n - h₀) / (n - 1)
    overlap_ratio = k_required / 1200
    
    error = |overlap_ratio - target_overlap_ratio|
    
    if error < best_error and 0.20 ≤ overlap_ratio ≤ 0.30:
      best_n = n
      best_error = error
  
  return best_n
```

---

## 8. Implementation Considerations

### 8.1 Memory Optimization

For large images, process patches sequentially:

```python
def process_image_memory_efficient(image_path, h0, n, k):
    """
    Stream processing to avoid loading entire image in memory
    """
    with open(image_path, 'rb') as f:
        for i in range(1, n + 1):
            y_start = (i - 1) * (1200 - k)
            y_end = min(y_start + 1200, h0)
            
            # Read only required rows
            f.seek(y_start * bytes_per_row)
            patch = read_rows(f, y_end - y_start)
            
            # Pad if necessary
            if patch.shape[0] < 1200:
                patch = np.pad(patch, ...)
            
            yield patch, i
```

### 8.2 Format Preservation

**CSV Metadata File**:

Store patch information in Geometry folder for reference:

```
patch_id,x_start,x_end,y_start,y_end,overlap_with_next,width,height
1,0,1200,0,1200,350,1200,1200
2,0,1200,850,2050,350,1200,1200
3,0,1200,1700,2900,350,1200,1200
...
```

---

## 9. Discussion

### 9.1 Comparison with Alternatives

| Method | Pros | Cons |
|--------|------|------|
| **Proposed (Overlap)** | Full coverage, contextual continuity, no distortion | 35% redundancy |
| **Resizing** | No redundancy | Information loss, aspect distortion |
| **Center Cropping** | Clean patches | 30-50% data loss |
| **Padding** | Preserves dimensions | Introduces artificial patterns |

### 9.2 Optimal Overlap Range Justification

**Why 20-30%?**

- **< 20%**: Insufficient context for boundary pixels, weak spatial continuity
- **20-30%**: Optimal balance between redundancy and contextual information
- **> 30%**: Excessive redundancy (memory waste), training inefficiency

Research in image segmentation (Ronneberger et al., U-Net) suggests 10-20% overlap is theoretical minimum; we use 20-30% for practical robustness.

### 9.3 Application Domains

1. **Satellite Imagery**: Slicing wide-area coverage scans
2. **Medical Pathology**: Sectioning histopathology slides
3. **Document OCR**: Processing long document scans
4. **Aerial LiDAR**: Preprocessing point cloud rasterizations
5. **Infrastructure Inspection**: Drone footage of pipelines, roads

---

## 10. Conclusion

This paper presents a mathematically rigorous and practically effective method for converting rectangular images into square ML-compatible patches with controlled overlap. The derived formulas enable:

1. **Deterministic calculation** of patch count and overlap
2. **100% coverage** of original data without loss
3. **Controlled redundancy** (20-30% overlap) for contextual continuity
4. **Closed-form solution** eliminating manual parameter tuning

The approach has been validated through practical examples and is immediately applicable to real-world preprocessing pipelines in geospatial, medical imaging, and document processing domains.

### 10.1 Future Work

1. **3D Extension**: Adapt methodology for volumetric medical data
2. **Hierarchical Patching**: Multi-scale patch extraction for feature hierarchies
3. **Adaptive Weighting**: Reduce redundancy penalty in overlapping regions during training
4. **Real-time Streaming**: GPU-accelerated patch extraction for live data feeds

---

## References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." *International Conference on Medical Image Computing and Computer-Assisted Intervention*, 234-241.

2. Long, J., Shelhamer, E., & Darrell, T. (2015). "Fully Convolutional Networks for Semantic Segmentation." *IEEE Conference on Computer Vision and Pattern Recognition*, 3431-3440.

3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition." *IEEE Conference on Computer Vision and Pattern Recognition*, 770-778.

4. Dosovitskiy, A., et al. (2021). "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale." *International Conference on Learning Representations*.

5. GDAL Documentation. (2023). "Raster Data I/O and Processing." https://gdal.org

6. OpenCV Documentation. (2023). "Image Processing." https://docs.opencv.org

---

## Appendix A: Python Implementation

```python
import numpy as np
from pathlib import Path

class ImageSlicer:
    """
    Convert rectangular images to square patches with controlled overlap
    """
    
    PATCH_SIZE = 1200
    OVERLAP_MIN = 0.20
    OVERLAP_MAX = 0.30
    
    def __init__(self, image_height, overlap_ratio=0.25):
        self.h0 = image_height
        self.k_min = int(self.PATCH_SIZE * self.OVERLAP_MIN)
        self.k_max = int(self.PATCH_SIZE * self.OVERLAP_MAX)
        self.overlap_ratio = overlap_ratio
        
    def calculate_patches(self):
        """Calculate optimal patch count and overlap"""
        best_n = None
        best_k = None
        best_error = float('inf')
        
        for k in range(self.k_min, self.k_max + 1):
            n = int(np.ceil((self.h0 + k) / (self.PATCH_SIZE - k)))
            
            # Verify with actual formula
            k_actual = (self.PATCH_SIZE * n - self.h0) / (n - 1)
            
            if self.k_min <= k_actual <= self.k_max:
                error = abs((k_actual / self.PATCH_SIZE) - self.overlap_ratio)
                
                if error < best_error:
                    best_error = error
                    best_n = n
                    best_k = k_actual
        
        return best_n, int(best_k)
    
    def generate_patch_coordinates(self):
        """Generate (y_start, y_end) for each patch"""
        n, k = self.calculate_patches()
        
        patches = []
        for i in range(1, n + 1):
            y_start = (i - 1) * (self.PATCH_SIZE - k)
            y_end = min(y_start + self.PATCH_SIZE, self.h0)
            
            patches.append({
                'patch_id': i,
                'y_start': y_start,
                'y_end': y_end,
                'height': y_end - y_start,
                'overlap_with_next': k if i < n else 0
            })
        
        return patches
    
    def verify_coverage(self):
        """Verify 100% coverage"""
        patches = self.generate_patch_coordinates()
        n, k = self.calculate_patches()
        
        total_coverage = n * self.PATCH_SIZE - (n - 1) * k
        
        assert total_coverage == self.h0, \
            f"Coverage mismatch: {total_coverage} != {self.h0}"
        
        return True

# Usage Example
if __name__ == "__main__":
    slicer = ImageSlicer(image_height=8000)
    patches = slicer.generate_patch_coordinates()
    
    print(f"Number of patches: {len(patches)}")
    print(f"Patch coordinates:\n")
    
    for p in patches:
        print(f"Patch {p['patch_id']}: "
              f"y=[{p['y_start']}, {p['y_end']}], "
              f"overlap={p['overlap_with_next']}px")
    
    slicer.verify_coverage()
    print("\n✓ Coverage verified: 100%")
```

---

**Document Version**: 1.0  
**Last Updated**: April 2026  
**Status**: Published
