# Image-Slicing
Image Slicing Implementation: Converting Rectangular to Square Patches
A Python implementation of a mathematically rigorous method to convert rectangular images (satellite, medical, document scans) into square ML-compatible patches with controlled overlap.


✅ Core Functionality

Calculate optimal patch count: n = ⌈(h₀ + k) / (1200 - k)⌉
Generate all patch coordinates with overlap
Verify 100% image coverage
Calculate redundancy and statistics
Extract patches from actual image files
Export metadata to CSV

📋 Overview
This project addresses the challenge of training machine learning models on rectangular images with extreme aspect ratios (1:6 to 1:8). The solution provides:

100% Coverage: Every pixel from the original image is included in at least one patch
Controlled Overlap: 20-30% overlap between patches for contextual continuity
Closed-form Equations: Mathematical framework eliminates manual parameter tuning
Production-Ready Code: Tested, documented, and optimized for performance


Project Directory
│
├── 🐍 PYTHON CODE
│   ├── image_slicer.py           ← Core implementation (main file!)
│   ├── example_usage.py          ← 10 examples (run this!)
│   └── test_image_slicer.py      ← 30+ tests
│
├── 📖 DOCUMENTATION
│   ├── QUICKSTART.md             ← Start here! (60 seconds)
│   ├── README.md                 ← Full documentation
│   ├── requirements.txt          ← Dependencies
│   └── Image_Slicing_Technical_Paper
│       ├── .md (19 KB)          ← Markdown version
│       └── .docx (14 KB)        ← Word version (for journals)
│
└── 📄 TECHNICAL PAPERS (previously created)
    ├── Image_Slicing_Technical_Paper.md
    └── Image_Slicing_Technical_Paper.docx


Program Execution 
Step 1: Install
bashpip install numpy Pillow
Step 2: Use
pythonfrom image_slicer import ImageSlicer

slicer = ImageSlicer(image_height=8000)
n, k, ratio = slicer.calculate_patches(verbose=True)
slicer.print_summary()
Step 3: Run Examples
bashpython example_usage.py
