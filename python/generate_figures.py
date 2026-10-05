"""
Generate publication-quality engineering diagrams and figures for GitHub README.
Figures generated:
1. assets/polar_unwrapping_concept.png (Cartesian Circular Rim vs Linear-Polar Unrolled Tape)
2. assets/projection_signal_analysis.png (1D Projection Comparison: Crack Defect Dip vs Smooth Normal Rim)
3. assets/dataset_distribution_and_samples.png (Dataset reference patches p00-p50 + distribution)
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("assets", exist_ok=True)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# -----------------------------------------------------------------------------
# Figure 1: Polar Unwrapping Concept
# -----------------------------------------------------------------------------
def make_polar_concept():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), facecolor='#ffffff')
    
    can_lid = cv2.imread(r'data\samples\reconstructed_can_crack1.jpg', cv2.IMREAD_GRAYSCALE)
    rim_strip = cv2.imread(r'data\samples\crack1.jpg', cv2.IMREAD_GRAYSCALE)
    
    # Left: Cartesian circle
    axes[0].imshow(can_lid, cmap='gray')
    axes[0].set_title("(A) Cartesian Coordinates (x, y)\n360° Circular Rim with Micro-Crack", fontsize=11, fontweight='bold', pad=10)
    axes[0].axis('off')
    
    # Highlight defect on circle
    circle = plt.Circle((287, 287), 260, color='#00ffcc', fill=False, linewidth=2, linestyle='--')
    axes[0].add_patch(circle)
    axes[0].annotate("Crack Defect\n(at ~1 o'clock)", xy=(460, 110), xytext=(350, 40),
                     arrowprops=dict(arrowstyle="->", color="red", lw=2),
                     color="red", fontweight='bold', fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", fc="#ffe6e6", ec="red", lw=1))

    # Right: Unwrapped strip
    axes[1].imshow(rim_strip, cmap='gray', aspect='auto')
    axes[1].set_title("(B) Linear-Polar Unwrapped Strip (r, θ)\nPerimeter Mapped to Straight 1D Tape", fontsize=11, fontweight='bold', pad=10)
    axes[1].set_xlabel("Angular Dimension θ (0° -> 360° along columns)", fontsize=10)
    axes[1].set_ylabel("Radial Depth (outer 50 px)", fontsize=10)
    
    # Highlight defect box
    rect = plt.Rectangle((219, 0), 79, 50, edgecolor='red', facecolor='none', linewidth=2.5)
    axes[1].add_patch(rect)
    axes[1].annotate("Defect Notch\n[x=219:298]", xy=(258, 25), xytext=(220, 75),
                     arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                     color="red", fontweight='bold', fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", fc="#ffe6e6", ec="red", lw=1))

    plt.tight_layout()
    out_path = "assets/polar_unwrapping_concept.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")

# -----------------------------------------------------------------------------
# Figure 2: Projection Signal Analysis
# -----------------------------------------------------------------------------
def make_projection_analysis():
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), gridspec_kw={'height_ratios': [1, 2]}, facecolor='#ffffff')
    
    crack_rim = cv2.imread(r'data\samples\crack1.jpg', cv2.IMREAD_GRAYSCALE)
    # create normal rim strip
    norm_patch = cv2.imread(r'data\samples\crack5.jpg', cv2.IMREAD_GRAYSCALE)
    repeats = int(np.ceil(575 / norm_patch.shape[1]))
    norm_rim = np.tile(norm_patch, (1, repeats))[:, :575]
    
    crack_col_sums = np.sum(crack_rim, axis=0)
    norm_col_sums = np.sum(norm_rim, axis=0)
    
    # (A) Crack Strip
    axes[0, 0].imshow(crack_rim, cmap='gray', aspect='auto')
    axes[0, 0].set_title("(A) Defective Rim Strip (Physical Micro-Crack)", fontsize=10, fontweight='bold')
    axes[0, 0].axis('off')
    
    # (B) Normal Strip
    axes[0, 1].imshow(norm_rim, cmap='gray', aspect='auto')
    axes[0, 1].set_title("(B) Normal Rim Strip (Smooth Metallic Reflection)", fontsize=10, fontweight='bold')
    axes[0, 1].axis('off')
    
    # (C) Crack Projection Signal
    axes[1, 0].plot(crack_col_sums, color='#1f77b4', lw=1.5, label='Intensity Sum H[x]')
    axes[1, 0].axhline(y=5000, color='red', linestyle='--', lw=1.2, label='Threshold = 5,000')
    axes[1, 0].axvspan(219, 298, color='red', alpha=0.25, label='Detected Crack Gap')
    axes[1, 0].set_title("Defect Signal: Sharp Valley Drop Below Threshold", fontsize=10, fontweight='bold')
    axes[1, 0].set_xlabel("Column Index (Perimeter x)", fontsize=9)
    axes[1, 0].set_ylabel("Column Intensity Sum", fontsize=9)
    axes[1, 0].set_ylim(0, 9500)
    axes[1, 0].grid(True, linestyle=':', alpha=0.6)
    axes[1, 0].legend(loc='lower left', fontsize=8.5)
    
    # (D) Normal Projection Signal
    axes[1, 1].plot(norm_col_sums, color='#2ca02c', lw=1.5, label='Intensity Sum H[x]')
    axes[1, 1].axhline(y=5000, color='red', linestyle='--', lw=1.2, label='Threshold = 5,000')
    axes[1, 1].set_title("Normal Signal: Continuous Baseline Above Threshold", fontsize=10, fontweight='bold')
    axes[1, 1].set_xlabel("Column Index (Perimeter x)", fontsize=9)
    axes[1, 1].set_ylabel("Column Intensity Sum", fontsize=9)
    axes[1, 1].set_ylim(0, 9500)
    axes[1, 1].grid(True, linestyle=':', alpha=0.6)
    axes[1, 1].legend(loc='lower left', fontsize=8.5)

    plt.tight_layout()
    out_path = "assets/projection_signal_analysis.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")

# -----------------------------------------------------------------------------
# Figure 3: Reference Defect and Normal Patches (p00-p50)
# -----------------------------------------------------------------------------
def make_reference_patches_figure():
    fig, axes = plt.subplots(2, 3, figsize=(9, 5), facecolor='#ffffff')
    
    patches = [
        ("p00.jpg", "Crack Sample #1 (Defect)", 'red'),
        ("p10.jpg", "Crack Sample #2 (Defect)", 'red'),
        ("p20.jpg", "Crack Sample #3 (Defect)", 'red'),
        ("p30.jpg", "Normal Sample #1 (Pass)", 'green'),
        ("p40.jpg", "Normal Sample #2 (Pass)", 'green'),
        ("p50.jpg", "Normal Sample #3 (Pass)", 'green'),
    ]
    
    for idx, (fname, label, col) in enumerate(patches):
        r = idx // 3
        c = idx % 3
        img_path = os.path.join(r"data\reference_patches", fname)
        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            axes[r, c].imshow(img, cmap='gray')
            axes[r, c].set_title(label, fontsize=9.5, fontweight='bold', color=col, pad=8)
            axes[r, c].axis('off')
            # outline
            for spine in axes[r, c].spines.values():
                spine.set_color(col)
                spine.set_linewidth(2)
                
    plt.suptitle("Reference Benchmark Patches: Micro-Crack Defects vs Normal Rim Textures", fontsize=11, fontweight='bold', y=0.98)
    plt.tight_layout()
    out_path = "assets/reference_defect_patches.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")

if __name__ == '__main__':
    make_polar_concept()
    make_projection_analysis()
    make_reference_patches_figure()
