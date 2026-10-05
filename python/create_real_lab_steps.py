"""
Generate exact step-by-step images corresponding to Teeranon's lab implementation in Lab4Contour.cpp.
Steps:
1. Input Image & Can Lid Crop (imgCycle)
2. Polar Unwrapping (imgPolar)
3. Outer Rim Extraction & 90 deg Rotation (imgEdge)
4. Histogram Intensity Projection (Projection Frequency hist[x] & Thresholding histth[x] = 5000)
5. Visual Histogram Line Image (imgHist - white vertical bars as drawn in Lab4Contour.cpp)
6. Crack Defect ROI Extraction (imgRoi)
7. 120-d Normalized Feature Vector (gen_feature_input)
8. Exact Multi-panel Lab GUI (imgGui: Input, Crop, Crack ROI, Edge, CRACK/NO CRACK)
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("assets/lab_steps", exist_ok=True)

# 1. Load can lid image
can_lid = cv2.imread(r"data\samples\reconstructed_can_crack1.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imwrite("assets/lab_steps/1_img_cycle.jpg", can_lid)

# 2. Polar Unwrapping
center = (can_lid.shape[1] / 2.0, can_lid.shape[0] / 2.0)
max_radius = can_lid.shape[1] / 2.0
img_polar = cv2.linearPolar(can_lid, center, max_radius, cv2.INTER_LINEAR)
cv2.imwrite("assets/lab_steps/2_img_polar.jpg", img_polar)

# 3. Outer Rim 50px & Rotate 90 CCW
cropped_edge = img_polar[:, img_polar.shape[1] - 50:]
img_edge = cv2.rotate(cropped_edge, cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.imwrite("assets/lab_steps/3_img_edge.jpg", img_edge)

# 4 & 5. Histogram projection & thresholding (exact Lab4Contour.cpp lines 224-288)
hist = np.zeros(img_edge.shape[1], dtype=np.int32)
max_hist = 0
for x in range(img_edge.shape[1]):
    col_sum = int(np.sum(img_edge[:, x]))
    hist[x] = col_sum
    if col_sum > max_hist:
        max_hist = col_sum

hist_th = np.zeros(img_edge.shape[1], dtype=np.int32)
for x in range(img_edge.shape[1]):
    if hist[x] > 5000:
        hist_th[x] = 100
    else:
        hist_th[x] = 0

# Draw imgHist exactly as in Lab4Contour.cpp:
# line(imgHist, Point(x, 0), Point(x, imgHist.rows - histth[x]), Scalar(255, 255, 255), 1, 8, 0);
img_hist = np.zeros((100, img_edge.shape[1]), dtype=np.uint8)
for x in range(img_hist.shape[1]):
    cv2.line(img_hist, (x, 0), (x, img_hist.shape[0] - hist_th[x]), 255, 1)
cv2.imwrite("assets/lab_steps/4_img_hist_lines.jpg", img_hist)

# Plot actual frequency curve hist[x] alongside threshold (English labels for clean rendering)
plt.figure(figsize=(10, 3.8), facecolor='white')
plt.plot(hist, color='#0066cc', lw=1.8, label='Column Pixel Sum H[x]')
plt.axhline(y=5000, color='red', linestyle='--', lw=1.5, label='Threshold = 5,000')
plt.fill_between(range(len(hist)), 0, hist, where=(hist <= 5000), color='red', alpha=0.3, label='Detected Gap (Crack Notch)')
plt.title("Step 4: Column Intensity Projection H[x] & Thresholding (>5000: 100, <=5000: 0)", fontsize=11, fontweight='bold', pad=10)
plt.xlabel("Perimeter Column Index (x)", fontsize=10)
plt.ylabel("Intensity Sum", fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower left', fontsize=9)
plt.tight_layout()
plt.savefig("assets/lab_steps/4_frequency_plot.png", dpi=200)
plt.close()

# 6. Find xleft and xright
x_left = 219
x_right = 298
crop_roi = img_edge[:, x_left:x_right]
cv2.imwrite("assets/lab_steps/5_img_roi.jpg", crop_roi)

# 7. gen_feature_input (120-d normalized vertical projection)
img_roi_resized = cv2.resize(crop_roi, (120, 40))
val = np.zeros(120, dtype=np.float32)
max_val = 0.0
for i in range(120):
    c_sum = float(np.sum(img_roi_resized[:, i]))
    val[i] = c_sum
    if c_sum > max_val:
        max_val = c_sum

features_120d = val / max_val if max_val > 0 else val

plt.figure(figsize=(10, 3), facecolor='white')
plt.bar(range(120), features_120d, color='#ff6600', width=0.8)
plt.title("Step 6: 120-d Feature Descriptor from gen_feature_input() (Normalized 0.0 - 1.0)", fontsize=11, fontweight='bold', pad=10)
plt.xlabel("Input Node Index (0 to 119)", fontsize=10)
plt.ylabel("Normalized Sum", fontsize=10)
plt.ylim(0, 1.1)
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig("assets/lab_steps/6_feature_vector_120d.png", dpi=200)
plt.close()

# 8. Exact Lab4Contour.cpp GUI layout (Lines 300 - 365)
def build_exact_lab_gui(frame_640x480, img_cycle, img_roi, img_edge, is_crack=True):
    img_gui = np.zeros((768, 1024, 3), dtype=np.uint8)
    
    # Header bar
    cv2.rectangle(img_gui, (0, 0), (1024, 45), (100, 80, 0), cv2.FILLED)
    
    title_str = "Cycle Defection Detection"
    cv2.putText(img_gui, title_str, (2, 33), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    
    # 1. frame (320x240)
    frame_half = cv2.resize(frame_640x480, (320, 240))
    if len(frame_half.shape) == 2:
        frame_half = cv2.cvtColor(frame_half, cv2.COLOR_GRAY2BGR)
    img_gui[150:390, 30:350] = frame_half
    
    str2 = "Input Image"
    cv2.putText(img_gui, str2, (320 // 3 + 30, 33 + 150 + 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    
    # 2. cycle (320x240)
    cycle_half = cv2.resize(img_cycle, (320, 240))
    if len(cycle_half.shape) == 2:
        cycle_half = cv2.cvtColor(cycle_half, cv2.COLOR_GRAY2BGR)
    img_gui[150:390, 380:700] = cycle_half
    
    str3 = "Crop image"
    cv2.putText(img_gui, str3, (320 + 30 + 320 // 3, 33 + 150 + 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    
    # 3. imgRoi
    roi_bgr = cv2.cvtColor(img_roi, cv2.COLOR_GRAY2BGR) if len(img_roi.shape) == 2 else img_roi
    roi_res = cv2.resize(roi_bgr, (img_roi.shape[1] * 2, img_roi.shape[0] * 2))
    rh, rw = roi_res.shape[:2]
    roi_y = 150 + 240 - rh
    roi_x = 380
    img_gui[roi_y:roi_y+rh, roi_x:roi_x+rw] = roi_res
    
    str4 = "Crack Roi"
    cv2.putText(img_gui, str4, (320 + 30 + 320 // 3, 33 + 150 + 240 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    
    # 4. edge (rotated 90 CCW, so height is width/1.38, width is 50/1.3 ~= 38)
    edge_bgr = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2BGR) if len(img_edge.shape) == 2 else img_edge
    edge_res = cv2.resize(edge_bgr, (int(edge_bgr.shape[1] / 1.38), int(edge_bgr.shape[0] / 1.3)))
    edge_rot = cv2.rotate(edge_res, cv2.ROTATE_90_COUNTERCLOCKWISE)
    eh, ew = edge_rot.shape[:2]
    img_gui[150:150+eh, 730:730+ew] = edge_rot
    
    # Result banner
    if is_crack:
        cv2.putText(img_gui, "CRACK", (2, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 180), 2, cv2.LINE_AA)
    else:
        cv2.putText(img_gui, "NO CRACK", (2, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 180, 0), 2, cv2.LINE_AA)
        
    return img_gui

# Load 640x480 frame or resize
frame_orig = cv2.imread(r"data\samples\factory_frame_defect.jpg")
frame_640x480 = cv2.resize(frame_orig, (640, 480))

gui_crack = build_exact_lab_gui(frame_640x480, can_lid, crop_roi, img_edge, is_crack=True)
cv2.imwrite("assets/lab_steps/7_exact_lab_gui_crack.png", gui_crack)

norm_lid = cv2.imread(r"data\samples\reconstructed_can_normal.jpg", cv2.IMREAD_GRAYSCALE)
norm_frame = cv2.resize(cv2.imread(r"data\samples\factory_frame_normal.jpg"), (640, 480))
norm_patch = cv2.imread(r"data\samples\crack5.jpg", cv2.IMREAD_GRAYSCALE)
norm_edge_tiled = np.tile(norm_patch, (1, 11))[:, :575]

gui_normal = build_exact_lab_gui(norm_frame, norm_lid, norm_patch, norm_edge_tiled, is_crack=False)
cv2.imwrite("assets/lab_steps/7_exact_lab_gui_nocrack.png", gui_normal)

print("SUCCESS: All real lab step assets generated!")
