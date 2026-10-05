"""
Generate genuine lab step figures directly from the actual lab images:
Bad (1).jpg (Real can with crack defect) and Good (1).jpg (Real normal can)
from SeamFlangeCrack dataset.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("assets/lab_steps", exist_ok=True)

# -----------------------------------------------------------------------------
# 0. Raw Input Comparison (Bad (1).jpg vs Good (1).jpg)
# -----------------------------------------------------------------------------
bad_can = cv2.imread(r"data\raw_cans\Bad (1).jpg")
good_can = cv2.imread(r"data\raw_cans\Good (1).jpg")

fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), facecolor='white')
axes[0].imshow(cv2.cvtColor(bad_can, cv2.COLOR_BGR2RGB))
axes[0].set_title("(A) Bad (1).jpg - Flange Crack Defect", fontsize=12, fontweight='bold', pad=10)
axes[0].axis('off')

# Circle around defect at true coordinates (838, 348) (~2 o'clock)
circle_bad = plt.Circle((838, 348), 35, color='red', fill=False, lw=2.5, linestyle='-')
axes[0].add_patch(circle_bad)

# Arrow from text to defect
axes[0].annotate("Crack Defect\n(~2 o'clock position)",
                 xy=(838, 348), xytext=(480, 140),
                 arrowprops=dict(arrowstyle="->", color="red", lw=2.2),
                 color="red", fontweight='bold', fontsize=10.5,
                 bbox=dict(boxstyle="round,pad=0.4", fc="#ffe6e6", ec="red", lw=1.5))

# Inset zoom of the crack defect
axins_bad = axes[0].inset_axes([0.04, 0.04, 0.38, 0.38])
crop_bad = bad_can[348-100:348+100, 838-120:838+80]
axins_bad.imshow(cv2.cvtColor(crop_bad, cv2.COLOR_BGR2RGB))
axins_bad.set_xticks([])
axins_bad.set_yticks([])
for spine in axins_bad.spines.values():
    spine.set_edgecolor('red')
    spine.set_linewidth(2.2)
axins_bad.set_title("Zoom: Crack Area", fontsize=9.5, color='red', fontweight='bold', pad=4)

axes[1].imshow(cv2.cvtColor(good_can, cv2.COLOR_BGR2RGB))
axes[1].set_title("(B) Good (1).jpg - Normal Flange (No Crack)", fontsize=12, fontweight='bold', pad=10)
axes[1].axis('off')

# Circle around corresponding normal rim at (773, 323)
circle_good = plt.Circle((773, 323), 35, color='#00aa00', fill=False, lw=2.5, linestyle='-')
axes[1].add_patch(circle_good)

axes[1].annotate("Normal Continuous Rim\n(No crack notch)",
                 xy=(773, 323), xytext=(430, 140),
                 arrowprops=dict(arrowstyle="->", color="#00aa00", lw=2.2),
                 color="#008800", fontweight='bold', fontsize=10.5,
                 bbox=dict(boxstyle="round,pad=0.4", fc="#eafbea", ec="#00aa00", lw=1.5))

# Normal seam zoom inset
axins_good = axes[1].inset_axes([0.04, 0.04, 0.38, 0.38])
crop_good = good_can[323-100:323+100, 773-100:773+100]
axins_good.imshow(cv2.cvtColor(crop_good, cv2.COLOR_BGR2RGB))
axins_good.set_xticks([])
axins_good.set_yticks([])
for spine in axins_good.spines.values():
    spine.set_edgecolor('#00aa00')
    spine.set_linewidth(2.2)
axins_good.set_title("Zoom: Normal Seam", fontsize=9.5, color='#008800', fontweight='bold', pad=4)

plt.tight_layout()
plt.savefig("assets/lab_steps/0_raw_input_comparison.png", dpi=200, bbox_inches='tight')
plt.close()
print("Generated 0_raw_input_comparison.png from real Bad (1).jpg and Good (1).jpg")


# -----------------------------------------------------------------------------
# Process Bad (1).jpg through exact Lab4Contour.cpp pipeline
# -----------------------------------------------------------------------------
def process_real_can(image_path, is_crack_expected=True):
    frame = cv2.imread(image_path)
    dst_image1 = cv2.GaussianBlur(frame, (5, 5), 0, 0)
    src_gray = cv2.cvtColor(dst_image1, cv2.COLOR_BGR2GRAY)

    grad_x = cv2.Sobel(src_gray, cv2.CV_16S, 1, 0, 3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    grad_y = cv2.Sobel(src_gray, cv2.CV_16S, 0, 1, 3)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    _, dst = cv2.threshold(grad, 30, 255, 1)

    contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_index = -1
    for i, ct in enumerate(contours):
        poly = cv2.approxPolyDP(ct, 1, True)
        if len(poly) > 5:
            mu = cv2.moments(poly)
            if mu['m00'] > 0:
                hu = cv2.HuMoments(mu).flatten()
                area = cv2.contourArea(poly)
                if hu[0] < 0.18 and area > max_area:
                    max_area = area
                    max_index = i

    best_poly = cv2.approxPolyDP(contours[max_index], 1, True)
    rect = cv2.boundingRect(best_poly)

    # 1. imgCycle
    img_cycle = src_gray[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

    # 2. imgPolar
    center = (img_cycle.shape[1] / 2.0, img_cycle.shape[0] / 2.0)
    radius = img_cycle.shape[1] / 2.0
    img_polar = cv2.linearPolar(img_cycle, center, radius, 1)

    # 3. imgEdge (50px rim rotated 90 CCW)
    img_edge = img_polar[:, img_polar.shape[1]-50:]
    img_edge = cv2.rotate(img_edge, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # 4. hist[x] & histth[x]
    col_sums = np.sum(img_edge, axis=0)
    hist_th = np.where(col_sums > 5000, 100, 0)

    # 5. Scan xleft & xright
    xleft = 0
    for x in range(img_edge.shape[1]):
        if hist_th[x] == 0:
            xleft = x
            break

    xright = img_edge.shape[1] - 1
    for x in range(img_edge.shape[1]-1, -1, -1):
        if hist_th[x] == 0:
            xright = x
            break

    # Seam recovery if needed
    if xleft == 0 or xright == 0:
        img_cycle = cv2.rotate(img_cycle, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_polar = cv2.linearPolar(img_cycle, center, radius, 1)
        img_edge = img_polar[:, img_polar.shape[1]-50:]
        img_edge = cv2.rotate(img_edge, cv2.ROTATE_90_COUNTERCLOCKWISE)
        col_sums = np.sum(img_edge, axis=0)
        hist_th = np.where(col_sums > 5000, 100, 0)
        for x in range(img_edge.shape[1]):
            if hist_th[x] == 0:
                xleft = x
                break
        for x in range(img_edge.shape[1]-1, -1, -1):
            if hist_th[x] == 0:
                xright = x
                break

    # 6. imgRoi
    img_roi = img_edge[:, xleft:xright]
    resized_roi = cv2.resize(img_roi, (120, 40))

    # 7. gen_feature_input
    col_s = np.sum(resized_roi, axis=0, dtype=np.float32)
    feat = (col_s / np.max(col_s)).reshape(1, 120)

    # ANN predict
    model = cv2.ml.ANN_MLP_load(r"models\model.xml")
    _, resp = model.predict(feat)
    pred_class = int(np.argmax(resp)) # 0=CRACK, 1=NORMAL

    # 8. imgGui (Exact Lab4Contour.cpp lines 300-365)
    img_gui = np.zeros((768, 1024, 3), dtype=np.uint8)
    cv2.rectangle(img_gui, (0, 0), (1024, 45), (100, 80, 0), cv2.FILLED)
    cv2.putText(img_gui, "Cycle Defection Detection", (2, 33), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # Input image
    f_res = cv2.resize(frame, (frame.shape[1] // 3, frame.shape[0] // 3))
    fh, fw = f_res.shape[:2]
    img_gui[150:150+fh, 30:30+fw] = f_res
    cv2.putText(img_gui, "Input Image", (30 + fw // 4, 33 + 150 + fh), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    # Crop image
    c_res = cv2.resize(img_cycle, (fw, fh))
    if len(c_res.shape) == 2:
        c_res = cv2.cvtColor(c_res, cv2.COLOR_GRAY2BGR)
    img_gui[150:150+fh, 60+fw:60+fw*2] = c_res
    cv2.putText(img_gui, "Crop image", (60 + fw + fw // 4, 33 + 150 + fh), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    # Crack Roi
    roi_bgr = cv2.cvtColor(img_roi, cv2.COLOR_GRAY2BGR) if len(img_roi.shape) == 2 else img_roi
    roi_res = cv2.resize(roi_bgr, (img_roi.shape[1] * 2, img_roi.shape[0] * 2))
    rh, rw = roi_res.shape[:2]
    img_gui[150+fh-rh:150+fh, 60+fw:60+fw+rw] = roi_res
    cv2.putText(img_gui, "Crack Roi", (60 + fw + fw // 4, 33 + 150 + fh + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    # imgEdge (rotated 90 CCW, vertical)
    edge_bgr = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2BGR) if len(img_edge.shape) == 2 else img_edge
    edge_res = cv2.resize(edge_bgr, (int(edge_bgr.shape[1] / 2.0), int(edge_bgr.shape[0] * 0.9)))
    edge_rot = cv2.rotate(edge_res, cv2.ROTATE_90_COUNTERCLOCKWISE)
    eh, ew = edge_rot.shape[:2]
    img_gui[150:150+eh, 90+fw*2:90+fw*2+ew] = edge_rot

    # Result text
    if pred_class == 0:
        cv2.putText(img_gui, "CRACK", (2, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 180), 2, cv2.LINE_AA)
    else:
        cv2.putText(img_gui, "NO CRACK", (2, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 180, 0), 2, cv2.LINE_AA)

    return {
        "img_cycle": img_cycle,
        "img_polar": img_polar,
        "img_edge": img_edge,
        "col_sums": col_sums,
        "hist_th": hist_th,
        "img_roi": img_roi,
        "features": feat.flatten(),
        "img_gui": img_gui,
        "pred": pred_class,
        "xleft": xleft,
        "xright": xright
    }

# Process real Bad (1).jpg
bad_res = process_real_can(r"data\raw_cans\Bad (1).jpg", is_crack_expected=True)
cv2.imwrite("assets/lab_steps/1_img_cycle.jpg", bad_res["img_cycle"])
cv2.imwrite("assets/lab_steps/2_img_polar.jpg", bad_res["img_polar"])
cv2.imwrite("assets/lab_steps/3_img_edge.jpg", bad_res["img_edge"])
cv2.imwrite("assets/lab_steps/5_img_roi.jpg", bad_res["img_roi"])
cv2.imwrite("assets/lab_steps/7_exact_lab_gui_crack.png", bad_res["img_gui"])

# Draw real line histogram
img_hist = np.zeros((100, bad_res["img_edge"].shape[1]), dtype=np.uint8)
for x in range(img_hist.shape[1]):
    cv2.line(img_hist, (x, 0), (x, img_hist.shape[0] - bad_res["hist_th"][x]), 255, 1)
cv2.imwrite("assets/lab_steps/4_img_hist_lines.jpg", img_hist)

# Plot real frequency curve
plt.figure(figsize=(10, 3.8), facecolor='white')
plt.plot(bad_res["col_sums"], color='#0066cc', lw=1.8, label='Column Pixel Sum H[x]')
plt.axhline(y=5000, color='red', linestyle='--', lw=1.5, label='Threshold = 5,000')
plt.axvspan(bad_res["xleft"], bad_res["xright"], color='red', alpha=0.3, label=f'Detected Crack Notch [x={bad_res["xleft"]}:{bad_res["xright"]}]')
plt.title("Real Lab Data: Column Intensity Projection H[x] & Thresholding on Bad (1).jpg", fontsize=11, fontweight='bold', pad=10)
plt.xlabel("Perimeter Column Index (x)", fontsize=10)
plt.ylabel("Intensity Sum", fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower left', fontsize=9)
plt.tight_layout()
plt.savefig("assets/lab_steps/4_frequency_plot.png", dpi=200)
plt.close()

# Plot real feature vector
plt.figure(figsize=(10, 3), facecolor='white')
plt.bar(range(120), bad_res["features"], color='#ff6600', width=0.8)
plt.title("Real 120-d Feature Vector from gen_feature_input() on Bad (1).jpg", fontsize=11, fontweight='bold', pad=10)
plt.xlabel("Input Node Index (0 to 119)", fontsize=10)
plt.ylabel("Normalized Sum (0.0 to 1.0)", fontsize=10)
plt.ylim(0, 1.1)
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig("assets/lab_steps/6_feature_vector_120d.png", dpi=200)
plt.close()

# Process real Good (1).jpg
good_res = process_real_can(r"data\raw_cans\Good (1).jpg", is_crack_expected=False)
cv2.imwrite("assets/lab_steps/7_exact_lab_gui_nocrack.png", good_res["img_gui"])

print("SUCCESS: All genuine lab step images generated from Bad (1).jpg and Good (1).jpg!")
