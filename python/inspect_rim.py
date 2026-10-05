"""
Automated Metal Can Rim Crack Defect Inspection Pipeline
Author: Teeranon (AI Engineering, Prince of Songkla University)
Description:
    Clean, human-readable implementation of the 8-stage industrial vision pipeline
    for high-speed canned food rim crack detection using Linear Polar unwrapping,
    1D intensity projection, and OpenCV ANN_MLP classification.
"""

import os
import sys
import argparse
import cv2
import numpy as np


def find_can_lid(image_gray):
    """
    Stage 1: Can Boundary Localization using Sobel gradient and Hu Moment invariant.
    A genuine circular lid has a theoretical first Hu moment hu[0] ~= 1/(2*pi) = 0.159.
    Returns: (cropped_lid, bounding_rect)
    """
    blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
    
    # Sobel gradient in X and Y
    grad_x = cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_16S, 0, 1, ksize=3)
    abs_x = cv2.convertScaleAbs(grad_x)
    abs_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
    
    # Binary inverse thresholding (inside circular boundary becomes solid)
    _, thresh = cv2.threshold(grad, 30, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    frame_area = image_gray.shape[0] * image_gray.shape[1]
    best_contour = None
    max_area = 0
    best_rect = None
    
    for ct in contours:
        area = cv2.contourArea(ct)
        # Filter out whole frame boundaries and tiny noise specs
        if 5000 < area < (frame_area * 0.9):
            mu = cv2.moments(ct)
            hu = cv2.HuMoments(mu).flatten()
            # Circular invariant check: ideal circle has hu[0] ~= 0.159
            if hu[0] < 0.22 and area > max_area:
                max_area = area
                best_contour = ct
                best_rect = cv2.boundingRect(ct)
                
    if best_rect is not None:
        x, y, w, h = best_rect
        cropped_lid = image_gray[y:y+h, x:x+w]
        return cropped_lid, best_rect
        
    return image_gray, (0, 0, image_gray.shape[1], image_gray.shape[0])


def unwrap_can_rim(lid_gray, rim_width=50):
    """
    Stage 2 & 3: Linear Polar unwrapping and outer rim extraction.
    Converts 360-degree circular rim into a flat rectangular strip.
    """
    h, w = lid_gray.shape[:2]
    center = (w / 2.0, h / 2.0)
    max_radius = min(w, h) / 2.0
    
    # Unroll circular lid into polar coordinates
    polar_img = cv2.linearPolar(lid_gray, center, max_radius, cv2.INTER_LINEAR)
    
    # Outer rim is located at the outermost rim_width pixels in radius
    outer_rim = polar_img[:, polar_img.shape[1] - rim_width:]
    
    # Rotate 90 CCW so perimeter trajectory runs horizontally along columns
    rim_horizontal = cv2.rotate(outer_rim, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    return rim_horizontal


def detect_crack_region(rim_strip, intensity_threshold=5000, pad=18):
    """
    Stage 4: Column-wise intensity projection and defect gap detection.
    Scans for drops in intensity caused by physical metal cracks or notches.
    
    Seam Recovery Logic:
    If a defect touches column 0 or the end column (split across the 0/360 deg seam),
    we flag seam_split=True so the caller can rotate the can 90 degrees and unwrap again.
    """
    col_sums = np.sum(rim_strip, axis=0, dtype=np.int32)
    high_mask = col_sums > intensity_threshold
    high_indices = np.where(high_mask)[0]
    
    hist_th = np.where(high_mask, 100, 0)
    
    if len(high_indices) == 0:
        return None, 0, 0, hist_th, False
        
    valleys = []
    in_valley = False
    v_start = 0
    for x in range(high_indices[0], high_indices[-1] + 1):
        if not high_mask[x] and not in_valley:
            in_valley = True
            v_start = x
        elif high_mask[x] and in_valley:
            in_valley = False
            valleys.append((v_start, x - 1))
            
    if len(valleys) > 0:
        valleys.sort(key=lambda v: (v[1] - v[0]), reverse=True)
        v_l, v_r = valleys[0]
        x_left = max(0, v_l - pad)
        x_right = min(rim_strip.shape[1], v_r + 1 + pad)
        is_seam_split = (x_left == 0 or x_right >= rim_strip.shape[1])
        return rim_strip[:, x_left:x_right], x_left, x_right, hist_th, is_seam_split
        
    # No valley found (smooth normal rim)
    mid = rim_strip.shape[1] // 2
    x_left = max(0, mid - 30)
    x_right = min(rim_strip.shape[1], mid + 30)
    return rim_strip[:, x_left:x_right], x_left, x_right, hist_th, False


def extract_projection_features(roi_patch, target_size=(120, 40)):
    """
    Stage 6: 120-dimensional normalized vertical intensity descriptor.
    Resizes patch to 120x40 and calculates column sums normalized by maximum.
    Provides illumination invariance against ambient metal reflections.
    """
    if len(roi_patch.shape) == 3:
        roi_patch = cv2.cvtColor(roi_patch, cv2.COLOR_BGR2GRAY)
        
    resized_patch = cv2.resize(roi_patch, target_size)
    col_sums = np.sum(resized_patch, axis=0, dtype=np.float32)
    
    max_val = np.max(col_sums)
    if max_val > 0:
        features = col_sums / max_val
    else:
        features = col_sums
        
    return features.reshape(1, 120), resized_patch


def classify_defect(features, model_path="models/model.xml"):
    """
    Stage 7: ANN MLP classification (120 -> 15 -> 2).
    Returns: (is_crack, confidence_scores, pred_index)
        pred_index 0 = CRACK (FAIL)
        pred_index 1 = NO CRACK (PASS)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    model = cv2.ml.ANN_MLP_load(model_path)
    _, response = model.predict(features)
    
    pred_idx = int(np.argmax(response))
    is_crack = (pred_idx == 0)
    
    return is_crack, response[0], pred_idx


def render_dashboard(input_frame, lid_crop, unwrapped_rim, crack_roi, hist_th, is_crack, x_left=0, x_right=0):
    """
    Stage 8: Industrial Inspection Dashboard (1024x768).
    Renders multi-panel live view with input camera, circular lid, unwrapped rim,
    zoomed crack patch, projection signal, and clear PASS/FAIL banner.
    """
    dashboard = np.zeros((768, 1024, 3), dtype=np.uint8)
    
    # Top header bar (Industrial dark slate)
    cv2.rectangle(dashboard, (0, 0), (1024, 55), (40, 35, 30), -1)
    cv2.putText(dashboard, "AUTOMATED CAN RIM CRACK METROLOGY SYSTEM", (20, 37),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(dashboard, "High-Speed In-Line Vision", (760, 37),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    
    # Status Banner (PASS / FAIL)
    banner_color = (30, 30, 200) if is_crack else (40, 160, 40) # BGR: Red / Green
    status_text = "STATUS: FAIL - CRACK DETECTED" if is_crack else "STATUS: PASS - RIM NORMAL"
    cv2.rectangle(dashboard, (0, 55), (1024, 105), banner_color, -1)
    cv2.putText(dashboard, status_text, (25, 92),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Panel 1: Input Camera Frame (300 x 225)
    f_disp = input_frame.copy()
    if len(f_disp.shape) == 2:
        f_disp = cv2.cvtColor(f_disp, cv2.COLOR_GRAY2BGR)
    f_disp = cv2.resize(f_disp, (300, 225))
    dashboard[140:365, 30:330] = f_disp
    cv2.rectangle(dashboard, (30, 140), (330, 365), (80, 80, 80), 1)
    cv2.putText(dashboard, "[1] Factory Camera Input", (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
                
    # Panel 2: Cropped Circular Lid (300 x 300)
    lid_disp = lid_crop.copy()
    if len(lid_disp.shape) == 2:
        lid_disp = cv2.cvtColor(lid_disp, cv2.COLOR_GRAY2BGR)
    lid_disp = cv2.resize(lid_disp, (240, 240))
    dashboard[140:380, 370:610] = lid_disp
    cv2.rectangle(dashboard, (370, 140), (610, 380), (80, 80, 80), 1)
    cv2.putText(dashboard, "[2] Extracted Can Lid ROI", (370, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
                
    # Panel 3: Zoomed Crack / Defect Patch
    if crack_roi is not None and crack_roi.size > 0:
        patch_disp = crack_roi.copy()
        if len(patch_disp.shape) == 2:
            patch_disp = cv2.cvtColor(patch_disp, cv2.COLOR_GRAY2BGR)
        patch_disp = cv2.resize(patch_disp, (320, 120))
        dashboard[140:260, 660:980] = patch_disp
        cv2.rectangle(dashboard, (660, 140), (980, 260), (0, 0, 255) if is_crack else (0, 255, 0), 2)
        cv2.putText(dashboard, "[3] Defect ROI Patch (Zoomed)", (660, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        
        defect_w = x_right - x_left
        cv2.putText(dashboard, f"Span: {defect_w} px [{x_left}:{x_right}]", (660, 285),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 200, 255), 1, cv2.LINE_AA)
                    
    # Panel 4: Linear Polar Unwrapped Rim (Full 360 deg tape)
    rim_disp = unwrapped_rim.copy()
    if len(rim_disp.shape) == 2:
        rim_disp = cv2.cvtColor(rim_disp, cv2.COLOR_GRAY2BGR)
    # Resize horizontally across bottom
    rim_disp = cv2.resize(rim_disp, (950, 75))
    dashboard[440:515, 30:980] = rim_disp
    cv2.rectangle(dashboard, (30, 440), (980, 515), (100, 100, 100), 1)
    cv2.putText(dashboard, "[4] Unwrapped Linear-Polar Rim Strip (360 deg perimeter trajectory)", (30, 430),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
                
    # Mark defect interval on unwrapped rim
    if x_right > x_left:
        scale_x = 950.0 / unwrapped_rim.shape[1]
        mark_x1 = int(30 + x_left * scale_x)
        mark_x2 = int(30 + x_right * scale_x)
        cv2.rectangle(dashboard, (mark_x1, 440), (mark_x2, 515), (0, 0, 255) if is_crack else (0, 255, 0), 2)
        
    # Panel 5: Horizontal Projection Profile Graph
    graph_h = 100
    graph_w = 950
    graph_img = np.full((graph_h, graph_w, 3), 20, dtype=np.uint8)
    
    if hist_th is not None and len(hist_th) > 0:
        step = graph_w / len(hist_th)
        for i in range(len(hist_th) - 1):
            pt1 = (int(i * step), int(graph_h - (hist_th[i] / 100.0 * 70 + 15)))
            pt2 = (int((i + 1) * step), int(graph_h - (hist_th[i + 1] / 100.0 * 70 + 15)))
            color = (0, 255, 255) if hist_th[i] > 0 else (0, 0, 255)
            cv2.line(graph_img, pt1, pt2, color, 2)
            
    dashboard[570:670, 30:980] = graph_img
    cv2.rectangle(dashboard, (30, 570), (980, 670), (80, 80, 80), 1)
    cv2.putText(dashboard, "[5] 1D Column-wise Intensity Projection (Notch/Drop Detection Signal)", (30, 560),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
                
    # Footer metadata
    footer_text = "Algorithm: Hu-Moments Localization -> Polar Unwrapping -> 120-d Projection -> OpenCV ANN_MLP (RPROP)"
    cv2.putText(dashboard, footer_text, (30, 725),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (140, 140, 140), 1, cv2.LINE_AA)
                
    return dashboard


def inspect_image(image_path, model_path="models/model.xml", output_dashboard=None, show_window=False):
    """
    Executes full inspection on a given input image (can be full frame, can lid, or rim strip).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
        
    frame = cv2.imread(image_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    
    # Detect input mode: pre-cropped patch, unwrapped strip, circular lid, or full frame
    h, w = gray.shape[:2]
    float_ar = float(w) / h
    if w < 200 and h < 100:
        # Mode 1: Pre-cropped defect/normal patch (crack0, crack5, etc.)
        lid_crop = gray
        rim_strip = gray
        crack_roi = gray
        x_left = 0
        x_right = w
        hist_th = None
        is_seam_split = False
    elif float_ar > 3.0:
        # Mode 2: Unwrapped rim strip (crack1, crack4)
        lid_crop = gray
        rim_strip = gray
        crack_roi, x_left, x_right, hist_th, is_seam_split = detect_crack_region(rim_strip)
    elif 0.8 < float_ar < 1.2 and w >= 200:
        # Mode 3: Circular lid crop
        lid_crop = gray
        rim_strip = unwrap_can_rim(lid_crop)
        crack_roi, x_left, x_right, hist_th, is_seam_split = detect_crack_region(rim_strip)
    else:
        # Mode 4: Full camera frame
        lid_crop, _ = find_can_lid(gray)
        rim_strip = unwrap_can_rim(lid_crop)
        crack_roi, x_left, x_right, hist_th, is_seam_split = detect_crack_region(rim_strip)
    
    # Seam recovery: if crack is split on boundary seam, rotate lid 90 CCW and re-run
    if is_seam_split and lid_crop is not None and lid_crop.shape[0] == lid_crop.shape[1]:
        lid_crop = cv2.rotate(lid_crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rim_strip = unwrap_can_rim(lid_crop)
        crack_roi, x_left, x_right, hist_th, _ = detect_crack_region(rim_strip)
        
    # If no defect gap found, extract middle section as normal reference patch
    if crack_roi is None or crack_roi.size == 0:
        mid = rim_strip.shape[1] // 2
        crack_roi = rim_strip[:, mid - 25:mid + 25]
        x_left = mid - 25
        x_right = mid + 25
        
    # Feature extraction & Classification
    features, resized_patch = extract_projection_features(crack_roi)
    is_crack, scores, pred_idx = classify_defect(features, model_path=model_path)
    
    # Render Dashboard
    dashboard = render_dashboard(frame, lid_crop, rim_strip, crack_roi, hist_th, is_crack, x_left, x_right)
    
    result_str = "CRACK (DEFECT)" if is_crack else "NORMAL (PASS)"
    print(f"[{os.path.basename(image_path)}] Result: {result_str} | Scores: {scores} | Defect Width: {x_right - x_left}px")
    
    if output_dashboard:
        os.makedirs(os.path.dirname(output_dashboard) or ".", exist_ok=True)
        cv2.imwrite(output_dashboard, dashboard)
        print(f"Saved inspection dashboard to: {output_dashboard}")
        
    if show_window:
        cv2.imshow("Inspection Dashboard", dashboard)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return is_crack, scores, dashboard


def main():
    parser = argparse.ArgumentParser(description="Automated Metal Can Rim Crack Defect Inspection")
    parser.add_argument("--image", type=str, default="data/samples/crack1.jpg", help="Path to input image")
    parser.add_argument("--model", type=str, default="models/model.xml", help="Path to OpenCV ANN_MLP model.xml")
    parser.add_argument("--output", type=str, default=None, help="Path to save dashboard image")
    parser.add_argument("--show", action="store_true", help="Display inspection window")
    args = parser.parse_args()
    
    inspect_image(args.image, model_path=args.model, output_dashboard=args.output, show_window=args.show)


if __name__ == "__main__":
    main()
