/**
 * ============================================================================
 * Project: Automated Metal Can Rim Crack Defect Metrology
 * Repository: https://github.com/teeranon124/can_crack
 * Author: Teeranon (AI Engineering, Prince of Songkla University)
 * Target: Industrial High-Speed In-Line Vision Inspection
 * ============================================================================
 * Pipeline Architecture:
 * 1. Boundary Localization & Circular Fitting (Hu Moments Invariant)
 * 2. Linear-Polar Coordinate Unwrapping (Curved Perimeter -> 1D Tape)
 * 3. Outer Rim ROI Isolation & CCW Alignment
 * 4. 1D Intensity Projection & Notch/Crack Valley Scanning (with 90° Seam Recovery)
 * 5. Normalized Defect Patch Extraction (120x40 standard size)
 * 6. Illumination-Invariant 120-d Vertical Feature Extraction
 * 7. Shallow Neural Network (ANN_MLP) Defect Classification (model.xml)
 * 8. Real-time Multi-Panel Inspection Dashboard GUI (1024x768)
 * ============================================================================
 */

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

using namespace cv;
using namespace cv::ml;
using namespace std;

// ----------------------------------------------------------------------------
// Stage 1: Circular Lid Contour Verification using Hu Moments
// ----------------------------------------------------------------------------
// Theoretical fact: An ideal solid circle has its first Hu moment hu[0] = 1 / (2 * pi) ~= 0.159.
// Machine conveyor borders, straight edges, and debris will produce hu[0] > 0.25.
bool isCircularContour(const vector<Point>& contour)
{
    if (contour.size() < 6) return false;

    Moments mu = moments(contour);
    if (mu.m00 <= 0.0) return false;

    double hu[7];
    HuMoments(mu, hu);

    // hu[0] < 0.22 accepts circular and slightly elliptical metal can lids
    return (hu[0] < 0.22);
}

// ----------------------------------------------------------------------------
// Stage 1: Can Boundary Localization & Circular Lid Extraction
// ----------------------------------------------------------------------------
Mat extractCanLid(const Mat& srcGray, Rect& outBoundingBox)
{
    Mat blurred, gradX, gradY, absGradX, absGradY, grad, binaryThresh;

    GaussianBlur(srcGray, blurred, Size(5, 5), 0, 0);

    // Compute gradients along X and Y axes (Sobel kernel size 3)
    Sobel(blurred, gradX, CV_16S, 1, 0, 3);
    Sobel(blurred, gradY, CV_16S, 0, 1, 3);
    convertScaleAbs(gradX, absGradX);
    convertScaleAbs(gradY, absGradY);
    addWeighted(absGradX, 0.5, absGradY, 0.5, 0, grad);

    // Invert threshold so solid can interior forms a closed white region
    threshold(grad, binaryThresh, 30, 255, THRESH_BINARY_INV);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(binaryThresh, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    double maxArea = 0;
    int bestIndex = -1;
    double frameArea = srcGray.cols * srcGray.rows;

    for (size_t i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        // Filter out whole frame borders (>90% frame) and tiny noise specs (<5000px)
        if (area > 5000 && area < (frameArea * 0.9))
        {
            if (isCircularContour(contours[i]) && area > maxArea)
            {
                maxArea = area;
                bestIndex = static_cast<int>(i);
            }
        }
    }

    if (bestIndex >= 0)
    {
        outBoundingBox = boundingRect(contours[bestIndex]);
        return srcGray(outBoundingBox).clone();
    }

    // Fallback: If no single contour dominates, assume entire image is already cropped lid
    outBoundingBox = Rect(0, 0, srcGray.cols, srcGray.rows);
    return srcGray.clone();
}

// ----------------------------------------------------------------------------
// Stage 2 & 3: Linear-Polar Unwrapping & Outer Rim Strip Extraction
// ----------------------------------------------------------------------------
Mat unwrapCanRim(const Mat& lidGray, int rimThickness = 50)
{
    Point2f center(lidGray.cols / 2.0f, lidGray.rows / 2.0f);
    double maxRadius = min(lidGray.cols, lidGray.rows) / 2.0;

    Mat imgPolar;
    linearPolar(lidGray, imgPolar, center, maxRadius, INTER_LINEAR);

    // Isolate the outermost rim boundary (outermost 'rimThickness' pixels in radius)
    int cropX = max(0, imgPolar.cols - rimThickness);
    Mat outerRim = imgPolar(Rect(cropX, 0, imgPolar.cols - cropX, imgPolar.rows));

    // Rotate 90 degrees CCW so that the circular trajectory aligns horizontally
    Mat horizontalRim;
    rotate(outerRim, horizontalRim, ROTATE_90_COUNTERCLOCKWISE);

    return horizontalRim;
}

// ----------------------------------------------------------------------------
// Stage 4: 1D Intensity Projection & Notch/Crack Valley Scanning
// ----------------------------------------------------------------------------
Mat detectDefectRegion(const Mat& rimStrip, int& xLeft, int& xRight, vector<int>& histTh, bool& isSeamSplit, int intensityThreshold = 5000, int pad = 18)
{
    histTh.assign(rimStrip.cols, 0);
    vector<int> colSums(rimStrip.cols, 0);

    for (int x = 0; x < rimStrip.cols; x++)
    {
        int sum = 0;
        for (int y = 0; y < rimStrip.rows; y++)
        {
            sum += rimStrip.at<uchar>(y, x);
        }
        colSums[x] = sum;
        histTh[x] = (sum > intensityThreshold) ? 100 : 0;
    }

    // Find valleys: contiguous intervals where intensity drops below threshold between bright metal shoulders
    vector<int> highIndices;
    for (int x = 0; x < rimStrip.cols; x++)
    {
        if (histTh[x] > 0) highIndices.push_back(x);
    }

    vector<pair<int, int>> valleys;
    if (!highIndices.empty())
    {
        bool inValley = false;
        int vStart = 0;
        for (int x = highIndices.front(); x <= highIndices.back(); x++)
        {
            if (histTh[x] == 0 && !inValley)
            {
                inValley = true;
                vStart = x;
            }
            else if (histTh[x] > 0 && inValley)
            {
                inValley = false;
                valleys.push_back({ vStart, x - 1 });
            }
        }
    }

    if (!valleys.empty())
    {
        // Pick the most significant valley by width
        auto bestValley = *max_element(valleys.begin(), valleys.end(),
            [](const pair<int, int>& a, const pair<int, int>& b) {
                return (a.second - a.first) < (b.second - b.first);
            });

        xLeft = max(0, bestValley.first - pad);
        xRight = min(rimStrip.cols, bestValley.second + 1 + pad);
        isSeamSplit = (xLeft == 0 || xRight >= rimStrip.cols);

        return rimStrip(Rect(xLeft, 0, xRight - xLeft, rimStrip.rows)).clone();
    }

    // If no prominent valley, extract center patch as normal reference
    int mid = rimStrip.cols / 2;
    xLeft = max(0, mid - 30);
    xRight = min(rimStrip.cols, mid + 30);
    isSeamSplit = false;

    return rimStrip(Rect(xLeft, 0, xRight - xLeft, rimStrip.rows)).clone();
}

// ----------------------------------------------------------------------------
// Stage 6: 120-d Illumination-Invariant Vertical Feature Extraction
// ----------------------------------------------------------------------------
Mat extract120dFeatures(const Mat& defectPatch)
{
    Mat resizedPatch;
    resize(defectPatch, resizedPatch, Size(120, 40));

    float maxVal = 0.0f;
    vector<float> colSums(120, 0.0f);

    for (int x = 0; x < 120; x++)
    {
        float sum = 0.0f;
        for (int y = 0; y < 40; y++)
        {
            sum += resizedPatch.at<uchar>(y, x);
        }
        colSums[x] = sum;
        if (sum > maxVal) maxVal = sum;
    }

    Mat features(1, 120, CV_32F);
    for (int x = 0; x < 120; x++)
    {
        features.at<float>(0, x) = (maxVal > 0.0f) ? (colSums[x] / maxVal) : colSums[x];
    }

    return features;
}

// ----------------------------------------------------------------------------
// Stage 7: Multi-Layer Perceptron (ANN_MLP) Defect Classification
// ----------------------------------------------------------------------------
int classifyDefect(const Ptr<ANN_MLP>& model, const Mat& featureVector, float& rawScoreCrack, float& rawScoreNormal)
{
    Mat response;
    model->predict(featureVector, response);

    rawScoreCrack = response.at<float>(0, 0);
    rawScoreNormal = response.at<float>(0, 1);

    // Class 0: CRACK (Defect / FAIL)
    // Class 1: NO CRACK (Normal / PASS)
    return (rawScoreCrack > rawScoreNormal) ? 0 : 1;
}

// ----------------------------------------------------------------------------
// Stage 8: Multi-Panel Industrial Inspection Dashboard (1024x768)
// ----------------------------------------------------------------------------
Mat renderDashboardGUI(const Mat& inputFrame, const Mat& lidCrop, const Mat& unwrappedRim, const Mat& crackRoi, const vector<int>& histTh, bool isCrack, int xLeft, int xRight)
{
    Mat gui = Mat::zeros(Size(1024, 768), CV_8UC3);

    // Header Bar
    rectangle(gui, Rect(0, 0, 1024, 55), Vec3b(40, 35, 30), FILLED);
    putText(gui, "AUTOMATED CAN RIM CRACK METROLOGY SYSTEM", Point(20, 37),
            FONT_HERSHEY_SIMPLEX, 0.85, Scalar(255, 255, 255), 2, LINE_AA);
    putText(gui, "High-Speed In-Line Vision", Point(760, 37),
            FONT_HERSHEY_SIMPLEX, 0.55, Scalar(180, 180, 180), 1, LINE_AA);

    // Status Banner (PASS = Green, FAIL = Red)
    Scalar bannerColor = isCrack ? Scalar(30, 30, 200) : Scalar(40, 160, 40);
    string statusText = isCrack ? "STATUS: FAIL - CRACK DETECTED" : "STATUS: PASS - RIM NORMAL";
    rectangle(gui, Rect(0, 55, 1024, 50), bannerColor, FILLED);
    putText(gui, statusText, Point(25, 92),
            FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2, LINE_AA);

    // Panel 1: Camera Input
    Mat inputDisp;
    if (inputFrame.channels() == 1) cvtColor(inputFrame, inputDisp, COLOR_GRAY2BGR);
    else inputDisp = inputFrame.clone();
    resize(inputDisp, inputDisp, Size(300, 225));
    inputDisp.copyTo(gui(Rect(30, 140, 300, 225)));
    rectangle(gui, Rect(30, 140, 300, 225), Scalar(80, 80, 80), 1);
    putText(gui, "[1] Factory Camera Input", Point(30, 130),
            FONT_HERSHEY_SIMPLEX, 0.55, Scalar(220, 220, 220), 1, LINE_AA);

    // Panel 2: Can Lid ROI
    Mat lidDisp;
    if (lidCrop.channels() == 1) cvtColor(lidCrop, lidDisp, COLOR_GRAY2BGR);
    else lidDisp = lidCrop.clone();
    resize(lidDisp, lidDisp, Size(240, 240));
    lidDisp.copyTo(gui(Rect(370, 140, 240, 240)));
    rectangle(gui, Rect(370, 140, 240, 240), Scalar(80, 80, 80), 1);
    putText(gui, "[2] Extracted Can Lid ROI", Point(370, 130),
            FONT_HERSHEY_SIMPLEX, 0.55, Scalar(220, 220, 220), 1, LINE_AA);

    // Panel 3: Zoomed Crack ROI
    if (!crackRoi.empty())
    {
        Mat patchDisp;
        if (crackRoi.channels() == 1) cvtColor(crackRoi, patchDisp, COLOR_GRAY2BGR);
        else patchDisp = crackRoi.clone();
        resize(patchDisp, patchDisp, Size(320, 120));
        patchDisp.copyTo(gui(Rect(660, 140, 320, 120)));
        Scalar boxColor = isCrack ? Scalar(0, 0, 255) : Scalar(0, 255, 0);
        rectangle(gui, Rect(660, 140, 320, 120), boxColor, 2);
        putText(gui, "[3] Defect ROI Patch (Zoomed)", Point(660, 130),
                FONT_HERSHEY_SIMPLEX, 0.55, Scalar(220, 220, 220), 1, LINE_AA);

        string spanText = "Span: " + to_string(xRight - xLeft) + " px [" + to_string(xLeft) + ":" + to_string(xRight) + "]";
        putText(gui, spanText, Point(660, 285),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(160, 200, 255), 1, LINE_AA);
    }

    // Panel 4: Linear-Polar Unwrapped Rim (Full 360 deg)
    Mat rimDisp;
    if (unwrappedRim.channels() == 1) cvtColor(unwrappedRim, rimDisp, COLOR_GRAY2BGR);
    else rimDisp = unwrappedRim.clone();
    resize(rimDisp, rimDisp, Size(950, 75));
    rimDisp.copyTo(gui(Rect(30, 440, 950, 75)));
    rectangle(gui, Rect(30, 440, 950, 75), Scalar(100, 100, 100), 1);
    putText(gui, "[4] Unwrapped Linear-Polar Rim Strip (360 deg perimeter trajectory)", Point(30, 430),
            FONT_HERSHEY_SIMPLEX, 0.55, Scalar(220, 220, 220), 1, LINE_AA);

    if (xRight > xLeft && unwrappedRim.cols > 0)
    {
        double scaleX = 950.0 / unwrappedRim.cols;
        int markX1 = static_cast<int>(30 + xLeft * scaleX);
        int markX2 = static_cast<int>(30 + xRight * scaleX);
        rectangle(gui, Rect(markX1, 440, markX2 - markX1, 75), isCrack ? Scalar(0, 0, 255) : Scalar(0, 255, 0), 2);
    }

    // Panel 5: Projection Profile Graph
    Mat graphImg(100, 950, CV_8UC3, Scalar(20, 20, 20));
    if (!histTh.empty())
    {
        double step = 950.0 / histTh.size();
        for (size_t i = 0; i < histTh.size() - 1; i++)
        {
            Point pt1(static_cast<int>(i * step), static_cast<int>(100 - (histTh[i] / 100.0 * 70 + 15)));
            Point pt2(static_cast<int>((i + 1) * step), static_cast<int>(100 - (histTh[i + 1] / 100.0 * 70 + 15)));
            Scalar color = (histTh[i] > 0) ? Scalar(0, 255, 255) : Scalar(0, 0, 255);
            line(graphImg, pt1, pt2, color, 2);
        }
    }
    graphImg.copyTo(gui(Rect(30, 570, 950, 100)));
    rectangle(gui, Rect(30, 570, 950, 100), Scalar(80, 80, 80), 1);
    putText(gui, "[5] 1D Column-wise Intensity Projection (Notch/Drop Detection Signal)", Point(30, 560),
            FONT_HERSHEY_SIMPLEX, 0.55, Scalar(220, 220, 220), 1, LINE_AA);

    // Footer
    putText(gui, "Algorithm: Hu-Moments Localization -> Polar Unwrapping -> 120-d Projection -> OpenCV ANN_MLP (RPROP)",
            Point(30, 725), FONT_HERSHEY_SIMPLEX, 0.48, Scalar(140, 140, 140), 1, LINE_AA);

    return gui;
}

// ----------------------------------------------------------------------------
// Main Execution Entrypoint
// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
    cout << "==========================================================" << endl;
    cout << "   Automated Metal Can Rim Crack Metrology Inspector      " << endl;
    cout << "   PSU AI Engineering | Target: High-Speed Canning Plant  " << endl;
    cout << "==========================================================" << endl;

    string imagePath = (argc > 1) ? argv[1] : "data/samples/crack1.jpg";
    string modelPath = (argc > 2) ? argv[2] : "models/model.xml";

    Mat frame = imread(imagePath);
    if (frame.empty())
    {
        cerr << "[ERROR] Cannot open image: " << imagePath << endl;
        return -1;
    }

    Ptr<ANN_MLP> model = StatModel::load<ANN_MLP>(modelPath);
    if (model.empty())
    {
        cerr << "[ERROR] Could not load classifier model from: " << modelPath << endl;
        return -1;
    }

    Mat gray;
    if (frame.channels() == 3) cvtColor(frame, gray, COLOR_BGR2GRAY);
    else gray = frame.clone();

    Mat lidCrop, rimStrip, crackRoi;
    Rect lidRect;
    int xLeft = 0, xRight = 0;
    vector<int> histTh;
    bool isSeamSplit = false;

    // Detect input mode: pre-cropped patch, unwrapped strip, circular lid, or full frame
    float aspectRatio = static_cast<float>(gray.cols) / gray.rows;
    if (gray.cols < 200 && gray.rows < 100)
    {
        // Mode 1: Image is already an extracted defect patch (e.g. crack0.jpg, crack5.jpg)
        lidCrop = gray.clone();
        rimStrip = gray.clone();
        crackRoi = gray.clone();
        xLeft = 0;
        xRight = gray.cols;
    }
    else if (aspectRatio > 3.0f)
    {
        // Mode 2: Image is an unwrapped rim strip (e.g. crack1.jpg, crack4.jpg)
        lidCrop = gray.clone();
        rimStrip = gray.clone();
        crackRoi = detectDefectRegion(rimStrip, xLeft, xRight, histTh, isSeamSplit);
    }
    else if (aspectRatio > 0.8f && aspectRatio < 1.2f && gray.cols >= 200)
    {
        // Mode 3: Image is a circular lid crop
        lidCrop = gray.clone();
        rimStrip = unwrapCanRim(lidCrop);
        crackRoi = detectDefectRegion(rimStrip, xLeft, xRight, histTh, isSeamSplit);
    }
    else
    {
        // Mode 4: Full factory camera frame
        lidCrop = extractCanLid(gray, lidRect);
        rimStrip = unwrapCanRim(lidCrop);
        crackRoi = detectDefectRegion(rimStrip, xLeft, xRight, histTh, isSeamSplit);
    }

    // Seam recovery: if defect touches boundary cut, rotate 90 CCW and re-unwrap
    if (isSeamSplit && lidCrop.rows == lidCrop.cols)
    {
        rotate(lidCrop, lidCrop, ROTATE_90_COUNTERCLOCKWISE);
        rimStrip = unwrapCanRim(lidCrop);
        crackRoi = detectDefectRegion(rimStrip, xLeft, xRight, histTh, isSeamSplit);
    }

    // Stage 6: 120-d Feature extraction
    Mat features = extract120dFeatures(crackRoi);

    // Stage 7: Defect classification
    float scoreCrack = 0.0f, scoreNormal = 0.0f;
    int prediction = classifyDefect(model, features, scoreCrack, scoreNormal);
    bool isCrack = (prediction == 0);

    cout << "\n>>> Inspection Results for [" << imagePath << "]:" << endl;
    cout << "    Classification : " << (isCrack ? "FAIL (CRACK DETECTED)" : "PASS (NORMAL RIM)") << endl;
    cout << "    ANN Raw Scores : Crack=" << scoreCrack << ", Normal=" << scoreNormal << endl;
    cout << "    Defect Width   : " << (xRight - xLeft) << " pixels [" << xLeft << " - " << xRight << "]" << endl;

    // Stage 8: Render Dashboard
    Mat dashboard = renderDashboardGUI(frame, lidCrop, rimStrip, crackRoi, histTh, isCrack, xLeft, xRight);

    imwrite("assets/cpp_dashboard_result.png", dashboard);
    cout << "    Dashboard GUI saved to: assets/cpp_dashboard_result.png" << endl;

    namedWindow("Inspection Dashboard", WINDOW_NORMAL);
    resizeWindow("Inspection Dashboard", 1024, 768);
    imshow("Inspection Dashboard", dashboard);
    cout << "\nShowing dashboard (auto-closing in 3s or press any key)..." << endl;
    waitKey(3000);

    return 0;
}
