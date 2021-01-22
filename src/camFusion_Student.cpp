
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}

double GetSeparation(const cv::Point2f &p1, const cv::Point2f &p2)
{
    return cv::norm(p1 - p2);
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    for (std::vector<cv::DMatch>::const_iterator it = kptMatches.begin(); it != kptMatches.end(); ++it)
    {
        const cv::DMatch match = *it;
        const cv::Point2f &matchPt = kptsCurr[match.trainIdx].pt;
        double separationRejectTolerance = 10.0;
        if (boundingBox.roi.contains(matchPt))
        {
            // This point is within the bounding box

            // Calculate the separation distance between the matched pointxs in this and the previous frame
            double separation = GetSeparation(matchPt, kptsPrev[match.queryIdx].pt);
            if (separation < separationRejectTolerance)
            {
                boundingBox.kptMatches.push_back(match);
                boundingBox.keypoints.push_back(kptsCurr[match.trainIdx]);
            }
        }
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // Outer loop through the keypoint matches
    std::vector<double> separationRatios;
    for (std::vector<cv::DMatch>::const_iterator it1 = kptMatches.begin(); it1 != kptMatches.end(); ++it1)
    {
        // Get the match
        const cv::DMatch &matchLeft = *it1;
        // .. and the point in the current and previous frame
        const cv::KeyPoint &currFrPtLeft = kptsCurr[matchLeft.trainIdx];
        const cv::KeyPoint &prevFrPtLeft = kptsPrev[matchLeft.queryIdx];

        // Now loop through the remaining points (that haven't already been considered) and calculate the separation between the two sets of keypoints
        // and then the separation ratios
        for (std::vector<cv::DMatch>::const_iterator it2 = it1; it2 != kptMatches.end(); ++it2)
        {
            // Get the match
            const cv::DMatch &matchRight = *it2;
            // .. and the point in the current and previous frame
            const cv::KeyPoint &currFrPtRight = kptsCurr[matchRight.trainIdx];
            const cv::KeyPoint &prevFrPtRight = kptsPrev[matchRight.queryIdx];

            // Calculate the separations
            double currFrSep = GetSeparation(currFrPtLeft.pt, currFrPtRight.pt);
            double prevFrSep = GetSeparation(prevFrPtLeft.pt, prevFrPtRight.pt);

            // Calculate and store the ration of the separations
            if (prevFrSep > 0)
            {
                separationRatios.push_back(currFrSep / prevFrSep);
            }
        }
    }

    // sort the separation ratios
    std::sort(separationRatios.begin(), separationRatios.end());
    int centreIndex = std::floor(separationRatios.size() / 2.0);
    double separationRatioMedian = separationRatios.size() % 2 == 0 ? (separationRatios[centreIndex] + separationRatios[centreIndex + 1]) / 2.0 : separationRatios[centreIndex];

    // Try using the average
    double separationRatioMean = 0.0;
    for (std::vector<double>::const_iterator it = separationRatios.begin(); it != separationRatios.end(); ++it)
    {
        separationRatioMean += *it;
    }

    separationRatioMean /= separationRatios.size();

    double frameTimeSpan = 1.0 / frameRate;
    TTC = -frameTimeSpan / (1.0 - separationRatioMean);
}

bool LidarPointCompare(const LidarPoint & i, const LidarPoint &j)
{
    return (i.x < j.x);
}

double FindClosestLidarPoint(const std::vector<LidarPoint> &points)
{
    std::vector<LidarPoint> distances(points);

    // now sort them in x
    std::sort(distances.begin(), distances.end(), LidarPointCompare);

    // Now search through the sorted list, returning the first point that is not considered an outlier
    const double OutlierTolerance = 0.001;
    double closest = 0.0;
    for (int index = 0; index < distances.size() - 1; ++index)
    {
        closest = distances[index].x;
        if (distances[index + 1].x - distances[index].x < OutlierTolerance)
        {
            break;
        }
    }

    return closest;
}

double FindClosest(const std::vector<LidarPoint> &points)
{
    std::vector<double> distances;
    for (std::vector<LidarPoint>::const_iterator it = points.begin(); it != points.end(); ++it)
    {
        distances.push_back((*it).x);
    }

    // now sort them
    std::sort(distances.begin(), distances.end());

    // Now search through the sorted list, returning the first point that is not considered an outlier
    const double OutlierTolerance = 0.001;
    double closest = 0.0;
    for (int index = 0; index < distances.size() - 1; ++index)
    {
        closest = distances[index];
        if (distances[index + 1] - distances[index] < OutlierTolerance)
        {
            break;
        }
    }

    return closest;
}

double FindClosestHistogramMethod(const std::vector<LidarPoint> &points)
{
    std::vector<double> distances;
    for (std::vector<LidarPoint>::const_iterator it = points.begin(); it != points.end(); ++it)
    {
        distances.push_back((*it).x);
    }

    // now sort them
    std::sort(distances.begin(), distances.end());
    double binWidth = 0.002; // 1mm
    std::map<double, int> histogram;
    double bin = distances[0];

    // Build the histogram. Note : This loop was adapted from the stackoverlow post https://stackoverflow.com/questions/49458662
    // Could make the histogram into a class - but I don't have time at the mo. Think boost may have one. Surprised if it doesn't
    for (std::vector<double>::const_iterator it = distances.begin(); it != distances.end(); ++it)
    {
        const double &dist = *it;
        while (dist > bin + binWidth)
            bin += binWidth;
        
        ++histogram[bin];
    }

    // Choose the lower %ile bin as the closest distance
    int histPop = distances.size();
    int lowerPercentileBinPop = 10.0/100.0 * histPop;

    // Now locate the bin
    int cumulativeHistPop = 0;
    double closest = 0.0;
    for (std::map<double, int>::const_iterator it = histogram.begin(); it != histogram.end(); ++it)
    {
        cumulativeHistPop += (*it).second;
        if (cumulativeHistPop > lowerPercentileBinPop)
        {
            closest = (*it).first + (binWidth / 2);
            break;
        }
    }

    return closest;
}

// I have tried 2 methods for rejecting outlier points 'FindClosest' simply iterates through a sorted array of distances until the delta between 
// adjacent distances is less than a tolerance.
// 'FindClosestHistogramMethod' generates a histogram and then rejects ditances that are below the specified percentile population tolerance.
// I settled on the histogram method as this appears to produce (slightly) more consistent results.
// I have also used an 'avg' speed when computing the TTC as there appears to be genuine fluctuations in the measured distances between scans, leading to
// significant variations in instantaneous speed and hence TTC.
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev, std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // Find the closest point from the previous frame
    static bool firstFrame = true;
    static double lastMinX = 0.0;
    static double firstX = 0.0;
    static int frameCount = 1;
    double prevFrameMinX;
    double currFrameMinX;
    if (firstFrame)
    {
        prevFrameMinX = FindClosestHistogramMethod(lidarPointsPrev);
        currFrameMinX = FindClosestHistogramMethod(lidarPointsCurr);
        firstX = prevFrameMinX;
        firstFrame = false;
    }
    else
    {
        prevFrameMinX = lastMinX;
        currFrameMinX = FindClosestHistogramMethod(lidarPointsCurr);
    }
    lastMinX = currFrameMinX;
    double delta = std::abs(currFrameMinX - prevFrameMinX);

    double speed = delta * frameRate;
    double avgSpeed = std::abs(currFrameMinX - firstX) * frameRate / frameCount;
    ++frameCount;
    TTC = currFrameMinX / speed;
//    TTC = currFrameMinX / avgSpeed;

    std::cout << "Delta X = " << delta << "m, Min X = " << currFrameMinX << "m, Speed = " << speed << "m, Avg Speed = " << avgSpeed << "m/s, TTC = " << TTC << "s" << std::endl;

}

bool FindBoxForPoint(const cv::Point2f &point, const std::vector<BoundingBox> &boxes, int &boxId)
{
    for (std::vector<BoundingBox>::const_iterator boxIt = boxes.begin(); boxIt != boxes.end(); ++boxIt)
    {
        const BoundingBox &box = *boxIt;
        if (box.roi.contains(point))
        {
            boxId = box.boxID;
            return true;
        }
    }

    return false;
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // ...
    std::map<int, int> boxMatches;
    for (std::vector<cv::DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it)
    {
        const cv::DMatch match = *it;
        const cv::Point2f &prevFrameMatchPt = prevFrame.keypoints[match.queryIdx].pt;

        // Find the bounding box this point belongs to
        int prevFrameBoxId = 0;
        if (FindBoxForPoint(prevFrameMatchPt, prevFrame.boundingBoxes, prevFrameBoxId))
        {
            cv::Point2f &currFrameMatchPt = currFrame.keypoints[match.trainIdx].pt;

            // .. and now find the bounding box this point belongs to
            int currFrameBoxId = 0;
            if (FindBoxForPoint(currFrameMatchPt, currFrame.boundingBoxes, currFrameBoxId))
            {
                // Increment the match count for this box pair
                // .. Generate a map key based upon the box ids
                int key = prevFrameBoxId << 16;
                key |= currFrameBoxId;

                std::map<int, int>::iterator existingMatch = boxMatches.find(key);

                if (existingMatch == boxMatches.end())
                {
                    boxMatches.insert({key, 1});
                }
                else
                {
                    boxMatches[key]++;
                }
            }
        }    
    }

    for (std::vector<BoundingBox>::const_iterator boxIt = prevFrame.boundingBoxes.begin(); boxIt != prevFrame.boundingBoxes.end(); ++boxIt)
    {
        int prevFrameBoxId = (*boxIt).boxID;
        int currFrameBoxId = 0;
        int score = 0;
        for (std::map<int, int>::const_iterator it1 = boxMatches.begin(); it1 != boxMatches.end(); ++it1)
        {
            int key = (*it1).first;
            int testId = (key & 0xFFFF0000) >> 16;
            if (prevFrameBoxId == testId)
            {
                int testScore = (*it1).second;
                if (testScore > score)
                {
                    score = testScore;
                    currFrameBoxId = key & 0xFFFF;
                }
            }
        }

        if (score > 10)
        {
            bbBestMatches.insert({prevFrameBoxId, currFrameBoxId});
        }

    }
}
