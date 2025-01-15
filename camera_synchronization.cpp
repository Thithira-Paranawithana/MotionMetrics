#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <numeric>

using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace Spinnaker::GenICam;
using namespace std;
using namespace cv;

// Add these new global variables after existing globals
atomic<bool> syncRequested{false};
const int64_t SYNC_THRESHOLD_NS = 1000000; // 1ms threshold for sync
mutex syncMutex;
condition_variable syncCV;

// Structure to hold frame data and timestamp
struct FrameData {
    Mat frame;
    uint64_t timestamp;
    int cameraIndex;

    FrameData(const Mat& f, uint64_t ts, int idx)
        : frame(f), timestamp(ts), cameraIndex(idx) {}

    // Add move constructor for efficient queue operations
    FrameData(FrameData&& other) noexcept
        : frame(std::move(other.frame)),
        timestamp(other.timestamp),
        cameraIndex(other.cameraIndex) {}

    // Add move assignment operator
    FrameData& operator=(FrameData&& other) noexcept {
        if (this != &other) {
            frame = std::move(other.frame);
            timestamp = other.timestamp;
            cameraIndex = other.cameraIndex;
        }
        return *this;
    }

    // Default constructor for vector initialization
    FrameData() : timestamp(0), cameraIndex(-1) {}
};

// Thread-safe queue with proper move semantics
class SafeQueue {
private:
    queue<FrameData> q;
    mutable mutex mtx;
    condition_variable cond;

public:
    // Default constructor
    SafeQueue() = default;

    // Delete copy constructor and assignment operator
    SafeQueue(const SafeQueue&) = delete;
    SafeQueue& operator=(const SafeQueue&) = delete;

    // Move constructor
    SafeQueue(SafeQueue&& other) noexcept {
        lock_guard<mutex> lock(other.mtx);
        q = move(other.q);
    }

    // Move assignment operator
    SafeQueue& operator=(SafeQueue&& other) noexcept {
        if (this != &other) {
            lock_guard<mutex> lock1(mtx);
            lock_guard<mutex> lock2(other.mtx);
            q = move(other.q);
        }
        return *this;
    }

    void push(FrameData&& item) {
        unique_lock<mutex> lock(mtx);
        q.push(move(item));
        lock.unlock();
        cond.notify_one();
    }

    bool pop(FrameData& item) {
        unique_lock<mutex> lock(mtx);
        if (q.empty()) {
            return false;
        }
        item = move(q.front());
        q.pop();
        return true;
    }

    bool empty() const {
        lock_guard<mutex> lock(mtx);
        return q.empty();
    }
};

// Global variables for synchronization
vector<SafeQueue> frameQueues;
atomic<bool> isRunning{ true };
mutex displayMutex;

// Configure trigger for software only with timestamp
int ConfigureTrigger(INodeMap& nodeMap)
{
    int result = 0;
    cout << endl << "* CONFIGURING TRIGGER AND TIMESTAMP *" << endl;

    try
    {
        // Enable timestamp
        CBooleanPtr ptrTimestampEnabled = nodeMap.GetNode("TimestampEnabled");
        if (IsWritable(ptrTimestampEnabled))
        {
            ptrTimestampEnabled->SetValue(true);
            cout << "Timestamp enabled..." << endl;
        }

        // Configure trigger same as before
        CEnumerationPtr ptrTriggerMode = nodeMap.GetNode("TriggerMode");
        if (!IsReadable(ptrTriggerMode))
        {
            cout << "Unable to disable trigger mode (node retrieval). Aborting..." << endl;
            return -1;
        }

        ptrTriggerMode->SetIntValue(ptrTriggerMode->GetEntryByName("Off")->GetValue());

        CEnumerationPtr ptrTriggerSelector = nodeMap.GetNode("TriggerSelector");
        if (IsWritable(ptrTriggerSelector))
        {
            ptrTriggerSelector->SetIntValue(ptrTriggerSelector->GetEntryByName("FrameStart")->GetValue());
        }

        CEnumerationPtr ptrTriggerSource = nodeMap.GetNode("TriggerSource");
        if (IsWritable(ptrTriggerSource))
        {
            ptrTriggerSource->SetIntValue(ptrTriggerSource->GetEntryByName("Software")->GetValue());
        }

        ptrTriggerMode->SetIntValue(ptrTriggerMode->GetEntryByName("On")->GetValue());
        cout << "Trigger configured for software trigger..." << endl;
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }
    return result;
}

// Function to acquire images from a single camera
void AcquireImages(CameraPtr pCam, int cameraIndex)
{
    ImageProcessor processor;
    processor.SetColorProcessing(SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR);

    while (isRunning)
    {
        try
        {
            // Execute software trigger
            CCommandPtr ptrSoftwareTriggerCommand = pCam->GetNodeMap().GetNode("TriggerSoftware");
            if (IsWritable(ptrSoftwareTriggerCommand))
            {
                ptrSoftwareTriggerCommand->Execute();
            }

            // Get the image
            ImagePtr pResultImage = pCam->GetNextImage(1000);

            if (pResultImage->IsIncomplete())
            {
                pResultImage->Release();
                continue;
            }

            // Get timestamp
            uint64_t timestamp = pResultImage->GetTimeStamp();

            // Convert to OpenCV format
            ImagePtr convertedImage = processor.Convert(pResultImage, PixelFormat_BGR8);
            Mat frame(convertedImage->GetHeight(), convertedImage->GetWidth(),
                CV_8UC3, convertedImage->GetData());

            // Create a copy of the frame
            Mat frameCopy;
            resize(frame, frameCopy, Size(640, 480));

            // Add timestamp to the frame
            string timestampStr = "Timestamp: " + to_string(timestamp) + " ns";
            putText(frameCopy, timestampStr, Point(10, 30),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);

            // Push to queue
            frameQueues[cameraIndex].push(FrameData(frameCopy, timestamp, cameraIndex));

            pResultImage->Release();
        }
        catch (Spinnaker::Exception& e)
        {
            cout << "Camera " << cameraIndex << " Error: " << e.what() << endl;
        }
    }
}

// Function to display synchronized frames
void DisplayFrames(int numCameras)
{
    vector<Mat> latestFrames(numCameras);
    vector<uint64_t> latestTimestamps(numCameras);

    int gridCols = ceil(sqrt(numCameras));
    int gridRows = ceil((float)numCameras / gridCols);

    namedWindow("Synchronized Camera Feeds", WINDOW_NORMAL);

    while (isRunning)
    {
        vector<FrameData> currentFrames;
        bool allFramesReceived = true;

        // Collect frames from all cameras
        for (int i = 0; i < numCameras; i++)
        {
            FrameData frameData;
            if (frameQueues[i].pop(frameData))
            {
                latestFrames[i] = frameData.frame;
                latestTimestamps[i] = frameData.timestamp;
            }
            else
            {
                allFramesReceived = false;
                break;
            }
        }

        if (!allFramesReceived)
        {
            this_thread::sleep_for(chrono::milliseconds(1));
            continue;
        }

        // Calculate timestamp differences
        if (numCameras > 1)
        {
            for (int i = 1; i < numCameras; i++)
            {
                int64_t diff = static_cast<int64_t>(latestTimestamps[i] - latestTimestamps[0]);
                string diffStr = "Diff with Cam0: " + to_string(diff) + " ns";
                putText(latestFrames[i], diffStr, Point(10, 60),
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
            }
        }

        // Arrange frames in grid
        Mat displayGrid;
        vector<Mat> rows;

        for (int i = 0; i < gridRows; i++)
        {
            Mat rowImg;
            vector<Mat> rowFrames;

            for (int j = 0; j < gridCols; j++)
            {
                int idx = i * gridCols + j;
                if (idx < numCameras)
                {
                    rowFrames.push_back(latestFrames[idx]);
                }
                else
                {
                    rowFrames.push_back(Mat::zeros(480, 640, CV_8UC3));
                }
            }

            hconcat(rowFrames, rowImg);
            rows.push_back(rowImg);
        }

        vconcat(rows, displayGrid);

        // Show the grid of camera feeds
        imshow("Synchronized Camera Feeds", displayGrid);

        char key = (char)waitKey(1);
        if (key == 'q' || key == 'Q')
        {
            isRunning = false;
        }
    }
}

int RunMultipleCameras(CameraList& camList)
{
    int result = 0;
    vector<CameraPtr> cameras;
    vector<thread> cameraThreads;
    unsigned int numCameras = camList.GetSize();

    // Initialize frame queues using emplace_back
    for (unsigned int i = 0; i < numCameras; i++) {
        frameQueues.emplace_back();
    }

    try
    {
        // Initialize cameras
        for (unsigned int i = 0; i < numCameras; i++)
        {
            CameraPtr pCam = camList.GetByIndex(i);
            pCam->Init();

            result |= ConfigureTrigger(pCam->GetNodeMap());

            CEnumerationPtr ptrAcquisitionMode = pCam->GetNodeMap().GetNode("AcquisitionMode");
            if (IsWritable(ptrAcquisitionMode))
            {
                ptrAcquisitionMode->SetIntValue(ptrAcquisitionMode->GetEntryByName("Continuous")->GetValue());
            }

            pCam->BeginAcquisition();
            cameras.push_back(pCam);
        }

        cout << "Press Enter to start synchronized video feeds (press 'q' to exit)..." << endl;
        getchar();

        // Start camera threads
        for (size_t i = 0; i < cameras.size(); i++)
        {
            cameraThreads.emplace_back(AcquireImages, cameras[i], i);
        }

        // Start display thread
        thread displayThread(DisplayFrames, numCameras);

        // Wait for threads to finish
        displayThread.join();
        for (auto& thread : cameraThreads)
        {
            thread.join();
        }

        // Clean up
        for (auto& cam : cameras)
        {
            cam->EndAcquisition();
            cam->DeInit();
        }
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

int main(int /*argc*/, char** /*argv*/)
{
    int result = 0;

    SystemPtr system = System::GetInstance();
    CameraList camList = system->GetCameras();
    unsigned int numCameras = camList.GetSize();

    cout << "Number of cameras detected: " << numCameras << endl;

    if (numCameras == 0)
    {
        camList.Clear();
        system->ReleaseInstance();
        cout << "No cameras detected!" << endl;
        cout << "Press Enter to exit..." << endl;
        getchar();
        return -1;
    }

    result = RunMultipleCameras(camList);

    camList.Clear();
    system->ReleaseInstance();

    cout << endl << "Press Enter to exit..." << endl;
    getchar();

    return result;
}
