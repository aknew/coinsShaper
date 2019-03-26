#include <string>
#include <iostream>
#include <sys/stat.h>
#include <unistd.h> //mkdir for Win32

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

bool rectCompare(CvRect rect1, CvRect rect2);

Mat grayBlurredImage(Mat &src, float blurCoefficient) {
    Mat src_gray;
    /// Convert it to gray
    cvtColor( src, src_gray, CV_BGR2GRAY );
    int size = min(src.rows,src.cols)*blurCoefficient*0.25;
    if ( size % 2 == 0) {
        ++size;
    }
    GaussianBlur(src_gray, src_gray, Size(size,size), 0, 0, BORDER_DEFAULT );
    return src_gray;
}

Mat sobelFilter(Mat &src_gray) {
    // based on https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html
    Mat grad;

    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    /// Generate grad_x and grad_y
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    /// Gradient X
    //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
    Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );

    /// Gradient Y
    //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );

    /// Total Gradient (approximate)
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
    return grad;
}

vector<Rect> findContoursRects(Mat &img, int minSize = 10, int rectAppend = 30){
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    //CV_RETR_EXTERNAL RETR_TREE
    cv::findContours(img, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    vector<Rect> rects;
    rects.reserve(contours.size());
    for (auto c : contours) {
        auto rect = boundingRect(c);
        if (rect.width > minSize && rect.height > minSize) {

            // add some space around contour

            int x = max(rect.x - rectAppend,0);
            int y = max(rect.y - rectAppend,0);
            int w = min(rect.width + rectAppend*2, img.cols - rect.x);
            int h = min(rect.height + rectAppend*2, img.rows - rect.y);
            rect = Rect(x,y,w,h);

            rects.push_back(rect);
        }
    }
    return rects;
}

void cropResultImages(Mat &firstImage, Mat &secondImage, vector<Rect> &firstRects, vector<Rect> &secondRects, string nameCore) {
    int i = 0;
    for (auto rect1: firstRects) {
        for (auto rect2: secondRects) {
            if (rectCompare(rect1, rect2)){
                int height = max(rect1.height,rect2.height);
                int width = rect1.width + rect2.width;
                Mat result(height, width, firstImage.type());

                Mat imgROI1 = firstImage(rect1);
                Rect roi = Rect(0, 0, rect1.width, rect1.height);
                Mat resultROI1 = result(roi);
                imgROI1.copyTo(resultROI1);


                Mat imgROI2 = secondImage(rect2);
                roi = Rect(rect1.width, 0, rect2.width, rect2.height);
                Mat resultROI2 = result(roi);
                imgROI2.copyTo(resultROI2);

                stringstream imname;
                imname << nameCore << i << ".jpg";
                ++i;
                imwrite(imname.str(), result);
            }
        }
    }
}

void cutSingleImage(Mat &image, vector<Rect> &rects, string nameCore) {
    int i = 0;
    for (auto rect1: rects) {
        Mat imgROI1 = image(rect1);
        stringstream imname;
        imname << nameCore << i << ".jpg";
        ++i;
        imwrite(imname.str(), imgROI1);
    }
}

Mat preprocessImage(Mat &src, float blurParam, bool useSobel, int thParam, bool saveProcessed, string nameCore = "") {
    Mat gray = grayBlurredImage(src, blurParam);
    Mat result;
    if (useSobel) {
        result = sobelFilter(gray);
    } else {
        threshold(gray, result, thParam, 255.0, THRESH_BINARY);
    }

    if (saveProcessed) {
        string grayName = nameCore + "Gray.jpg";
        imwrite(grayName, gray);
        string resultName = nameCore;
        if (useSobel) {
            resultName = resultName + "Sobel.jpg";
        } else {
            resultName = resultName + "Threshold.jpg";
        }

        imwrite(resultName, result);
    }
    return result;
}

void printHelp() {
    cout << "usage: coinsShaper [-o OutputDir -saveProcessed -blur N.N -th N] inputFile1 [inputFile2]" << endl;
    cout << "   -o OutputDir    path to directory where result will saved" << endl;
    cout << "   -saveProcessed  app will save preprocessed images (blurred gray and thresholded" << endl;
    cout << "   -blur N.N   blur level from 0 to 1.0, default value is 0.05" << endl;
    cout << "   -th N   threshold level, default value is 50" << endl;
}

int main(int argc, char* argv[]) {

    string outputDir="", inputFile1="", inputFile2="";
    bool saveProcessed = false;
    int thLevel = 50;
    float blurLevel = 0.05;
    bool useSobel = false;

    int i = 1;
    // I know about boost/program_options, but I don't want add all boost only for this small issue
    while (i<argc) {
        string str = string(argv[i]);

        if (str == "-o"){ // output directory
            ++i;
            outputDir=string(argv[i]);
        } else if (str == "-help") {
            printHelp();
            return 0;
        } else if (str == "-saveProcessed") {
            saveProcessed = true;
        } else if (str == "-th"){ // output directory
            ++i;
            thLevel = atoi(argv[i]);
        }else if (str == "-blur"){ // output directory
            ++i;
            blurLevel = atof(argv[i]);
        } else if (inputFile1==""){
            inputFile1=str;
        } else {
            inputFile2=str;
        }
        ++i;
    }

    if (inputFile1==""){
        cout<<"Error: Need at list one input file"<<endl;
        return -1;
    }

    if (outputDir==""){
        outputDir="tempCoinsShaper";
    }
#ifdef _WIN32
     mkdir(outputDir.c_str());//Windows
#else
    mkdir(outputDir.c_str(),0777);
#endif

    Mat firstImage = imread(inputFile1);
    Mat preprocessed = preprocessImage(firstImage, blurLevel, useSobel, thLevel, saveProcessed, outputDir + "/first");
    auto rects1 = findContoursRects(preprocessed);
    cout<<"find contours in 1 new: "<<rects1.capacity()<<endl;

    if (inputFile2 != "") {

        Mat secondImage = imread(inputFile2);
        Mat preprocessed2 = preprocessImage(secondImage, blurLevel, useSobel, thLevel, saveProcessed, outputDir + "/second");
        auto rects2 = findContoursRects(preprocessed2);
        cout<<"find contours in 2 new: "<<rects2.capacity()<<endl;

        cropResultImages(firstImage, secondImage, rects1, rects2, outputDir + "/coinCpp");
    } else {
        cutSingleImage(firstImage, rects1, outputDir + "/coinCpp");
    }
    return 0;
}

bool rectCompare(CvRect rect1, CvRect rect2){

    int S1=rect1.width*rect1.height;
    int S2=rect2.width*rect2.height;

    float ratio = float(S1)/float(S2);

    const float kRatioThreashold = 0.2;

    if (abs(1.0-ratio) > kRatioThreashold) {
        // square ratio is too big, it can't be different side of one object
        return false;
    }

    int x=max(rect1.x,rect2.x);
    int y=max(rect1.y,rect2.y);
    int w=min(rect1.x+rect1.width, rect2.x+rect2.width);
    int h=min(rect1.y+rect1.height, rect2.y+rect2.height);

    if (x>w || y>h)
        return false;

    int S=(w-x)*(h-y);

    return S>0;
}
