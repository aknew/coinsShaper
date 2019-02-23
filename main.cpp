#include <string>
#include <iostream>
#include <sys/stat.h>
#include <unistd.h> //mkdir for Win32

#include <opencv2/opencv.hpp>

#define rectAdding 10 //увеличение ректа, чтобы избежать ошибок и не обрезать вплотную

using namespace std;
using namespace cv;

int findCoins(CvMemStorage* storage,CvSeq** contours, IplImage* image, bool isBlack=false);

bool rectCompare(CvRect rect1, CvRect rect2);

Mat grayBlurredImage(Mat &src) {
    Mat src_gray;
    /// Convert it to gray
    cvtColor( src, src_gray, CV_BGR2GRAY );
    GaussianBlur(src_gray, src_gray, Size(15,15), 0, 0, BORDER_DEFAULT );
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

vector<Rect> findContoursRects(Mat &img){
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    //CV_RETR_EXTERNAL RETR_TREE
    cv::findContours(img, contours, hierarchy, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    vector<Rect> rects;
    rects.reserve(contours.size());
    const int kMinSize = 50;
    for (auto c : contours) {
        auto rect = boundingRect(c);
        if (rect.width > kMinSize && rect.height > kMinSize) {
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

Mat preprocessImage(Mat &src, bool saveTransitionStage = false, string nameCore = "") {
    Mat gray = grayBlurredImage(src);
    Mat result;
    threshold(gray, result, 128, 255.0, THRESH_OTSU);

    if (saveTransitionStage) {
        string grayName = nameCore + "Gray.jpg";
        imwrite(grayName, gray);
        string tsName = nameCore + "Threshold.jpg";
        imwrite(tsName, result);
    }
    return result;
}

int main(int argc, char* argv[])
{

    string outputDir="", inputFile1="", inputFile2="";
    bool blackBackground=false;
    bool nextKeyisOutput=false;

    for (int i=1; i<argc; ++i){
        if (nextKeyisOutput){
            outputDir=string(argv[i]);
            nextKeyisOutput=false;
            continue;
        }
        string str=string(argv[i]);
        if (str=="-b"){
            blackBackground=true;
        }
        else if (str.find("-o=")!=string::npos){
            outputDir=str.substr(3);
            if (outputDir=="")
                nextKeyisOutput=true;
        }else if (inputFile1==""){
            inputFile1=str;
        }else{
            inputFile2=str;
        }
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
    Mat preprocessed = preprocessImage(firstImage, true, outputDir + "/first");
    auto rects1 = findContoursRects(preprocessed);
    cout<<"find contours in 1 new: "<<rects1.capacity()<<endl;

    if (inputFile2 != "") {

        Mat secondImage = imread(inputFile2);
        Mat preprocessed = preprocessImage(secondImage, true, outputDir + "/second");
        auto rects2 = findContoursRects(preprocessed);
        cout<<"find contours in 2 new: "<<rects2.capacity()<<endl;

        cropResultImages(firstImage, secondImage, rects1, rects2, outputDir + "/coinsNew");
    }

    IplImage* image1 = 0;

    // получаем первую картинку
    image1 = cvLoadImage(inputFile1.c_str(),1);

    assert( image1 != 0 );

    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* contours=0;

    int c=findCoins(storage, &contours, image1, blackBackground);

    cout<<"find coins in file1:"<<c<<endl;

    if (inputFile2!=""){
        IplImage* image2 = 0;

        // получаем первую картинку
        image2 = cvLoadImage(inputFile2.c_str(),1);

        assert( image2 != 0 );

        CvMemStorage* storage2 = cvCreateMemStorage(0);
        CvSeq* contours2=0;

        c=findCoins(storage2, &contours2, image2, blackBackground);

        cout<<"find coins in file2:"<<c<<endl;

        int i=0;
        for(CvSeq* seq0 = contours2;seq0!=0;seq0 = seq0->h_next){
            CvRect rect=cvBoundingRect(seq0);
            rect.x-=rectAdding;
            rect.y-=rectAdding;
            rect.width+=2*rectAdding;
            rect.height+=2*rectAdding;
            for(CvSeq* seq1 = contours;seq1!=0;seq1 = seq1->h_next){
                CvRect rect1=cvBoundingRect(seq1);
                rect1.x-=rectAdding;
                rect1.y-=rectAdding;
                rect1.width+=2*rectAdding;
                rect1.height+=2*rectAdding;
                if (rectCompare(rect1,rect)){
                    cvSetImageROI(image2, rect);
                    cvSetImageROI(image1, rect1);

                    IplImage* imgRoi   = cvCreateImage( cvSize(rect.width+rect1.width, max(rect1.height,rect.height)),
                                                        image1->depth,
                                                        image1->nChannels
                                                      );
                    cvSetImageROI(imgRoi, cvRect(0, 0, rect1.width, rect1.height));
                    cvResize(image1,imgRoi);
                    cvResetImageROI(imgRoi);

                    cvSetImageROI(imgRoi, cvRect(rect1.width, 0, rect.width, rect.height));
                    cvResize(image2,imgRoi);
                    cvResetImageROI(imgRoi);


                    stringstream convert;
                    convert<<outputDir<<"/coin"<<i<<"Old.jpg";
                    ++i;
                    cvSaveImage(convert.str().c_str(), imgRoi);
                    cvReleaseImage(&imgRoi);

                    cvResetImageROI(image1);
                    cvResetImageROI(image2);

                    //break;
                }
            }

        }

        cvReleaseImage(&image2);
    }
    else  {
        int i=0;
        for(CvSeq* seq0 = contours;seq0!=0;seq0 = seq0->h_next){
            stringstream convert;
            convert<<outputDir<<"/coin"<<i<<"Old.jpg";
            CvRect rect=cvBoundingRect(seq0);
            rect.x-=rectAdding;
            rect.y-=rectAdding;
            rect.width+=2*rectAdding;
            rect.height+=2*rectAdding;
            IplImage* imgRoi   = cvCloneImage(image1);
            cvSetImageROI(imgRoi, rect);
            cvSaveImage(convert.str().c_str(), imgRoi);
            cvReleaseImage(&imgRoi);
            ++i;
        }
    }

    cvReleaseImage(&image1);

    return 0;
}

int findCoins(CvMemStorage* storage,CvSeq** contours, IplImage* image, bool isBlack){

    IplImage* gray = 0;
    IplImage* dst = 0;

    // клонируем
    dst = cvCloneImage(image);
    // создаём одноканальные картинки
    gray = cvCreateImage( cvGetSize(image), IPL_DEPTH_8U, 1 );

    //сглаживаем
    cvSmooth(dst,dst, CV_GAUSSIAN, 15, 15);
    // преобразуем в градации серого
    cvCvtColor(dst, gray, CV_RGB2GRAY);
    cvSaveImage("grayOld.jpg", gray);
    // преобразуем в двоичное
    cvThreshold( gray, gray, 128, 255, CV_THRESH_OTSU);
    cvSaveImage("thresholdOld.jpg", gray);

    //если фон черный - инвертируем изображение
    if (!isBlack)
        cvNot(gray,gray);

    // находим контуры
    int contoursCont = cvFindContours( gray, storage,contours,sizeof(CvContour),CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE,cvPoint(0,0));

    cvReleaseImage(&gray);
    cvReleaseImage(&dst);

    return contoursCont;
}


bool rectCompare(CvRect rect1, CvRect rect2){

    //проверяем что площадь пресечения больше половины от площади прямоугольника
    int S0=rect1.width*rect1.height;

    int x=max(rect1.x,rect2.x);
    int y=max(rect1.y,rect2.y);
    int w=min(rect1.x+rect1.width,rect2.x+rect2.width);
    int h=min(rect1.y+rect1.height, rect2.y+rect2.height);

    if (x>w || y>h)
        return false;

    int S=(w-x)*(h-y);

    return S>0.5*S0;
}
