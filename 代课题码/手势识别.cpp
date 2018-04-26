
/************************************************************************/
/*
Description:    ���Ƽ��
���˲�ȥ��
-->ת����HSV�ռ�
-->����Ƥ����HSV�ռ�ķֲ�������ֵ�жϣ������õ���inRange������
Ȼ�����һ����̬ѧ�Ĳ�����ȥ���������ţ����ֵı߽��������ƽ��
-->�õ���2ֵͼ�����findContours�ҳ��ֵ�������ȥ��α����������convexHull�����õ�͹����
Author:         Yang Xian
Email:          yang_xian521@163.com
Version:        2011-11-2
History:
*/
/*
#include <iostream>   // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion
#include <windows.h>
#include <opencv2/opencv.hpp>
//#include "cv.h"
//#include "highgui.h"
//#include "cxcore.h"
//#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
//#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
//#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
int delay = 1;
char c;
int frameNum = -1;          // Frame counter
bool lastImgHasHand = false;

int previousX = 0;
int previousY = 0;
CvCapture* pCapture = NULL;//
pCapture = cvCaptureFromCAM(0);

//Size refS = Size( (int) captRefrnc.get(CV_CAP_PROP_FRAME_WIDTH),
//  (int) captRefrnc.get(CV_CAP_PROP_FRAME_HEIGHT) );

bool bHandFlag = false;

const char* WIN_SRC = "Source";
const char* WIN_RESULT = "Result";

// Windows
namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
namedWindow(WIN_RESULT, CV_WINDOW_AUTOSIZE);

Mat frame;  // ������Ƶ֡����
Mat frameHSV;   // hsv�ռ�
Mat mask(frame.rows, frame.cols, CV_8UC1);  // 2ֵ��Ĥ
Mat dst(frame); // ���ͼ��

//  Mat frameSplit[4];

vector< vector<Point> > contours;   // ����
vector< vector<Point> > filterContours; // ɸѡ�������
vector< Vec4i > hierarchy;    // �����Ľṹ��Ϣ
vector< Point > hull; // ͹����ĵ㼯

bool movement = false;
int count = 0;

int presentX = 0;
int presentY = 0;

while (true) //Show the image captured in the window and repeat
{
//captRefrnc >> frame;
int minX = 320;//��Ļ��һ��
int maxX = 240;
int minY = 320;
int maxY = 240;

frame = cvQueryFrame(pCapture);
if (frame.empty())
{
cout << " < < <  Game over!  > > > ";
break;
}
imshow(WIN_SRC, frame);

// Begin

// ��ֵ�˲���ȥ����������
medianBlur(frame, frame, 5);
cvtColor(frame, frameHSV, CV_BGR2HSV);

Mat dstTemp1(frame.rows, frame.cols, CV_8UC1);
Mat dstTemp2(frame.rows, frame.cols, CV_8UC1);
// ��HSV�ռ�����������õ�2ֵͼ�����Ĳ���Ϊ�ֵ���״
inRange(frameHSV, Scalar(0, 30, 30), Scalar(40, 170, 256), dstTemp1);
inRange(frameHSV, Scalar(156, 30, 30), Scalar(180, 170, 256), dstTemp2);
bitwise_or(dstTemp1, dstTemp2, mask);
//  inRange(frameHSV, Scalar(0,30,30), Scalar(180,170,256), dst);

// ��̬ѧ������ȥ����������ʹ�ֵı߽��������
Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
erode(mask, mask, element);
morphologyEx(mask, mask, MORPH_OPEN, element);
dilate(mask, mask, element);
morphologyEx(mask, mask, MORPH_CLOSE, element);
frame.copyTo(dst, mask);
contours.clear();
hierarchy.clear();
filterContours.clear();
// �õ��ֵ�����
findContours(mask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
// ȥ��α����

for (size_t i = 0; i < contours.size(); i++)
{
//  approxPolyDP(Mat(contours[i]), Mat(approxContours[i]), arcLength(Mat(contours[i]), true)*0.02, true);
if (fabs(contourArea(Mat(contours[i]))) > 30000) //�ж��ֽ����������ֵ
{
filterContours.push_back(contours[i]);
}
}
// ������
if (filterContours.size()>0)
{

count++;
lastImgHasHand = true;
drawContours(dst, filterContours, -1, Scalar(255, 0, 255), 3/*, 8, hierarchy*//*);
for (size_t j = 0; j<filterContours.size(); j++)
{
convexHull(Mat(filterContours[j]), hull, true);
int hullcount = (int)hull.size();
for (int i = 0; i<hullcount - 1; i++)
{
line(dst, hull[i + 1], hull[i], Scalar(255, 255, 255), 2, CV_AA);//��ɫ
printf("num%d:x=%d\ty=%d\t\n", i, hull[i].x, hull[i].y);
if (hull[i].x>maxX)
maxX = hull[i].x;
if (hull[i].x<minX)
minX = hull[i].x;
if (hull[i].y>maxY)
maxY = hull[i].y;
if (hull[i].y<minY)
minY = hull[i].y;
printf("miniX=%d\tminiY=%d\tmaxX=%d\tmaxY=%d\t\n", minX, minY, maxX, maxY);

}

line(dst, hull[hullcount - 1], hull[0], Scalar(0, 255, 0), 2, CV_AA);//��ɫ�����һ��

if (count == 1)//��һ������������λ�ô���ȫ�ֱ����У������һ���ٸ����ȡ�
{
previousX = (minX + maxX) / 2;
printf("previousX=%d\n", previousX);
previousY = (minY + maxY) / 2;
printf("previousY=%d\n", previousY);
}
else
{
presentX = (minX + maxY) / 2;
presentY = (minY + maxY) / 2;
}
}
}
else
{
if (lastImgHasHand == true)
{
if ((previousX - presentX)<0)//���ĵĴ����ź�Ӣ�ĵĴ����������ۿ����������Ӱ�
{
printf("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<left\n");//����û��flip��������������ע��㡣
keybd_event(VK_LEFT, (BYTE)0, 0, 0);
keybd_event(VK_LEFT, (BYTE)0, KEYEVENTF_KEYUP, 0);
}
if ((previousX - presentX)>0)
{
printf(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>right\n");
keybd_event(VK_RIGHT, (BYTE)0, 0, 0);
keybd_event(VK_RIGHT, (BYTE)0, KEYEVENTF_KEYUP, 0);
}
if ((previousY - presentY)<0)
{
printf("downVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV\n");
}
if ((previousY - presentY)>0)
{
printf("upAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa\n");
}

count = 0;
lastImgHasHand = false;
}
}

imshow(WIN_RESULT, dst);
dst.release();


printf("previousX=%d\tpresentX=%d\tpreviousY=%d\tpresentY=%d\n", previousX, presentX, previousY, presentY);
printf("count=%d\n", count);
// End
c = cvWaitKey(1);
if (c == 27)
break;
}
}
*/