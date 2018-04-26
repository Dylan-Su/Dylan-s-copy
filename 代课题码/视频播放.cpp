/*
#include <cstdio>
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main()
{
IplImage *img;
char filename[1000];
CvVideoWriter *writer = NULL;

for (int i = 1; i <= 11775; i++)
{
sprintf_s(filename, "D:\opencv\output1.avi", i);
img = cvLoadImage(filename);

if (img == NULL)
continue;

if (writer == NULL)
{
writer = cvCreateVideoWriter("out.avi", CV_FOURCC('D', 'I', 'V', 'X'), 25, cvGetSize(img));
}

if (writer == NULL)
{
cout << "ÊÓÆµ´´½¨Ê§°Ü£¡" << endl;
exit(0);
}
cvWriteFrame(writer, img);
cvReleaseImage(&img);
}

cvReleaseVideoWriter(&writer);
return 0;
}*/