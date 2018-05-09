#include<opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

int main()
{
	VideoCapture capture("C:\\Users\\DylanSu\\Desktop\\6.MP4");//获取视频
	if (!capture.isOpened())
		return -1;
	double rate = capture.get(CV_CAP_PROP_FPS);//获取视频帧率
	int delay = 1000 / rate;
	Mat framepro, frame, dframe;
	bool flag = false;
	namedWindow("image");
	namedWindow("test");
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	while (capture.read(frame))
	{	
		//imshow("原图", frame);
		cvtColor(frame, frame, CV_BGR2GRAY);//将每帧的图像转为灰度图
		if (false == flag)
		{
			framepro = frame.clone();//将第一帧图像拷贝给framePro
			flag = true;
		}
		else
		{
			
			absdiff(frame, framepro, dframe);//帧间差分计算两幅图像各个通道的相对应元素的差的绝对值
			framepro = frame.clone();//将当前帧拷贝给framepro
			Canny(dframe,dframe, 45, 135, 3);
			findContours(dframe,contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
			drawContours();
			imshow("image", frame);
			imshow("test", dframe);
			waitKey(delay);
		}
	}
	waitKey();
	return 0;
}