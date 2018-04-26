//#include <iostream>
//#include <opencv2/opencv.hpp>
//
//using namespace std;
//using namespace cv;
//
//int main()
//{
//	Mat srcImage = imread("C:/Users/DylanSu/Desktop/2.jpg");//imread 函数读取图片路径时，支持双右斜杠\\、双左斜杠//
//	                                                        //单左斜杠/三种方式均可以;三种方法混合亦可以进行读取
//	                                                        //或者将图片文件与.cpp文件放在一个文件夹中时，只需要相对路径即可
//	if (!srcImage.data)
//	{
//		cout << "图片加载失败" << endl;
//		return false;
//	}
//	else
//	{
//		cout << "图片加载成功" << endl;
//	}
//	namedWindow("原图像",WINDOW_AUTOSIZE);
//	//imshow("原图像", srcImage);
//
//
//
//
//	                                           //将图像转换为灰度图，使用CV前缀；
//	Mat grayImage;
//	cvtColor(srcImage, grayImage, CV_BGR2GRAY);
//	namedWindow("灰度图", WINDOW_AUTOSIZE);
//	imshow("灰度图", grayImage);
//
//
//
//
//	                                           //将图片转换为HSV空间图像，采用COLOR前缀;
//	Mat hsvImage;
//	cvtColor(srcImage, hsvImage, COLOR_BGR2HSV);
//	namedWindow("HSV空间图像", WINDOW_AUTOSIZE);
//	imshow("HSV空间图像", hsvImage);
//
//
//
//
//	                                          //对转化的hsv图像进行分割去噪和滤波、平滑处理;
//
//
//
//
//
//
//
//
//
//	                                               ////将图片从hsv颜色空间转换为二值图像，使用threshold函数（输入图像，输出图像，145,255，THRESH_BINARY);
//	                                                //Mat binImage;
//	//threshold(grayImage, binImage, 145, 255, THRESH_BINARY);
//	//namedWindow("二值图像", WINDOW_AUTOSIZE);
//	//imshow("二值图像", binImage);
//
//
//	//使用inrange函数对hsv颜色空间的图像进行二值化，要确定上边界和下边界
//	Mat binImage;
//	inRange(hsvImage,Scalar(0,30,30),Scalar(40, 170,256),binImage);
//	imshow("利用inrange函数输出二值函数", binImage);
//	//inRange(hsvImage, Scalar(0, 30, 30), Scalar(40, 170, 256), dstImage);
//
//
//
//	//绘出图像中手势的轮廓；
//	IplImage src = binImage;
//	IplImage* newImage = cvCloneImage(&src);
//	CvMemStorage* storage = cvCreateMemStorage(0);
//	CvSeq* first_contour = nullptr;
//	cvFindContours(newImage,storage, &first_contour,sizeof(CvChain),CV_CHAIN_CODE);
//
//	void cvStartReadChainPoints(CvChain* chain, CvChainPtReader* reader);	
//
//	CvPoint cvReadChainPoint(CvChainPtReader* reader);
//
//
//
//
//
//
//	
//	system("pause");
//	return 0;
//
//
//}