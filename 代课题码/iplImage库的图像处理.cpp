//#include <iostream>
//#include <opencv2/opencv.hpp>
//
//using namespace std;
//using namespace cv;
//
//int main()
//{
//	Mat srcImage = imread("C:/Users/DylanSu/Desktop/2.jpg");//imread ������ȡͼƬ·��ʱ��֧��˫��б��\\��˫��б��//
//	                                                        //����б��/���ַ�ʽ������;���ַ����������Խ��ж�ȡ
//	                                                        //���߽�ͼƬ�ļ���.cpp�ļ�����һ���ļ�����ʱ��ֻ��Ҫ���·������
//	if (!srcImage.data)
//	{
//		cout << "ͼƬ����ʧ��" << endl;
//		return false;
//	}
//	else
//	{
//		cout << "ͼƬ���سɹ�" << endl;
//	}
//	namedWindow("ԭͼ��",WINDOW_AUTOSIZE);
//	//imshow("ԭͼ��", srcImage);
//
//
//
//
//	                                           //��ͼ��ת��Ϊ�Ҷ�ͼ��ʹ��CVǰ׺��
//	Mat grayImage;
//	cvtColor(srcImage, grayImage, CV_BGR2GRAY);
//	namedWindow("�Ҷ�ͼ", WINDOW_AUTOSIZE);
//	imshow("�Ҷ�ͼ", grayImage);
//
//
//
//
//	                                           //��ͼƬת��ΪHSV�ռ�ͼ�񣬲���COLORǰ׺;
//	Mat hsvImage;
//	cvtColor(srcImage, hsvImage, COLOR_BGR2HSV);
//	namedWindow("HSV�ռ�ͼ��", WINDOW_AUTOSIZE);
//	imshow("HSV�ռ�ͼ��", hsvImage);
//
//
//
//
//	                                          //��ת����hsvͼ����зָ�ȥ����˲���ƽ������;
//
//
//
//
//
//
//
//
//
//	                                               ////��ͼƬ��hsv��ɫ�ռ�ת��Ϊ��ֵͼ��ʹ��threshold����������ͼ�����ͼ��145,255��THRESH_BINARY);
//	                                                //Mat binImage;
//	//threshold(grayImage, binImage, 145, 255, THRESH_BINARY);
//	//namedWindow("��ֵͼ��", WINDOW_AUTOSIZE);
//	//imshow("��ֵͼ��", binImage);
//
//
//	//ʹ��inrange������hsv��ɫ�ռ��ͼ����ж�ֵ����Ҫȷ���ϱ߽���±߽�
//	Mat binImage;
//	inRange(hsvImage,Scalar(0,30,30),Scalar(40, 170,256),binImage);
//	imshow("����inrange���������ֵ����", binImage);
//	//inRange(hsvImage, Scalar(0, 30, 30), Scalar(40, 170, 256), dstImage);
//
//
//
//	//���ͼ�������Ƶ�������
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