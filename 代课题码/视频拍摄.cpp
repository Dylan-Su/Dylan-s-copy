//
//#include <iostream>
//#include <opencv2/opencv.hpp>
//using namespace cv;
//using namespace std;
//
//int main()
//{
//	VideoCapture cap(0);
//	if (!cap.isOpened())
//	{
//		cout << "�޷���������ͷ" << endl;
//		return -2;
//	}
//	//ȡ��ͼ��֡�Ĵ�С
//	Size S = Size((int)cap.get(CV_CAP_PROP_FRAME_HEIGHT),
//		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));
//	//��������ʼ����Ƶ�洢����
//	VideoWriter put("F:\\opencv\\1.avi", CV_FOURCC('M', 'P', 'E', 'G'), 50, S);
//	namedWindow("Video");
//	//��ʼ����
//	while (char(waitKey(1)) != 'q' && cap.isOpened())
//	{
//		Mat frame;
//		cap >> frame;
//		//�������ͷ�Ƿ��������
//		if (frame.empty())
//		{
//			break;
//		}
//		imshow("Video", frame);
//		put << frame;
//	}
//	return 0;
//}
