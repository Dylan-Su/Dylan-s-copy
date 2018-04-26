//#include <iostream>
//#include <opencv2/opencv.hpp>
//using namespace cv;
//using namespace std;
//
//struct FM
//{
//	int x;
//	int y;
//	int direction;
//};
//
//
//
//void Freemans(Mat srcimage, vector<vector<FM>>&FMS)//定义一个二维动态数组FMS
//{
//	CvMat _srcimage = srcimage;
//	CvMemStorage* storage = cvCreateMemStorage();//采用默认大小，即：0.
//	CvSeq* first_contour = NULL;
//	//cvFindContours函数返回轮廓的个数：
//	int Nc = cvFindContours(&_srcimage, storage, &first_contour, sizeof(CvContour),
//		CV_RETR_CCOMP,
//		CV_CHAIN_CODE,///*这个是关键参数*/
//		cvPoint(0, 0)
//	);
//
//	//定义一个空链表；
//	CvChain* chain = 0;
//	vector<CvSeq*>c1;//定义一下向量序列
//	CvSeq* h;//定义一个序列 h；
//	for (CvSeq* c = first_contour; c != NULL; c = c->h_next)
//	{
//		vector<FM>fms;
//		int total = c->total;
//
//		if (total<20)
//			c = c->h_next;
//		else
//		{
//			//定义一个序列读取器：
//			CvSeqReader reader;
//			//开始顺序读取序列c中的元素到reader读取器中
//			cvStartReadSeq((CvSeq*)c, &reader, 0);//读取序列
//			CvChainPtReader reader1;//定义一个坐标点的读取器
//			cvStartReadChainPoints((CvChain*)c, &reader1);//读取链表
//			FM fm;
//			for (int i = 0; i < total; i++)
//			{
//				char code;//定义一个字符串来接收Freeman链码
//				CV_READ_SEQ_ELEM(code, reader); //printf(" %d,",code); //得到轮廓的Freeman链码序列//按顺序把reader中的点读入code中；
//				fm.direction = code;
//				CvPoint pt;
//				CV_READ_CHAIN_POINT(pt, reader1);
//				fm.x = pt.x;
//				fm.y = pt.y;
//				fms.push_back(fm);
//			}
//			FMS.push_back(fms);
//		}
//	}//for
//}
//
//
//int main()
//{
//
//}