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
//void Freemans(Mat srcimage, vector<vector<FM>>&FMS)//����һ����ά��̬����FMS
//{
//	CvMat _srcimage = srcimage;
//	CvMemStorage* storage = cvCreateMemStorage();//����Ĭ�ϴ�С������0.
//	CvSeq* first_contour = NULL;
//	//cvFindContours�������������ĸ�����
//	int Nc = cvFindContours(&_srcimage, storage, &first_contour, sizeof(CvContour),
//		CV_RETR_CCOMP,
//		CV_CHAIN_CODE,///*����ǹؼ�����*/
//		cvPoint(0, 0)
//	);
//
//	//����һ��������
//	CvChain* chain = 0;
//	vector<CvSeq*>c1;//����һ����������
//	CvSeq* h;//����һ������ h��
//	for (CvSeq* c = first_contour; c != NULL; c = c->h_next)
//	{
//		vector<FM>fms;
//		int total = c->total;
//
//		if (total<20)
//			c = c->h_next;
//		else
//		{
//			//����һ�����ж�ȡ����
//			CvSeqReader reader;
//			//��ʼ˳���ȡ����c�е�Ԫ�ص�reader��ȡ����
//			cvStartReadSeq((CvSeq*)c, &reader, 0);//��ȡ����
//			CvChainPtReader reader1;//����һ�������Ķ�ȡ��
//			cvStartReadChainPoints((CvChain*)c, &reader1);//��ȡ����
//			FM fm;
//			for (int i = 0; i < total; i++)
//			{
//				char code;//����һ���ַ���������Freeman����
//				CV_READ_SEQ_ELEM(code, reader); //printf(" %d,",code); //�õ�������Freeman��������//��˳���reader�еĵ����code�У�
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