/*
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main() {
VideoCapture cap("C:\\Users\\不懂别瞎说\\source\\repos\\OpencvText\\OpencvText\\output1.avi");
if (!cap.isOpened())
return -1;
//计算帧的总数；
int frnb = cap.get(CV_CAP_PROP_FRAME_COUNT);
cout << "帧总数="<<frnb << endl;
Mat frame;
//起始帧数
unsigned int fIdx = 10;
//设置帧位置
cap.set(CV_CAP_PROP_POS_FRAMES, fIdx);
//读取帧
cap >> frame;
//显示帧
namedWindow("frame", CV_WINDOW_AUTOSIZE);
imshow("frame", frame);
waitKey(3);
cout << "请输入 0 到 " << frnb - 1 << "的数字，输入0 就结束\n" << endl;
for (;;) {
cout << "第几个帧？";
cin >> fIdx;
if (fIdx > 0 && fIdx < frnb) {
cap.set(CV_CAP_PROP_POS_FRAMES, fIdx);
cap >> frame;
imshow("frame", frame);
waitKey(3);
}
else
return 0;
}
return 0;
}
*/