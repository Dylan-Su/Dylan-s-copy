//#include "FFTtrans.h"
//
//complex<double> X[OUT_BUFFER_SIZE];
//double X_A[OUT_BUFFER_SIZE];
//double X_F[OUT_BUFFER_SIZE];
//
//complex<double> E_k(int k, int N, double x[]) {
//	complex<double> e_temp(0, 0);
//	for (int m  = 0; m  <= N  / 2 - 1; m ++) {
//		e_temp  += x[2 * m] * complex<double>(cos(-2 * PI * m * k  / (N  / 2)), sin(-2 * PI * m * k  / (N  / 2)));
//	}
//	return e_temp;
//}
//
//complex<double> O_k(int k, int N, double x[]) {
//	complex<double> o_temp(0, 0);
//	for (int m  = 0; m  <= N  / 2 - 1; m ++) {
//		o_temp  += x[2 * m  + 1] * complex<double>(cos(-2 * PI * m * k) / (N  / 2), sin(-2 * PI * m * k  / (N  / 2)));
//	}
//	return o_temp;
//}
//
//complex<double> X_k(int k, int N, double x[]) {
//	if (k < N  / 2) {
//		return E_k(k, N, x) +
//			complex<double>(cos(-2 * PI * k  / N), sin(-2 * PI * k  / N)) * O_k(k, N, x);
//	}
//	else {
//		return E_k(k  - N  / 2, N, x) -
//			complex<double>(cos(-2 * PI *(k  - N  / 2) / N), sin(-2 * PI * (k  - N  / 2) / N)) * O_k(k  - N  / 2, N, x);
//	}
//}
//
//void FFT_Caculate(int N, double x[]) {
//	for (int i  = 0; i < N; i ++) {
//		X[i] = X_k(i, N, x);
//	}
//	for (int i  = 0; i < N; i ++) {
//		X_A[i] = sqrt(X[i].real() * X[i].real() + X[i].imag() * X[i].imag());
//		X_F[i] = i;
//	}
//};