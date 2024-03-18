#include <stdio.h>
#include <stdlib.h>
int main() {
	const int nElem = 10;
	float *d_a, *d_b, *h_a, *h_b, *h_c;
	int nBytes = nElem * sizeof(float);	

	h_a = (float*)malloc(nBytes);

//	initialData(h_a, nElem);
//	initialData(h_b, nElem);
	for (int i = 0; i < nElem; ++i) {
		h_a[i] = (float)i;
	}
	for (int i = 0; i < nElem; ++i) {
		printf("%f ", h_a[i]);
	}
}