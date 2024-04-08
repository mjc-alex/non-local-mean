#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#define NLM_FILTER
typedef unsigned char uchar;

const double h2 = 100;
const int search_size = 15, block_size = 7;//31 7
const int search_radius = search_size >> 1;
const int block_radius = block_size >> 1;
const int bb = search_radius + block_radius;
const int sNum = search_size * search_size;
const int bNum = block_size * block_size;

uchar* paddingImage(uchar *buf, uchar* pad, int row, int col) {
	int newCol = col + 2 * bb;
	int newRow = row + 2 * bb;
	int  newNum = newCol * newRow;
	memset(pad, 0, sizeof(pad));
	int newPos = bb * newCol + bb;
	for (int r = 0; r < row; r++) {
		for (int c = 0; c < col; c++) {
			pad[newPos+r*newCol+c] = buf[r*col+c]; 
		}
	}
	return pad;
}
void getBuf(uchar* pad, uchar* buf, int row, int col) {
	int newCol = col + 2 * bb;
	int newRow = row + 2 * bb;
	int  newNum = newCol * newRow;
	int newPos = bb * newCol + bb;
	for (int r = 0; r < row; r++) {
		for (int c = 0; c < col; c++) {
			buf[r*col+c]= pad[newPos+r*newCol+c]; 
		}
	}
}
#ifdef NLM_FILTER
void nlmFilter(uchar *buf, int row, int col) {
	int newCol = col + 2 * bb;
	int newRow = row + 2 * bb;
	int newNum = newCol * newRow;
	uchar *pad = (uchar*)malloc(newNum * sizeof(uchar));
	double *mse = (double*)malloc(sNum*sizeof(double));					
	memset(mse, 0, sizeof(mse));
	pad = paddingImage(buf, pad, row, col); 

	//double *w = (double*)malloc(bNum*sizeof(double));					
	//memset(w, 0, sizeof(w));
	// filter every pixel
	for (int i = bb; i < bb + row; i++) {
		for (int j = bb; j < bb + col; j++) {
			int pos = i * newCol + j; 
			double sum = 0;
			double res = 0;
			int cnt = 0;
			int np = (i - block_radius) * newCol + j - block_radius;
			// every search window
			for (int r = i - search_radius; r <= i + search_radius; r++) {
				for (int c = j - search_radius; c <= j + search_radius; c++) {
					// every zone
					for (int m = r - block_radius; m <= r + block_radius; m++) {
						for (int n = c - block_radius; n <= c + block_radius; n++) {
							int bp = m * newCol + n;
							int pp = (m - r + i) * newCol + n - c + j;
							mse[cnt] += (double)(pad[bp] - pad[pp]) * (pad[bp] - pad[pp]);
						}
					}

					mse[cnt] = mse[cnt] / (double)bNum;
					mse[cnt] = exp(-mse[cnt] / h2);
					sum += mse[cnt];
//					if (pos % 10000 == 0) {
//						printf("at pos%d, mse[%d] = %f\n",pos, cnt, mse[cnt]);
//					}
					cnt++;
				}
			}
			int cth = 0;
			for (int r = i - search_radius; r <= i + search_radius; r++) {
				for (int c = j - search_radius; c <= j + search_radius; c++) {
					int sp = r * newCol + c;
					res += pad[sp] * (mse[cth++] / sum);
				}
			}
			pad[pos] = res;
			// if (pos % 1000 == 0)
			// printf("%f ",mse[pos]);
		}
	}
	getBuf(pad, buf, row, col);
	free(pad);
	free(mse);
	//free(w);
}	
#endif
int main(int argc, char *argv[])
{
	clock_t start, stop;
	start = clock();
		
	FILE *fi = fopen("../noise.bmp", "r");
	FILE *fo = fopen("../output.bmp", "wb");

	if (fi == (FILE*) 0) {
		printf("File opening error ocurred.\n");
		exit(0);
	}
	uchar header[54];
	uchar colorTable[1024];

	int i = 0;
	for (i = 0; i < 54; i++) {
		header[i] = getc(fi);
	}
	int width = *(int*)&header[18];
	int height = *(int*)&header[22];
	int bitDepth = *(int*)&header[28];

	if(bitDepth <= 8)
		fread(colorTable, sizeof(uchar), 1024, fi);

	printf("width: %d\n",width);
	printf("height: %d\n",height );

	fwrite(header, sizeof(uchar), 54, fo); // write the image header to output file
 	uchar buf[height * width]; // to store the image data
	fread(buf, sizeof(uchar), (height * width), fi);
	if(bitDepth <= 8)
		fwrite(colorTable, sizeof(uchar), 1024, fo);	

#ifndef NLM_FILTER
	int col = width;
	int row = height;
	int newCol = col + 2 * bb;
	int newRow = row + 2 * bb;
	int newNum = newCol * newRow;
	uchar *pad = (uchar*)malloc(newNum * sizeof(uchar));
	pad = paddingImage(buf, pad, row, col); 
	getBuf(pad, buf, row, col);
	free(pad);
#else
	nlmFilter(buf, height, width);
#endif
	fwrite(buf, sizeof(uchar), (height * width), fo);
 	
 	fclose(fi);
	fclose(fo);

	stop = clock(); 
	printf("Time: %lf ms\n",((double)(stop - start) * 1000.0 )/ CLOCKS_PER_SEC);

	return 0;
}
