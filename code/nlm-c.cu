#include <getopt.h>
#include "image.h"

#define W 3
#define P 2
#define R (2*(W+P))
#define H 100.0

#define MAX_SHMEM (48*1024)

__global__ void nlm(
	const float *src, float *dst,
	unsigned pitch, unsigned X, unsigned Y, unsigned Z)
{
	extern __shared__ float block[];

	const unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= X || y >= Y) return;

	const unsigned xy = x + y*pitch;
	const unsigned slab = pitch * (Y + R);
	const unsigned pad = R*(1 + pitch + slab);

	/* the difference of this threads position relative to the block
	 * boundary as a flat index */
	const unsigned dt = threadIdx.x + threadIdx.y*pitch;

	/* "b" variables are in coordinates local to the cache block */
	const unsigned BX = blockDim.x;
	const unsigned BY = blockDim.y;
	const unsigned bpitch = BX + R;
	const unsigned bslab = bpitch * (BY + R);
	const unsigned bpad = 1 + bpitch + bslab;

	for (unsigned z = 0; z < Z; z++)
	{
		unsigned start = (xy - dt) + z*slab;

		/* prefetch block with BZ depth */
		__syncthreads();

		for (unsigned bz = 0; bz < (1+R); bz++)
		{
			/* the first R threads will wrap around and execute these
			 * loops twice */
			for (unsigned by = threadIdx.y; by < (BY+R); by += BY)
			{
				for (unsigned bx = threadIdx.x; bx < (BX+R); bx += BX)
				{
					/* copy from global memory into the shared memory block */
					block[bx + bpitch*by + bslab*bz] =
						src[start + bx + pitch*by + slab*bz];
				}
			}
		}

		__syncthreads();

		/* perform calculations across BZ depth */

		const unsigned bstart = threadIdx.x + threadIdx.y*bpitch;
		const unsigned wstart = bstart + P * bpad;
		const unsigned pstart = bstart + W * bpad;

		float sum = 0.0f;
		float weights = 0.0f;

		for (unsigned wz = 0; wz <= 2*W; wz++){
		for (unsigned wy = 0; wy <= 2*W; wy++){
		for (unsigned wx = 0; wx <= 2*W; wx++)
		{
			float weight = 0.0f;

			for (unsigned pz = 0; pz <= 2*P; pz++){
			for (unsigned py = 0; py <= 2*P; py++){
			for (unsigned px = 0; px <= 2*P; px++)
			{
				float d =
					block[pstart + px + py*bpitch + pz*bslab] -
					block[bstart + wx+px + (wy+py)*bpitch + (wz+pz)*bslab];
				weight += d*d;
			}
			}
			}

			weight = exp( -(weight*weight*H) );

			weights += weight;
			sum += weight * block[wstart + wx + wy*bpitch + wz*bslab];
		}
		}
		}

		dst[xy + z*slab + pad] = sum / weights;
	}
}

void print_usage()
{
	printf(
"usage: nlm -i INPUT -x XSize -y ySize\n"
"\n"
"    -i  input file in bmp format\n"
"\n"
"   Blocking parameters to tune (specify with comma-separated list):\n"
"    -x  block size in X dimension\n"
"    -y  block size in Y dimension\n");
}
void split_int(char* str, const char* delim, int *result)
{
	char* token = strtok(str, delim);
    int i = 0;
	while(token != NULL)
	{
		result[i++] = (atoi(token));
		token = strtok(NULL, delim);
	}
}
void check_arg(char c, bool test)
{
	if (!test) {
	    printf("error: missing argument '-%c'\n", c);
		exit(-1);
	}
}
int main(int argc, char **argv)
{
	const char* input = NULL;
	int xs[100];
	int ys[100];

	int c;
	while ((c = getopt(argc, argv, "i:x:y:")) != -1)
		switch (c) {
			case 'i':
				input = optarg;
				break;
			case 'x':
				split_int(optarg, ",", xs);
				break;
			case 'y':
				split_int(optarg, ",", ys);
				break;
			default:
				print_usage();
		}
	int xsNum = sizeof(xs) / sizeof(int);
	int ysNum = sizeof(ys) / sizeof(int);
	check_arg('i', input != NULL);
	check_arg('x', xsNum > 0);
	check_arg('y', ysNum > 0);

	FILE *fi = fopen(input, "r");
	FILE *fo = fopen("./output.bmp", "wb");

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
 	uchar *buf = (uchar*)malloc(height*width*sizeof(uchar)); // to store the image data
	fread(buf, sizeof(uchar), (height * width), fi);
	if(bitDepth <= 8)
		fwrite(colorTable, sizeof(uchar), 1024, fo);
	cudaSetDevice(0);

	struct Image I, J;
	InitImage(&I, height, width, buf);

	const unsigned X = I.dims[0];
	const unsigned Y = I.dims[1];
	const unsigned Z = I.dims[2];

	double start, alloc_time, xfer1_time, filter_time, xfer2_time;
	bool status;

	for (unsigned j=0; j<ysNum; j++)
	{
		unsigned BY = ys[j];
		if (BY > Y) continue;

		for (unsigned i=0; i<xsNum; i++)
		{
			unsigned BX = xs[i];
			unsigned shmem = (BX+R)*(BY+R)*(1+R)*sizeof(float);

			if (BX > X || BX*BY < 32 || shmem > MAX_SHMEM) continue;

			dim3 block(BX, BY, 1);
			dim3 grid((X-1)/BX+1, (Y-1)/BY+1, 1);

			// Allocate GPU memory and transfer image
			start = timestamp();
			alloc_gpu(W+P, &I);
			alloc_gpu(W+P, &J);
			alloc_time = timestamp() - start;

			start = timestamp();
			to_gpu(W+P, &I);
			xfer1_time = timestamp() - start;

			// Launch filter kernel
			start = timestamp();
			nlm<<<grid, block, shmem>>>(I.ptr, J.ptr, I.pitch, X, Y, Z);
			CUCALL(cudaThreadSynchronize());
			CUCALL_RETURN(cudaGetLastError(), status);
			filter_time = timestamp() - start;

			// Transfer image back and compute similarity with reference
			start = timestamp();
			from_gpu(W+P, &J);
			xfer2_time = timestamp() - start;
			printf("%u, %u, %d, %lf, %lf, %lf, %lf\n",\
			BX, BY, status, alloc_time, xfer1_time, filter_time, xfer2_time);
		}
	}
    //printf("x = %d, y = %d\n", xs[0], ys[0]);
	fwrite(J.data, sizeof(uchar), (height * width), fo);
 	
 	fclose(fi);
	fclose(fo);
	destroyImage(&I);
	free(buf);
	return 0;
}