#ifndef _MAIN_H_
#define _MAIN_H_

class NLMeansProcessor{
public:
	static void NL_Means(unsigned char *GPU_input, float *GPU_result, int W, int H, int sr_size, int nb_size, int IMAGE_SIZE_X , int IMAGE_SIZE_Y);
};

#endif
