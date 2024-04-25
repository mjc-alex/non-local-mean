#include <iostream>
#include <opencv2/opencv.hpp>
#include "main.h"
#include <getopt.h>
using namespace cv;

void processImage(const std::string& imagePath, int sr_size, int nb_size) {
    Mat img, img_input;

    // Read image
    img = imread(imagePath, 0);
    int img_W = img.rows;
    int img_H = img.cols;

    // Padding
    int top = (sr_size + nb_size - 2) / 2;
    int bottom = (sr_size + nb_size - 2) / 2;
    int left = (sr_size + nb_size - 2) / 2;
    int right = (sr_size + nb_size - 2) / 2;
    copyMakeBorder(img, img_input, top, bottom, left, right, BORDER_REFLECT_101);

    // Padded image size
    int W = img_input.rows;
    int H = img_input.cols;

    // Allocate memory
    unsigned char* GPU_input = new unsigned char[W * H];
    float* GPU_result = new float[img_W * img_H];

    // Write padded input image
    imwrite("../data/input.bmp", img_input);

    // Copy data to GPU input
    memcpy(GPU_input, img_input.data, W * H * sizeof(unsigned char));

    // Perform filtering on GPU
    NLMeansProcessor::NL_Means(GPU_input, GPU_result, W, H, sr_size, nb_size, img_H, img_W);

    // Create output image
    Mat temp(img_W, img_H, CV_32FC1, GPU_result);
    Mat output = temp.clone();

    // Write output image
    imwrite("../data/output.bmp", output);

    // Free memory
    delete[] GPU_input;
    delete[] GPU_result;
}

void print_usage()
{
	std::cout <<
"usage: nlm -i INPUT -x xSize -y ySize\n"
"\n"
"    -i  input file in BMP format\n"
"\n"
"   Blocking parameters to tune (specify with comma-separated list):\n"
"    -x  block size in X dimension\n"
"    -y  block size in Y dimension\n";
	exit(EXIT_SUCCESS);
}

void split_int(char* str, const char* delim, std::vector<int>& result)
{
	char* token = strtok(str, delim);
	while(token != NULL)
	{
		result.push_back(atoi(token));
		token = strtok(NULL, delim);
	}
}

void check_arg(char c, bool test)
{
	if (!test) {
		std::cerr << "error: missing argument '-" << c << "'" << std::endl;
		exit(EXIT_FAILURE);
	}
}
int main(int argc, char* argv[]) {
	const char* input = NULL;
	std::vector<int> xs;
	std::vector<int> ys;

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
    check_arg('i', input != NULL);
	//check_arg('x', xs.size() > 0);
	//check_arg('y', ys.size() > 0);
    processImage(input, 21,5);
//	processImage("../data/sample_640*426.bmp",21,5);
//  	processImage("/home/alex/cuda/noise.bmp",21,5);
}


