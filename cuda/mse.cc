#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#define CV_LOAD_IMAGE_COLOR 1

double compute_MSE(cv::Mat Mat1, cv::Mat Mat2)
{

    cv::Mat M1 = Mat1.clone();
    cv::Mat M2 = Mat2.clone();
    cv::Mat Diff;
    // 提前转换为32F精度
    M1.convertTo(M1,CV_32F);
    M2.convertTo(M2,CV_32F);
    Diff.convertTo(Diff,CV_32F);
    
    cv::absdiff(M1,M2,Diff); //  Diff = | M1 - M2 |

    Diff=Diff.mul(Diff);     // | M1 - M2 |.^2
    cv::Scalar S = cv::sum(Diff);  //分别计算每个通道的元素之和

    double sse;   // square error
    if (Diff.channels()==3)
        sse = S.val[0] +S.val[1] + S.val[2];  // sum of all channels
    else
        sse = S.val[0];

    int nTotalElement = M2.channels()*M2.total();

    double mse = ( sse / (double)nTotalElement );  //

    return mse;
}

int main(int argc, char *argv[]) {
  std::string input_img_path1 = argv[1];
  std::string input_img_path2 = argv[2];
  cv::Mat img1, img2;
  img1 = cv::imread(input_img_path1, CV_LOAD_IMAGE_COLOR);
  img2 = cv::imread(input_img_path2, CV_LOAD_IMAGE_COLOR);
  // 注意两张图片大小需要一致
  double mse = compute_MSE(img1, img2);
  std::cout << "MSE: "<< mse << std::endl;
  return 0;
}
