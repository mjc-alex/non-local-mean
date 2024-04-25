# README
目前实现了两个版本的算法，一个是传统的串行算法(`nlm-cpu.c`)。另一个是cuda-c，(build目录下)
- `nlm-cpu.c`: 读取bmp图片，串行nlm滤波后输出，这个代码没有问题。
-  `build/main.cu`: 有两个版本的算法，分别是使用全局内存和共享内存的算法，在`main.cu`里用`#define _GLOBAL_`或者`#define _SHARED_`控制使用哪个版本。
- `build/nlm.cpp`: 主函数，读入图片，读入预设参数。目前可以使用`-i [image path]`来决定输入，例如：`./nl-means -i ../data/1024.bmp`将../data/1024.bmp文件作为输入。