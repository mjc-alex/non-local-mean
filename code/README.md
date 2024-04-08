# README
目前实现了两个版本的算法，一个是传统的串行算法(`nlm-cpu.c`)。另一个在写的是cuda-c，(`nlm-c.cu`, `image.h`, `image.c`)
- `nlm-cpu.c`: 读取bmp图片，串行nlm滤波后输出，这个代码没有问题。
- `nlm-c.cu`: 在命令行中用-x, -y指定参数，分别是x, y的块大小。`usage: nlm -i INPUT -x XSize -y ySize`
- `image.h`: 定义image的结构体
- `image.c`: 实现基本的函数，包括图片存取、时间统计等