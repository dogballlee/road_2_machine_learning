#include "swap.h"	//表示该源文件与swap.h头文件存在关联



//函数的定义
void swap(int num1, int num2)
{
	int t = num1;
	num1 = num2;
	num2 = t;

	cout << "num1 = " << num1 << endl;
	cout << "num2 = " << num2 << endl;
}