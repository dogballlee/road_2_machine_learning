#include<iostream>
#include "swap.h"
using namespace std;


//函数的分文件编写
//实现两个数字进行交换的函数

//函数的声明
//void swap(int num1, int num2);
//函数的定义
//void swap(int num1, int num2)
//{
//	int t = num1;
//	num1 = num2;
//	num2 = t;
//}


//1、创建.h为后缀名的头文件
//2、创建.cpp为后缀名的源文件
//3、在头文件中写函数的声明(记得写所需的框架)
//4、在源文件中写函数的定义(记得写包含头文件)

int main() {
	int a = 10;
	int b = 20;

	swap(a, b);

	system("pause");

	return 0;
}