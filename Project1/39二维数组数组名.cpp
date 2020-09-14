#include<iostream>
using namespace std;

/*二维数组数组名的用途：
	1、可用于统计二维数组在内存中的空间		例如：sizeof(arr)
	2、可以获取二维数组在内存中的首地址*/

int main39() {

	//1、计算二维数组占用内存大小
	
	int arr[2][3] =
	{ 1,2,3,4,5,6 };
	cout << "二维数组arr所占内存空间：" << sizeof(arr) << endl;
	cout << "二维数组arr第一行所占内存空间：" << sizeof(arr[0]) << endl;
	cout << "二维数组arr第一个元素所占内存空间：" << sizeof(arr[0][0]) << endl;

	//2、查看二维数组首地址

	cout << "二维数组arr的首地址" << (int)arr << endl;
	cout << "二维数组arr第一行的首地址" << (int)arr[0] << endl;
	cout << "二维数组arr第一个元素的首地址" << (int)&arr[0][0] << endl;

	system("pause");

	return 0;
}