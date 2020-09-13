#include<iostream>
using namespace std;

/*一维数组数组名的用途：
	1、可用于统计数组在内存中的空间		例如：sizeof(arr)
	2、可以获取数组在内存中的首地址*/

int main34() {

	//1、计算数组占用内存大小
	int arr[] = { 1,2,3,4,5,6,7,8 };
	cout << "数组arr在内存中占用的空间是:" << sizeof(arr) << endl;

	cout << "数组arr中一个元素在内存中占用的空间是:" << sizeof(arr[0]) << endl;

	cout << "数组arr中元素个数是:" << sizeof(arr)/ sizeof(arr[0]) << endl;


	//2、查看数组首地址
	cout << "数组首地址为：" << (int)arr << endl;	//这里的(int)表示把16进制的地址强转换成10进制显示
	cout << "数组中第一个元素地址为：" << (int)&arr[0] << endl;

	system("pause");

	return 0;
}