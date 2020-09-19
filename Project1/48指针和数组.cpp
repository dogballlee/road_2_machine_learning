#include <iostream>
using namespace std;

//目的：利用指针访问数组中的元素


int main48() {

	int arr[10] = { 1,2,3,4,5,6,7,8,9 };

	cout << "数组arr的第一个元素为：" << arr[0] << endl;	//常规操作

	int* p = arr;	//arr就是数组的首地址

	cout << "数组arr的第一个元素为(利用指针解引用获取)：" << *p << endl;

	int* p1 = &arr[1];	//或者使用p++使指针向后偏移1个位置(4个字节)

	cout << "数组arr的第二个元素为(利用指针解引用获取)：" << *p1 << endl;

	system("pause");

	return 0;
}