#include<iostream>
using namespace std;

//把数组中的元素倒着排列

int main36() {

	int t = 0;
	int arr[] = { 1,3,2,5,4 };
	int start = 0;		//起始元素的下标
	int end = sizeof(arr) / sizeof(arr[0]) - 1;		//末尾元素的下标

	cout << "原始数组为："  << endl;
	for (int k = 0; k < 5; k++)
	{
		cout << arr[k] << endl;
	}

	for (int i = 0; i < end / 2; i++)
	{
		t = arr[i];
		arr[i] = arr[end - i];
		arr[end - i] = t;

	}
	cout << "逆置后的数组为：" << endl;
	for (int j = 0; j < 5; j++)
	{
		cout << arr[j] << endl;
	}

	system("pause");

	return 0;
}