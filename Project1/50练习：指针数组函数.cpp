#include <iostream>
using namespace std;

//封装一个函数，利用冒泡排序，实现对整型数组的升序排列

int arr[] = { 3,5,1,7,6,8,2,9,4,0 };

int len = sizeof(arr) / sizeof(arr[0]);


//利用指针对arr进行排序
void bubblesort(int * arr, int len)		//参数1：数组的首地址  //参数2：数组长度
{
	for (int i = 0; i < 10; i++) 
	{
		for (int j = 0; j < 10 - i - 1; j++) 
		{
			if (arr[j + 1] < arr[j]) 
			{
				int t = arr[j];
				arr[j] = arr[j + 1];
				arr[j + 1] = t;
			}
		}
	
	}
}

//打印数组
void printarray(int* arr, int len)
{
	for (int i = 0; i < len; i++) {
		cout << "sort后的arr是：" << arr[i] << endl;
	}
}

int main50() 
{

	bubblesort(arr, 10);

	printarray(arr, 10);

	system("pause");

	return 0;
}