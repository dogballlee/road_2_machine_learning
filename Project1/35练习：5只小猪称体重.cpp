#include<iostream>
using namespace std;

//一个数组中记录了五只小猪的体重，请找出并打印出最终的小猪体重

int main35() {

	int max = 0;
	int arr[] = { 300,350,200,400,250 };
	
	for (int i = 0; i < sizeof(arr); i++)
	{
		if (arr[i] > max)
		{
			max = arr[i];
		}
		
	}
	cout << "最大体重是：" << max << endl;

	system("pause");

	return 0;
}