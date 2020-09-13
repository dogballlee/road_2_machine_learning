#include<iostream>
using namespace std;



int main() {

	int t = 0;
	int arr[] = { 4,2,8,0,5,7,1,3,9 };
	int length = sizeof(arr) / sizeof(arr[0]);
	for (int i = 0; i < length; i++)
	{
		cout << arr[i] << " ";
	} cout << endl;
	for (int i = 0; i < length - 1; i++)
	{
		for (int j = 0; j < length - i - 1; j++)	//内层循环“元素个数-当前轮数-1”次
		{
			if (arr[j] > arr[j+1])
			{
				t = arr[j];
				arr[j] = arr[j+1];
				arr[j+1] = t;
			}
		}
	}
	cout << "冒泡后的数组是：" << endl;
	for (int i = 0; i < length; i++)
	{
		cout << arr[i] << " ";
	} cout << endl;

	system("pause");

	return 0;
}