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
		for (int j = 0; j < length - i - 1; j++)	//�ڲ�ѭ����Ԫ�ظ���-��ǰ����-1����
		{
			if (arr[j] > arr[j+1])
			{
				t = arr[j];
				arr[j] = arr[j+1];
				arr[j+1] = t;
			}
		}
	}
	cout << "ð�ݺ�������ǣ�" << endl;
	for (int i = 0; i < length; i++)
	{
		cout << arr[i] << " ";
	} cout << endl;

	system("pause");

	return 0;
}