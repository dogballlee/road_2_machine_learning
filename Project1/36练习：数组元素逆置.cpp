#include<iostream>
using namespace std;

//�������е�Ԫ�ص�������

int main36() {

	int t = 0;
	int arr[] = { 1,3,2,5,4 };
	int start = 0;		//��ʼԪ�ص��±�
	int end = sizeof(arr) / sizeof(arr[0]) - 1;		//ĩβԪ�ص��±�

	cout << "ԭʼ����Ϊ��"  << endl;
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
	cout << "���ú������Ϊ��" << endl;
	for (int j = 0; j < 5; j++)
	{
		cout << arr[j] << endl;
	}

	system("pause");

	return 0;
}