#include <iostream>
using namespace std;

//��װһ������������ð������ʵ�ֶ������������������

int arr[] = { 3,5,1,7,6,8,2,9,4,0 };

int len = sizeof(arr) / sizeof(arr[0]);


//����ָ���arr��������
void bubblesort(int * arr, int len)		//����1��������׵�ַ  //����2�����鳤��
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

//��ӡ����
void printarray(int* arr, int len)
{
	for (int i = 0; i < len; i++) {
		cout << "sort���arr�ǣ�" << arr[i] << endl;
	}
}

int main50() 
{

	bubblesort(arr, 10);

	printarray(arr, 10);

	system("pause");

	return 0;
}