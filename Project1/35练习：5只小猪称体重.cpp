#include<iostream>
using namespace std;

//һ�������м�¼����ֻС������أ����ҳ�����ӡ�����յ�С������

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
	cout << "��������ǣ�" << max << endl;

	system("pause");

	return 0;
}