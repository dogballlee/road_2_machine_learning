#include<iostream>
using namespace std;

//��ѭ������У�������continue���򲻻�ִ�к�����䣬����������ǰѭ����ִ����һѭ��
//break��ͬ����ֱ����������ѭ����
int main31() {


	for (int i = 0; i < 100; i++)
	{
		//ֻ���ż��
		if (i % 2 == 1) 
		{
			continue;
		}
		else
		{
			cout << i << endl;
		}
	}

	system("pause");

	return 0;
}