#include<iostream>
using namespace std;

/*��ֱ������λͬѧ���ܳɼ�*/

int main() {

	//1�������ά����ռ���ڴ��С

	int scores[3][3] =
	{ 100,100,100,90,50,100,60,70,80 };

	string names[3] = { "����","����","����" };

	int sum = 0;

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			sum = sum + scores[i][j];
		}cout << names[i] << "���ܷ�Ϊ��" << sum << endl;
	}

	system("pause");

	return 0;
}