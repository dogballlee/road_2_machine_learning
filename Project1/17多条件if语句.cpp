#include<iostream>
#include<string>

using namespace std;

int main17() {

	//�û��������
	int score = 0;
	cout << "������һ��������" << endl;
	cin >> score;

	//��ӡ�û�����
	cout << "������ķ���Ϊ��" << score << endl;

	//�жϷ����Ƿ����600��������ڣ������ͨ��һ����������500�����ͨ���������������400�����ͨ���������������δͨ��
	if (score > 600)
	{
		cout << "��ϲͨ��һ��" << endl;
	}
	else if(score > 500)
	{
		cout << "��ϲͨ������" << endl;
	}
	else if (score > 400)
	{
		cout << "��ϲͨ������" << endl;
	}
	else
	{
		cout << "Ŷ��������" << endl;
	}

	system("pause");
	return 0;
}