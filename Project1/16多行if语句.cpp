#include<iostream>
#include<string>

using namespace std;

int main16() {

	//�û��������
	int score = 0;
	cout << "������һ��������" << endl;
	cin >> score;

	//��ӡ�û�����
	cout << "������ķ���Ϊ��" << score << endl;

	//�жϷ����Ƿ����600��������ڣ������ͨ�����������δͨ��
	if (score > 600)
	{
		cout << "��ϲ����ͨ��" << endl;
	}
	else
	{
		cout << "�����ɵ��ɣ�δͨ��" << endl;
	}

	system("pause");
	return 0;
}