#include<iostream>
#include<string>

using namespace std;

int main15() {

	//�û��������
	int score = 0;
	cout << "������һ��������" << endl;
	cin >> score;

	//��ӡ�û�����
	cout << "������ķ���Ϊ��" << score << endl;

	//�жϷ����Ƿ����600��������ڣ������
	if (score > 600)
	{
		cout << "��ϲ����ͨ��" << endl;
	}

	system("pause");
	return 0;
}