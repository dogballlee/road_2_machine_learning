#include<iostream>

using namespace std;

//switch��ȱ�㣺ֻ���ж����ͻ��ַ���(if else����)����������һ������
//switch���ŵ㣺�ṹ������ִ��Ч�ʸ���if else


int main21() {

	cout << "���TENET��֣�" << endl;
	int score = 0;
	cin >> score;

	cout << "���Ĵ��Ϊ��" << score << endl;

	switch (score) {
	case 10:
		cout << "����Ϊ��������" << endl;
		break;
	case 9:
		cout << "����Ϊ���ǲ��ɶ�õļ���" << endl;
		break;
	case 8:
		cout << "����Ϊ��Ƭ����" << endl;
		break;
	case 7:
		cout << "����Ϊ��Ƭ����" << endl;
		break;
	case 6:
		cout << "����Ϊ��Ƭһ��" << endl;
		break;
	default:
		cout << "��������Ƭ��̫��" << endl;
		break;
	}

	system("pause");
	return 0;
}