#include<iostream>
#include<string>

using namespace std;

int main19() {

	//��ֻС������أ���ֻ����

	int num1 = 0;
	int num2 = 0;
	int num3 = 0;

	//���û�������ֻС�������
	cout << "������С��A������" << endl;
	cin >> num1;

	cout << "������С��B������" << endl;
	cin >> num2;

	cout << "������С��C������" << endl;
	cin >> num3;

	cout << "С��A������Ϊ��" <<num1 << endl;
	cout << "С��B������Ϊ��" << num2 << endl;
	cout << "С��C������Ϊ��" << num3 << endl;

	if (num1 > num2)
	{
		if (num1 > num3)
		{
			cout << "С��A����" << endl;
		}
		else
		{
			cout << "С��C����" << endl;
		}
	}
	else
	{
		if (num2 > num3)
		{
			cout << "С��B����" << endl;
		}
		else
		{
			cout << "С��C����" << endl;
		}
	}

	system("pause");
	return 0;
}