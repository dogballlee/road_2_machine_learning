#include<iostream>
#include<string>

using namespace std;

int main12() {

	//ǰ�õ���(���ñ���+1,�ٽ��б��ʽ����)
	int a = 10;
	++a;	//����+1
	cout << a << endl;

	//���õ���(�Ƚ��б��ʽ����,���ñ���+1)
	int b = 10;
	b++;	//����+1
	cout << b << endl;

	//ǰ������õ�����
	int a2 = 10;
	int b2 = ++a2 * 10;
	cout << "a2=" << a2 << endl;
	cout << "b2=" << b2 << endl;

	int a3 = 10;
	int b3 = a3++ * 10;
	cout << "a3=" << a3 << endl;
	cout << "b3=" << b3 << endl;



	system("pause");
	return 0;
}