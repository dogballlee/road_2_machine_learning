#include <iostream>
using namespace std;

//��ʹ�� ֵ���� ����ʹ�� ָ�� ʵ���������ֽ��н���

void swap02(int *p1, int *p2) {

	int t = *p1;
	*p1 = *p2;
	*p2 = t;

	cout << "swap02���*p1=" << *p1 << endl;
	cout << "swap02���*p2=" << *p2 << endl;
}

int main49() {

	int a = 10;
	int b = 20;
	swap02(&a, &b);

	cout << "swap02���a=" << a << endl;
	cout << "swap02���b=" << b << endl;

	//��Ү����ַ���ݻ��޸�ʵ�Σ�(ֵ���ݴ��㣡)

	system("pause");

	return 0;
}