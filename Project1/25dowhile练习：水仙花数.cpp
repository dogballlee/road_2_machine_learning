#include<iostream>
using namespace std;



int main25() {

	int num = 100;

	int a = 0;
	int b = 0;
	int c = 0;

	do {

		a = num % 10;	//��ȡ��λ����
		b = num / 10 % 10;	//��ȡʮλ����
		c = num / 100;	//��ȡ��λ����

		if (a*a*a + b*b*b + c*c*c == num) {
			cout << "num��һ��ˮ�ɻ�����" << num << endl;
		}
		else {
			cout << "num����ˮ�ɻ���" << endl;
		}
		num++;
	} while (num < 1000);

	system("pause");
	return 0;
}