#include "swap.h"	//��ʾ��Դ�ļ���swap.hͷ�ļ����ڹ���



//�����Ķ���
void swap(int num1, int num2)
{
	int t = num1;
	num1 = num2;
	num2 = t;

	cout << "num1 = " << num1 << endl;
	cout << "num2 = " << num2 << endl;
}