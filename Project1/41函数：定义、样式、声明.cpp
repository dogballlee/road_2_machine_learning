#include<iostream>
using namespace std;

//������Ҫ�أ�����ֵ���͡��������������б���������䡢return���ʽ

//����������������δ��������������ǰ��֪�����������Ĵ��ڣ��ö����ں󷽵ĺ����ܹ���ǰ���ĳ����ִ�ж�������

//��������������д��Σ����Ǻ���ֻ�ܶ���һ��


int add(int num1, int num2) 
{
	int sum = num1 + num2;
	return sum;
}


//�����������Ҫ����ֵ��������ʱ�����дvoid����ʱ����Ҫ��дreturn��

void swap1(int num1, int num2)
{
	cout << "����ǰ��" << endl;
	cout << "num1=" << num1 << endl;
	cout << "num2=" << num2 << endl;

	int t = num2;
	num2 = num1;
	num1 = t;

	cout << "������" << endl;
	cout << "num1=" << num1 << endl;
	cout << "num2=" << num2 << endl;
}

//������������ʽ
/*
1���޲��з�
2���в��޷�
3���޲��޷�
4���в��з�
*/


int main41() {
	int a = 10;
	int b = 20;

	int sum = add(a, b);

	swap1(a, b);

	cout << "sum:" << sum << endl;

	cout << "aת����" << a << endl;
	cout << "bת����" << b << endl;	
	//����ֵ����ֻ��ı��βΣ�ʵ�β��ᱻ�ı�

	system("pause");

	return 0;
}