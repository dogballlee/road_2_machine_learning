#include<iostream>
#include "swap.h"
using namespace std;


//�����ķ��ļ���д
//ʵ���������ֽ��н����ĺ���

//����������
//void swap(int num1, int num2);
//�����Ķ���
//void swap(int num1, int num2)
//{
//	int t = num1;
//	num1 = num2;
//	num2 = t;
//}


//1������.hΪ��׺����ͷ�ļ�
//2������.cppΪ��׺����Դ�ļ�
//3����ͷ�ļ���д����������(�ǵ�д����Ŀ��)
//4����Դ�ļ���д�����Ķ���(�ǵ�д����ͷ�ļ�)

int main() {
	int a = 10;
	int b = 20;

	swap(a, b);

	system("pause");

	return 0;
}