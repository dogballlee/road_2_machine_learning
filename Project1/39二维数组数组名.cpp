#include<iostream>
using namespace std;

/*��ά��������������;��
	1��������ͳ�ƶ�ά�������ڴ��еĿռ�		���磺sizeof(arr)
	2�����Ի�ȡ��ά�������ڴ��е��׵�ַ*/

int main39() {

	//1�������ά����ռ���ڴ��С
	
	int arr[2][3] =
	{ 1,2,3,4,5,6 };
	cout << "��ά����arr��ռ�ڴ�ռ䣺" << sizeof(arr) << endl;
	cout << "��ά����arr��һ����ռ�ڴ�ռ䣺" << sizeof(arr[0]) << endl;
	cout << "��ά����arr��һ��Ԫ����ռ�ڴ�ռ䣺" << sizeof(arr[0][0]) << endl;

	//2���鿴��ά�����׵�ַ

	cout << "��ά����arr���׵�ַ" << (int)arr << endl;
	cout << "��ά����arr��һ�е��׵�ַ" << (int)arr[0] << endl;
	cout << "��ά����arr��һ��Ԫ�ص��׵�ַ" << (int)&arr[0][0] << endl;

	system("pause");

	return 0;
}