#include<iostream>
using namespace std;

/*һά��������������;��
	1��������ͳ���������ڴ��еĿռ�		���磺sizeof(arr)
	2�����Ի�ȡ�������ڴ��е��׵�ַ*/

int main34() {

	//1����������ռ���ڴ��С
	int arr[] = { 1,2,3,4,5,6,7,8 };
	cout << "����arr���ڴ���ռ�õĿռ���:" << sizeof(arr) << endl;

	cout << "����arr��һ��Ԫ�����ڴ���ռ�õĿռ���:" << sizeof(arr[0]) << endl;

	cout << "����arr��Ԫ�ظ�����:" << sizeof(arr)/ sizeof(arr[0]) << endl;


	//2���鿴�����׵�ַ
	cout << "�����׵�ַΪ��" << (int)arr << endl;	//�����(int)��ʾ��16���Ƶĵ�ַǿת����10������ʾ
	cout << "�����е�һ��Ԫ�ص�ַΪ��" << (int)&arr[0] << endl;

	system("pause");

	return 0;
}