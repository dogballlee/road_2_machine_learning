#include <iostream>
using namespace std;

//Ŀ�ģ�����ָ����������е�Ԫ��


int main48() {

	int arr[10] = { 1,2,3,4,5,6,7,8,9 };

	cout << "����arr�ĵ�һ��Ԫ��Ϊ��" << arr[0] << endl;	//�������

	int* p = arr;	//arr����������׵�ַ

	cout << "����arr�ĵ�һ��Ԫ��Ϊ(����ָ������û�ȡ)��" << *p << endl;

	int* p1 = &arr[1];	//����ʹ��p++ʹָ�����ƫ��1��λ��(4���ֽ�)

	cout << "����arr�ĵڶ���Ԫ��Ϊ(����ָ������û�ȡ)��" << *p1 << endl;

	system("pause");

	return 0;
}