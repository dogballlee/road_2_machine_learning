#include<iostream>
using namespace std;

//ִ�е�goto���ʱ���������������ĳ����У���ת����Ǵ�����ִ��
//�������ʹ�ô�д��ĸ��������һ��Լ���׳ɵ�ϰ�ߡ�goto��Ǻ��;��ʵ�ʱ�Ǻ��:
//ʵ�ʹ�����ʹ�ý��٣�������ȥ�����׻���

int main32() {

	cout << "1test" << endl;
	cout << "2test" << endl;
	goto FLAG;
	cout << "3test" << endl;
	cout << "4test" << endl;
	FLAG:
	cout << "5test" << endl;

	system("pause");

	return 0;
}