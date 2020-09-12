#include<iostream>
using namespace std;


int main6() {
	//创建字符型变量，要使用单引号
	//创建字符型变量，单引号内只能有一个字符

	char ch = 'a';

	cout << ch << endl;

	cout << (int)ch << endl;//字符型强转对应的ASCII码(a:97  A:65)

	system("pause");

	return 0;
}