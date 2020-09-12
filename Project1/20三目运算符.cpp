#include<iostream>
#include<string>

using namespace std;


int main20() {

	int a = 10;
	int b = 20;
	int c = 0;

	c = (a > b ? a : b);
	cout << "c=" << c << endl;

	//三目运算后可以对返回的变量继续进行赋值操作
	(a > b ? a : b) = 100;
	cout << "a=" << a << endl;
	cout << "b=" << b << endl;

	system("pause");
	return 0;
}