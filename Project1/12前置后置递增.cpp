#include<iostream>
#include<string>

using namespace std;

int main12() {

	//前置递增(先让变量+1,再进行表达式运算)
	int a = 10;
	++a;	//变量+1
	cout << a << endl;

	//后置递增(先进行表达式运算,再让变量+1)
	int b = 10;
	b++;	//变量+1
	cout << b << endl;

	//前置与后置的区别
	int a2 = 10;
	int b2 = ++a2 * 10;
	cout << "a2=" << a2 << endl;
	cout << "b2=" << b2 << endl;

	int a3 = 10;
	int b3 = a3++ * 10;
	cout << "a3=" << a3 << endl;
	cout << "b3=" << b3 << endl;



	system("pause");
	return 0;
}