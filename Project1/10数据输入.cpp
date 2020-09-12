#include<iostream>
#include<string>
using namespace std;

int main10() {
	//1、整型
	int a = 0;	//赋初始值
	cout << "请给整型变量a赋值" << endl;
	cin >> a;
	cout << "整型变量a=" << a << endl;

	//2、浮点型
	float f = 3.14f;
	cout << "请给浮点型变量f赋值" << endl;
	cin >> f;
	cout << "浮点型变量f=" << f << endl;

	//3、字符型

	char ch = 'a';

	//4、字符串型

	string str = "helloworld";
	
	system("pause");
	return 0;
}