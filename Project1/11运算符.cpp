#include<iostream>
#include<string>
using namespace std;

int main11() {
	//加减乘除
	int a1 = 10;
	int b1 = 3;

	cout << a1 + b1 << endl;//整数相除得整数，小数相除得小数，若分母为0，相除会报错

	//取模（其实就是取余数）

	cout << a1 % b1 << endl;

	system("pause");
	return 0;
}