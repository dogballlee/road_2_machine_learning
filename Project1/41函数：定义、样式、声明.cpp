#include<iostream>
using namespace std;

//函数的要素：返回值类型、函数名、参数列表、函数体语句、return表达式

//函数的声明（以下未举例）：用于提前告知编译器函数的存在，让定义于后方的函数能够被前方的程序段执行而不报错

//函数的声明可以写多次，但是函数只能定义一次


int add(int num1, int num2) 
{
	int sum = num1 + num2;
	return sum;
}


//如果函数不需要返回值，声明的时候可以写void，这时不需要再写return了

void swap1(int num1, int num2)
{
	cout << "交换前：" << endl;
	cout << "num1=" << num1 << endl;
	cout << "num2=" << num2 << endl;

	int t = num2;
	num2 = num1;
	num1 = t;

	cout << "交换后：" << endl;
	cout << "num1=" << num1 << endl;
	cout << "num2=" << num2 << endl;
}

//函数常见的样式
/*
1、无参有反
2、有参无反
3、无参无反
4、有参有返
*/


int main41() {
	int a = 10;
	int b = 20;

	int sum = add(a, b);

	swap1(a, b);

	cout << "sum:" << sum << endl;

	cout << "a转换后：" << a << endl;
	cout << "b转换后：" << b << endl;	
	//函数值传递只会改变形参，实参不会被改变

	system("pause");

	return 0;
}