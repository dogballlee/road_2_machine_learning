#include<iostream>
using namespace std;

int main4() {
	//整型：short(2字节) int(4字节) long(4字节) long long(8字节)
	//sizeof可用来求出数据类型占用的内存大小
	//语法：sizeof(数据类型/变量)

	short num1 = 10;
	cout << "short占用的内存空间为：" << sizeof(num1) << endl;

	system("pause");

	return 0;
}