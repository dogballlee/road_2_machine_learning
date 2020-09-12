#include<iostream>
#include<string>	//使用C++风格字符串时要包含该头文件
using namespace std;

int main8() {
	//1、	C风格字符串
	//注意事项1	char字符串名加[]
	//注意事项2 等号后面要用双引号包含起来字符串
	char str[] = "hello world";
	cout << str << endl;

	//2、	C++风格字符串
	string str2 = "hello world";
	cout << str2 << endl;

	system("pause");

	return 0;
}