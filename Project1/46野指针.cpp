#include <iostream>
using namespace std;

//定义：指针变量指向非法的内存空间，在程序中一定要避免出现野指针，语法莫得问题，但是执行会崩溃滴

int main46() {

	//指针变量p指向内存地址编号为0x1100的空间(在这之前还没申请使用过该空间，怎么可以瞎指呢)
	int* p = (int*)0x1100;

	//访问野指针报错（多行不义必自毙，该）
	cout << *p << endl;

	system("pause");

	return 0;
}

//总结：空指针和野指针都指向了非用户申请的内存空间（系统占用的&未申请的），因此访问不能