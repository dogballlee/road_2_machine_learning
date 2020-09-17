#include <iostream>
using namespace std;

//定义：指针变量p指向  内存地址编号为0  的空间，则该指针称为空指针
//空指针指向的内存是不可以访问的
//空指针可用来初始化指针变量

int main45() {

	int* p = NULL;

	//0~255之间的内存编号是系统占用的，不可以访问

	//*p = 100;		//此行执行会报错滴，因为莫得权限

	system("pause");

	return 0;
}