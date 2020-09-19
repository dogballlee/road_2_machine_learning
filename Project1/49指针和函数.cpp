#include <iostream>
using namespace std;

//不使用 值传递 ，而使用 指针 实现两个数字进行交换

void swap02(int *p1, int *p2) {

	int t = *p1;
	*p1 = *p2;
	*p2 = t;

	cout << "swap02后的*p1=" << *p1 << endl;
	cout << "swap02后的*p2=" << *p2 << endl;
}

int main49() {

	int a = 10;
	int b = 20;
	swap02(&a, &b);

	cout << "swap02后的a=" << a << endl;
	cout << "swap02后的b=" << b << endl;

	//妈耶，地址传递会修改实参！(值传递打咩！)

	system("pause");

	return 0;
}