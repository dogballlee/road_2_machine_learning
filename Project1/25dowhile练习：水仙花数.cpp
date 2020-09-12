#include<iostream>
using namespace std;



int main25() {

	int num = 100;

	int a = 0;
	int b = 0;
	int c = 0;

	do {

		a = num % 10;	//获取个位数字
		b = num / 10 % 10;	//获取十位数字
		c = num / 100;	//获取百位数字

		if (a*a*a + b*b*b + c*c*c == num) {
			cout << "num是一个水仙花数：" << num << endl;
		}
		else {
			cout << "num不是水仙花数" << endl;
		}
		num++;
	} while (num < 1000);

	system("pause");
	return 0;
}