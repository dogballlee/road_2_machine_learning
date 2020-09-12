#include<iostream>
#include<ctime>
using namespace std;

//do...while与while的区别在于：do...while会先执行一次循环语句

int main24() {
	
	int num = 0;
	do {
		cout << num << endl;
		num++;
	} while (num < 10);

	system("pause");
	return 0;
}