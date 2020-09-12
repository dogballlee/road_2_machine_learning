#include<iostream>
using namespace std;

//在循环语句中，若遇到continue，则不会执行后续语句，而是跳出当前循环，执行下一循环
//break不同，会直接跳出整个循环体
int main31() {


	for (int i = 0; i < 100; i++)
	{
		//只输出偶数
		if (i % 2 == 1) 
		{
			continue;
		}
		else
		{
			cout << i << endl;
		}
	}

	system("pause");

	return 0;
}