#include<iostream>
using namespace std;

/*请分别输出三位同学的总成绩*/

int main() {

	//1、计算二维数组占用内存大小

	int scores[3][3] =
	{ 100,100,100,90,50,100,60,70,80 };

	string names[3] = { "张三","李四","王五" };

	int sum = 0;

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			sum = sum + scores[i][j];
		}cout << names[i] << "的总分为：" << sum << endl;
	}

	system("pause");

	return 0;
}