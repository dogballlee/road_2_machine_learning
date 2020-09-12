#include<iostream>
#include<string>

using namespace std;

int main16() {

	//用户输入分数
	int score = 0;
	cout << "请输入一个分数：" << endl;
	cin >> score;

	//打印用户分数
	cout << "您输入的分数为：" << score << endl;

	//判断分数是否大于600，如果大于，则输出通过；否则输出未通过
	if (score > 600)
	{
		cout << "恭喜考试通过" << endl;
	}
	else
	{
		cout << "垃圾吧倒吧，未通过" << endl;
	}

	system("pause");
	return 0;
}