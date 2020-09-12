#include<iostream>
#include<string>

using namespace std;

int main17() {

	//用户输入分数
	int score = 0;
	cout << "请输入一个分数：" << endl;
	cin >> score;

	//打印用户分数
	cout << "您输入的分数为：" << score << endl;

	//判断分数是否大于600，如果大于，则输出通过一本；若大于500，输出通过二本；如果大于400，输出通过三本；否则，输出未通过
	if (score > 600)
	{
		cout << "恭喜通过一本" << endl;
	}
	else if(score > 500)
	{
		cout << "恭喜通过二本" << endl;
	}
	else if (score > 400)
	{
		cout << "恭喜通过三本" << endl;
	}
	else
	{
		cout << "哦豁，人上人" << endl;
	}

	system("pause");
	return 0;
}