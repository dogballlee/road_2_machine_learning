#include<iostream>

using namespace std;

//switch的缺点：只能判断整型或字符型(if else可以)，不可以是一个区间
//switch的优点：结构清晰，执行效率高于if else


int main21() {

	cout << "请给TENET打分：" << endl;
	int score = 0;
	cin >> score;

	cout << "您的打分为：" << score << endl;

	switch (score) {
	case 10:
		cout << "您认为这是神作" << endl;
		break;
	case 9:
		cout << "您认为这是不可多得的佳作" << endl;
		break;
	case 8:
		cout << "您认为这片不错" << endl;
		break;
	case 7:
		cout << "您认为这片还行" << endl;
		break;
	case 6:
		cout << "您认为这片一般" << endl;
		break;
	default:
		cout << "您觉得这片不太行" << endl;
		break;
	}

	system("pause");
	return 0;
}