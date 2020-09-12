#include<iostream>
#include<ctime>
using namespace std;

int main23() {
	//添加一个随机种子，破除伪随机情况的发生
	srand((unsigned int)time(NULL));

	int num = rand() % 100 + 1;	//生成一个1~100之间的随机数
	//cout << num << endl;

	int val = 0;



	while (1) {

		cout << "猜猜我是几？" << endl;
		cin >> val;

		if (val > num) {
			cout << "猜大咧" << endl;
		}
		else if (val < num) {
			cout << "猜小咯" << endl;
		}
		else {
			cout << "那是真滴牛批，你猜对了老铁" << endl;
			break;
		}
	}
	system("pause");
	return 0;
}