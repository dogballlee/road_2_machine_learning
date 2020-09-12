#include<iostream>
using namespace std;

//游戏规则：过7

int main26() {


	for(int i = 1; i < 101; i ++)
	{
		if (i%10 == 7 || i / 10 == 7 || i % 7 == 0) {
			cout << "敲桌子" << endl;
		}

		else {
			cout << i << endl;
		}
	}

	system("pause");

	return 0;
}