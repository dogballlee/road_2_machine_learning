#include<iostream>
using namespace std;

int main30() {
//switch语句中的使用
//用选择游戏难度举例：普通、困难、噩梦
//
//	int select = 0;
//
//	cout << "please select a game model" << endl;
//
//	cin >> select;
//
//	switch (select) {
//	case 1:
//		cout << "普通难度" << endl;
//		break;
//	case 2:
//		cout << "困难难度" << endl;
//		break;
//	case 3:
//		cout << "噩梦难度" << endl;
//		break;
//	defalt:
//		break;
//	}


//在for循环语句中的使用
	//for (int i = 0; i < 10; i++) {

	//	if (i == 5) {
	//		break;
	//	}
	//	cout << i << endl;

	//}

//出现在嵌套循环语句中
	for (int i = 0; i < 10; i++) 
	{
		for (int j = 0; j < 10; j++) 
		{
			if (j == 5) 
			{
				break;
			}
			cout << "* ";
		}
		cout << endl;
	}
	system("pause");

	return 0;


}

