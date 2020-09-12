#include<iostream>
using namespace std;

//执行到goto语句时，会跳过接下来的程序行，跳转到标记处继续执行
//标记往往使用大写字母命名，是一个约定俗成的习惯。goto标记后跟;，实际标记后跟:
//实际工作中使用较少，跳来跳去的容易混淆

int main32() {

	cout << "1test" << endl;
	cout << "2test" << endl;
	goto FLAG;
	cout << "3test" << endl;
	cout << "4test" << endl;
	FLAG:
	cout << "5test" << endl;

	system("pause");

	return 0;
}