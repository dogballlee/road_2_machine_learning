#include<iostream>
#include<ctime>
using namespace std;

int main23() {
	//���һ��������ӣ��Ƴ�α�������ķ���
	srand((unsigned int)time(NULL));

	int num = rand() % 100 + 1;	//����һ��1~100֮��������
	//cout << num << endl;

	int val = 0;



	while (1) {

		cout << "�²����Ǽ���" << endl;
		cin >> val;

		if (val > num) {
			cout << "�´���" << endl;
		}
		else if (val < num) {
			cout << "��С��" << endl;
		}
		else {
			cout << "�������ţ������¶�������" << endl;
			break;
		}
	}
	system("pause");
	return 0;
}