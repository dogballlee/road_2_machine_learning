#include<iostream>
using namespace std;

int main30() {
//switch����е�ʹ��
//��ѡ����Ϸ�ѶȾ�������ͨ�����ѡ�ج��
//
//	int select = 0;
//
//	cout << "please select a game model" << endl;
//
//	cin >> select;
//
//	switch (select) {
//	case 1:
//		cout << "��ͨ�Ѷ�" << endl;
//		break;
//	case 2:
//		cout << "�����Ѷ�" << endl;
//		break;
//	case 3:
//		cout << "ج���Ѷ�" << endl;
//		break;
//	defalt:
//		break;
//	}


//��forѭ������е�ʹ��
	//for (int i = 0; i < 10; i++) {

	//	if (i == 5) {
	//		break;
	//	}
	//	cout << i << endl;

	//}

//������Ƕ��ѭ�������
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

