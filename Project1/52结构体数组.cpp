#include<iostream>
using namespace std;

//���ã����Զ���Ľṹ����뵽�����з���ά��

//�﷨��struct �ṹ���� ������[Ԫ�ظ���] = {{},{},...{}}

struct student	//����ṹ��
{
	string name;
	int age;
	int score;
};

int main52()
{
	//�����ṹ������
	struct student stuarray[3] = 
	{ 
		{"����",18,90},
		{"����",19,80},
		{"����",20,70} 
	};

	//���ṹ�������е�Ԫ�ظ�ֵ
	stuarray[2].name = "����";
	stuarray[2].age = 80;
	stuarray[2].score = 60;

	//�����ṹ������
	for (int i = 0; i < 3; i++)
	{
		cout << "������" << stuarray[i].name <<"  " 
			<< "���䣺" << stuarray[i].age << "  "
			<< "������" << stuarray[i].score << "  " << endl;
	};
	system("pause");

	return 0;
}