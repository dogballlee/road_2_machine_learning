#include<iostream>
using namespace std;

//�ṹ��ָ��

struct student	//����ṹ��
{
	string name;
	int age;
	int score;
};

int main53()
{
	//1������ѧ���ṹ�����

	struct student s = { "����",18,100 };

	//2��ͨ��ָ��ָ��ṹ�����

	struct student * p = &s;
	
	//3��ͨ��ָ����ʽṹ������е�����
	//��ȥ��ѧ��ָ����ʷ�ʽ���ܷ��ʽṹ���е�����(��Ϊ�������ͻ������ͬ)����Ҫ����"->"

	cout << " ������" << p->name
		<< " ���䣺" << p->age
		<< " ������" << p->score << endl;

	system("pause");

	return 0;
}