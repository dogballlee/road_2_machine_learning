#include<iostream>
using namespace std;

//��һ���ṹ���Ӵ�����һ���ṹ��
//����ÿ����ʦ����һ��ѧ����һ����ʦ�ṹ���У���¼��һ��ѧ���ṹ��

struct student	//����ṹ��
{
	string name;
	int age;
	int score;
};

struct teacher 
{
	int id;
	string name;
	int age;
	struct student stu;
};

int main54()
{
	//������ʦ
	teacher t;
	t.id = 1000;
	t.name = "��˾";
	t.age = 25;
	t.stu.name = "����";
	t.stu.age = 20;
	t.stu.score = 100;

	cout << "��ʦ�����֣�" << t.name << endl;
	cout << "��ʦ��id��" << t.id << endl;
	cout << "��ʦ�����䣺" << t.age << endl;
	cout << "��ʦ����ѧ�������֣�" << t.stu.name << endl;
	cout << "��ʦ����ѧ�������䣺" << t.stu.age << endl;
	cout << "��ʦ����ѧ���ķ�����" << t.stu.score << endl;

	system("pause");

	return 0;
}