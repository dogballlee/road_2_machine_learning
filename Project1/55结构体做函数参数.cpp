#include<iostream>
using namespace std;

//����ѧ���ṹ��
struct student 
{
	string name;
	int age;
	int score;
};

//ֵ����
void printstudent1(struct student s) 
{
	//s.age = 100;	��д���д��룬��ֻ���޸��βΣ�ʵ��ֵ����
	cout << "�Ӻ����� ������" << s.name
		<< "���䣺 " << s.age
		<< " ������ " << s.score << endl;
}

//��ַ����
void printstudent2(struct student * p)
{
	//p->age = 100;	��д���д���,�βκ�ʵ��ֵ���ᱻ�޸�
	cout << "�Ӻ����� ������" << p->name
		<< "���䣺 " << p->age
		<< " ������ " << p->score << endl;
}


int main55() 
{
	struct student s;
	s.name = "����";
	s.age = 20;
	s.score = 100;

	printstudent1(s);
	printstudent2(&s);
	
	cout << "��main�����д�ӡ ������" << s.name 
		<< "���䣺 " << s.age 
		<< " ������ " << s.score << endl;

	system("pause");

	return 0;
}