#include<iostream>
using namespace std;

//����ѧ���ṹ��
struct student 
{
	string name;
	int age;
	int score;
};


int main() 
{
	struct student s;
	s.name = "����";
	s.age = 20;
	s.score = 100;
	
	cout << "��main�����д�ӡ ������" << s.name 
		<< "���䣺 " << s.age 
		<< " ������ " << s.score << endl;

	system("pause");

	return 0;
}