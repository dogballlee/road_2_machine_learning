#include<iostream>
using namespace std;

//在一个结构体钟存在另一个结构体
//例：每个老师辅导一个学生，一个老师结构体中，记录着一个学生结构体

struct student	//定义结构体
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
	//创建老师
	teacher t;
	t.id = 1000;
	t.name = "葵司";
	t.age = 25;
	t.stu.name = "张三";
	t.stu.age = 20;
	t.stu.score = 100;

	cout << "老师的名字：" << t.name << endl;
	cout << "老师的id：" << t.id << endl;
	cout << "老师的年龄：" << t.age << endl;
	cout << "老师辅导学生的名字：" << t.stu.name << endl;
	cout << "老师辅导学生的年龄：" << t.stu.age << endl;
	cout << "老师辅导学生的分数：" << t.stu.score << endl;

	system("pause");

	return 0;
}