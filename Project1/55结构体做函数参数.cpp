#include<iostream>
using namespace std;

//定义学生结构体
struct student 
{
	string name;
	int age;
	int score;
};

//值传递
void printstudent1(struct student s) 
{
	//s.age = 100;	若写此行代码，则只会修改形参，实参值不变
	cout << "子函数中 姓名：" << s.name
		<< "年龄： " << s.age
		<< " 分数： " << s.score << endl;
}

//地址传递
void printstudent2(struct student * p)
{
	//p->age = 100;	若写此行代码,形参和实参值都会被修改
	cout << "子函数中 姓名：" << p->name
		<< "年龄： " << p->age
		<< " 分数： " << p->score << endl;
}


int main55() 
{
	struct student s;
	s.name = "张三";
	s.age = 20;
	s.score = 100;

	printstudent1(s);
	printstudent2(&s);
	
	cout << "在main函数中打印 姓名：" << s.name 
		<< "年龄： " << s.age 
		<< " 分数： " << s.score << endl;

	system("pause");

	return 0;
}