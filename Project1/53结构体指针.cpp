#include<iostream>
using namespace std;

//结构体指针

struct student	//定义结构体
{
	string name;
	int age;
	int score;
};

int main53()
{
	//1、创建学生结构体变量

	struct student s = { "张三",18,100 };

	//2、通过指针指向结构体变量

	struct student * p = &s;
	
	//3、通过指针访问结构体变量中的数据
	//过去所学的指针访问方式不能访问结构体中的属性(因为数据类型或各不相同)，需要利用"->"

	cout << " 姓名：" << p->name
		<< " 年龄：" << p->age
		<< " 分数：" << p->score << endl;

	system("pause");

	return 0;
}