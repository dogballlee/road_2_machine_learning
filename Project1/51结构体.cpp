//结构体的基本概念：
//用户 自定义 的 数据类型，允许用户存储不同的数据类型

//语法：struct 结构体名{结构体成员列表};

/*
通过结构体创建变量的方式有三种：

1、struct 结构体名 变量名
2、struct 结构体名 变量名 = {成员值1, 成员值2...}
3、定义结构体时顺便创建变量
*/


#include<iostream>

using namespace std;

//1、创建一个 学生 数据类型：学生包括（姓名、年龄、分数）
//自定义数据类型，一些类型集合组成的一个烈性
//语法 struct 类型 { 成员列表 }
struct student 
{
	//成员列表

	//姓名
	string name;
	//年龄
	int age;
	//分数
	int score;
}s3;	//顺便创建结构体变量

//2、通过 学生 类型创建具体的学生

//2.1 strduct student s1
//2.2 strduct student s2 = {...}
//2.3 定义结构体时顺便创建变量

int main51() 
{
	//2.1 strduct student s1
	//创建变量时，struct关键字可以省略不写(创建数据类型时，不可以省！)
	struct student s1;
	//给s1属性赋值，通过.访问结构体变量中的属性
	s1.name = "张三";
	s1.age = 18;
	s1.score = 100;
	cout << "姓名：" << s1.name << "年龄：" << s1.age << "分数：" << s1.score << endl;
	
	//2.2 strduct student s2 = {...}
	struct student s2 = {"李四",19,80};
	cout << "姓名：" << s2.name << "年龄：" << s2.age << "分数：" << s2.score << endl;

	//2.3 定义结构体时顺便创建变量
	s3.name = "王五";
	s3.age = 20;
	s3.score = 60;
	cout << "姓名：" << s3.name << "年龄：" << s3.age << "分数：" << s3.score << endl;

	system("pause");

	return 0;
}