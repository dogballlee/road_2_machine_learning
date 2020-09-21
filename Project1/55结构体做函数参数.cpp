#include<iostream>
using namespace std;

//定义学生结构体
struct student 
{
	string name;
	int age;
	int score;
};


int main() 
{
	struct student s;
	s.name = "张三";
	s.age = 20;
	s.score = 100;
	
	cout << "在main函数中打印 姓名：" << s.name 
		<< "年龄： " << s.age 
		<< " 分数： " << s.score << endl;

	system("pause");

	return 0;
}