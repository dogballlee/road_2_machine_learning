#include<iostream>
using namespace std;

//作用：将自定义的结构体放入到数组中方便维护

//语法：struct 结构体名 数组名[元素个数] = {{},{},...{}}

struct student	//定义结构体
{
	string name;
	int age;
	int score;
};

int main52()
{
	//创建结构体数组
	struct student stuarray[3] = 
	{ 
		{"张三",18,90},
		{"李四",19,80},
		{"王五",20,70} 
	};

	//给结构体数组中的元素赋值
	stuarray[2].name = "赵六";
	stuarray[2].age = 80;
	stuarray[2].score = 60;

	//遍历结构体数组
	for (int i = 0; i < 3; i++)
	{
		cout << "姓名：" << stuarray[i].name <<"  " 
			<< "年龄：" << stuarray[i].age << "  "
			<< "分数：" << stuarray[i].score << "  " << endl;
	};
	system("pause");

	return 0;
}