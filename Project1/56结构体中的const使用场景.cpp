#include<iostream>
using namespace std;

//定义学生结构体
struct student
{
	string name;
	int age;
	int score;
};

//若使用值传递的方式获取原数据，会在内存中复制一份新的副本，增加开销，降低速度，因此不建议使用
//void printstudent(student s) 
//{
//	cout << "子函数中 姓名：" << s.name
//		<< "年龄： " << s.age
//		<< " 分数： " << s.score << endl;
//}

//与值传递不同，使用指针进行地址传递(将上一段代码中的形参改为指针)，可以大大减少开销（指针大小始终为4个字节），推荐使用
void printstudent1(const student *s) 
{
	//s->age = 150;	//此行罪大恶极，可怕！加入const后，一旦有修改的操作就会报错，可以防止我们的误操作
	cout << "子函数中 姓名：" << s->name
		<< "年龄： " << s->age
		<< " 分数： " << s->score << endl;
}
//但是，以上代码存在风险，如果不小心输入了一段赋值语句呢？那可是会改变实参的值的，有点野哦
//得想个辙，此处有请const大佬

int main57()
{
	struct student s = { "张三" ,20,100 };

	//printstudent(s);
	printstudent1(&s);

	system("pause");

	return 0;
}