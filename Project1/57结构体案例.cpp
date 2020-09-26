#include<iostream>
#include<string>
#include<ctime>
using namespace std;


//定义学生结构体
struct student
{
	string sname;
	int score;
};

//定义老师结构体
struct teacher 
{
	string tname;
	struct student sarray[5];
};

//给老师和学生赋值的函数
void allocatespace(struct teacher tarray[], int len) 
{
	string nameseed = "ABCDE";


	//开始给老师赋值
	for (int i = 0; i < len; i++) 
	{
		tarray[i].tname = "teacher_";
		tarray[i].tname = tarray[i].tname + nameseed[i];


		//通过循环给每名老师所带的学生赋值
		for (int j = 0; j < 5; j++) 
		{
			tarray[i].sarray[j].sname = "student_";
			tarray[i].sarray[j].sname += nameseed[j];
			int random = rand() % 61 + 40;	//40~100之间的一个随机数
			tarray[i].sarray[j].score = random;
		}
		
	}
}



//打印所有信息
void printinfo(struct teacher tarray[], int len) 
{
	for (int i = 0; i < len; i++) 
	{
		cout << "老师姓名：" << tarray[i].tname << endl;
		for (int j = 0; j < 5; j++) 
		{
			cout << "\t老师所带学生姓名： " << tarray[i].sarray[j].sname << "\t考试分数： " << tarray[i].sarray[j].score << endl;
		}
	}
}


int main57()

{
	//随机数种子(制造真随机数，跟随当前时间)
	srand((unsigned int)time(NULL));

	//创建3名老师的数组
	struct teacher tarray[3];

	//通过函数给3名老师的信息赋值，并给老师带的学生信息赋值
	int len = sizeof(tarray) / sizeof(tarray[0]);
	allocatespace(tarray, len);

	//打印所有老师及所带的学生信息;
	printinfo(tarray, len);

	system("pause");

	return 0;
}