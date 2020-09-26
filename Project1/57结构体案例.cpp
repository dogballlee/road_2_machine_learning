#include<iostream>
#include<string>
#include<ctime>
using namespace std;


//����ѧ���ṹ��
struct student
{
	string sname;
	int score;
};

//������ʦ�ṹ��
struct teacher 
{
	string tname;
	struct student sarray[5];
};

//����ʦ��ѧ����ֵ�ĺ���
void allocatespace(struct teacher tarray[], int len) 
{
	string nameseed = "ABCDE";


	//��ʼ����ʦ��ֵ
	for (int i = 0; i < len; i++) 
	{
		tarray[i].tname = "teacher_";
		tarray[i].tname = tarray[i].tname + nameseed[i];


		//ͨ��ѭ����ÿ����ʦ������ѧ����ֵ
		for (int j = 0; j < 5; j++) 
		{
			tarray[i].sarray[j].sname = "student_";
			tarray[i].sarray[j].sname += nameseed[j];
			int random = rand() % 61 + 40;	//40~100֮���һ�������
			tarray[i].sarray[j].score = random;
		}
		
	}
}



//��ӡ������Ϣ
void printinfo(struct teacher tarray[], int len) 
{
	for (int i = 0; i < len; i++) 
	{
		cout << "��ʦ������" << tarray[i].tname << endl;
		for (int j = 0; j < 5; j++) 
		{
			cout << "\t��ʦ����ѧ�������� " << tarray[i].sarray[j].sname << "\t���Է����� " << tarray[i].sarray[j].score << endl;
		}
	}
}


int main57()

{
	//���������(����������������浱ǰʱ��)
	srand((unsigned int)time(NULL));

	//����3����ʦ������
	struct teacher tarray[3];

	//ͨ��������3����ʦ����Ϣ��ֵ��������ʦ����ѧ����Ϣ��ֵ
	int len = sizeof(tarray) / sizeof(tarray[0]);
	allocatespace(tarray, len);

	//��ӡ������ʦ��������ѧ����Ϣ;
	printinfo(tarray, len);

	system("pause");

	return 0;
}