//�ṹ��Ļ������
//�û� �Զ��� �� �������ͣ������û��洢��ͬ����������

//�﷨��struct �ṹ����{�ṹ���Ա�б�};

/*
ͨ���ṹ�崴�������ķ�ʽ�����֣�

1��struct �ṹ���� ������
2��struct �ṹ���� ������ = {��Աֵ1, ��Աֵ2...}
3������ṹ��ʱ˳�㴴������
*/


#include<iostream>

using namespace std;

//1������һ�� ѧ�� �������ͣ�ѧ�����������������䡢������
//�Զ����������ͣ�һЩ���ͼ�����ɵ�һ������
//�﷨ struct ���� { ��Ա�б� }
struct student 
{
	//��Ա�б�

	//����
	string name;
	//����
	int age;
	//����
	int score;
}s3;	//˳�㴴���ṹ�����

//2��ͨ�� ѧ�� ���ʹ��������ѧ��

//2.1 strduct student s1
//2.2 strduct student s2 = {...}
//2.3 ����ṹ��ʱ˳�㴴������

int main51() 
{
	//2.1 strduct student s1
	//��������ʱ��struct�ؼ��ֿ���ʡ�Բ�д(������������ʱ��������ʡ��)
	struct student s1;
	//��s1���Ը�ֵ��ͨ��.���ʽṹ������е�����
	s1.name = "����";
	s1.age = 18;
	s1.score = 100;
	cout << "������" << s1.name << "���䣺" << s1.age << "������" << s1.score << endl;
	
	//2.2 strduct student s2 = {...}
	struct student s2 = {"����",19,80};
	cout << "������" << s2.name << "���䣺" << s2.age << "������" << s2.score << endl;

	//2.3 ����ṹ��ʱ˳�㴴������
	s3.name = "����";
	s3.age = 20;
	s3.score = 60;
	cout << "������" << s3.name << "���䣺" << s3.age << "������" << s3.score << endl;

	system("pause");

	return 0;
}