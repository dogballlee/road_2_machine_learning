#include<iostream>
using namespace std;

//����ѧ���ṹ��
struct student
{
	string name;
	int age;
	int score;
};

//��ʹ��ֵ���ݵķ�ʽ��ȡԭ���ݣ������ڴ��и���һ���µĸ��������ӿ����������ٶȣ���˲�����ʹ��
//void printstudent(student s) 
//{
//	cout << "�Ӻ����� ������" << s.name
//		<< "���䣺 " << s.age
//		<< " ������ " << s.score << endl;
//}

//��ֵ���ݲ�ͬ��ʹ��ָ����е�ַ����(����һ�δ����е��βθ�Ϊָ��)�����Դ����ٿ�����ָ���Сʼ��Ϊ4���ֽڣ����Ƽ�ʹ��
void printstudent1(const student *s) 
{
	//s->age = 150;	//�������񼫣����£�����const��һ�����޸ĵĲ����ͻᱨ�����Է�ֹ���ǵ������
	cout << "�Ӻ����� ������" << s->name
		<< "���䣺 " << s->age
		<< " ������ " << s->score << endl;
}
//���ǣ����ϴ�����ڷ��գ������С��������һ�θ�ֵ����أ��ǿ��ǻ�ı�ʵ�ε�ֵ�ģ��е�ҰŶ
//������ޣ��˴�����const����

int main57()
{
	struct student s = { "����" ,20,100 };

	//printstudent(s);
	printstudent1(&s);

	system("pause");

	return 0;
}