#include<iostream>
#include<string>
using namespace std;

struct heros 
{
	string name;
	int age;
	string sex;
};

void herosortbyage(struct heros harray[], int len)
{
	for (int i = 0;i < len; i++) 
	{
		for (int j = 0;j < len-i-1;j++) 
		{
			
			if (harray[j].age > harray[j+1].age)
			{
				struct heros t = harray[j];
				harray[j] = harray[j+1];
				harray[j + 1] = t;
			}
		}
	}
}

void printherosinfo(struct heros harray[], int len) 
{
	for (int i = 0; i < len; i++)
	{
		cout << harray[i].name << harray[i].age << harray[i].sex << endl;
	}
}

int main() 
{
	struct heros harray[5] = 
	{ {"����",23,"��"},
	{"����",22,"��"}, 
	{"�ŷ�",20,"��"}, 
	{"����",21,"��"}, 
	{"����",19,"Ů"} };

	cout << "����ǰ��" << endl;
	printherosinfo(harray, 5);

	herosortbyage(harray,5);

	cout << "�����" << endl;
	printherosinfo(harray, 5);

	system("pause");

	return 0;
}