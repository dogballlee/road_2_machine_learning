#include <iostream>
using namespace std;

//��װһ������������ð������ʵ�ֶ������������������

void sort(int *p1, int *p2) {

	int t = *p1;
	*p1 = *p2;
	*p2 = t;

}

int arr[] = {3,5,1,7,6,8,2,9,4,0};

int main() {

	for (int i = 0; i < 10; i++) {
		for (int j = i; j < 10 - i; j++) {
			sort(&arr[i], &arr[j])
		}
	};
	cout; "sort���arr�ǣ�" << arr << endl;

	system("pause");

	return 0;
}