#include <iostream>
using namespace std;
int main(){ int a=10; int *p=&a; int b=sizeof(p); cout<<"int*字节数："<<b<<endl;
cout<<sizeof(char*)<<sizeof(double*)<<sizeof(int)<<sizeof(double)<<endl; return 0;}
