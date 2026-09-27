#include <iostream>
using namespace std;
int main(){ int a=10; int *p; p=&a; cout<<"指针p为："<<p<<endl; cout<<"p指向的值："<<*p<<endl; *p=1000; cout<<a<<endl; return 0;}
