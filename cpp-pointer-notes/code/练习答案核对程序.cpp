#include <iostream>
using namespace std;
struct Node { int val; Node *next; };
void f(int *q){ *q = *q * 2; q = nullptr; }
void show(int *a){ cout << "Q29 show " << sizeof(a) << endl; }
void minmax(const int *arr,int n,int *pmin,int *pmax){ *pmin=arr[0]; *pmax=arr[0]; for(int i=1;i<n;i++){ if(arr[i]<*pmin) *pmin=arr[i]; if(arr[i]>*pmax) *pmax=arr[i]; } }
int cmp=0,sw=0;
void bs(int *arr,int n,bool trace,bool flag){ cmp=sw=0; for(int i=0;i<n-1;i++){ bool s=false; for(int j=0;j<n-i-1;j++){ cmp++; if(arr[j]>arr[j+1]){int t=arr[j];arr[j]=arr[j+1];arr[j+1]=t;sw++;s=true;} if(trace&&i==0){cout<<"  j="<<j<<":";for(int k=0;k<n;k++)cout<<" "<<arr[k];cout<<endl;} } if(trace){cout<<" round"<<i+1<<":";for(int k=0;k<n;k++)cout<<" "<<arr[k];cout<<endl;} if(flag&&!s)break;} }
int main(){
 {int a=10,b=20;int *p=&a;*p=30;p=&b;*p=40;cout<<"Q21 "<<a<<" "<<b<<" "<<*p<<endl;}
 {int arr[]={1,2,3,4,5};int *p=arr+1;cout<<"Q26 "<<*(p+2)<<endl;}
 {int a=5;int *p=&a;f(p);cout<<"Q27 a="<<a<<" p==&a "<<(p==&a)<<endl;}
 {int arr[10];cout<<"Q29 main "<<sizeof(arr)<<endl;show(arr);}
 {int arr[]={10,20,30,40};int *p=arr;*(p+1)+=5;p+=2;*p=*(p-1)+1;cout<<"Q31";for(int x:arr)cout<<" "<<x;cout<<" idx "<<(p-arr)<<endl;}
 {int a[]={1,2,3,4,5,6};bs(a,6,false,false);cout<<"Q32 sorted cmp="<<cmp<<" sw="<<sw<<endl;int b[]={6,5,4,3,2,1};bs(b,6,false,false);cout<<"Q32 rev cmp="<<cmp<<" sw="<<sw<<endl;}
 {double arr[4];double *q=&arr[3];cout<<"Q33 off="<<((char*)q-(char*)arr)<<" diff="<<(q-arr)<<" size="<<sizeof(q)<<endl;}
 {int a[]={4,9,1,7};int mn,mx;minmax(a,4,&mn,&mx);cout<<"Q34 "<<mn<<" "<<mx<<endl;}
 {cout<<"Q35"<<endl;int a[]={3,7,9,12,5,1};bs(a,6,true,false);int c[]={1,2,3,4,5,6};bs(c,6,false,true);cout<<"Q35 flag cmp="<<cmp<<endl;}
 {cout<<"Q25"<<endl;int a[]={6,2,8,4,1};bs(a,5,true,false);}
 {Node n3={3,nullptr};Node n2={2,&n3};Node n1={1,&n2};Node *p=&n1;p=p->next;cout<<"Q36 "<<p->val<<endl;}
 {int arr[]={2,4,6,8};int *p=arr;cout<<"Q24";for(int i=0;i<4;i++){cout<<" "<<*p;p++;}cout<<" end==arr+4 "<<(p==arr+4)<<endl;}
}
