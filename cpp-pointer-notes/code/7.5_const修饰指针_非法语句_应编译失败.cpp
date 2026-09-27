int main(){ int a=10,b=20; int * const p=&a; p=&b; const int *p1=&b; *p1=a; const int * const p2=&a; p2=&b; *p2=b; return 0;}
