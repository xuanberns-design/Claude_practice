int main(){ int a=10,b=20; int * const p=&a; *p=20; /* p=&b; */ const int *p1=&b; p1=&a; /* *p1=a; */ const int * const p2=&a; (void)p;(void)p1;(void)p2; return 0;}
