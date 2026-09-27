#include <iostream>
using namespace std;
void BubbleSort(int *arr, int n){ for (int i = 0; i < n - 1; i++){ for (int j = 0; j < n - i - 1; j++){ if (arr[j] > arr[j + 1]){ int temp = arr[j]; arr[j] = arr[j + 1]; arr[j + 1] = temp; } } } }
void PrintArray(const int *arr, int n){ for (int i = 0; i < n; i++) cout << arr[i] << " "; cout << endl; }
int main(){ int arr[] = {3, 7, 9, 12, 5, 1}; int len = sizeof(arr) / sizeof(arr[0]); BubbleSort(arr, len); PrintArray(arr, len); return 0; }
