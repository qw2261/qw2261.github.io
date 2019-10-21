

[TOC]

# C++笔记

1 scanf不支持string，或者说，不支持c++的类

2 $max(a, b) = \frac{a + b + abs(a - b)}{2}$

3 while (cir >> x, x): 先输入x再判断x

4 xcode-select --install: 装载g++

## 1. AcWing 78.  左旋字符串

分解操作，先整个翻转，再把前n-k个翻转，再把后k个翻转

```c++
string leftRotateString(string str, int n)
{
  int size = str.size();
  reverse(str.begin(), str.end());
  reverse(str.begin(), str.begin() + size - n);
  reverse(str.begin() + size - n, str.begin());
}
```

## 2. AcWing 87. 把字符串转换为整数

![2](./1. C++/Graphs/2.png)

```c++
class Solution {
public:
    int strToInt(string str) {
        int k = 0;
        while (k < str.size() && str[k] == ' ') k ++;

        long long number = 0;
        bool is_minus = false;
        if (str[k] == '+') k ++;
        else if (str[k] == '-') k ++, is_minus = true;

        while (k < str.size() && str[k] >= '0' && str[k] <= '9')
        {
            number = number * 10 + str[k] - '0';
            k ++;
        }

        if (is_minus) number *= -1;
        if (number > INT_MAX) return INT_MAX;
        else if (number < INT_MIN) return INT_MIN;
        else return (int)number;
    }
};
```

## 3. AcWing 84. 求1+2+…+n

![3](./1. C++/Graphs/3.png)

```c++
class Solution {
public:
    int getSum(int n) {
        int res = n;
        n > 0 && (res += getSum(n - 1));
        return res;
    }
};
```

##4. AcWing 28. 在O(1)时间删除链表结点

![4](./1. C++/Graphs/4.png)

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    void deleteNode(ListNode* node) {
        
        node->val = node->next->val;
        node->next = node->next->next;
    }
};
```

## 5. AcWing 36. 合并两个排序的链表

![5](./1. C++/Graphs/5.png)

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* merge(ListNode* l1, ListNode* l2) {
        ListNode* dummy = new ListNode(0);
        ListNode* cur = dummy;
        while (l1 != NULL && l2 != NULL)
        {
            if (l1->val < l2->val)
            {
                cur->next = l1;
                l1 = l1->next;
            }
            else
            {
                cur->next = l2;
                l2 = l2->next;
            }
            cur = cur->next;
        }
        if (l1 != NULL) cur->next = l1;
        else cur->next = l2;

        return dummy->next;
    }
};
```

##6. AcWing 35. 反转链表

![6](./1. C++/Graphs/6.png)

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
      ListNode* prev = nullptr;
      auto cur = head;
      
      while (cur)
      {
        auto next = cur->next;
        cur->next = prev;
        prev = cur;
        cur = next;
      }
      return prev;
    }
};
```

##7. AcWing 66. 两个链表的第一个公共结点

![7](./1. C++/Graphs/7.png)

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode *findFirstCommonNode(ListNode *headA, ListNode *headB) {
        auto p = headA, q = headB;
        while(p != q)
        {
            if (p) p = p->next;
            else p = headB;
            if (q) q = q->next;
            else q = headA;
        }
        return q;
    }
};
```

## 8. AcWing 29. 删除链表中重复的节点

![8](./1. C++/Graphs/8.png)

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* deleteDuplication(ListNode* head) {
        auto dummy = new ListNode(0);
        dummy->next = head;

        auto p = head;
        while (p->next)
        {
            auto q = p->next;
            while (q && p->next->val == q->val) q = q->next;
            if (p->next->next == q) p = p->next;
            else p->next = q;
        }
        return dummy->next;
    }
};
```

## 9. AcWing 68. 0到n-1中缺失的数字

![9](./1. C++/Graphs/9.png)

```c++
class Solution {
public:
    int getMissingNumber(vector<int>& nums) {
        if (nums.empty()) return 0;

        int l = 0, r = nums.size();

        while (l < r)
        {
            int mid = l + r >> 1;
            if (nums[mid] != mid) r = mid;
            else l = mid + 1;
        }
        return l;
    }
};
```

## 10. AcWing 17. 从尾到头打印链表

![10](./1. C++/Graphs/10.png)

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    vector<int> printListReversingly(ListNode* head) {
        vector<int> res;
        while (head)
        {
            res.push_back(head->val);
            head = head->next;
        }
        return vector<int>(res.rbegin(), res.rend());
    }
};
```

## 11. AcWing 20. 用两个栈实现队列

![11](./1. C++/Graphs/11.png)

```c++
class MyQueue {
public:
    /** Initialize your data structure here. */
    stack<int> stk, cache;
    MyQueue() {

    }
    void copy(stack<int> &a, stack<int> &b)
    {
        while (a.size())
        {
            b.push(a.top());
            a.pop();
        }
    }

    /** Push element x to the back of queue. */
    void push(int x) {
        stk.push(x);
    }

    /** Removes the element from in front of queue and returns that element. */
    int pop() {
        copy(stk, cache);
        int res = cache.top();
        cache.pop();
        copy(cache, stk);
        return res;
    }

    /** Get the front element. */
    int peek() {
        copy(stk, cache);
        int res = cache.top();
        copy(cache, stk);
        return res;
    }

    /** Returns whether the queue is empty. */
    bool empty() {
        return stk.empty();
    }
};

/**
 * Your MyQueue object will be instantiated and called as such:
 * MyQueue obj = MyQueue();
 * obj.push(x);
 * int param_2 = obj.pop();
 * int param_3 = obj.peek();
 * bool param_4 = obj.empty(); 
 */
```

## 12. AcWing 32. 调整数组顺序使奇数位于偶数前面

![12](./1. C++/Graphs/12.png)

```c++
class Solution {
public:
    void reOrderArray(vector<int> &array) {
        int l = 0, r = array.size() - 1;
        while (l < r)
        {
            while (l < r && array[l] % 2 != 0) l ++;
            while (l < r && array[r] % 2 == 0) r --;
            if (l < r) swap(array[l], array[r]);
        }

    }
};
```

## 13. AcWing 53. 最小的k个数

![13](./1. C++/Graphs/13.png)

```c++
class Solution {
public:
    vector<int> getLeastNumbers_Solution(vector<int> input, int k) {
        priority_queue<int> heap;
        for (auto x : input)
        {
            heap.push(x);
            if (heap.size() > k) heap.pop();
        }

        vector<int> res;
        while (heap.size()) res.push_back(heap.top()), heap.pop();
        reverse(res.begin(), res.end());


        return res;
    }
};
```

## 14. AcWing 75. 和为S的两个数字

![14](./1. C++/Graphs/14.png)

```c++
class Solution {
public:
    vector<int> findNumbersWithSum(vector<int>& nums, int target) {
        unordered_set<int> hash;
        for (int i = 0; i < nums.size(); i ++)
        {
            if (hash.count(target - nums[i])) return vector<int>{target - nums[i], nums[i]};
            hash.insert(nums[i]);
        }
    }
};
```

## 15. AcWing 40. 顺时针打印矩阵

![15](./1. C++/Graphs/15.png)

```c++
class Solution {
public:
    vector<int> printMatrix(vector<vector<int> > matrix) {
        vector<int> res;
        if (matrix.empty()) return res;
        int n = matrix.size(), m = matrix[0].size();
        vector<vector<bool>> st(n, vector<bool>(m, false));
        int dx[4] = {0, 1, 0, -1}, dy[4] = {1, 0, -1, 0};
        int x = 0, y = 0, d = 0;
        for (int k = 0; k < n * m; k ++)
        {
            res.push_back(matrix[x][y]);
            st[x][y] = true;
            int a = x + dx[d], b = y + dy[d];
            if (a < 0 || a >= n || b < 0 || b >= m || st[a][b])
            {
                d = (d + 1) % 4;
                a = x + dx[d], b = y + dy[d];
            }
            x = a, y = b;
        }
        return res;
    }
};
```

# 数据结构和算法







# 算法基础课

## 第一章 基础算法（一）

### 快排

```c++
#include <iostream>

using namespace std;

const int N = 1e6 + 10;

int n;
int q[N];


void quick_sort(int q[], int l, int r)
{
  if (l >= r) return;
  
  int x = q[l], i = l - 1, j = r + 1;
  while (i < j)
  {
    do i ++; while (q[i] < x);
    do j --; while (q[j] > x);
    if (i < j) swap(q[i], q[j]);
  }
  
  quick_sort(q, l, j), quick_sort(q, j + 1, r);
  // quick_sort(q, l, i - 1), quick_sort(q, i, r);
  // 之前的x变为q[r], q[(l + r + 1) / 2], 边界问题
}

int main()
{
  scanf("%d", &n);
  
  for (int i = 0; i < n; i ++) scanf("%d", &q[i]);
  
  quick_sort(q, 0, n - 1);
  
  for (int i = 0; i < n; i ++) printf("%d", q[i]);
  
  return 0;
  
}
```

### 归并排序

```c++
#include <iostream>

using namespace std;

const int N = 1000010;
int n;
int q[N], tmp[N];

void merge_sort(int q[], int l, int r)
{
  if (l >= r) return;
  
  int mid = l + r >> 1;
  
  merge_sort(q, l , mid), merge_sort(q, mid + 1, r);
  
  int k = 0, i = 1, j = mid + 1;
  
  while (i <= mid && j <= r)
    if (q[i] <= q[j]) tmp[k ++] = q[i ++];
  	else tmp[k ++] = q[j ++];
  
  while (i <= mid) tmp[k ++] = q[i ++];
  while (j <= r) tmp[k ++] = q[j ++];
  
  for (i = 1, j = 0; i <= r; i ++, j ++) q[i] = tmp[j];
}

int main()
{
  scanf("%d", n);
  
  for (int i = 0; i < n; i ++) scanf("%d", &q[i]);
  
  merge_sort(q, 0, n - 1);
  
  for (int i = 0; i < n; i ++) printf("%d", q[i]);
  
  return 0;
}
```

### 整数二分

```c++
bool check(int x) {/* ... */} // 检查x是否满足某种性质

// 区间[l, r]被划分成[l, mid]和[mid + 1, r]时使用：
int bsearch_1(int l, int r)
{
    while (l < r)
    {
        int mid = l + r >> 1;
        if (check(mid)) r = mid;    // check()判断mid是否满足性质
        else l = mid + 1;
    }
    return l;
}
// 区间[l, r]被划分成[l, mid - 1]和[mid, r]时使用：
int bsearch_2(int l, int r)
{
    while (l < r)
    {
        int mid = l + r + 1 >> 1;
        if (check(mid)) l = mid;
        else r = mid - 1;
    }
    return l;
}
```

```c++
// AcWing 789, 数的范围
#include <iostream>

using namespace std;

const int N = 100010;

int n, m;
int q[N];

int main()
{
  scanf("%d%d", &n, &m);
  for (int n = 0; i < n; i ++) scanf("%d", &q[i]);
  
  while (m --)
  {
    int x;
    scanf("%d", &x);
    
    int l = 0, r = n - 1;
    while (l < r)
    {
      int mid = l + r >> 1;
      if (q[mid >= x]) r = mid;
      else l = mid + 1;
    }
    if (q[l] != x) cout << "-1 -1" << endl;
    else
    {
      cout << l << ' ';
      
      int l = 0, r = n - 1;
      while (l < r)
      {
        int mid = l + r + 1 >> 1;
        if (q[mid] <= x) l = mid;
        else r = mid - 1;
      }
      
      cout << l << endl;
    }
  }
  
  return 0;
}
```

### 浮点数二分

```c++
bool check(double x) {/* ... */} // 检查x是否满足某种性质

double bsearch_3(double l, double r)
{
    const double eps = 1e-6;   // eps 表示精度，取决于题目对精度的要求
    while (r - l > eps)
    {
        double mid = (l + r) / 2;
        if (check(mid)) r = mid;
        else l = mid;
    }
    return l;
}
```

```c++
// Sqrt
#include <iostream>

using namespace std;

int main()
{
  double x;
  cin >> x;
  
  double l = 0, r = x;
  while (r - l > 1e-8) //要求高一些
  {
    doubel mid = (l + r) / 2;
    if (mid * mid >= x) r = mid;
    else l = mid;
  }
  
  printf("%lf\n", l);
  
  return 0;
}
```

## 第一章 基础算法（二）

### 高精度加法

```c++
#include <iostream>
#include <vector>

using namespace std;

// C = A + B
vector<int> add(vector<int> &A, vector<int> &B)
{
  vector<int> C;
  
  int t = 0;
  for (int i = 0; i < A.size() || i < B.size(); i ++)
  {
    if (i < A.size()) t += A[i];
    if (i < B.size()) t += B[i];
    C.push_back(t % 10);
    t /= 10;
  }
  
  if (t) C.push_back(1);
  return C;
}

int main()
{
  string a, b;
  vector<int> A, B;
  
  cin >> a >> b; // a = "123456"
  for (int i = a.size() - 1; i >= 0; i --) A.push_back(a[i] - '0'); // A = [6, 5, 4, 3, 2, 1]
  for (int i = b.size() - 1; i >= 0; i --) B.push_back(b[i] - '0');
  
  auto C = add(A, B);
  
  for (int i = C.size(); i >= 0; i --) printf("%d", C[i]);
  
  return 0;
}
```

```c++
vector<int> add(vector<int> &A, vector<int> &B)
{
    if (A.size() < B.size()) return add(B, A);
    
    vector<int> C;
    int t = 0;
    for (int i = 0; i < A.size(); i ++ )
    {
        t += A[i];
        if (i < B.size()) t += B[i];
        C.push_back(t % 10);
        t /= 10;
    }
    
    if (t) C.push_back(t);
    return C;
}
```

### 高精度减法（假定A和B都是正数）->可以转换成绝对值相减，和相加的情况

```c++
#include <iostream>
#include <vector>

using namespace std;

// 判断A是否大于等于B
bool cmp(vector<int> &A, vector<int> &B)
{
  if (A.size() != B.size()) return A.size() > B.size();
  for (int i = A.size() - 1; i >= 0; i --)
  {
    if (A[i] != B[i])
      return A[i] > B[i];
  }
  return true;
}


// C = A - B
vector<int> sub(vector<int> &A, vector<int> &B)
{
  vector<int> C;
  for (int i = 0, t = 0; i < A.size(); i ++)
  {
    t = A[i] - t;
    if (i < B.size()) t -= B[i];
    C.push_back((t + 10) % 10);
    if (t < 0) t = 1;
    else t = 0;
  }
  
  while (C.size() > 1 && C.back() == 0) C.pop_back(); 
  //去掉多余的0，前导0
  
  return C;
}

int main()
{
  string a, b;
  vector<int> A, B;
  
  cin >> a >> b; // a = "123456"
  for (int i = a.size() - 1; i >= 0; i --) A.push_back(a[i] - '0'); // A = [6, 5, 4, 3, 2, 1]
  for (int i = b.size() - 1; i >= 0; i --) B.push_back(a[i] - '0');
  
  if (cmp(A, B))
  {
    auto C = sub(A, B);
    for (int i = C.size(); i >= 0; i --) printf("%d", C[i]);
  }
  else
  {
    auto C = sub(B, A);
    printf("-");
    for (int i = C.size(); i >= 0; i --) printf("%d", C[i]);
  }
  
  
  
  return 0;
}
```

### 高精度乘法

```c++
#include <iostream>
#include <vector>

using namespace std;


// C = A * b
vector<int> mul(vector<int> &A, int b)
{
  vector<int> C;
  
  int t = 0;
  for (int i = 0; i < A.size() || t; i ++)
  {
    if (i < A.size()) t += A[i] * b;
    C.push_back(t % 10);
    t /= 10;
  }
  
  return C;
}



int main()
{
  string a;
  int b;
  cin >> a >> b;
  
  vector<int> A;
  for (int i = a.size() - 1; i >= 0; i --) A.push_back(a[i] - '0');
  
  auto C = mul(A, b);
  
  for (int i = C.size() - 1; i >= 0; i --) printf("%d", C[i]);
  
  return 0;
}
```

### 高精度除法

```c++
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;


// C = A / b，商是C，余数是r
vector<int> div(vector<int> &A, int b, int &r)
{
  vector<int> C;
  
  r = 0;
  for (int i = A.size() - 1; i >= 0; i --)
  {
    r = r * 10 + A[i];
    C.push_back(r / b);
    r = r % b;
  }
  
  reverse(C.begin(), C.end());
  
  while (C.size() >  1 && C.back() == 0) C.pop_back();
  
  return C;
}



int main()
{
  string a;
  int b;
  cin >> a >> b;
  
  vector<int> A;
  for (int i = a.size() - 1; i >= 0; i --) A.push_back(a[i] - '0');
  
  int r;
  auto C = div(A, b, r);
  
  for (int i = C.size() - 1; i >= 0; i --) printf("%d", C[i]);
  cout << endl << r << endl;
  
  return 0;
}
```

### 前缀和

```c++
// AcWing 795. 前缀和
#include <iostream>

using namespace std;

const int N = 100010;

int n, m;
int a[N], s[N];

int main()
{
  ios:sync_with_stdio(false);
  // 让cin和标准输入输出不同步，提高cin读取速度
  
  scanf("%d%d", &n, &m);
  for (int i = 1; i <= n; i ++) scanf("%d", &a[i]);
  
  for (int i = 1; i <= n; i ++) s[i] = s[i - 1] + a[i]; // 前缀和的计算
  
  while (m --)
  {
    int l, r;
    scanf("%d%d", &l, &r);
    printf("%d\n", s[r] - s[l - 1]); // 区间和的计算
  }
  
  
  return 0;
}
```

### 二维前缀和

```c++
// AcWing 796. 子矩阵的和
#include <iostream>

const N = 1010;

int n, m, q;
int a[N][N], s[N][N];

int main()
{
  scanf("%d%d%d", &n, &m, &q);
  
  for (int i = 1; i <= n; i ++)
  {
    for (int j = 1; j <= m; j ++)
    {
      scanf("%d", &a[i][j]);
    }
  }
  
  for (int i = 1; i <= n; i ++)
  {
    for (int j = 1; j <= m; j ++)
    {
      s[i][j] = s[i - 1][j] + s[i][j - 1] - s[i - 1][j - 1] + a[i][j]; // 前缀和
    }
  }
  
  while (q --)
  {
    int x1, y1, x2, y2;
    scanf("%d%d%d%d", &x1, &y1, &x2, &y2);
    printf("%d\n", s[x2][y2] - s[x1 - 1][y2] - s[x2][y1 -1] + s[x1 - 1][y1 - 1]); // 算子矩阵的和
  }
  
  return 0;
}
```

### 差分

```c++
// AcWing 797. 差分

#include <iostream>

using namespace std;

const int N = 100010;

int n, m;
int a[N], b[N];

void insert(int l, int r, int c)
{
  b[l] += c;
  b[r + 1] -= c;
}


int main()
{
  scanf("%d%d", &n, &m);
  for (int i = 1; i < n; i ++) scanf("%d", &a[i]);
  
  for (int i = 1; i < n; i ++) insert(i, i, a[i]);
  
  while (m --)
  {
    int l, r, c;
    scanf("%d%d%d", &l, &r, &c);
    insert(l, r, c);
  }
  
  for (int i = 1; i <= n; i ++) b[i] += b[i - 1];
  
  for (int i = 1; i <= n; i ++) printf("%d", b[i]);
  
  
  
  return 0;
}

```

### 二维差分

```c++
// AcWing 798. 差分矩阵
#incude <iostream>

using namespace std;

const int N = 1010;

int n, m, q;
int a[N][N], b[N][N];

void insert(int x1, int y1, int x2, int y2)
{
  b[x1][y1] += c;
  b[x2 + 1][y1] -= c;
  b[x1][y2 + 1] -= c;
  b[x2 + 1][y2 + 1] += c;
}


int main()
{
  scanf("%d%d", &n, &m, &q);
  
  for (int i = 1; i <= n; i ++)
  {
    for (int j = 1; j <= m; j ++)
    {
      scanf("%d", &a[i][j]);
    }
  }
  
  for (int i = 1; i <= n; i ++)
  {
    for (int j = 1; j <= m; j ++)
    {
      insert(i, j, i, j, a[i][j]);
    }
  }
  
  while (q --)
  {
    int x1, y1, x2, y2, c;
    cin >> x1 >> y1 >> x2 >> y2 >> c;
    insert(x1, y1, x2, y2, c);
  }
  
  for (int i = 1; i <= n; i ++)
  {
    for (int j = 1; j <= m ; j ++)
    {
      b[i][j] += b[i - 1][j] + b[i][j - 1] - b[i - 1][j - 1];
    }
  }
  
  for (int i = 1; i <= n; i ++)
  {
    for (int j = 1; j <= m; j ++) printf("%d", b[i][j]);
    puts("");
  }
  
  return 0;
}
```

## Week 1. 习题课

### 1. AcWing 786. 第k个数

![](/Users/wangqi/Desktop/Coding/2. 算法基础课/Graphs/786.png)

```c++
#include <iostream>

using namespace std;

const int N = 1e5 + 10;

int n, k;
int q[N];

int quick_sort(int l, int r, int k)
{
  if (l == r) return q[l];
  
  int q[l], i = l - 1, j = r + 1;
  
  while (i < j)
  {
    while (q[++ i] < x);
    while (q[-- j] > x);
    if (i < j) swap(q[i], q[j]);
  }
  
  int sl = j - 1 + 1;
  if (k <= sl) return quick_sort(l, j, k);
  
  return quick_sort(j + 1, r, k - sl);
}

int main()
{
  cin >> n >> k;
  
  for (int i = 0; i < n; i ++) cin >> q[i];
  
  cout<< quick_sort(0, n - 1, k) << endl;
  
  return 0;
}
```

### 2. AcWing 788. 逆序对的数量

![](/Users/wangqi/Desktop/Coding/2. 算法基础课/Graphs/788.png)

```c++
#inlcude <iostream>

using namespace std;

typedef long long LL;

const int N = 100010;

int n, q[N], tmp[N];

LL merge_sort(int l, int r)
{
  if (l >= r) return 0;
  
  int mid = l + r >> 1;
  LL res = merge_sort(l, mid) + merge_sort(mid + 1, r);
  
  // 归并过程
  int k = 0, i = l, j = mid + 1;
  while (i <= mid && j <= r)
  {
    if (q[i] < q[j]) tmp[k ++] = q[i ++];
    else
    {
      tmp[k ++] = q[j ++];
      res += mid - i + 1;
    }
  }
  
  // 物归原主
  while (i <= mid) tmp[k ++] = q[i ++];
  while (j <= r) tmp[k ++] = q[j ++];
  
  for (int i = l, j = 0; i <= r; i ++, j ++) q[i] = tmp[j];
  return res;
}

int main()
{
  cin >> n;
  for (int i = 0; i < n; i ++) cin >> q[i];
  cout << merge_sort(0, n - 1);
  return 0;
}
```

### 3. AcWing 790. 数的三次方根

![](/Users/wangqi/Desktop/Coding/2. 算法基础课/Graphs/790.png)

```c++
#include <iostream>

using namespace std;

int main()
{
  double x;
  cin >> x;
  
  double l = -10000, r = 10000;
  while (r - l > 1e-8)
  {
    double mid = (l + r) / 2;
    if (mid * mid * mid >= x) r = mid;
    else l = mid;
  }
  
  printf("%lf\n", l);
  return 0;
}
```

### 4. AcWing 795. 前缀和

![](/Users/wangqi/Desktop/Coding/2. 算法基础课/Graphs/795.png)

```c++
#include <iostream>

using namespace std;

const int N = 100010;

int n, m;
int a[N], s[N];

int main()
{
  cin >> n >> m;
  for (int i = 1; i <= n; i ++) cin >> a[i];
  
  for (int i = 1; i <= n; i ++) s[i] = s[i - 1] + a[i];
  
  while (m --)
  {
    int l, r;
    cin >> l >> r;
    cout << s[r] - s[l - 1] << endl;
  }
  
  return 0;
}
```

### 5. AcWing 796. 子矩阵的和

![](/Users/wangqi/Desktop/Coding/2. 算法基础课/Graphs/796.png)

```c++
#include <iostream>

using namespace std;

const int N = 1010;

int n, m, q;
int a[N][N], s[N][N];

int main()
{
  scanf("%d%d%d", &n, &m, &q);
  
  for (int i = 1; i <= n; i ++)
    for (int j = 1; j <= m; j ++)
      scanf("%d", &a[i][j]);
  
  for (int i = 1; i <= n; i ++)
    for (int j = 1; j <= m; j ++)
      s[i][j] = s[i - 1][j] + s[i][j - 1] - s[i - 1][j - 1] + a[i][j];
  
  while (q --)
  {
    int x1, y1, x2, y2;
    scanf("%d%d%d%d", &x1, &y1, &x2, &y2);
    printf("%d\n", s[x2][y2] - s[x2][y1 - 1] - s[x1 - 1][y2] + s[x1 - 1][y1 - 1]);
  }
  
  return 0;
}
```

### 6. AcWing 797. 差分

![](/Users/wangqi/Desktop/Coding/2. 算法基础课/Graphs/797.png)

```c++
#include <iostream>

using namespace std;

const int N = 100010;

int n, m;
int a[N], b[N];

void insert(int l, int r, int c)
{
  b[l] += c;
  b[r + 1] -= c;
}

int main()
{
  cin >> n >> m;
  
  for (int i = 1; i <= n; i ++) cin >> a[i];
  
  for (int i = 1; i <= n; i ++) insert(i, i, a[i]);
  
  while (m --)
  {
    int l, r, c;
    cin >> l >> r >> c;
    insert(l, r, c);
  }
  
  for (int i = 1; i <= n; i ++) a[i] = a[i - 1] + b[i];
  
  for (int i = 1; i <= n; i ++) printf("%d ", a[i]);
  puts("");
  
  return 0;
}
```

### 7. AcWing 798. 差分矩阵

![](/Users/wangqi/Desktop/Coding/2. 算法基础课/Graphs/798.png)

```c++
#include <iostream>
using namespace std;

const int N = 1010;

int n, m, q;
int a[N][N], b[N][N];

void insert(int x1, int y1, int x2, int y2, int c)
{
  b[x1][y1] += c;
  b[x2 + 1][y1] -= c;
  b[x1][y2 + 1] -= c;
  b[x2 + 1][y2 + 1] += c;
}

int main()
{
  scanf("%d%d%d", &n, &m, &q);
  
  for (int i = 1; i <= n; i ++)
    for (int j = 1; j <= m; j ++)
      scanf("%d", &a[i][j]);
  
  for (int i = 1; i <= n; i ++)
    for (int j = 1; j <= m; j ++)
      insert(i, j, i, j, a[i][j]);
  
  while (q --)
  {
    int x1, y1, x2, y2, c;
    scanf("%d%d%d%d%d", &x1, &y1, &x2, &y2, &c);
    insert(x1, y1, x2, y2, c);
  }
  
  for (int i = 1; i <= n; i ++)
    for (int j = 1; j <= m; j ++)
      a[i][j] = a[i - 1][j] + a[i][j - 1] - a[i - 1][j - 1] + b[i][j];
  
  for (int i = 1; i <= n; i ++)
  {
    for (int j = 1; j <-= m; j ++) printf("%d", a[i][j]);
    puts("");
  }
  
  return 0;
}
```

### 8. AcWing 789. 数的范围

![](/Users/wangqi/Desktop/Coding/2. 算法基础课/Graphs/789.png)

```c++
#include <iostream>

using namespace std;

const int N = 1e5 + 10;

int q[N], n, m;

int main()
{
    cin >> n >> m;

    for (int i = 0; i < n; i ++) cin >> q[i];

    while (m --)
    {
        int x; cin >> x;

        int l = 0, r = n - 1;
        while (l < r)
        {
            int mid = l + r >> 1;
            if (q[mid] >= x) r = mid;
            else l = mid + 1;
        }

        if (q[l] != x) cout << "-1 -1" << endl;
        else
        {
            cout << l << " ";

            int l = 0, r = n - 1;
            while (l < r)
            {
                int mid = l + r + 1 >> 1;
                if (q[mid] > x) r = mid - 1;
                else l = mid;
            }
            cout << l << endl;
        }
    }
    return 0;
}
```

## 第一章 基础算法（三）

### 双指针算法

```c++
for (int i = 0, j = 0; i < n; i ++)
{
  for (int j < i && check(i, j)) j ++;
}

// 暴力
for (int i = 0; i < n; i ++)
{
  for (int j =0; j < n; j ++)
}
```

核心思想：

把暴力的$O(n^2)$ -> $O(n)$

```c++
// abc def ghi，输出abc ghi
#include <iostream>
#include <string.h>

using namespace std;

int main()
{
  char str[1000];
  gets(str);
  
  int n = strlen(str);
  
  for (int i = 0; i < n; i ++)
  {
    int j = i;
    while (j < n && str[j] != ' ') j ++;
    
    for (int k = i; k < j; k ++) cout << str[k];
    cout << endl;
    
    i = j;
  }
}
```

```c++
// AcWing 799. 最长连续不重复子序列

// 思路
// j: j往左最远能到什么地方

// Naive, O(n^2)
for (int i = 0; i < n; i ++)
{
  for (int j = 0; j <= i; j ++)
  {
    if (check(j, i))
    {
      res = max(res, i - j + 1);
    }
  }
}

// 双指针
for (int i = 0, j = 0; i < n; i ++)
{
  while (j <= i && check(j, i)) j ++;
  res = max(res, i - j + 1)
}
```

```c++
// AcWing 799. 最长连续不重复子序列

// 解答

#include <iostream>

using namespace std;

const int N = 100010;

int n;
int a[N], s[N];

int main()
{
  cin >> n;
  for (int i = 0; i < n; i ++) cin >> a[i];
  
  int res = 0;
  
  for (int i = 0, j = 0; i < n; i ++)
  {
    s[a[i]] ++;
    while (s[a[i]] > 1)
    {
      s[a[j]] --;
      j ++;
    }
    res = max(res, i - j + 1);
  }
  
  cout << res << endl;
  
  return 0;
}

```

### 位运算

```c++
// 求n的第k位数字: 
n >> k & 1
```

```c++
// 返回n的最后一位1：
lowbit(n) = n & -n
-x = ~x + 1
```

```c++
// AcWing 801. 二进制中1的个数

#include <iostream>

using namespace std;

int lowbit(int x)
{
  return x & -x;
}

int main()
{
  int n;
  cin >> n;
  while (n --)
  {
    int x;
    cin >> x;
    
    int res = 0;
    while (x) x -= lowbit(x), res ++; //每次减去x的最后一位1
    
    cout << res << endl;
  }
}
```

### 离散化

```c++
vector<int> alls; // 存储所有待离散化的值
sort(alls.begin(), alls.end()); // 将所有值排序
alls.erase(unique(alls.begin(), alls.end()), alls.end());	// 去掉重复元素
	
// 二分求出x对应的离散化的值
int find(int x) // 找到第一个大于等于x的位置
{
  int l = 0, r = alls.size() - 1;
  while (l < r)
  {
    int mid = l + r >> 1;
    if (alls[mid] >= x) r = mid;
    else l = mid + 1;
  }
  return r + 1; // 映射到1, 2, ...n; 不加1则从过0开始映射
}
```

```c++
// AcWing 802. 区间和
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

typedef pair<int, int> PII;

const int N = 300010;
int n, m;
int a[N], s[N];

vector<int> alls;
vecor<PII> add, query;


vector<int>::iterator unique(vector<int> &a)
{
    int j = 0;
    for (int i = 0; i < a.size(); i ++ )
        if (!i || a[i] != a[i - 1])
            a[j ++ ] = a[i];
    // a[0] ~ a[j - 1] 所有a中不重复的数

    return a.begin() + j;
}

int find(int x)
{
  int l = 0, r = alls.size() - 1;
  while (l < r)
  {
    int mid = l + r >> 1;
    if (alls[mid] >= x) r = mid;
    else l = mid + 1;
  }
  return r + 1;
}

int main()
{
  cin >> n >> m;
  
  for (int i = 0; i < n; i ++)
  {
    int x, c;
    cin >> x >> c;
    add.push_back({x, c});
    
    alls.push_back(x);
  }
  
  for (int i = 0; i < m; i ++)
  {
    int l, r;
    cin >> l >> r;
    query.push_back({l, r});
    
    alls.push_back(l);
    alls.push_back(r);
  }
  
  // 去重
  sort(alls.begin(), alls.end());
  alls.erase(unique(alls.begin(), alls.end()), alls.end());
  
  
  for (auto item : add)
  {
    int x  = find(item.first);
    a[x] += item.second;
  }
  
  // 预处理前缀和
  for (int i = 1; i <= alls.size(); i ++) s[i] = s[i - 1] + a[i];
  
  // 处理询问
  for (auto item : query)
  {
    int l = find(item.first), r = find(item.second);
    cout << s[r] - s[l - 1] << endl;
  }
  
  
  return 0;
}


```

```c++
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

typedef pair<int, int> PII;
const int N = 300010;
int a[N], s[N];
int n, m;

vector<int> alls;
vector<PII> add, query;

int find(int x)
{
    int l = 0, r = alls.size() - 1;
    while(l < r)
    {
        int mid = l + r >> 1;
        if(alls[mid] >= x) r = mid;
        else l = mid + 1;
    }
    return r + 1;
}
vector<int>:: iterator unique(vector<int> &a)
{
    int j = 0;
    for(int i = 0; i < a.size(); i ++)
        if(!i || a[i] != a[i - 1])
            a[j ++ ] = a[i];
    return a.begin() + j;
}

int main()
{

    cin >> n >> m;

    for(int i = 0; i < n; i ++ )
    {
        int x, c;
        cin >> x >> c;
        add.push_back({x, c});

        alls.push_back(x);
    }

    for(int i = 0; i < m; i ++ )
    {
        int l, r;
        cin >> l >> r;
        query.push_back({l, r});

        alls.push_back(l);
        alls.push_back(r);
    }

    sort(alls.begin(), alls.end());
    alls.erase(unique(alls), alls.end());

    for(auto item : add)
    {
        int x = find(item.first);
        a[x] += item.second;
    }

    for(int i = 1; i <= alls.size(); i ++ ) s[i] = s[i - 1] + a[i];

    for(auto item : query)
    {
        int l = find(item.first), r = find(item.second);
        cout << s[r] - s[l - 1] << endl;
    }

    return 0;
}


作者：此题有解否
链接：https://www.acwing.com/solution/AcWing/content/2321/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```



### 区间合并

```c++
// 将所有存在交集的区间合并
void merge(vector<PII> &segs)
{
  vector<PII> res;

  sort(segs.begin(), segs.end());

  int st = -2e9, ed = -2e9;
  for (auto seg : segs)
    if (ed < seg.first)
    {
      if (st != -2e9) res.push_back({st, ed});
      st = seg.first, ed = seg.second;
    }
  else ed = max(ed, seg.second);

  if (st != -2e9) res.push_back({st, ed});

  segs = res;
}
```

```c++
// AcWing 803. 区间合并

#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;


typedef pair<int, int> PII;

const int N = 100010;

int n;
vector<PII> segs;

void merge(vector<PII> &segs)
{
  vector<PII> res;
  sort(segs.begin(), segs.end());
  
  int st = -2e9, ed = -2e9;
  for (auto seg : segs)
  {
    if (ed < seg.first)
    {
      if (st != -2e9) res.push_back({st, ed});
      st = seg.first, ed = seg.second;
      
    }
    else ed = max(ed, seg.second);
  }
  if (st != -2e9) res.push_back({st, ed});
  
  segs = res;
}

int main()
{
  cin >> n;
  
  for (int i = 0; i < n; i ++)
  {
    int l, r;
    cin >> l >> r;
    segs.push_back({l, r});
  }
  merge(segs);
  
  cout << segs.size() << endl;
  
  return 0;
}
```











# 算法提高课





# Leetcode刷题

### 1. [Two Sum](https://leetcode.com/problems/two-sum/submissions/)

```c++
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        vector<int> res;
        unordered_map<int, int> hash;
        for (int i = 0; i < nums.size(); i ++)
        {
            int another = target - nums[i];
            if (hash.count(another))
            {
                res = vector<int>({hash[another], i});
                break;
            }
            hash[nums[i]] = i;
        }
        return res;
    }
};
```







# Leetcode周赛及各大比赛题目







# 面试及剑指offer





# 技术总结









