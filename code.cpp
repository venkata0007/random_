#include <bits/stdc++.h>
using namespace std;

// max network rank
class Solution {
public:
    int maximalNetworkRank(int n,vector<vector<int>>& edges) {
    vector<int> degree(n, 0);

    // Calculate the degrees of nodes
    for (const vector<int>& edge : edges) {
        degree[edge[0]]++;
        degree[edge[1]]++;
    }

    int max_degree = 0;
    int count_max = 0;
    int sec_max = 0;
    int count_sec = 0;
    for(int i =0 ;i<n;i++)
    {
        if(degree[i]>max_degree)
        {
            max_degree = degree[i];
        }
    }
    for(int i =0;i<n;i++)
    {
        if(degree[i] == max_degree)
        {
            count_max++;
        }
        if(degree[i]>sec_max && degree[i] != max_degree)
        {
            sec_max = degree[i];
        }
    }
    for(int i =0 ;i<n;i++)
    {
        if(degree[i] == sec_max)
        {
            count_sec++;
        }
    }
    int maxRank = 0;
    if(count_max >1)
    {
        int sum =0;
        for(int i =0;i<edges.size();i++)
        {
            int u = edges[i][0],v = edges[i][1];
            if(degree[u] == max_degree && degree[v] == max_degree)
            {
                sum+=1;
            }
        }
        if(sum == (count_max*(count_max-1))/2) return  2*max_degree-1;
        return 2*max_degree;
    }
    // if(count_sec>=1)
    // {
    int sum = 0;
    for(int i =0;i<edges.size();i++)
    {
        int u = edges[i][0],v = edges[i][1];
        if((degree[u] == sec_max && degree[v] == max_degree)||(degree[u] == max_degree && degree[v] == sec_max))
        {
            sum++;
        }
    }
    if(sum == count_max*count_sec)
    {
        return max_degree + sec_max-1;
    }
    return max_degree +sec_max;
}

};

// next permutation
class Solution {
public:
    void nextPermutation(vector<int>& nums) {
        int i = nums.size()-1;
        while((i>0)&&(nums[i-1]>=nums[i]))
        {
            i--;
        }
        if(i==0)
        {
            reverse(nums.begin(),nums.end());
            return;
        }
        for(int j =nums.size()-1;j>=i;j--)
        {
            if(nums[j]>nums[i-1])
            {
                int temp = nums[i-1];
                nums[i-1] = nums[j];
                nums[j] = temp;
                break;
            }
        }
        reverse(nums.begin()+i,nums.end());
        return;
    }
};
// kadane's algorithm
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int curr_sum = 0;
        int sum = nums[0];
        for(int i  =0;i<nums.size();i++)
        {
            curr_sum += nums[i];
            
            sum = max(sum,curr_sum);
            if(curr_sum<0)
            {
                curr_sum  = 0;
            }
        }
        return sum;
    }
};
// repeateed substring pattern
class Solution {
public:
    bool repeatedSubstringPattern(string s) {
        int n = s.size();
        for(int i =1;i<=n/2;i++)
        {
            if(n%i == 0)
            {
                string temp = s.substr(0,i);
                string ans = "";
                for(int j =0;j<n/i;j++)
                {
                    ans+=temp;
                }
                if(ans == s)
                {
                    return true;
                }
            }
        }
        return false;
    }
};

//merge_sort array
class Solution {
public:
    void merge_sort(vector<int>& nums,int start,int end)
    {
        if(end <= start) return;
        int mid = start + (end-start)/2;
        merge_sort(nums,start,mid);
        merge_sort(nums,mid+1,end);
        merge(nums,start,mid,end);
    }
    void merge(vector<int>& nums,int start,int mid,int end)
    {
        int n1 = mid-start+1,n2=end-mid;
        vector<int> left(n1);
        vector<int> right(n2);

        for(int i =0;i<n1;i++)
        {
            left[i] = nums[start+ i];
        }
        for(int j =0;j<n2;j++)
        {
            right[j] = nums[mid+ j+1];
        }
        int i = 0,j=0,k = start;
        while(i<n1 && j<n2)
        {
            if(left[i]<right[j])
            {
                nums[k] = left[i];
                i++;
            }
            else
            {
                nums[k] = right[j];
                j++;
            }
            k++;
        }
        while(i<n1)
        {
            nums[k] = left[i];
            i++;
            k++;
        }
        while(j<n2)
        {
            nums[k] = right[j];
            j++;
            k++;
        }
        return;
    }
    vector<int> sortArray(vector<int>& nums) {
        vector<int> ans = nums;
        int n = nums.size();
        merge_sort(ans,0,n-1);
        return ans;
    }
};
// duplicate number in array

class Solution {
public:
    int findDuplicate(vector<int>& a) {
        int slow = a[0];
        int fast = a[0]; // to avoid first check slow !=fast
        do{
            slow = a[slow];
            fast = a[a[fast]];
        }while(slow != fast);
        fast = a[0];
        while(fast!=slow)
        {
            fast = a[fast];
            slow = a[slow];
        }
        return slow;
    }
};
// Reorganize String
class Solution {
public:
    string reorganizeString(string s) {
        unordered_map<char,int>mp;
        string ans = "";
        for(int i =0;i<s.size();i++)
        {
            if(mp.find(s[i])!=mp.end())mp[s[i]]++;
            else mp[s[i]] = 1;
            if(mp[s[i]]>s.size()-mp[s[i]]+1)
            {
                return ans;
            }
        }
        priority_queue<pair<int,int>>pq;
        for(auto it : mp)
        {
            pq.push({it.second,it.first});
        }
        while(pq.size()>=2)
        {
            int freq1 = pq.top().first,char1 = pq.top().second;
            pq.pop();
            int freq2 = pq.top().first,char2 = pq.top().second;
            pq.pop();
            ans += char1;
            ans += char2;
            if(freq1>1)
            {
                pq.push({freq1-1,char1});
            }
            if(freq2>1)
            {
                pq.push({freq2-1,char2});
            }
        }
        if(!pq.empty())
        {
            ans+= pq.top().second;
        }
        return ans;
    }
};
 // error but fine
class Solution {
public:
    string reorganizeString(string s) {
        unordered_map<char, int> mp;
        int n = s.size();
        for (char c : s) {
            mp[c]++;
            if(mp[c]> (n+1)/2) return "";
        }
        vector<char> sorted_chars;
        for (auto it: mp) {
            sorted_chars.push_back(it.first);
        }
        sort(sorted_chars.begin(),sorted_chars.begin(),[&](char a, char b){return mp[a] > mp[b];});
        if(mp[sorted_chars[0]]>(n+1)/2)return "";
        // cout<< sorted_chars[0];
        
        string ans(n,' ');
        int k =0;
        for(int i =0;i<n;i++)
        {
            for(int j =0;j<mp[sorted_chars[i]];j++)
            {
                if(k>=n)
                {
                    k = 1;
                }
                ans[k] = sorted_chars[i];
                k += 2;
            }
        }
        return ans;
    }
};
// Interleaving String with recursion
class Solution {
public:
    bool isInterleave(string s1, string s2, string s3) {
        if(s1.size()+s2.size() != s3.size()) return false;
        return helper(s1,s2,s3,0,0);
    }
    bool helper(string s1,string s2,string s3,int i,int j)
    {
        if(i+j == s3.size()) return true;
        bool ans = false;
        int k = i+j;
        if(i<s1.size() && s1[i] == s3[k])
        {
            ans |= helper(s1,s2,s3,i+1,j);
        }
        if(j<s2.size() && s2[j] == s3[k])
        {
            ans |= helper(s1,s2,s3,i,j+1);
        }
        return ans;
    }
};
// count inversions
//{ Driver Code Starts
#include <bits/stdc++.h>
using namespace std;


// } Driver Code Ends
class Solution{
  public:
  long long count = 0;
  void merge_sort(long long nums[],int start,int end)
    {
        if(end <= start) return;
        int mid = start + (end-start)/2;
        
        merge_sort(nums, start, mid);
        merge_sort(nums, mid + 1, end);
        merge(nums, start, mid, end);
        return ;
    }
    void merge(long long nums[],int start,int mid,int end)
    {
        // long long count = 0;
        long long n1 = mid-start+1,n2=end-mid;
        vector<long long> left(n1);
        vector<long long> right(n2);

        for(long long i =0;i<n1;i++)
        {
            left[i] = nums[start+ i];
        }
        for(long long j =0;j<n2;j++)
        {
            right[j] = nums[mid+ j+1];
        }
        
        long long i = 0,j=0,k = start;
        while(i<n1 && j<n2)
        {
            if(left[i]<=right[j])
            {
                nums[k] = left[i];
                i++;
            }
            else
            {
                count += n1-i;
                nums[k] = right[j];
                j++;
            }
            k++;
        }
        // if(i < n1) count += (n1-i-1)*n2;
        while(i<n1)
        {
            nums[k] = left[i];
            i++;
            k++;
        }
        while(j<n2)
        {
            nums[k] = right[j];
            j++;
            k++;
        }
        return ;
    }
    // arr[]: Input Array
    // N : Size of the Array arr[]
    // Function to count inversions in the array.
    long long int inversionCount(long long nums[], long long n)
    {
        // Your Code Here
        // long long global =  
        merge_sort(nums,0,n-1);
        return count;
    }
};

// global and local inversion
class Solution {
public:
    bool isIdealPermutation(vector<int>& nums) {
        for(int i =0;i<nums.size();i++)
        {
            if(abs(nums[i]-i)>1)
            {
                return false;
            }
        }
        return true;
    }
};
//minimize penality
class Solution {
public:
    int bestClosingTime(string customers) {
        int n = customers.size();
        int min_penality = 0;
        int pen = INT_MAX-1;
        int count = 0;
        int min_hour = -1;
        for(int i =n-1;i>=0;i--)
        {
            if(customers[i]=='Y') count++;
        }
        min_penality = count;
        min_hour = -1;
        for(int i =0;i<n;i++)
        {
            if(customers[i]=='Y') count--;
            else count++;
            if(count<min_penality)
            {
                min_hour = i;
                min_penality = count;
            }
        }
        return min_hour+1;
    }
};
//maximize profit
class Solution {
public:
    int bestClosingTime(string customers) {
        int n = customers.size(),min_penality = 0,count = 0,min_hour = -1;
        for(int i =0;i<n;i++)
        {
            if(customers[i]=='Y') count++;
            else count--;
            if(count>min_penality)
            {
                min_hour = i;
                min_penality = count;
            }
        }
        return min_hour+1;
    }
};

//Flip Bits kadanes algorithm
int maxOnes(int a[], int n)
    {
        // Your code goes here
        int curr = 0;
        int max = 0;
        int count = 0;
        for(int i =0;i<n;i++)
        {
            if(a[i]==1)
            {
                curr--; // to make it 0
                count++;
            }
            else if(a[i]==0)
            {
                curr++; // to make it 1
            }
            if(curr>max)
            {
                max = curr;
            }
            if(curr <0)
            {
                curr = 0;
            }
        }
        return max+count;
    }
// remove nodes having greater value on right
class solution{
   Node* reverse(Node *head)
    {
        Node * temp = head;
        Node * dummy = new Node(-1);
        dummy->next = temp;
        while(temp)
        {
            Node * temp1 = temp->next;
            // Node * temp2 = temp1->next;
            temp->next = dummy;
            dummy = temp;
            temp = temp1;
        }
        head->next = NULL;
        // head = dummy;
        return dummy ;
    }
    Node *compute(Node *head)
    {
        // your code goes here
        if(!head || !head->next) return head;
        
        Node*temp =  new Node(-1);
        // temp->next = reverse(head);
        Node*temp1 = reverse(head);
        Node*ans = temp1;
        
        int maxi =0;
        while(temp1)
        {
            if(temp1->data >=maxi)
            {
                temp->next = temp1;
                temp = temp1;
                maxi = temp1->data;
            }
            temp1 = temp1->next;
        }
        temp->next = NULL;
        return reverse(ans);
        // return 
    }
};
// one repeated and one missing
vector<int> repeatedNumber(const vector<int> &nums) {
    long long diff = 0, sq_diff = 0, n1 = nums.size();
    for (int i = 0; i < n1; i++) {
        diff += nums[i];
        diff -= (i + 1);
        sq_diff += (long long)nums[i] * (long long)nums[i];
        sq_diff -= (long long)(i + 1) * (long long)(i + 1);
    }

    if (diff == 0) {
        // Handle division by zero case
        return {-1, -1}; // Or any other appropriate response
    }

    // diff = n2 - n1
    // sq_diff = n2^2 - n1^2
    long long sum = sq_diff / diff; // n2 + n1

    return {(sum + diff) / 2, (sum - diff) / 2};
}
//non repeating numbers
class Solution
{
public:
    vector<int> singleNumber(vector<int> nums) 
    {
        // Code here.
        int xor_ele = 0;
        
        for(int i =0;i<nums.size();i++)
        {
            xor_ele ^= nums[i];
        }
        
        int right = 1;
        while(!(right & xor_ele))
        {
            right <<= 1;
        }
        int x = 0,y=0;
        for(int i =0 ;i<nums.size();i++)
        {
            if(right & nums[i])
            {
                x^=nums[i];
            }
            else
            {
                y^=nums[i];
            }
        }
        if(x>y)
        {
            return {y,x};
        }
        return {x,y};
    }
};
vector<int> repeatedNumber(const vector<int> &A) {
    int xr = 0, n1 = A.size();
    for (int i = 0; i < n1; i++) {
        int k = i + 1;
        xr ^= A[i];
        xr ^= k;
    }

    int check = 1;
    while (!(xr & check)) {
        check <<= 1;
    }

    int a = 0, b = 0;
    for (int i = 0; i < n1; i++) {
        if (A[i] & check) {
            a ^= A[i];
        } else {
            b ^= A[i];
        }
        int k = i + 1;
        if (k & check) {
            a ^= k;
        } else {
            b ^= k;
        }
    }
    int count = 0;
    for(int i =0;i<n1;i++)
    {
        if(A[i]==a)
        {
            count++;
        }
    }
    return (count == 2) ? vector<int>{a, b} : vector<int>{b, a};
}
//merging intervals
class Solution {
public:
    static bool comp(vector<int>a,vector<int>b)
    {
        return a[0]<b[0];
    }
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        int n = intervals.size();
         if(n<2) return intervals;
         sort(intervals.begin(),intervals.end(),comp);
         vector<vector<int>>ans;
         int i =0,j=1;
         int a = intervals[i][0];
         int b = intervals[i][1];
         while(i<n)
         {
             if(intervals[i][0]>b)
             {
                ans.push_back({a,b});
                a = intervals[i][0];
                b = intervals[i][1];
             }
             else
             {
                 b = max(b,intervals[i][1]);
             }
             i++;
         }
         ans.push_back({a,b});

         return ans;
    }
};
// binary search in a matrix
class Solution {
public:
    bool search(vector<vector<int>>& a, int t,int s,int e)
    {
        int m = a.size(); //1
        int n = a[0].size(); //2
        int mid = (s+e)/2;
        int pre = a[mid/n][mid%n];
        cout<<pre<<endl;
        if(pre == t)
        {
            return true;
        }
        if(s == e) return false;
        if(pre > t)
        {
            return search(a,t,s,mid);
        }
        else
        {
            return search(a,t,mid+1,e);
        }
    }
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        cout<<matrix[0].size()<<" "<<matrix.size()<<endl;
        return search(matrix,target,0,matrix[0].size()*matrix.size()-1);
    }
};
// reverse pairs

class Solution {
public:
    long long merge_sort(vector<int> &nums,int start,int end)
    {
        if(end <= start) return 0;
        int mid = start + (end-start)/2;
        long long count = 0;
        count+= merge_sort(nums, start, mid);
        count+= merge_sort(nums, mid + 1, end);
        count+= merge(nums, start, mid, end);
        return count;
    }
    long long merge(vector<int> &nums,int start,int mid,int end)
    {
        long long count = 0;
        int n1 = mid-start+1,n2=end-mid;
        vector<int> left(n1);
        vector<int> right(n2);

        for(int i =0;i<n1;i++)
        {
            left[i] = nums[start+ i];
        }
        for(int j =0;j<n2;j++)
        {
            right[j] = nums[mid+ j+1];
        }
        int i =0,j=0,k =0;
        for(int i =0;i<n1;i++)
        {
            while(j<n2 && (left[i] >2*(long long)right[j]))
            {
                j++;
            }
            count+= j;
        }
        i = 0,j=0,k = start;
        while(i<n1 && j<n2)
        {
            
            if(left[i]<=right[j])
            {
                nums[k] = left[i];
                i++;
            }
            else
            {
                nums[k] = right[j];
                j++;
            }
            k++;
        }
        while(i<n1)
        {
            nums[k] = left[i];
            i++;
            k++;
        }
        while(j<n2)
        {
            nums[k] = right[j];
            j++;
            k++;
        }
        return count;
    }
    int reversePairs(vector<int>& nums) {
        int n = nums.size();
        return merge_sort(nums,0,n-1);
    }
};

//Longest Consecutive Sequence
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        unordered_map<int,int>mp;
        int ans = 0;
        for(int i =0;i<nums.size();i++)
        {
            mp[nums[i]] = 1;
        }
        for(int i =0;i<nums.size();i++)
        {
            int start;
            int length = 1;
            int num = nums[i];
            if(mp.find(num)!=mp.end()){
                while(mp.find(num-1)!=mp.end())
                {
                    num -=1;
                }
                start = num;
                while(mp.find(num+1)!=mp.end())
                {
                    mp.erase(num);
                    length++;
                    num+=1;
                }
                ans = max(ans,length);
                }
        }
        return ans;
    }
};
// longest substring without repeating characters
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_map<char,int>mp;
        int i =0,j = 0,n = s.size();
        int ans = 0;
        while(j<n)
        {
            if(mp.find(s[j]) == mp.end())
            {
                mp[s[j]] = j;
            }
            else
            {
                i = max(i,mp[s[j]]+1);
                mp[s[j]] = j;
            }
            ans = max(ans,j-i+1);
            j++;
        }
        return ans;
    }
};
//Largest subarray with 0 sum 
class Solution{
    public:
    int maxLen(vector<int>&A, int n)
    {   
        // Your code here
        unordered_map<int,int>mp;
        int sum = 0,ans=0;
        mp[0] = -1;
        for(int i =0;i<n;i++)
        {
            sum+= A[i];
            if(mp.find(sum)!=mp.end())
            {
                ans = max(ans,i-mp[sum]);
            }
            else
            {
                mp[sum] = i;
            }
        }
        return ans;
    }
};
// reverse an linked list
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        if(!head || !head->next)
        {
            return head;
        }
        ListNode* prev = head;
        ListNode* curr = head->next;

        prev->next = NULL;
        while(curr)
        {
            ListNode* nxt = curr->next;
            curr->next = prev;

            prev = curr;
            curr = nxt;
        }
        return prev;
        
    }
    ListNode* reverse(ListNode* head)
    {
        ListNode*prev = NULL;
        while(head)
        {
            ListNode* nxt = head->next;
            head->next = prev;
            prev = head;
            head  = nxt;
        }
        return head = prev;
    }
};

// split linked list into k parts
class Solution {
public:
    vector<ListNode*> splitListToParts(ListNode* head, int k) {
        vector<ListNode*>ans;
        int len = 0;
        ListNode* temp = head;
        while(temp)
        {
            len++;
            temp = temp->next;
        }
        temp =  head;
        int quo = len/k,rem = len%k;
        while(temp)
        {
            ans.push_back(temp);
            int rand = quo;
            if(rem<=0)rand = quo-1;
            while(rand && temp)
            {
                temp = temp->next;
                rand--;
            }
            rem--;
            if(!temp) break;
            ListNode* nxt = temp->next;
            temp->next = NULL;
            temp = nxt;
        }
        temp  = NULL;
        cout<<k<<" "<<len<<endl;
        if(k>len){    for(int i =0;i<k-len;i++)
            {
                ans.push_back(temp);
            }
            }
        return ans;
    }
};
//  kth largest element in bst
class Solution
{
    private:
    int search(Node *root, int K,int &temp)
    {
        if(!root) return 0;
        int y = search(root->right,K,temp);
        temp++;
        if(temp == K) return root->data;
        int x = search(root->left,K,temp);
        
        if(x)
        {
            return x;
        }
        if(y)
        {
            return y;
        }
        return 0;
        
    }
    public:
    int kthLargest(Node *root, int K)
    {
        //Your code here
        int r =0;
        return search(root,K,r);
    }
};
//combination sum4
class Solution {
public:
    int combinationSum4(vector<int>& nums, int target) {
        int ans = 0;
        vector<vector<unsigned long long>>dp;
        dp.resize(target+1,vector<unsigned long long>(nums.size()+1,0));
        for(int i =0;i<=nums.size();i++)
        {
            dp[0][i] = 1;
        }
        for(int j =0;j<=target;j++)
        {
            for(int i = nums.size()-1;i>=0;i--)
            {
            
                if(nums[i]<=j)
                {
                    dp[j][i] = dp[j-nums[i]][0]+dp[j][i+1];
                }
                else
                {
                    dp[j][i] = dp[j][i+1];
                }
            }
        }
            
        return dp[target][0];
    }
};//revisit
class Solution {
public:
    int combinationSum4(vector<int>& nums, int target) {
        vector<unsigned long long> dp(target + 1, 0); // dp[i] represents the number of combinations to make sum i
        
        dp[0] = 1; // There is one way to make sum 0, which is by not selecting any element
        
        for (int i = 1; i <= target; i++) {
            for (int j = 0; j < nums.size(); j++) {
                if (i >= nums[j]) {
                    dp[i] += dp[i - nums[j]];
                }
            }
        }
        
        return dp[target];
    }
};
//check linked list is palindrome or not
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* reverse(ListNode* head)
    {
        ListNode*prev = NULL;
        while(head)
        {
            ListNode* nxt = head->next;
            head->next = prev;
            prev = head;
            head  = nxt;
        }
        return prev;
    }
    bool isPalindrome(ListNode* head) {

        if(!head || !head->next)
        {
            return true;
        }
        ListNode* slow = head;
        ListNode* fast = head->next;
        while(fast && fast->next)
        {
            slow = slow->next;
            fast = fast->next->next;
        }
        fast = slow->next;
        fast = reverse(fast);
        slow = head;
        while(fast)
        {
            if(fast->val != slow->val)
            {
                return false;
            }
            fast = fast->next;
            slow = slow->next;
        }
        return true;
    }
};
//smallest palindrome string after adding some characters (KMP ALGO) lps
class Solution {
public:
    int kmp(string &txt,string &patt)
    {
        string str = patt + '#' + txt;
        int i=1,len=0,n=str.size();
        vector<int>lps(n,0);
        while(i<n)
        {
            if(str[i]==str[len])
            {
                len++;
                lps[i] = len;
                i++;
            }
            else
            {
                if(len>0)

                {
                    len = lps[len-1];
                }
                else
                {
                    lps[i] = 0;
                    i++;
                }
            }
        }
        return lps.back();
    }
    string shortestPalindrome(string s) {
        int n = s.size(),i=0, j = n-1;
        string so = s;//string(s.rbegin(),s.rend());
        reverse(so.begin(),so.end());
        int p = kmp(so,s);
        return so.substr(0,n-p)+s;
    }
};
// sorting in lexicographical order 1 to n
class Solution {
public:
    int n;
    void dfs(int i,vector<int>&ans)
    {
        if(i>n) return;
        if(i<=n) ans.push_back(i);
        for(int j =0;j<10;j++)
        {
            if(10*i<=n)dfs(10*i+j,ans);
        }
        return;
    } 
    vector<int> lexicalOrder(int n) {
        this->n = n;
        vector<int>ans;
        for(int i =1;i<10;i++)dfs(i,ans);
        return ans;
    }
};
// kth smallestlexicographical number
class Solution {
public:
    long count(long n , long prefix){
        if(prefix > n){
            return 0;
        }else if(prefix == n){
           return 1; 
        }
        long minPrefix = prefix , maxPrefix = prefix;
        long count = 1;
        while(1){
            minPrefix = 10*minPrefix;
            maxPrefix = 10*maxPrefix + 9;
            if(n < minPrefix)break;
            if(minPrefix <= n && n <= maxPrefix){
                count += (n - minPrefix + 1);
                break;
            }else{
                count += (maxPrefix - minPrefix + 1);
            }
        }
        
        return count;
    }
    int findKthNumber(int n, int k, int prefix =0) {
        for(int i = (prefix == 0 ? 1 : 0) ; i<=9; i++){
            if(k == 0){
                return prefix;
            }
            int64_t numbers_prefix_i_less_n = count(n,prefix*10 + i);
            if(numbers_prefix_i_less_n >= k){
                return findKthNumber(n , k-1 , prefix*10 + i);
            }else{
                k -= numbers_prefix_i_less_n;
            }
        }
        return prefix;
    }
}; 
// min max binary search
class Solution {
public:
    bool possible(long long time,vector<int>&times,int ht)
    {
        long long tot_ht =0;
        for(int wt:times)
        {
            long long low = 0,high=1e6;
            while(low<=high)
            {
                long long mid = low+(high-low)/2;
                if(wt*mid*(mid+1)/2 <=time)
                {
                    low = mid+1;
                }
                else
                {
                    high = mid-1;
                }
            }
            // detect ht reduced by this time(time)
            tot_ht +=high;
            if(tot_ht>=ht) return true;
        }
        return tot_ht>=ht;
    }
    long long minNumberOfSeconds(int mountainHeight, vector<int>& workerTimes) {
        long long st = 0,end = 1e18,ans=0;
        while(st<=end)
        {
            long long mid = st+(end-st)/2;//time
            if(possible(mid,workerTimes,mountainHeight))
            {
                end = mid-1;
                ans=mid;
            }
            else
            {
                st = mid+1;
            }
        }
        return ans;

    }
};
// trie 
struct Node{
    public:
    Node*child[26];
    bool is_word;
    int count;
    Node()
    {
        is_word = false;
        count = 0;
        for(auto &it:child)
        {
            it = NULL;
        }
    }
    ~Node() { 
        for (auto ch : child) 
            delete ch; // Recursively delete children
    }
};
class Trie { // destructor important
public:
    Node*root;
    Trie() {
        root = new Node();
    }
    ~Trie() {
        delete root; // Free memory of the entire trie
    }
    void insert(string word) {
        Node*p = root;
        for(auto &ch:word)
        {
            auto it = ch-'a';
            if(!p->child[it])
            {
                p->child[it] = new Node();
            }
            p = p->child[it];
            p->count++;
        }
        p->is_word = true;
    }
    
    int search(string word,bool prefix = false) {
        Node*p = root;
        int ans = 0;
        for(auto &ch:word)
        {
            auto it = ch-'a';
            if(p->child[it]==NULL)
            {
                return -1;
            }
            p = p->child[it];
            ans+= p->count;
        }
        // if(prefix) return true;
        return ans;
    }
    
    bool startsWith(string prefix) {
        return search(prefix,true);
    }
};
class Solution {
public:
    vector<int> sumPrefixScores(vector<string>& words) {
        Trie*tr = new Trie();
        for(auto it:words)
        {
            tr->insert(it);
        }
        vector<int>ans;
        for(auto it:words)
        {
            ans.push_back(tr->search(it));
        }
        delete tr;
        return ans;
    }
};
// inserting intervals mycalender i
class MyCalendar {
public:
    map<int,int>st;
    MyCalendar() {
        st.empty();
    }
    bool book(int start, int end) {
        auto nxt = st.lower_bound(start);
        // 1 3 5 7  lower_bound(4) = 5
        if(nxt!=st.end()&& nxt->first<end)
        {
            return false;
        }
        if(nxt!=st.begin()&& prev(nxt)->second>start)
        {
            return false;
        }
        st[start] = end;
        return true;
    }
};
//mycalender ii similar to min number of train platforms
class MyCalendarTwo {
public:
    map<int,int>mp;
    MyCalendarTwo() {
        mp.empty();
    }
    
    bool book(int start, int end) {
        mp[start]++;
        mp[end]--;
        int bookings =0;
        for(auto it:mp)
        {
            bookings += it.second;
            if(bookings>=3)
            {
                mp[start]--;
                mp[end]++;
                return false;
            }
        }
        return true;
    }
};
//circular deque
class MyCircularDeque {
public:
    vector<int>dq;
    int front,back,size,k;
    MyCircularDeque(int k) {
        this->size = 0;
        this->dq.resize(k);
        this->k = k;
        this->front = -1;
        this->back = k;
    }
    
    bool insertFront(int value) {
        // if(size==0) back = 0;
        if(size==k) return false;
        front = (front+1)%k;
        size++;
        cout<<"front"<<front<<" "<<value<<" "<<size<<endl;
        dq[front] = value;
        return true;
    }
    
    bool insertLast(int value) {
        if(size==k) return false;
        back = (back-1+k)%k;
        cout<<"back"<<back<<" "<<value<<" "<<size<<endl;
        dq[back] = value;
        size++;
        return true;
    }
    
    bool deleteFront() {
        if(size==0)return false;
        front = (front-1+k)%k;
        size--;
        cout<<"delFront"<<endl;
        return true;
    }
    
    bool deleteLast() {
        if(size==0)return false;
        back = (back+1)%k;
        size--;
        cout<<"delback"<<endl;
        return true;
    }
    
    int getFront() {
        if(size==0) return -1;
        front = (front+k)%k;
        return dq[front];
    }
    
    int getRear() {
        // cout<<back<<" "<<k<<endl;
        if(size==0) return -1;
        back = (back+k)%k;
        return dq[back];
    }
    
    bool isEmpty() {
        return size==0;
    }
    
    bool isFull() {
        return size==k ;
    }
};

// O(1) data structure lfu similar

struct Node {
    string word;
    int freq;
    Node* prev;
    Node* next;

    Node(string k) : word(k), freq(1), prev(nullptr), next(nullptr) {}
};

class AllOne {
public:
    Node *head, *tail;
    unordered_map<string, Node*> um;
    AllOne() {
        head = new Node("");
        tail = new Node("");
        head->next = tail;
        tail->prev = head;
    }
    void moveToCorrectNextPosition(Node* node) {
        Node* ptr = node->next;
        // checkig if any node exist with current frequency
        while (ptr != tail && node->freq > ptr->freq) {
            ptr = ptr->next;
        }

        if (ptr != node->next) {
            // remove node from current place
            node->prev->next = node->next;
            node->next->prev = node->prev;

            // add it to new place before ptr
            ptr->prev->next = node;
            node->prev = ptr->prev;
            node->next = ptr;
            ptr->prev = node;
        }
    }
    void moveToCorrectPrevPosition(Node* node) {
        Node* ptr = node->prev;
        // checkig if any node exist with current frequency
        while (ptr != head && node->freq < ptr->freq) {
            ptr = ptr->prev;
        }

        if (ptr != node->prev) {
            // remove node from current place
            node->prev->next = node->next;
            node->next->prev = node->prev;

            // add it to new place before ptr
            ptr->next->prev = node;
            node->next = ptr->next;
            node->prev = ptr;
            ptr->next = node;
        }
    }
    void inc(string word) {
        if (um.find(word) != um.end()) {
            Node* node = um[word];
            node->freq++;
            moveToCorrectNextPosition(node);
        } 
        else {
            Node* node = new Node(word);
            node->next = head->next;
            node->prev = head;
            head->next->prev = node;
            head->next = node;
            um[word] = node;
            moveToCorrectNextPosition(node);
        }
    }
    void dec(string word) {
        Node* node = um[word];
        node->freq--;
        moveToCorrectPrevPosition(node);
        if (node->freq == 0) {
            node->next->prev = node->prev;
            node->prev->next = node->next;
            um.erase(word);
            delete node;
        }
    }
    string getMaxKey() {
        string ans = "";
        if (tail->prev != head)
            ans = tail->prev->word;
        return ans;
    }
    string getMinKey() {
        string ans = "";
        if (head->next != tail)
            ans = head->next->word;
        return ans;
    }
};
// min subarray need to be removed to have its sum divisible by p
class Solution {
public:
    int minSubarray(vector<int>& nums, int p) {
        unordered_map<int,int>rem;
        rem[0] = -1;
        int ans = nums.size();
        long long sum=0;
        for(int i =0;i<nums.size();i++)
        {
            sum+=nums[i];
            int curr_rem = sum%p;
        }
        int find = sum%p;
        sum=0;
        for(int i =0;i<nums.size();i++)
        {
            sum+= nums[i];
            int curr_rem = sum%p;
            int prev  = (curr_rem-find+p)%p;
            rem[curr_rem] = i;
            if(rem.find(prev)!=rem.end())
            {
                ans = min(ans,i-rem[prev]);
            }
        }
        return (ans==nums.size()) ?-1:ans;
    }
};
// find if strin permutation(s1) is substr of s2
class Solution {
public:
    bool is_same(vector<int>&f1,vector<int>&f2)

    {
        for(int i =0;i<26;i++)
        {
            if(f1[i]!=f2[i])
            {
                return false;
            }
        }
        return true;
    }
    bool checkInclusion(string s1, string s2) {
        if(s1.size()>s2.size())return false;
        vector<int>f1(26,0);
        vector<int>f2(26,0);
        int k = s1.size();
        for(int i =0;i<k;i++)
        {
            f1[s1[i]-'a']++;
            f2[s2[i]-'a']++;
        }
        if(is_same(f1,f2)) return true;
        for(int i = k;i<s2.size();i++)
        {
            f2[s2[i]-'a']++;f2[s2[i-k]-'a']--;
            if(is_same(f1,f2))
            {
                return true;
            }
        }
        return false;
    }
};
// min swaps to get paranthesis balanced
class Solution {
public:
    int minSwaps(string s) {
        int count = 0,temp =0,n=s.size();
        for(int i =0;i<n;i++)
        {
            if(s[i]==']')
            {
                temp--;
            }
            else
            {
                temp++;
            }
            if(temp<0)
            {
                count++;
                temp+=2;
            }
        }
        return count;
    }
};
// balace parenthesis by adding ) or (
class Solution {
public:
    int minAddToMakeValid(string s) {
     int closed=0;
     int open=0;
     for(int i=0;i<s.size();i++){
        if(s[i]=='(')open++;
        else if(s[i]==')'&& open>0) open--;
        else closed++;
     }
     return closed+open;
    }
};
// monotonic stack largest ramp i,j that i<j and nums[i]<=nums[j]
class Solution {
public:
    int maxWidthRamp(vector<int>& nums) {
        stack<pair<int,int>>st;
        int ans = 0,n= nums.size();

        for(int i =0;i<nums.size();i++)
        {
            if(st.empty() || nums[i] < st.top().first)
            {
                st.push({nums[i],i});//storing minimum till now 
            }
        }
        for(int i=n - 1;i >= 0;i--)
        {
            while(!st.empty() && nums[i] >= st.top().first)
            {
                int ind = st.top().second;
                ans = max(ans, i - ind);//checking min till index with curr
                st.pop();
            }
        }
        return ans;

    }
};
//smallest number of uncapped chair sim to calender problem
class Solution {
public:
    int smallestChair(vector<vector<int>>& times, int tf) {
        int n = times.size();  // Number of people
        vector<pair<pair<int, int>, int>> events;  // To store events (arrival, departure)
        
        // Create events: (time, type), person index
        for (int i = 0; i < n; i++) {
            events.push_back({{times[i][0], 1}, i});  // Arrival event (1 means arrival)
            events.push_back({{times[i][1], -1}, i}); // Departure event (-1 means departure)
        }
        
        // Sort events by time; for same time, departure comes before arrival
        sort(events.begin(), events.end());
        
        vector<int> assignedChairs(n, -1);  // Track which chair each person gets
        priority_queue<int, vector<int>, greater<int>> availableChairs;  // Min-heap for free chairs
        
        // Initially all chairs are available
        for (int i = 0; i < n; i++) {
            availableChairs.push(i);
        }

        // Process each event
        for (auto& event : events) {
            int time = event.first.first;
            int type = event.first.second;
            int person = event.second;

            // If it's the target friend's arrival
            if (person == tf && type == 1) {
                return availableChairs.top();  // Return the smallest chair available
            }
            
            // If it's an arrival event
            if (type == 1) {
                assignedChairs[person] = availableChairs.top();  // Assign the smallest chair
                availableChairs.pop();  // Remove the chair from available
            }
            // If it's a departure event
            else {
                availableChairs.push(assignedChairs[person]);  // Free up the chair
                assignedChairs[person] = -1;  // Mark chair as free
            }
        }
        return -1;  // This line should not be reached
    }
};
//minimum number of platforms to have trains
class Solution {
public:
    int minGroups(vector<vector<int>>& intervals) {
        vector<int>times1;
        vector<int>times2;
        int ans = 0,temp=0;
        for(int i =0;i<intervals.size();i++)
        {
            times1.push_back(intervals[i][0]);
            times2.push_back(intervals[i][1]);
        }
        sort(times1.begin(),times1.end());
        sort(times2.begin(),times2.end());
        int i =0,j=0,n=times1.size();
        while(i<n && j<n)
        {
            if(times1[i]<=times2[j])
            {
                temp++;
                i++;
            }
            else
            {
                temp--;
                j++;
            }
            ans = max(ans,temp);
        }
        return ans;
    }
};
//smallest range covering atleast single element from k sorted lists in nums
class Solution {
public:
    vector<int> smallestRange(vector<vector<int>>& nums) {
        priority_queue<vector<int>,vector<vector<int>>,greater<vector<int>>>pq;
        int mini = INT_MAX,maxi=INT_MIN,k=nums.size();
        for(int i =0;i<k;i++)
        {
            pq.push({nums[i][0],i,0});
            mini = min(mini,nums[i][0]);
            maxi = max(maxi,nums[i][0]);
        }
        vector<int>ans{mini,maxi};
        while(pq.size()==k)
        {
            auto tp = pq.top();
            pq.pop();
            mini = max(tp[0],mini);
            if(maxi-mini <ans[1]-ans[0])
            {
                ans[0] = mini;
                ans[1] = maxi;
            }
            if(tp[2]<nums[tp[1]].size()-1)
            {
                pq.push({nums[tp[1]][tp[2]+1],tp[1],tp[2]+1});
                maxi = max(maxi,nums[tp[1]][tp[2]+1]);
            }
        }
        return ans;
    }
};
// largest good string using a,b.,c with max counts a b c  (cant have 3 same consecutive characters)
class Solution {
public:
    string longestDiverseString(int a, int b, int c) {
        priority_queue<pair<int,char>>pq;
        if(a)pq.push({a,'a'});
        if(b)pq.push({b,'b'});
        if(c)pq.push({c,'c'});
        string ans = "";
        char prev = '#';
        char count = 1;
        while(!pq.empty())
        {
            auto tp1 = pq.top();
            pq.pop();
            int freq1 = tp1.first;
            char ch1 = tp1.second;
            if(ch1==prev && count==2)
            {
                if(pq.size()>0)
                {
                    auto tp2 = pq.top();
                    pq.pop();
                    int freq2 = tp2.first;
                    char ch2 = tp2.second;
                    cout<<ch1<<" "<<freq1<<" "<<ch2<<" "<<freq2<<endl;
                    freq2--;
                    ans+= ch2;
                    prev = ch2;
                    count = 1;
                    if(freq2>0)pq.push({freq2,ch2});
                }
                else
                {
                    break;
                }
            }
            else
            {
                    if(ch1!=prev)
                    {
                        prev = ch1;
                        count = 1;
                    }
                    else
                    {
                        prev = ch1;
                        count++;
                    }
                    freq1--;
                    ans += ch1;
            }
            if(freq1>0)pq.push({freq1,ch1});
            cout<<pq.size()<<endl;
        }
        return ans;
    }
};
//count number of sub arrays with maximum or value
class Solution {
public:
    int n;
    unsigned max_OR;
    int f(int i, unsigned acc_or, vector<int>& nums){
        if (i<0) return (acc_or==max_OR)?1:0;
        int skip=f(i-1, acc_or, nums);
        int take=f(i-1, acc_or| nums[i], nums);
        return skip+take;
    }
    int countMaxOrSubsets(vector<int>& nums) {
        n=nums.size();
        max_OR=accumulate(nums.begin(), nums.end(), 0, bit_or<>());
        return f(n-1, 0, nums);
    }
};
// split string to unique sub strings
class Solution {
public:
    int solve(int i,unordered_map<string,int>&mp,string s)
    {
        if(i==s.size())
        {
            return 0;
        }
        string temp = "";
        int ans =  INT_MIN;
        for(int k = i;k<s.size();k++)
        {
            temp+= s[k];
            if(mp[temp]==0)
            {
                mp[temp]++;
                ans = max(ans,1+solve(k+1,mp,s));
                mp[temp]--;
            }
        }
        return ans;
    }
    int maxUniqueSplit(string s){
        unordered_map<string,int>mp;
        return solve(0,mp,s);;
    }
};
// count square sub matrices of ones in the grid
class Solution {
public:
    int countSquares(vector<vector<int>>& matrix) {
        int n=matrix.size(),m=matrix[0].size(),ans=0;
        vector<vector<int>>dp(n,vector<int>(m,0));
        for(int i =0;i<n;i++)
        {
            dp[i][0] = matrix[i][0];
            ans+=  dp[i][0];
        }
        
        for(int i =1;i<m;i++)
        {
            dp[0][i] = matrix[0][i];
            ans+=  dp[0][i];
        }
        for(int i = 1;i<n;i++)
        {
            for(int j =1;j<m;j++)
            {
                if(matrix[i][j]==1){
                    dp[i][j] = 1+ min(dp[i-1][j-1],min(dp[i-1][j],dp[i][j-1]));
                }
                else
                {
                    dp[i][j]=0;
                }
                // cout<<dp[i][j]<<" ";
                ans+= dp[i][j];
            }
            cout<<endl;
        }
        return ans;
    }
};
//cousins in binary tree ii - replacing value of node with sum of all its cousins
class Solution {
public:
    TreeNode* replaceValueInTree(TreeNode* root) {
        queue<pair<TreeNode*,int>>q;
        q.push({root,root->val});
        int layer_sum = root->val;
        while(!q.empty())
        {
            int k = q.size();
            int next_sum = 0;
            while(k--)
            {
                auto it = q.front();
                q.pop();
                TreeNode* node = it.first;
                node->val = layer_sum-it.second;
                int temp_sum = 0;
                if(node->left)
                {
                    temp_sum+= node->left->val;
                }
                if(node->right)
                {
                    temp_sum+= node->right->val;
                }
                next_sum += temp_sum;
                if(node->left)
                {
                    q.push({node->left,temp_sum});
                }
                if(node->right)
                {
                    q.push({node->right,temp_sum});
                }
            }
            layer_sum = next_sum;
        }
        return root;
    }
};
// mountain array >>>> increasing and decreasing<<<< should have a peak
class Solution {
public:
    int minimumMountainRemovals(vector<int>& nums) {
        int n = nums.size();
        vector<int> LIS(n, 1), LDS(n, 1);

        // Compute LIS up to each index
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                if (nums[i] > nums[j]) {
                    LIS[i] = max(LIS[i], LIS[j] + 1);
                }
            }
        }

        // Compute LDS from each index
        for (int i = n - 1; i >= 0; --i) {
            for (int j = n - 1; j > i; --j) {
                if (nums[i] > nums[j]) {
                    LDS[i] = max(LDS[i], LDS[j] + 1);
                }
            }
        }

        int maxMountainLength = 0;

        // Find the maximum mountain length
        for (int i = 1; i < n - 1; ++i) {
            if (LIS[i] > 1 && LDS[i] > 1) {  // Valid peak
                maxMountainLength = max(maxMountainLength, LIS[i] + LDS[i] - 1);
            }
        }

        return n - maxMountainLength;
    }
};
// ifind if goal is rotation of string s
class Solution {
public:
    bool rotateString(string s, string goal){
        if(s.size()!=goal.size()) return false;
        return (s + s).find(goal) != string::npos;//(no position in string refers to no character found)
        //return (s + s).find(goal) != -1;
        //
    }
};
//sorting possible? if we can swap adjacent numbers with same number of set bits?
class Solution {
public:
    bool canSortArray(vector<int>& nums) {
        pair<int,int> prev={INT_MAX,INT_MIN},curr;
        int prev_count = -1;
        for(int i =0;i<nums.size();i++)
        {
            int x = nums[i];
            int curr_count = __builtin_popcount(x);
            if(curr_count!=prev_count)
            {
                if(prev_count!=-1 && prev.second > curr.first) return false;
                prev_count = curr_count;
                prev = curr;
                curr = {x,x};
            }
            else
            {
                curr.first=min(curr.first, x);
                curr.second=max(curr.second, x);
            }
        }
        return curr.first>=prev.second;
    }
};
//** maximum number of elemnts set with and >0
class Solution {
public:
    int largestCombination(vector<int>& candidates) {
        int ans = 0;
        for(int i =0;i<32;i++)
        {
            int count =0 ;
            for(auto cand:candidates)
            {
                if(cand & (1<<i))
                {
                    count++;
                }
            }
            ans = max(ans,count);
        }
        return ans;
    }
};
// minimum end of the array of n elements with or >x
class Solution {
public:
    long long minEnd(int n, int x) {
        long long num = x;
        for(int i = 1; i < n; i++) {
            num = (num+1) | x;
            // cout<<num<<endl;
            // 10111 +1 = 11000 |10101
        }
        return num;
    }
};
// least subarray size with or >=k 
class Solution {
public:
    int minimumSubarrayLength(vector<int>& nums, int k) {
        int n = nums.size();
        vector<int>count(32,0);
        int i =0,min_length = INT_MAX,curr_or=0;
        for(int j =0;j<nums.size();j++)
        {
            curr_or |= nums[j];
            for(int bit =0;bit<32;bit++)
            {
                if(nums[j]&(1<<bit))
                {
                    count[bit]++;
                }
            }
            while(i<=j && curr_or>=k)
            {
                min_length = min(min_length,j-i+1);
                for(int bit =0;bit<32;bit++)
                {
                    if(nums[i]&(1<<bit))count[bit]--;
                    if(count[bit]<=0)
                    {
                        curr_or &= ~(1<<bit);
                    }
                }
                i++;
            }
        }
        return (min_length == INT_MAX)?-1:min_length;
    }
};
// number of pairs of elements with sum in between lower and upper
class Solution {
public:
    long long countFairPairs(vector<int>& nums, int lower, int upper) {
        sort(nums.begin(),nums.end());
        long long n = nums.size(),ans=0;
        for(int i =0;i<n;i++)
        {
            int left = lower_bound(nums.begin()+i+1,nums.end(),lower-nums[i])-nums.begin();
            int right = upper_bound(nums.begin()+i+1,nums.end(),upper-nums[i])-nums.begin();
            ans+= (right-left);
        }
        return ans;
    }
};
// lower bound and upper bound
int main() {
    vector<int> v = {1,2,3,4,5,6,7,9,11,15,17};
  
      // Sorting v before finding upper and lower bound
      sort(v.begin(), v.end());
      cout<<v.size()<<endl;

     cout << lower_bound(v.begin(), v.end(), 15)-v.begin() // gives starting element maximum element <= 15
       << endl; 

    // Finding the upper bound of value 30
    cout << upper_bound(v.begin(), v.end(), 16)-v.begin(); // gives starting mimimum element > 16 (element)
  
    return 0;
}
// min max 
class Solution {
public:
    // int sum = 0;
    bool possible(int x,vector<int>&quantities,int n){
        int sum=0;
        for(int a: quantities)
            sum+=(a+x-1)/x;// ceil(a/x)
        return sum <= n;
    }
    int minimizedMaximum(int n, vector<int>& quantities) {
        int low = 1,high = *max_element(quantities.begin(),quantities.end());
        // this->sum = accumulate(quantities.begin(),quantities.end(),0);
        while(low<high)
        {
            int mid = low + (high-low)/2;
            if(possible(mid,quantities,n))
            {
                high = mid;
            }
            else
            {
                low = mid+1;
            }
        }
        return high;
    }
};
// min subarray to be removed to have non decreasing array
class Solution {
public:
    int findLengthOfShortestSubarray(vector<int>& arr) {
        const int n = arr.size();
        int left= 0;
        for (; left+1 < n && arr[left] <= arr[left+1]; left++); 
        
        if (left== n-1) return 0;
        
        int right = n-1;
        for (; right>left && arr[right-1] <= arr[right]; right--);     
        
        int remove = min(n-left-1, right);
        for (int i = 0, j = right; i <= left && j < n; ) {
            if (arr[i] <= arr[j]) {
                remove = min(remove, j-i-1);
                i++;
            } 
            else j++;  
        }
        
        return remove;
    }
};
// power of each k sized subarray! if increasing replace highest form array

class Solution {
public:
    vector<int> resultsArray(vector<int>& nums, int k) {
        int n = nums.size(),j=n-k;
        vector<int>ans(j+1,-1);
        if(k==1)
        {
            return nums;
        }
        int i =0;j;
        for(j =0;j<=n-1;j++)
        {
            if(j-i+1>=k)
            {
                ans[j-k+1] = nums[j];
            }
            if(j<n-1 && nums[j] != nums[j+1]-1)
            {
                i = j+1;
            }
        }
        return ans;
    
    }
};
// min length of subarray with sum atleast k ****

class Solution {
public:
    int shortestSubarray(vector<int>& nums, int k) {
        vector<long>prefix;
        deque<long>q;
        long n = nums.size(),sum=0,ans=n+1;
        for(long i =0;i<n;i++)
        {
            sum+= nums[i];
            if(sum>=k)
            {
                ans = min(ans,i+1);
            }
            while(!q.empty() && sum >= prefix[q.front()]+k)
            {
                cout<<i<<" "<<q.front()<<endl;
                ans = min(ans,i-q.front());
                q.pop_front();
            }
            while(!q.empty() && sum < prefix[q.back()])
            {
                q.pop_back();
            }
            q.push_back(i);
            prefix.push_back(sum);
        }
        return ans==n+1?-1:ans;

    }
};

//Take K of Each Character From Left and Right ****
class Solution {
public:
    int takeCharacters(string s, int k) {
        int a=0,b=0,c=0,n=s.size(),i=0,j=0;
        vector<int>freq(3,0);
        for(int i=0;i<n;i++)
        {
            freq[s[i]-'a']++;
        }
        if(*min_element(freq.begin(),freq.end())<k)
        {
            return -1;
        }
        int ans =n;
        while(j<n)
        {
            freq[s[j]-'a']--;
            while(i<=j && *min_element(freq.begin(),freq.end())<k)
            {
                freq[s[i]-'a']++;
                i++;
            }
            ans = min(ans,n-(j-i+1));
            j++;
        }
        return ans;
    }
};
// number puzzle 123450
class Solution {
public:
    int slidingPuzzle(vector<vector<int>>& board) {
        string target = "123450";
        vector<vector<int>>dir = {{1,3},{0,2,4},{1,5},{0,4},{1,3,5},{2,4}};
        set<string>vis;
        queue<string>q;
        string start = "";
        for(int i =0;i<board.size();i++)
        {
            for(int j =0;j<board[0].size();j++)
            {
                start+= board[i][j]+'0';
            }
        }
        int count =0;
        q.push(start);
        while(!q.empty())
        {
            int k = q.size();
            while(k--)
            {
                string tp = q.front();
                q.pop();
                if(tp==target) return count;
                int pl = tp.find('0');
                for(auto move:dir[pl])
                {
                    string next = tp;
                    swap(next[pl],next[move]);
                    if(vis.find(next)==vis.end())
                    {
                        vis.insert(next);
                        q.push(next);
                    }
                }
            }
            count++;
        }
        return -1;
    }
};
//Find Champion II  Node from which we can reach all nodes.

class Solution {
public:
    int findChampion(int n, vector<vector<int>>& edges) {
        vector<int>Indegree(n,0);
        for(int i =0;i<edges.size();i++)
        {
            Indegree[edges[i][1]]++;
        }
        int count =0,node = -1;
        for(int i =0;i<n;i++)
        {
            if(Indegree[i]==0)
            {
                count++;
                node = i;
            }
        }
        return count==1? node : -1;
    }
};
//sub arrays with max - min <=2
class Solution {
public:
    long long continuousSubarrays(vector<int>& nums) {
        int l = 0;
        long long res = 0;  
        deque<int> minD, maxD;

        for (int r = 0; r < nums.size(); r++) {
            while (!minD.empty() && nums[minD.back()] >= nums[r]) minD.pop_back();
            while (!maxD.empty() && nums[maxD.back()] <= nums[r]) maxD.pop_back();
            minD.push_back(r);
            maxD.push_back(r);

            while (nums[maxD.front()] - nums[minD.front()] > 2) {
                l++;
                if (minD.front() < l) minD.pop_front();
                if (maxD.front() < l) maxD.pop_front();
            }

            res += r - l + 1;
        }

        return res;
    }
};
// maximum avg pass ratio
class Solution {
public:
    double maxAverageRatio(vector<vector<int>>& classes, int extraStudents) {
        auto profit = [&](double pass, double total) {
            return (pass + 1)/(total + 1) - pass/total;
        };

        double total = 0;
        priority_queue<pair<double, array<int, 2>>> pq;
        for(int i = 0; i < classes.size(); i++) {
            total += (double)classes[i][0]/classes[i][1];
            pq.push({profit(classes[i][0], classes[i][1]), {classes[i][0], classes[i][1]}});
        }

        while(extraStudents--) {
            auto [pfit, arr] = pq.top(); pq.pop();
            total += pfit;
            pq.push({profit(arr[0]+1, arr[1]+1), {arr[0]+1, arr[1]+1}});
        }
        return total / classes.size();
    }
};
// rearrange string with  maximum consecutive limit L and should be lexicographical longest
class Solution {
public:
    string repeatLimitedString(string s, int Limit) {
        priority_queue<pair<int,int>>pq;
        vector<int>freq(26,0);
        string ans = "";
        for(int i =0;i<s.size();i++)
        {
            freq[s[i]-'a']++;
        }
        for(int i =0;i<26;i++)
        {
            if(freq[i]!=0)pq.push({i,freq[i]});
        }
        while(!pq.empty())
        {
            auto [i,freq] = pq.top();
            pq.pop();
            char c = 'a'+i;
            ans+= string(min(freq,Limit),c);
            // cout<<ans<<endl;
            if(freq>Limit)
            {
                if(pq.empty())
                {
                    return ans;
                }
                else
                {
                    auto [j,freq2] = pq.top();
                    pq.pop();
                    char d = 'a'+j;
                    ans+= d;
                    // cout<<ans<<endl;
                    if(freq2>1)
                    {
                        pq.push({j,freq2-1});
                    }
                }
                pq.push({i,freq-Limit});
            }
        }
        return ans;
    }
};
// max chunks to make it sorted(0 to n-1)
class Solution {
public:
    int maxChunksToSorted(vector<int>& arr) {
        int maxi = 0,n=arr.size(),count =0,mini = n,len=0;
        for(int i =0;i<n;i++)
        {
            mini = min(mini,arr[i]);
            maxi = max(maxi,arr[i]);
            len++;
            if(maxi==i && mini ==i-len+1)
            {
                count++;
                mini = n,maxi =0;
                len =0;
            }
        }
        return count;
    }
};
//(random numbers)
class Solution {
public:
    int maxChunksToSorted(vector<int>& arr) {
        stack<int> st;
        int n = arr.size();
        int curmax = -1;
        for(int i = 0;i<n;i++){
            curmax = max(curmax,arr[i]);
            while(st.size() && st.top()>arr[i]){
                st.pop();
            }
            st.push(curmax);
        }

        return st.size();


    }
};
// spit tree to have parts divisible by k
class Solution {
public:
    int maxKDivisibleComponents(int n, vector<vector<int>>& edges, vector<int>& values, int k) {
        if(n<2)
        {
            return 1;
        }
        vector<vector<int>>adj(n);
        vector<int>degree(n,0);
        int ans =0;
        vector<long long>node(values.begin(),values.end());
        for(auto edge:edges)
        {
            int a = edge[0],b=edge[1];
            adj[a].push_back(b);
            adj[b].push_back(a);
            degree[a]++;
            degree[b]++;
        }
        queue<int>leaf;
        for(int i =0;i<n;i++)
        {
            if(degree[i]==1)
            {
                leaf.push(i);
            }
        }
        while(!leaf.empty())
        {
            int curr = leaf.front();
            leaf.pop();
            degree[curr]--;
            long long carry = 0;
            if(node[curr]%k==0)ans++;
            else
            {
                carry = node[curr];
            }
            for(auto neighbour:adj[curr])
            {
                if(degree[neighbour]==0)continue;
                node[neighbour]+= carry;
                degree[neighbour]--;
                if(degree[neighbour]==1)
                {
                    leaf.push(neighbour);
                }
            }
        }
        return ans;
    }
};
// queries finding element such that c>a & b and index greater than them
class Solution {
public:
    vector<int> leftmostBuildingQueries(vector<int>& heights, vector<vector<int>>& queries) {
        int sz= queries.size(),n=heights.size();
        vector<int>ans(sz,-1);
        vector<vector<pair<int,int>>>defer(n);
        for(int i =0;i<sz;i++)
        {
            int a = queries[i][0],b=queries[i][1];
            if(a>b)
            {
                swap(a,b);
            }
            if(a==b || heights[a] < heights[b])
            {
                ans[i] = b;
            }
            else
            {
                defer[b].push_back({heights[a],i});
            }
        }
        priority_queue<pair<int,int>,vector<pair<int,int>>,greater<pair<int,int>>>pq;
        for (int i =0;i<n;i++)
        {
            for(int j =0;j<defer[i].size();j++)
            {
                pq.push(defer[i][j]);
            }
            while(!pq.empty() && pq.top().first < heights[i])
            {
                ans[pq.top().second] = i;
                pq.pop();
            }
        }
        return ans;
    }
};
// diameter of a graph ~ khans algo
class Solution {
public:
    int find_d(vector<vector<int>>& edges)
    {
        int n = edges.size()+1;
        vector<vector<int>>adj(n);
        vector<int>deg(n,0);
        for(auto edge:edges)
        {
            int a = edge[0],b=edge[1];
            deg[a]++;
            deg[b]++;
            adj[a].push_back(b);
            adj[b].push_back(a);
        }
        int total = n,level = 0;
        queue<int>q;
        for(int i = 0;i<n;i++)
        {
            if(deg[i]==1)
            {
                q.push(i);
            }
        }
        while(total>2)
        {
            int k = q.size();
            total -=k;
            while(k--)
            {
                int node = q.front();
                q.pop();
                for(auto nbr:adj[node])
                {
                    deg[nbr]--;
                    if(deg[nbr]==1)
                    {
                        q.push(nbr);
                    }
                }
            }
            level++;
        }
        return total==2 ? 2*level+1 : 2*level;
    }
    int minimumDiameterAfterMerge(vector<vector<int>>& edges1, vector<vector<int>>& edges2) {
        int d1 = find_d(edges1),d2=find_d(edges2);
        return max(d1,max(d2,(d1+1)/2 + (d2+1)/2+1));
    }
};
// best sight seeing pair
class Solution {
public:
    int maxScoreSightseeingPair(vector<int>& nums) {
        int n = nums.size(),maxi = -1000,j=-1,ans = 0;
        for(int i =0;i<n;i++)
        {
            ans = max(ans,maxi+nums[i]+j-i);
            if(maxi -i + j < nums[i])
            {
                maxi = nums[i];
                j = i;
            }
        }
        return ans;
    }
};
// 3 intervals sum maximum
class Solution {
public:
    vector<int> maxSumOfThreeSubarrays(vector<int>& nums, int k) {
        int n = nums.size();
        vector<int> nums2; // Sums of subarrays of size k
        int sum = 0;

        // Calculate sliding window sums
        for (int i = 0; i < n; i++) {
            sum += nums[i];
            if (i >= k - 1) {
                nums2.push_back(sum);
                sum -= nums[i - k + 1];
            }
        }

        int m = nums2.size();
        vector<int> left(m), right(m);
        int bestLeft = 0;

        // Populate left array
        for (int i = 0; i < m; i++) {
            if (nums2[i] > nums2[bestLeft]) {
                bestLeft = i;
            }
            left[i] = bestLeft;
        }

        // Populate right array
        int bestRight = m - 1;
        for (int i = m - 1; i >= 0; i--) {
            if (nums2[i] >= nums2[bestRight]) {
                bestRight = i;
            }
            right[i] = bestRight;
        }

        // Find the maximum sum of three subarrays
        vector<int> ans(3, -1);
        int maxSum = 0;
        for (int i = k; i <= m - k - 1; i++) {
            int l = left[i - k];
            int r = right[i + k];
            int totalSum = nums2[l] + nums2[i] + nums2[r];
            if (totalSum > maxSum) {
                maxSum = totalSum;
                ans = {l, i, r};
            }
        }

        return ans;
    }
};
// maximum gap between two consecutive elements when array is sorted without sorting
class Solution {
public:
    int maximumGap(vector<int>& arr) {
        int high = *max_element(arr.begin(),arr.end());
        int low = *min_element(arr.begin(),arr.end()),n=arr.size();
        if(n<2)return 0;
        int bsize = max(int((high-low)/(n-1)),1);
        int num_buckets = int((high-low)/bsize)+1;
        vector<vector<int>>buckets(num_buckets+1,vector<int>());
        for(auto &ele:arr)
        {
            buckets[(ele-low)/bsize].push_back(ele);
        }
        int curr_high = 0,prev_high = 0,ans = 0;
        cout<<bsize<<endl;
        for(auto &bucket:buckets)
        {
            if(bucket.size()==0) continue;
            if(!curr_high){
                prev_high = bucket[0];
            }
            int curr_low = bucket[0];
            for(int &num:bucket)
            {
                curr_high = max(curr_high,num);
                curr_low = min(curr_low,num);
            }
            cout<<curr_low<<" "<<prev_high<<endl;
            ans = max(ans,curr_low-prev_high);
            prev_high = curr_high;
        }
        return ans;
    }
};
//circular tour (node from which i can complete circular tour)
class Solution {
  public:
    int circularTour(vector<int>& a1, vector<int>& a2) {
        // Your code here
        int sum = 0;
        int n = a1.size();
        // cout<<n<<endl;
        for(int i =0;i<a1.size();i++)
        {
            sum+= (a1[i]-a2[i]);
            // cout<< a1[i]-a2[i]<<"###";
        }
        // cout<<sum<<endl;
        if(sum<0)
        {
            return -1;
        }
        int i =0,j=0,x=0;
        while(j<n)
        {
            x+= a1[j]-a2[j];
            if(x<0)
            {
                i = j+1;
                x = 0;
            }
            j++;
        }
        return i;
    }
};
// number of 3 sized palindrome substrings from given string
class Solution {
public:
    int countPalindromicSubsequence(string s) {
        vector<vector<int>>range(26,vector<int>{INT_MAX,INT_MIN});
        int n = s.size(),ans=0;
        for(int i =0;i<n;i++)
        {
            range[s[i]-'a'][0] = min(range[s[i]-'a'][0],i);
            range[s[i]-'a'][1] = max(range[s[i]-'a'][1],i);
        }
        for(int i= 0;i<26;i++)
        {
            int st =  range[i][0],e=range[i][1];
            int mask = 0;
            if(st!=INT_MAX && e!=INT_MIN && e-st>1)
            {
                for(int k = st+1;k<e;k++)
                {
                    mask = mask | (1<<(s[k]-'a'));
                }
            }
            ans+= __builtin_popcount(mask);
        }
        return ans;
    }
};
//shifting letters
class Solution {
public:
    string shiftingLetters(string s, vector<vector<int>>& shifts) {
        int n = s.size();
        vector<int> shift(n + 1, 0);
        for (auto& shiftOp : shifts) {
            int start = shiftOp[0], end = shiftOp[1], direction = shiftOp[2];
            shift[start] += (direction == 1 ? 1 : -1);
            shift[end + 1] -= (direction == 1 ? 1 : -1);
        }
        int currentShift = 0;
        for (int i = 0; i < n; ++i) {
            currentShift += shift[i];
            shift[i] = currentShift;
        }

        for (int i = 0; i < n; ++i) {
            int netShift = (shift[i] % 26 + 26) % 26;
            s[i] = 'a' + (s[i] - 'a' + netShift) % 26;
        }

        return s;
    }
};
// min operations to make it to all i's
class Solution {
public:
    vector<int> minOperations(string boxes) {
        int value = 0,count1=0,count2=0,n=boxes.size();
        for(int i =0;i<n;i++)
        {
            if(boxes[i]=='1')
            {
                count1++;
                value+= i;
            }
        }
        vector<int>ans;
        for(int i =0;i<n;i++)
        {
            ans.push_back(value);
            // cout<<count1<<" "<<count2<<" "<<i<<endl;
            if(boxes[i]=='1')
            {
                count2++;
                count1--;
            }
            value += count2-count1;
        }
        return ans;
    }
};
//kmp algorithm
class Solution {
public:
    bool kmp(string &a,string &b)
    {
        if(a.size()>b.size())
        {
            return false;
        }
        int i =1,len=0,n=a.size();
        vector<int>lps(n,0);
        while(i<n)
        {
            if(a[i]==a[len])
            {
                len++;
                lps[i] = len;
                i++;
            }
            else
            {
                if(len>0)
                {
                    len = lps[len-1];
                }
                else 
                {
                 lps[i] =0;
                 i++;
                }
            }
        }
        //search
        i =0,len=0;
        while(i<b.size())
        {
            if(b[i]==a[len])
            {
                len++;
                i++;
                if(len==a.size())
                {
                    return true;
                }
            }
            else
            {
                if(len>0)
                {
                    len = lps[len-1];
                }
                else
                {
                    i++;
                }
            }
        }
        return false;
    }
    vector<string> stringMatching(vector<string>& words) {
        int n =words.size();
        vector<string>ans;
        for(int i =0;i<n;i++)
        {
            for(int j =0;j<n;j++)
            {
                if(j!=i){
                    string a = words[i];
                    string b = words[j];
                    if(kmp(a,b))
                    {
                        ans.push_back(a);
                        break;
                    }
                }
            }
        }
        return ans;
    }
};
// can divide string into k palindromes?
class Solution {
public:
    bool canConstruct(string s, int k) {
        vector<int>freq(26,0);
        int count =0;
        for(int i =0;i<s.size();i++)
        {
            freq[s[i]-'a']++;
        }
        for(int i =0;i<26;i++)
        {
            if(freq[i]%2!=0)
            {
                count++;
            }
        }
        if(count>k)
        {
            return false;
        }
        return s.size()>=k;
    }
};
// Check if a Parentheses String Can Be Valid
class Solution {
public:
    bool canBeValid(string s, string locked) {
        int n = s.size();
        if(n%2==1)
        {
            return false;
        }
        int temp = 0;
        for(int i =0;i<n;i++)
        {
            int op = s[i]=='(',yz=locked[i]=='0';
            temp+= (op || yz)?1:-1;
            if(temp<0)
            {
                return false;
            }
        }
        temp = 0;
        for(int i =n-1;i>=0;i--)
        {
            int op = s[i]==')',yz=locked[i]=='0';
            temp+= (op || yz)?1:-1;
            if(temp<0)
            {
                return false;
            }
        }
        return true;

    }
};
// dijkstras change some values in the grid that we can reach bottom left;
class Solution {
public:
    int minCost(vector<vector<int>>& grid) {
        int n=grid.size();
        int m=grid[0].size();

        int dx[4]={0,0,1,-1};
        int dy[4]={1,-1,0,0};

        int dist[n][m];

        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                dist[i][j]=INT_MAX;
            }
        }
        deque<pair<int,int>>dq;
        dq.push_back({0,0});
        dist[0][0]=0;

        while(dq.size()>0){
            auto curr=dq.front();
            dq.pop_front();
            int x=curr.first;
            int y=curr.second;
            int dir=grid[x][y];

            for(int i=0;i<4;i++){
                int nx=x+dx[i];
                int ny=y+dy[i];

                int edgewt=1;
                if(i+1==dir) edgewt=0;

                if(nx<n and ny<m and nx>=0 and ny>=0){
                    if(dist[nx][ny]>dist[x][y]+edgewt){
                        dist[nx][ny]=dist[x][y]+edgewt;
                        if(edgewt==1){
                            dq.push_back({nx,ny});
                        }else{
                            dq.push_front({nx,ny});
                        }
                    }
                }
            }
        }
        return dist[n-1][m-1];
    }
};
class Solution {
public:
    int m,n;
    int solve(vector<vector<int>>& grid,int i,int j)
    {
        vector<vector<int>>visited(n,vector<int>(m,INT_MAX));
        priority_queue<pair<int,pair<int,int>>,vector<pair<int,pair<int,int>>>,greater<pair<int,pair<int,int>>>>pq;
        pq.push({0,{0,0}});
        while(!pq.empty())
        {
            auto tp = pq.top();
            auto [a,b] = tp.second;
            int c = tp.first;
            pq.pop(); 
            if(visited[a][b]>c)
            {
                visited[a][b] = c;
                cout<<a<<" "<<b<<endl;
            }
            else
            {
                continue;
            }
            int k1 = 1+c,k2=k1,k3=k2,k4=k3;
            if(grid[a][b] == 1)k1--;
            if(grid[a][b] == 2)k2--;
            if(grid[a][b] == 3)k3--;
            if(grid[a][b] == 4)k4--;
            if(b<m-1 && (visited[a][b+1] > k1)) pq.push({k1,{a,b+1}});
            if((b>0) && visited[a][b-1] > k2)pq.push({k2,{a,b-1}});
            if((a<n-1)&&(visited[a+1][b] > k3))pq.push({k3,{a+1,b}});
            if((a>0) && (visited[a-1][b] > k4)) pq.push({k4,{a-1,b}});

        }
        return visited[n-1][m-1];
    }

    int minCost(vector<vector<int>>& grid) {
        n = grid.size(),m=grid[0].size();
        return solve(grid,0,0);
    }
};
//MINIMIZE SCORE OF SECOND ROBOT
class Solution {
public:
    long long gridGame(vector<vector<int>>& grid) {
        int n = grid[0].size();
        vector<long long>prefix(n,0),suffix(n+1,0);
        long long temp1 = 0,temp2=0;
        for(int i =0;i<n;i++)
        {
            temp1 += grid[0][i];
            temp2+= grid[1][n-i-1];
            prefix[i] = temp1;
            suffix[n-i-1] = temp2;
        }
        long long bridge =0,maxi =0,ans=  LLONG_MAX;
        for(int i =0;i<n;i++)
        {
            // if(prefix[i]+suffix[i]>=maxi)
            // {

                bridge = i;
                maxi = prefix[i]+suffix[i];
                ans= min(ans,max(prefix[n-1]-prefix[bridge],suffix[0]-suffix[bridge]));
            // }
        }
        // cout<<bridge<<endl;
        return ans;
    }
};
//Trapping rain water 2 bfs + heap
class Solution {
public:
    int dfs(int i,int j,vector<vector<int>>& ht,vector<vector<int>>& visited)
    {
        vector<int>dx{-1,+1,0,0};
        vector<int>dy{0,0,+1,-1};
        int n = ht.size(),m=ht[0].size(),count=0;
        queue<pair<int,int>>q;
        for(auto x:dx)
        {
            for(auto y:dy)
            {
                int a = i+x,b=j+y;
                if(a<n && b<m && a>=0 && b>=0)
                {
                    q.push({a,b});
                }
                else
                {
                    return 0;
                }
            }
        }
        visited[i][j] = 1;
        int mini = 0;
        while(!q.empty())
        {
            auto [a,b] = q.front();
            q.pop();
            cout<<a<<" "<<b<<" "<<endl;
            if(ht[a][b]>ht[i][j])
            {
                mini = min(mini,ht[a][b]-ht[i][j]);
            }
            else
            {
                if(ht[a][b]==ht[i][j])
                {
                    visited[a][b] = 1;
                }
                count++;
                for(auto x:dx)
                {
                    for(auto y:dy)
                    {
                        int a1 = a+x,b1=b+y;
                        // cout<<
                        if(a1>=n || b1>=m || a1<0 || b1<0) return 0;
                        if(visited[a1][b1]==-1)
                        {
                            q.push({a1,b1});
                        }
                    }
                }
            }
        }
        return count*mini
    }
    int trapRainWater(vector<vector<int>>& heightMap) {
        int n = heightMap.size(),m=heightMap[0].size(),ans=0;
        vector<vector<int>>visited(n,vector<int>(m,-1));
        for(int i =0;i<n;i++)
        {
            for(int j =0;j<m;j++)
            {
                ans+= dfs(i,j,heightMap,visited);
            }
        }
        return ans;
    }
};
//course schedule 4
class Solution {
public:
    vector<bool> checkIfPrerequisite(int numCourses, vector<vector<int>>& prerequisites, vector<vector<int>>& queries) {
        int n = numCourses;
        vector<int>indeg(n,0);
        queue<int>q;
        vector<bitset<100>>prereq(n,0);
        vector<vector<int>> adj(n);
        for(auto pre:prerequisites)
        {
            indeg[pre[1]]++;
            prereq[pre[1]][pre[0]] = 1;
            adj[pre[0]].push_back(pre[1]);
        }
        for(int i =0;i<n;i++)
        {
            if(indeg[i]==0)
            {
                q.push(i);
            }
        }
        while(!q.empty())
        {
                int a = q.front();
                q.pop();
                for(auto nei:adj[a])
                {
                    prereq[nei] |= prereq[a];
                    if(--indeg[nei]==0)
                    {
                        q.push(nei);
                    }
                }
        }
        vector<bool>ans;
        for(auto que:queries)
        {
            int a = que[0],b=que[1];
            ans.push_back(prereq[b][a]);
        }
        return ans;
    }
};
// maximum employees to be invited to a meeting
class Solution {
public:
    int maximumInvitations(vector<int>& favourite) {
        int n = favourite.size();
        vector<int>in_degree(n,0),lengths(n,0);
        vector<bool>vis(n,false);
        vector<int>map(n,0);
        for(int i: favourite)
        {
            in_degree[i]++;
        }
        queue<int>q;
        for(int i = 0;i<n;i++)
        {
            if(in_degree[i]==0)
            {
                q.push(i);
            }
        }
        while(!q.empty())
        {
            int node =  q.front();
            q.pop();
            vis[node] = true;
            int next = favourite[node];
            lengths[next] = 1+lengths[node];
            if(--in_degree[next]==0)
            {
                q.push(next);
            }

        }
        int max_cycle =0,chain =0;
        for(int i =0;i<n;i++)
        {
            int curr = i,cycle_length = 0;
            while(!vis[curr])
            {
                vis[curr] = true;
                curr = favourite[curr];
                cycle_length++;
            }
            if(cycle_length ==2)
            {
                chain += 2+lengths[i]+lengths[favourite[i]];
            }
            else
            {
                max_cycle = max(max_cycle,cycle_length);
            }
        }
        return max(max_cycle,chain);
    }
};
// union find
class Solution {
public:
    vector<int>parent;
    vector<int>rank;
    int Find(int ch)
    {
        int p = parent[ch];
        while(p!=parent[p])
        {
            // parent[p] = parent[parent[p]];
            p = parent[p];
        }
        return p;
    }
    bool Union(vector<int>edge)
    {
        int a = edge[0];
        int b = edge[1];
        int p1 = Find(a);
        int p2 = Find(b);
        if(p1==p2)
        {
            return false;
        }
        if(rank[p1]>rank[p2])
        {
            rank[p1]+=rank[p2];
            parent[p2] = p1;
        }
        else
        {
            rank[p2]+=rank[p1];
            parent[p1] = p2;
        }
        return true;
    }
    vector<int> findRedundantConnection(vector<vector<int>>& edges) {
        int n=edges.size();
        parent.resize(n+1,0);
        rank.resize(n+1,1);
        for(int i =0;i<n+1;i++)
        {
            parent[i] = i;
        }
        for(int i =0;i<n;i++)
        {
            if(!Union(edges[i]))
            {
                return edges[i];
            }
        }
        return {};
    }
};
// longest increasing/decreasing subarray in array
class Solution {
public:
    int longestMonotonicSubarray(vector<int>& nums) {
        int i =0,j=0,n=nums.size(),inc = 1,dec = 1,ans = 1;
        if(n<2)
        {
            return n;
        }
        for(int j =0;j<n-1;j++)
        {
            if(nums[j]<nums[j+1])
            {
                inc++;
                dec = 1;
            }
            else if (nums[j]>nums[j+1])
            {
                dec++;
                inc = 1;
            }
            else
            {
                inc = dec = 1;
            }
            ans = max({ans,inc,dec});
        }

        return ans;
    }
};
// ideal arrays of size n with max value 
class Solution {
public:
    const int mod = 1e9+7;
    int count[15][10001];
    int pref[15][10001];
    int options[15];
    void count_options(int i,int idx,int maxValue)
    {
        options[idx]++;
        for(int j = 2;i*j<=maxValue;j++)
        {
            count_options(i*j,idx+1,maxValue);
        }
    }
    int idealArrays(int n, int maxValue) {
        memset(count,0,sizeof(count));
        memset(pref,0,sizeof(pref));
        memset(options,0,sizeof(options));
        
            
        for(int i=1;i<=n;++i){
            count[1][i]=1;
            pref[1][i]=i;
        }
        for(int j = 2;j<15;j++)
        {
            for(int i = j;i<n+1;i++)
            {
                count[j][i] = pref[j-1][i-1];
                pref[j][i] = (pref[j][i-1]+count[j][i])%mod;
            }
        }
        for(int i =1;i<=maxValue;i++)
        {
            count_options(i,1,maxValue);
        }
        long long ans = 0;
        for(int j =1;j<15;j++)
        {
            ans+= (long long)options[j]*(long long)count[j][n];
            ans%=mod;
        }
       return ans;
    }
};
