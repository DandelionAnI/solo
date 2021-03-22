
import java.math.BigInteger;
import java.util.*;

public class Solution {

    //给定一个非空且只包含非负数的整数数组 nums，数组的度的定义是指数组里任一元素出现频数的最大值。
    //你的任务是在 nums 中找到与 nums 拥有相同大小的度的最短连续子数组，返回其长度。
    public int findShortestSubArry(int[] nums) {
        Map<Integer, int[]> map = new HashMap<Integer, int[]>();
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            if (map.containsKey(nums[i])) {
                map.get(nums[i])[0]++;
                map.get(nums[i])[2] = i;
            } else {
                map.put(nums[i], new int[]{1, i, i});
            }
        }
        int maxNum = 0, minLen = 0;
        for (Map.Entry<Integer, int[]> entry : map.entrySet()) {
            int[] arr = entry.getValue();
            if (maxNum < arr[0]) {
                maxNum = arr[0];
                minLen = arr[2] - arr[1] + 1;
            } else if (maxNum == arr[0]) {
                if (minLen > arr[2] - arr[1] + 1) {
                    minLen = arr[2] - arr[1] + 1;
                }
            }

        }
        return minLen;
    }


    //1. 两数之和
    public int[] twoSum(int[] nums, int target) {
        int[] ans = new int[2];
        for (int i = 0; i < nums.length; i++) {
            for (int j = i + 1; j < nums.length; j++) {
                if (target == nums[i] + nums[j]) {

                    ans[0] = i;
                    ans[1] = j;
                    return ans;
                }
            }
        }
        return ans;
    }

    //1. 两数之和hash
    public int[] twoSumHash(int[] nums, int target) {
        Map<Integer, Integer> hashtable = new HashMap<Integer, Integer>();
        for (int i = 0; i < nums.length; i++) {
            if (hashtable.containsKey(target - nums[i])) {
                return new int[]{hashtable.get(target - nums[i]), i};
            }
            hashtable.put(nums[i], i);
        }
        return new int[0];
    }

    //46 全排列
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        List<Integer> output = new ArrayList<Integer>();
        for (int num : nums) {
            output.add(num);
        }

        int n = nums.length;
        backtrack(n, output, res, 0);
        return res;
    }

    public void backtrack(int n, List<Integer> output, List<List<Integer>> res, int first) {
        // 所有数都填完了
        if (first == n) {
            res.add(new ArrayList<Integer>(output));
        }
        for (int i = first; i < n; i++) {
            // 动态维护数组
            Collections.swap(output, first, i);
            // 继续递归填下一个数
            backtrack(n, output, res, first + 1);
            // 撤销操作
            Collections.swap(output, first, i);
        }
    }
    //时间复杂度：O(n \times n!)O(n×n!)，其中 nn 为序列的长度。
    //空间复杂度：O(n)O(n)，其中 nn 为序列的长度。

    //832 翻转图像
    public int[][] flipAndInvertImage(int[][] A) {
        int n = A.length;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n / 2; j++) {
                if (A[i][j] == A[i][n - j - 1]) {
                    A[i][j] ^= 1;
                    A[i][n - j - 1] ^= 1;
                }
            }
            if (n % 2 != 0)
                A[i][n / 2] ^= 1;
        }
        return A;
    }

    //2.两数相加
    private class ListNode {
        int val;
        ListNode next;

        ListNode() {
        }

        ListNode(int val) {
            this.val = val;
        }

        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }
    }

    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode head = null, tail = null;
        int carry = 0;
        while (l1 != null || l2 != null) {
            int n1 = l1 != null ? l1.val : 0;
            int n2 = l2 != null ? l2.val : 0;
            int sum = n1 + n2 + carry;
            if (head == null) {
                head = tail = new ListNode(sum % 10);
            } else {
                tail.next = new ListNode(sum % 10);
                tail = tail.next;
            }
            carry = sum / 10;
            if (l1 != null) l1 = l1.next;
            if (l2 != null) l2 = l2.next;
        }
        //链表最后需要进位
        if (carry > 0)
            tail.next = new ListNode(carry);
        return head;
    }

    //4 中位数【mark】
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int length1 = nums1.length, length2 = nums2.length;
        int totalLength = length1 + length2;
        if (totalLength % 2 == 1) {
            int midIndex = totalLength / 2;
            double median = getKthElement(nums1, nums2, midIndex + 1);
            return median;
        } else {
            int midIndex1 = totalLength / 2 - 1, midIndex2 = totalLength / 2;
            double median = (getKthElement(nums1, nums2, midIndex1 + 1) + getKthElement(nums1, nums2, midIndex2 + 1)) / 2.0;
            return median;
        }
    }

    public int getKthElement(int[] nums1, int[] nums2, int k) {
        /* 主要思路：要找到第 k (k>1) 小的元素，那么就取 pivot1 = nums1[k/2-1] 和 pivot2 = nums2[k/2-1] 进行比较
         * 这里的 "/" 表示整除
         * nums1 中小于等于 pivot1 的元素有 nums1[0 .. k/2-2] 共计 k/2-1 个
         * nums2 中小于等于 pivot2 的元素有 nums2[0 .. k/2-2] 共计 k/2-1 个
         * 取 pivot = min(pivot1, pivot2)，两个数组中小于等于 pivot 的元素共计不会超过 (k/2-1) + (k/2-1) <= k-2 个
         * 这样 pivot 本身最大也只能是第 k-1 小的元素
         * 如果 pivot = pivot1，那么 nums1[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums1 数组
         * 如果 pivot = pivot2，那么 nums2[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums2 数组
         * 由于我们 "删除" 了一些元素（这些元素都比第 k 小的元素要小），因此需要修改 k 的值，减去删除的数的个数
         */

        int length1 = nums1.length, length2 = nums2.length;
        int index1 = 0, index2 = 0;
        int kthElement = 0;

        while (true) {
            //边界情况
            if (index1 == length1)
                return nums2[index2 + k - 1];
            if (index2 == length2)
                return nums1[index1 + k - 1];
            if (k == 1)
                return Math.min(nums1[index1], nums2[index2]);

            //正常情况
            int half = k / 2;
            int newIndex1 = Math.min(index1 + half, length1) - 1;
            int newIndex2 = Math.min(index2 + half, length2) - 1;
            int pivot1 = nums1[newIndex1], pivot2 = nums2[newIndex2];
            if (pivot1 <= pivot2) {
                k -= (newIndex1 - index1 + 1);
                index1 = newIndex1 + 1;
            } else {
                k -= (newIndex2 - index2 + 1);
                index2 = newIndex2 + 1;
            }

        }
    }


    //867. 转置矩阵
    public int[][] transpose(int[][] matrix) {
        int n = matrix.length;
        int m = matrix[0].length;
        int[][] ans = new int[m][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                ans[j][i] = matrix[i][j];
            }
        }
        return ans;
    }

    //5. 最长回文子串
    //dp方法,substring的endindex要+1
    public String longestPalindrome(String s) {
        int n = s.length();
        boolean[][] dp = new boolean[n][n];
        String ans = "";
        for (int len = 0; len < n; len++) {
            for (int i = 0; i + len < n; i++) {
                int j = i + len;
                if (len == 0)
                    dp[i][j] = true;
                else if (len == 1)
                    dp[i][j] = (s.charAt(i) == s.charAt(j));
                else
                    dp[i][j] = (s.charAt(i) == s.charAt(j) && dp[i + 1][j - 1]);
                if (dp[i][j] && len + 1 > ans.length())
                    ans = s.substring(i, j + 1);
            }
        }
        return ans;
    }

    //104二叉树最大深度 深度优先、广度优先搜索
    private class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode() {
        }

        TreeNode(int val) {
            this.val = val;
        }

        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

    public int maxDepth1(TreeNode root) {
        if (root == null) return 0;
        else {
            int leftHeight = maxDepth1(root.left);
            int rightHeight = maxDepth1(root.right);
            return Math.max(leftHeight, rightHeight) + 1;
        }
    }

    public int maxDepth2(TreeNode root) {
        if (root == null)
            return 0;
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        queue.offer(root);
        int ans = 0;
        while (!queue.isEmpty()) {
            int size = queue.size();
            while (size > 0) {
                TreeNode node = queue.poll();
                if (node.left != null)
                    queue.offer(node.left);
                if (node.right != null)
                    queue.offer(node.right);
                size--;
            }
            ans++;
        }
        return ans;
    }

    //14.最长公共前缀
    public String longestCommonPrefix(String[] strs) {
        if (strs == null || strs.length == 0)
            return "";
        for (int i = 0; i < strs[0].length(); i++) {
            char c = strs[0].charAt(i);
            for (int j = 1; j < strs.length; j++) {
                if (i == strs[j].length() || (i != 0 && strs[j].charAt(i) != c))
                    return strs[0].substring(0, i);
                else if (i == 0 && strs[j].charAt(i) != c)
                    return "";
            }
        }
        return strs[0];
    }


    //395至少有K个重复字符的最长子串 递归
    public int longestSubstring(String s, int k) {
        if (s.length() < k) return 0;
        Map<Character, Integer> map = new HashMap<Character, Integer>();
        for (int i = 0; i < s.length(); i++)
            map.put(s.charAt(i), map.getOrDefault(s.charAt(i), 0) + 1);
        for (char c : map.keySet()) {
            if (map.get(c) < k) {
                int ans = 0;
                for (String t : s.split(String.valueOf(c)))
                    ans = Math.max(ans, longestSubstring(t, k));
                return ans;
            }
        }
        return s.length();
    }

    //395至少有K个重复字符的最长子串 分治法
    public int longestSubstring2(String s, int k) {
        int n = s.length();
        return dfs(s, 0, n - 1, k);
    }

    public int dfs(String s, int l, int r, int k) {
        int[] cnt = new int[26];
        for (int i = l; i <= r; i++)
            cnt[s.charAt(i) - 'a']++;

        char split = 0;
        for (int i = 0; i < 26; i++) {
            if (cnt[i] > 0 && cnt[i] < k) {
                split = (char) (i + 'a');
                break;
            }
        }
        if (split == 0)
            return r - l + 1;

        int i = l;
        int ans = 0;
        while (i <= r) {
            while (i <= r && s.charAt(i) == split)
                i++;
            if (i > r)
                break;

            int start = i;
            while (i <= r && s.charAt(i) != split)
                i++;

            int length = dfs(s, start, i - 1, k);
            ans = Math.max(ans, length);
        }
        return ans;
    }


    //395至少有K个重复字符的最长子串 滑动窗口
    public int longestSubstring1(String s, int k) {
        int ret = 0;
        int n = s.length();
        for (int t = 1; t < 27; t++) {
            int l = 0, r = 0;
            int[] cnt = new int[26];
            int tot = 0;
            int less = 0;
            while (r < n) {
                int i = s.charAt(r) - 'a';
                cnt[i]++;
                if (cnt[i] == 1) {
                    tot++;
                    less++;
                }
                if (cnt[i] == k)
                    less--;

                while (tot > t) {
                    int j = s.charAt(l) - 'a';
                    cnt[j]--;
                    if (cnt[i] == k - 1)
                        less++;

                    if (cnt[i] == 0) {
                        tot--;
                        less--;
                    }
                    l++;
                }
                if (less == 0)
                    ret = Math.max(ret, r - l + 1);
                r++;
            }
        }
        return ret;
    }

    //896单调数列
    public boolean isMonotonic(int[] A) {
        int n = A.length - 1;
        if (A[0] < A[n]) {
            for (int i = 0; i < n; i++) {
                if (A[i] > A[i + 1])
                    return false;
            }
            return true;
        } else {
            for (int i = 0; i < n; i++) {
                if (A[i] < A[i + 1])
                    return false;
            }
            return true;
        }
    }

    //11盛最多水的容器 双指针
    public int maxArea(int[] height) {
        int left = 0, right = height.length - 1;
        int ans = 0;
        while (left < right) {
            int area = Math.min(height[left], height[right]) * (right - left);
            ans = Math.max(ans, area);
            if (height[left] <= height[right])
                left++;
            else right--;
        }
        return ans;
    }


    //303 区域和检索-数组不可变
    private class NumArray {
        int[] num;

        public NumArray(int[] nums) {
            num = new int[nums.length + 1];
            for (int i = 0; i < num.length; ++i)
                num[i + 1] = num[i] + nums[i];
        }

        public int sumRange(int i, int j) {
            return num[j + 1] - num[i];
        }
    }

    //20有效的括号
    public boolean isValid(String s) {
        int n = s.length();
        if (n % 2 == 1) return false;
        Map<Character, Character> pairs = new HashMap<Character, Character>() {{
            put(')', '(');
            put(']', '[');
            put('}', '{');
        }};
        Deque<Character> stack = new LinkedList<Character>();
        for (int i = 0; i < n; ++i) {
            char ch = s.charAt(i);
            if (pairs.containsKey(ch)) {
                if (stack.isEmpty() || stack.peek() != pairs.get(ch))
                    return false;
                stack.pop();
            } else
                stack.push(ch);
        }
        return stack.isEmpty();
    }

    //70爬楼梯
    public int climbStairs(int n) {
        if (n == 1 || n == 2) return n;
        int[] dp = new int[n];
        dp[0] = 1;
        dp[1] = 2;
        for (int i = 2; i < n; ++i)
            dp[i] = dp[i - 2] + dp[i - 1];
        return dp[n - 1];
    }

    //1178. 猜字谜
    public List<Integer> findNumOfValidWords(String[] words, String[] puzzles) {
        Map<Integer, Integer> frequency = new HashMap<Integer, Integer>();

        for (String word : words) {
            int mask = 0;
            for (int i = 0; i < word.length(); ++i) {
                char ch = word.charAt(i);
                mask |= (1 << (ch - 'a'));
            }
            if (Integer.bitCount(mask) <= 7) {
                frequency.put(mask, frequency.getOrDefault(mask, 0) + 1);
            }
        }

        List<Integer> ans = new ArrayList<Integer>();
        for (String puzzle : puzzles) {
            int total = 0;

            // 枚举子集方法一
            // for (int choose = 0; choose < (1 << 6); ++choose) {
            //     int mask = 0;
            //     for (int i = 0; i < 6; ++i) {
            //         if ((choose & (1 << i)) != 0) {
            //             mask |= (1 << (puzzle.charAt(i + 1) - 'a'));
            //         }
            //     }
            //     mask |= (1 << (puzzle.charAt(0) - 'a'));
            //     if (frequency.containsKey(mask)) {
            //         total += frequency.get(mask);
            //     }
            // }

            // 枚举子集方法二
            int mask = 0;
            for (int i = 1; i < 7; ++i) {
                mask |= (1 << (puzzle.charAt(i) - 'a'));
            }
            int subset = mask;
            do {
                int s = subset | (1 << (puzzle.charAt(0) - 'a'));
                if (frequency.containsKey(s)) {
                    total += frequency.get(s);
                }
                subset = (subset - 1) & mask;
            } while (subset != mask);

            ans.add(total);
        }
        return ans;
    }


    //7整数反转
    public int reverse(int x) {
        int rev = 0;
        while (x != 0) {
            int pop = x % 10;
            x /= 10;
            if (rev > Integer.MAX_VALUE / 10 || (rev == Integer.MAX_VALUE / 10 && pop > 7)) return 0;
            if (rev < Integer.MIN_VALUE / 10 || (rev == Integer.MIN_VALUE / 10 && pop < -8)) return 0;
            rev = rev * 10 + pop;
        }
        return rev;
    }

    //8字符串转换整数
    //DFA自动机
    private class Automaton {
        public int sign = 1;
        public long ans = 0;
        private String state = "start";
        private Map<String, String[]> table = new HashMap<String, String[]>() {{
            put("start", new String[]{"start", "signed", "in_number", "end"});
            put("signed", new String[]{"end", "end", "in_number", "end"});
            put("in_number", new String[]{"end", "end", "in_number", "end"});
            put("end", new String[]{"end", "end", "end", "end"});
        }};

        public void get(char c) {
            state = table.get(state)[get_col(c)];
            if ("in_number".equals(state)) {
                ans = ans * 10 + c - '0';

                ans = sign == 1 ? Math.min(ans, (long) Integer.MAX_VALUE) : Math.min(ans, -(long) Integer.MIN_VALUE);
            } else if ("signed".equals(state))
                sign = c == '+' ? 1 : -1;
        }


        private int get_col(char c) {
            if (c == ' ')
                return 0;
            if (c == '+' || c == '-')
                return 1;
            if (Character.isDigit(c))
                return 2;
            return 3;
        }

    }

    public int myAtoi(String str) {
        Automaton automaton = new Automaton();
        int length = str.length();
        for (int i = 0; i < length; i++)
            automaton.get(str.charAt(i));
        return (int) (automaton.sign * automaton.ans);
    }

    //9回文数--转字符串
    public boolean isPalindrome1(int x) {
        if (x < 0) return false;
        String s = String.valueOf(x);
        String ans = new StringBuffer(s).reverse().toString();
        if (s.equals(ans)) return true;
        else return false;
    }

    //9回文数 不转字符串
    public boolean isPalindrome2(int x) {
        if (x < 0) return false;
        int ans = 0;
        int y = x;
        while (y != 0) {
            ans = ans * 10 + y % 10;
            y = y / 10;
        }
        if (ans == x) return true;
        else return false;
    }


    //15 三数之和 双指针
    public List<List<Integer>> threeSum(int[] nums) {
        int n = nums.length;
        Arrays.sort(nums);
        List<List<Integer>> ans = new ArrayList<List<Integer>>();
        if (n < 3) return ans;
        for (int i = 0; i < n; ++i) {
            if (i > 0 && nums[i] == nums[i - 1]) continue;
            int k = n - 1;
            int need = -nums[i];
            for (int j = i + 1; j < n; ++j) {
                if (j > i + 1 && nums[j] == nums[j - 1]) continue;
                while (j < k && nums[j] + nums[k] > need) --k;
                if (j == k) break;
                if (nums[j] + nums[k] == need) {
                    List<Integer> adde = new ArrayList<Integer>();
                    adde.add(nums[i]);
                    adde.add(nums[j]);
                    adde.add(nums[k]);
                    ans.add(adde);
                }
            }
        }
        return ans;
    }

    //16三数之和
    public int threeSumClosest(int[] nums, int target) {
        int n = nums.length;
        int best = 10000000;
        Arrays.sort(nums);
        for (int i = 0; i < n; ++i) {
            if (i > 0 && nums[i] == nums[i - 1]) continue;
            int j = i + 1;
            int k = n - 1;
            int need = target - nums[i];
            while (j < k) {
                int sum = nums[i] + nums[j] + nums[k];
                if (sum == target) return target;
                if (Math.abs(sum - target) < Math.abs(best - target))
                    best = sum;
                if (sum > target) {
                    int k0 = k - 1;
                    while (j < k0 && nums[k0] == nums[k]) --k0;
                    k = k0;
                } else {
                    int j0 = j + 1;
                    while (j0 < k && nums[j0] == nums[j]) ++j0;
                    j = j0;
                }
            }
        }
        return best;
    }


    //304二维数组子数组
    private class NumMatrix {


        int[][] sums;

        public NumMatrix(int[][] matrix) {
            int m = matrix.length;
            if (m > 0) {
                int n = matrix[0].length;
                sums = new int[m + 1][n + 1];
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        sums[i + 1][j + 1] = sums[i][j + 1] + sums[i + 1][j + 1] - sums[i][j] + matrix[i][j];
                    }
                }
            }
        }

        public int sumRegion(int row1, int col1, int row2, int col2) {
            return sums[row2 + 1][col2 + 1] - sums[row1][col2 + 1] - sums[row2 + 1][col1] + sums[row1][col1];
        }
    }

    //21合并两个有序链表 递归
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null) return l2;
        else if (l2 == null) return l1;
        else if (l1.val < l2.val) {
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;
        } else {
            l2.next = mergeTwoLists(l1, l2.next);
            return l2;
        }
    }

    //21合并两个有序链表 迭代
    public ListNode mergeTwoLists2(ListNode l1, ListNode l2) {
        ListNode prehead = new ListNode(-1);
        ListNode prev = prehead;
        while (l1 != null && l2 != null) {
            if (l1.val > l2.val) {
                prev.next = l2;
                l2 = l2.next;
            } else {
                prev.next = l1;
                l1 = l1.next;
            }
            prev = prev.next;
        }
        prev.next = l1 == null ? l2 : l1;
        return prehead.next;
    }

    //338比特位计数
    public int[] countBits(int num) {
        int[] bits = new int[num + 1];
        int highBit = 0;
        for (int i = 1; i <= num; i++) {
            if ((i & (i - 1)) == 0)
                highBit = i;
            bits[i] = bits[i - highBit] + 1;
        }
        return bits;
    }

    //354俄罗斯套娃
    public int maxEnvelopes(int[][] envelopes) {
        return 0;
    }

    //232用栈实现队列

    /**
     * Your MyQueue object will be instantiated and called as such:
     * MyQueue obj = new MyQueue();
     * obj.push(x);
     * int param_2 = obj.pop();
     * int param_3 = obj.peek();
     * boolean param_4 = obj.empty();
     */
    private class MyQueue {
        Deque<Integer> inStack;
        Deque<Integer> outStack;

        public MyQueue() {
            inStack = new LinkedList<Integer>();
            outStack = new LinkedList<Integer>();
        }

        /**
         * Push element x to the back of queue.
         */
        public void push(int x) {
            inStack.push(x);
        }

        /**
         * Removes the element from in front of queue and returns that element.
         */
        public int pop() {
            if (outStack.isEmpty())
                in2out();
            return outStack.pop();
        }

        /**
         * Get the front element.
         */
        public int peek() {
            if (outStack.isEmpty())
                in2out();
            return outStack.peek();
        }

        /**
         * Returns whether the queue is empty.
         */
        public boolean empty() {
            return inStack.isEmpty() && outStack.isEmpty();
        }

        private void in2out() {
            while (!inStack.isEmpty())
                outStack.push(inStack.pop());
        }
    }

    //26 删除排序数组中的重复项
    public int removeDuplicates(int[] nums) {
        int i = 0;
        int sum = 0;
        for (int j = 1; j < nums.length; j++) {
            if (nums[j] != nums[i]) {
                i++;
                nums[i] = nums[j];
            }
        }
        return i + 1;
    }

    //206翻转链表 迭代
    public ListNode reverseList(ListNode head) {
        ListNode prev = null;
        ListNode curr = head;
        while (curr != null) {
            ListNode next = curr.next;
            curr.next = prev;
            prev = curr;
            curr = next;
        }
        return prev;
    }

    //206翻转链表 递归
    public ListNode reverseList1(ListNode head) {
        if (head == null || head.next == null)
            return head;
        ListNode newHead = reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return newHead;
    }

    //1047删除字符串中的所有相邻重复项
    public String removeDuplicates(String S) {
        StringBuilder builder = new StringBuilder();
        int top = -1;
        for (int i = 0; i < S.length(); ++i) {
            char ch = S.charAt(i);
            if (top >= 0 && builder.charAt(top) == ch) {
                builder.deleteCharAt(top);
                --top;
            } else {
                builder.append(ch);
                ++top;
            }
        }
        return builder.toString();
    }

    //225 两个队列实现栈
    private class MyStack {

        Queue<Integer> queue1;
        Queue<Integer> queue2;

        /**
         * Initialize your data structure here.
         */
        public MyStack() {
            queue1 = new LinkedList<>();
            queue2 = new LinkedList<>();
        }

        /**
         * Push element x onto stack.
         */
        public void push(int x) {
            queue2.offer(x);
            while (!queue1.isEmpty()) queue2.offer(queue1.poll());
            Queue<Integer> temp = queue1;
            queue1 = queue2;
            queue2 = temp;
        }

        /**
         * Removes the element on top of the stack and returns that element.
         */
        public int pop() {
            return queue1.poll();
        }

        /**
         * Get the top element.
         */
        public int top() {
            return queue1.peek();
        }

        /**
         * Returns whether the stack is empty.
         */
        public boolean empty() {
            return queue1.isEmpty();
        }
    }

    //225 一个队列实现栈
    private class MyStack2 {

        Queue<Integer> queue;

        /**
         * Initialize your data structure here.
         */
        public MyStack2() {
            queue = new LinkedList<>();
        }

        /**
         * Push element x onto stack.
         */
        public void push(int x) {
            int n = queue.size();
            queue.offer(x);
            for (int i = 0; i < n; i++)
                queue.offer(queue.poll());
        }

        /**
         * Removes the element on top of the stack and returns that element.
         */
        public int pop() {
            return queue.poll();
        }

        /**
         * Get the top element.
         */
        public int top() {
            return queue.peek();
        }

        /**
         * Returns whether the stack is empty.
         */
        public boolean empty() {
            return queue.isEmpty();
        }
    }

    /**
     * Your MyStack object will be instantiated and called as such:
     * MyStack obj = new MyStack();
     * obj.push(x);
     * int param_2 = obj.pop();
     * int param_3 = obj.top();
     * boolean param_4 = obj.empty();
     */

    //146 LRU缓存机制
    private class LRUCache {

        class DLinkedNode {
            int key;
            int value;
            DLinkedNode prev;
            DLinkedNode next;

            public DLinkedNode() {
            }

            public DLinkedNode(int key, int value) {
                this.key = key;
                this.value = value;
            }

        }

        private Map<Integer, DLinkedNode> cache = new HashMap<>();
        private int size;
        private int capacity;
        private DLinkedNode head, tail;

        public LRUCache(int capacity) {
            this.size = 0;
            this.capacity = capacity;
            //使用伪头部和伪尾部节点
            head = new DLinkedNode();
            tail = new DLinkedNode();
            head.next = tail;
            tail.prev = head;
        }

        public int get(int key) {
            DLinkedNode node = cache.get(key);
            if (node == null) return -1;
            //如果key存在，先通过哈希表定位，再移到头部
            moveToHead(node);
            return node.value;
        }

        public void put(int key, int value) {
            DLinkedNode node = cache.get(key);
            if (node == null) {
                DLinkedNode newNode = new DLinkedNode(key, value);
                cache.put(key, newNode);
                addToHead(newNode);
                ++size;
                if (size > capacity) {
                    DLinkedNode tail = removeTail();
                    cache.remove(tail.key);
                    --size;
                }
            } else {
                node.value = value;
                moveToHead(node);
            }
        }

        private void addToHead(DLinkedNode node) {
            node.prev = head;
            node.next = head.next;
            head.next.prev = node;
            head.next = node;
        }

        private void removeNode(DLinkedNode node) {
            node.prev.next = node.next;
            node.next.prev = node.prev;
        }

        private void moveToHead(DLinkedNode node) {
            removeNode(node);
            addToHead(node);
        }

        private DLinkedNode removeTail() {
            DLinkedNode res = tail.prev;
            removeNode(res);
            return res;
        }

    }

    /**
     * Your LRUCache object will be instantiated and called as such:
     * LRUCache obj = new LRUCache(capacity);
     * int param_1 = obj.get(key);
     * obj.put(key,value);
     */

    //136 只出现一次的数字
    public int singleNumber(int[] nums) {
        int single = 0;
        for (int num : nums)
            single ^= num;
        return single;
    }

    //53 最大子序和
    public int maxSubArray(int[] nums) {
        int pre = 0, maxAns = nums[0];
        for (int x : nums) {
            pre = Math.max(pre + x, x);
            maxAns = Math.max(maxAns, pre);
        }
        return maxAns;
    }


    //41缺失的第一个正数
    public int firstMissingPositive(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n; ++i)
            if (nums[i] <= 0)
                nums[i] = n + 1;
        for (int i = 0; i < n; ++i) {
            int num = Math.abs(nums[i]);
            if (num <= n)
                nums[num - 1] = -Math.abs(nums[num - 1]);
        }
        for (int i = 0; i < n; ++i)
            if (nums[i] > 0)
                return i + 1;
        return n + 1;
    }

    //88 合并两个有序数组 双指针从前往后
    public void merge1(int[] nums1, int m, int[] nums2, int n) {
        int[] nums1_copy = new int[m];
        System.arraycopy(nums1, 0, nums1_copy, 0, m);
        int p1 = 0, p2 = 0;
        int p = 0;
        while (p1 < m && p2 < n)
            nums1[p++] = (nums1_copy[p1] < nums2[p2]) ? nums1_copy[p1++] : nums2[p2++];
        if (p1 < m)
            System.arraycopy(nums1_copy, p1, nums1, p1 + p2, m + n - p1 - p2);
        if (p2 < n)
            System.arraycopy(nums2, p2, nums2, p1 + p2, m + n - p1 - p2);
    }

    //88 合并两个有序数组 双指针从后往前
    public void merge(int[] nums1, int m, int[] nums2, int n) {

        int p1 = m - 1, p2 = n - 1;
        int p = m + n - 1;
        while (p1 > 0 && p2 > 0)
            nums1[p--] = (nums1[p1] > nums2[p2]) ? nums1[p1--] : nums2[p2--];

        System.arraycopy(nums2, 0, nums1, 0, p2 + 1);
    }

    //160相交链表

    /**
     * Definition for singly-linked list.
     * public class ListNode {
     * int val;
     * ListNode next;
     * ListNode(int x) {
     * val = x;
     * next = null;
     * }
     * }
     */
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) return null;

        ListNode p1 = headA;
        ListNode p2 = headB;

        while (p1 != p2) {
            p1 = p1 == null ? headB : p1.next;
            p2 = p2 == null ? headA : p2.next;
        }
        return p1;
    }

    //128最长连续序列
    public int longestConsecutive(int[] nums) {
        Set<Integer> num_set = new HashSet<>();
        for (int num : nums)
            num_set.add(num);
        int max = 0;
        for (int num : num_set) {
            if (!num_set.contains(num - 1)) {
                int curnum = num;
                int curstr = 1;

                while (num_set.contains(curnum + 1)) {
                    curnum += 1;
                    curstr += 1;
                }

                max = Math.max(max, curstr);
            }
        }
        return max;
    }


    //121.买卖股票的最佳时机
    public int maxProfit(int[] prices) {
        int minprice = Integer.MAX_VALUE;
        int maxprofit = 0;
        for (int i = 0; i < prices.length; ++i) {
            if (prices[i] < minprice) minprice = prices[i];
            else if (prices[i] - minprice > maxprofit) maxprofit = prices[i] - minprice;
        }
        return maxprofit;
    }

    //122.买卖股票的最佳时机 2
    public int maxProfit2(int[] prices) {
        int ans = 0;
        for (int i = 0; i < prices.length; ++i)
            ans += Math.max(0, prices[i] - prices[i - 1]);
        return ans;
    }

    //33搜索旋转排序数组
    public int search(int[] nums, int target) {
        int n = nums.length;
        if (n == 0) return -1;
        if (n == 1) return nums[0] == target ? 0 : -1;
        int l = 0, r = n - 1;
        while (l <= r) {
            int mid = (l + r) / 2;
            if (nums[mid] == target) return mid;
            if (nums[0] <= nums[mid]) {
                if (nums[0] <= target && nums[mid] > target)
                    r = mid - 1;
                else
                    l = mid + 1;
            } else {
                if (nums[mid] < target && nums[n - 1] >= target)
                    l = mid + 1;
                else
                    r = mid - 1;
            }
        }
        return -1;
    }

    //215 数组中的第K个大元素 堆

    public int findKthLargest(int[] nums, int k) {
        int heapSize = nums.length;
        buildMaxHeap(nums, heapSize);
        for (int i = nums.length - 1; i >= nums.length - k + 1; --i) {
            swap(nums, 0, i);
            --heapSize;
            maxHeapify(nums, 0, heapSize);
        }
        return nums[0];
    }

    public void buildMaxHeap(int[] a, int heapSize) {
        for (int i = heapSize / 2; i >= 0; --i)
            maxHeapify(a, i, heapSize);
    }

    public void maxHeapify(int[] a, int i, int heapSize) {
        int l = i * 2 + 1, r = i * 2 + 2, largest = i;
        if (l < heapSize && a[l] > a[largest])
            largest = l;
        if (r < heapSize && a[r] > a[largest])
            largest = r;
        if (largest != i) {
            swap(a, i, largest);
            maxHeapify(a, largest, heapSize);
        }
    }

    public void swap(int[] a, int i, int j) {
        int temp = a[i];
        a[i] = a[j];
        a[j] = temp;
    }


    //字节一面
    public static int solution(String s) {
        if (s == null) return 0;
        int n = s.length();
        int[] dp = new int[n + 1];
        dp[0] = 1;
        if (s.charAt(0) == '0')
            dp[1] = 0;
        else dp[1] = 1;

        for (int i = 2; i <= n; i++) {
            int p1 = Integer.valueOf(s.substring(i - 1, i));
            int p2 = Integer.valueOf(s.substring(i - 2, i));
            //1-9
            if (p1 > 0 && p1 <= 9) {
                dp[i] = dp[i] + dp[i - 1];
            }
            //10-26   10/20
            if (p2 > 9 && p2 < 27) {
                dp[i] = dp[i] + dp[i - 2];
            }
        }
        return dp[n];
    }

    //91
    public int numDecodings(String s) {
        if (s == null && s.charAt(0) == '0') return 0;
        int n = s.length();
        int dp[] = new int[n + 1];
        dp[0] = 0;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            int one = Integer.valueOf(s.substring(i - 1, i));
            int two = Integer.valueOf(s.substring(i - 2, i));
            if (one > 0 && one <= 9)
                dp[i] = dp[i - 1] + dp[i];
            if (two > 9 && one <= 26)
                dp[i] = dp[i - 2] + dp[i];
        }
        return dp[n];
    }

    //1143.最长公共子序列
    public int longestCommonSubsequence(String text1, String text2) {
        int n = text1.length(), m = text2.length();
        int[][] dp = new int[n + 1][m + 1];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                char c1 = text1.charAt(i);
                char c2 = text2.charAt(j);
                if (c1 == c2)
                    dp[i + 1][j + 1] = dp[i][j] + 1;
                else
                    dp[i + 1][j + 1] = Math.max(dp[i + 1][j], dp[i][j + 1]);
            }
        }
        return dp[n][m];
    }

    //31下一个排列
    public void nextPermutation(int[] nums) {
        int n = nums.length;
        for (int i = n - 1; i > 0; i--) {
            if (nums[i] > nums[i - 1]) {
                Arrays.sort(nums, i, n);
                for (int j = i; j < n; ++j) {
                    if (nums[j] > nums[i - 1]) {
                        int temp = nums[i - 1];
                        nums[i - 1] = nums[j];
                        nums[j] = temp;
                        return;
                    }
                }
            }
        }
        Arrays.sort(nums);
        return;
    }

    //459 重复的字符串
    public boolean repeatedSubstringPattern(String s) {
        String str = s + s;
        return str.substring(1, str.length() - 1).contains(s);
    }

    //3.无重复字符的最长子串
    public int lengthOfLongestSubstring(String s) {
        if (s == null) return 0;
        Set<Character> set = new HashSet<>();
        int n = s.length();
        int right = -1;
        int ans = 0;
        for (int left = 0; left < n; ++left) {
            if (left != 0)
                set.remove(s.charAt(left - 1));
            while (right + 1 < n && !set.contains(s.charAt(right + 1))) {
                set.add(s.charAt(right + 1));
                ++right;
            }
            ans = Math.max(ans, right - left + 1);
        }
        return ans;
    }

    //208实现前缀树Trie
    class TrieNode {

        private TrieNode[] links;

        private final int R = 26;

        private boolean isEnd;

        public TrieNode() {
            links = new TrieNode[R];
        }

        public boolean containsKey(char ch) {
            return links[ch - 'a'] != null;
        }

        public TrieNode get(char ch) {
            return links[ch - 'a'];
        }

        public void put(char ch, TrieNode node) {
            links[ch - 'a'] = node;
        }

        public void setEnd() {
            isEnd = true;
        }

        public boolean isEnd() {
            return isEnd;
        }
    }

    class Trie {
        private TrieNode root;

        public Trie() {
            root = new TrieNode();
        }

        //插入键到树中
        public void insert(String word) {
            TrieNode node = root;
            for (int i = 0; i < word.length(); ++i) {
                char currentChar = word.charAt(i);
                if (!node.containsKey(currentChar)) {
                    node.put(currentChar, new TrieNode());
                }
                node = node.get(currentChar);
            }
            node.setEnd();
        }

        //查找键
        private TrieNode searchPrefix(String word) {
            TrieNode node = root;
            for (int i = 0; i < word.length(); ++i) {
                char currentChar = word.charAt(i);
                if (node.containsKey(currentChar)) {
                    node = node.get(currentChar);
                } else {
                    return null;
                }
            }
            return node;
        }

        // Returns if the word is in the trie.
        public boolean search(String word) {
            TrieNode node = searchPrefix(word);
            return node != null && node.isEnd();
        }

        //查找键前缀
        public boolean startsWith(String prefix) {
            TrieNode node = searchPrefix(prefix);
            return node != null;
        }

    }

    /**
     * Your Trie object will be instantiated and called as such:
     * Trie obj = new Trie();
     * obj.insert(word);
     * boolean param_2 = obj.search(word);
     * boolean param_3 = obj.startsWith(prefix);
     */

    //微软
    public static int solution1(int N) {
        int[] arr = new int[10];
        String s = Integer.toString(N);

        for (char c : s.toCharArray()) {
            arr[c - '0']++;
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 9; i >= 0; i--) {
            for (int j = 0; j < arr[i]; j++) {
                sb.append((char) ('0' + i));
            }
        }
        BigInteger res = new BigInteger(sb.toString());
        if (res.compareTo(new BigInteger("100000000")) >= 0) {
            return -1;
        } else {
            return res.intValue();
        }
    }

    //大数操作
    public void Bignumber() {
        String s = "1837519383";
        BigInteger b1 = new BigInteger(s);

    }

    //887 丢鸡蛋 递归
    Map<Integer, Integer> memo = new HashMap<Integer, Integer>();

    public int superEggDrop(int k, int n) {
        return dp(k, n);
    }

    public int dp(int k, int n) {
        if (!memo.containsKey(n * 100 + k)) {
            int ans;
            if (n == 0)
                ans = 0;
            else if (k == 1)
                ans = n;
            else {
                int low = 1, high = n;
                while (low + 1 < high) {
                    int x = (low + high) / 2;
                    int t1 = dp(k - 1, x - 1);
                    int t2 = dp(k, n - x);

                    if (t1 < t2) low = x;
                    else if (t1 > t2) high = x;
                    else low = high = x;
                }
                ans = 1 + Math.min(Math.max(dp(k - 1, low - 1), dp(k, n - low)), Math.max(dp(k - 1, high - 1), dp(k, n - high)));
            }
            memo.put(n * 100 + k, ans);
        }
        return memo.get(n * 100 + k);
    }

    //887 丢鸡蛋 状态表
    public int superEggDrop1(int K, int N) {
        // dp[i][j]：一共有 i 层楼梯的情况下，使用 j 个鸡蛋的最少仍的次数
        int[][] dp = new int[N + 1][K + 1];

        // 初始化
        for (int i = 0; i <= N; i++) {
            Arrays.fill(dp[i], i);
        }
        for (int j = 0; j <= K; j++) {
            dp[0][j] = 0;
        }

        dp[1][0] = 0;
        for (int j = 1; j <= K; j++) {
            dp[1][j] = 1;
        }
        for (int i = 0; i <= N; i++) {
            dp[i][0] = 0;
            dp[i][1] = i;
        }

        // 开始递推
        for (int i = 2; i <= N; i++) {
            for (int j = 2; j <= K; j++) {
                // 在区间 [1, i] 里确定一个最优值
                int left = 1;
                int right = i;
                while (left < right) {
                    // 找 dp[k - 1][j - 1] <= dp[i - mid][j] 的最大值 k
                    int mid = left + (right - left + 1) / 2;

                    int breakCount = dp[mid - 1][j - 1];
                    int notBreakCount = dp[i - mid][j];
                    if (breakCount > notBreakCount) {
                        // 排除法（减治思想）写对二分见第 35 题，先想什么时候不是解
                        // 严格大于的时候一定不是解，此时 mid 一定不是解
                        // 下一轮搜索区间是 [left, mid - 1]
                        right = mid - 1;
                    } else {
                        // 这个区间一定是上一个区间的反面，即 [mid, right]
                        // 注意这个时候取中间数要上取整，int mid = left + (right - left + 1) / 2;
                        left = mid;
                    }
                }
                // left 这个下标就是最优的 k 值，把它代入转移方程 Math.max(dp[k - 1][j - 1], dp[i - k][j]) + 1) 即可
                dp[i][j] = Math.max(dp[left - 1][j - 1], dp[i - left][j]) + 1;
            }
        }
        return dp[N][K];
    }

    //141 环形链表
    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null) return false;
        ListNode first = head.next;
        ListNode second = head;

        while (first != second) {
            if (first == null || first.next == null)
                return false;
            first = first.next.next;
            second = second.next;
        }
        return true;
    }

    //169多数元素
    public int majorityElement(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums)
            if (!map.containsKey(num))
                map.put(num, 1);
            else
                map.put(num, map.get(num) + 1);

        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            if (entry.getValue() > nums.length / 2)
                return entry.getKey();
        }
        return 0;
    }

    //217存在重复元素
    public boolean containsDuplicate(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            if (set.contains(num)) return true;
            else set.add(num);
        }
        return false;
    }

    //235二叉树的最近公共祖先 一次便利
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        TreeNode ancestor = root;
        while (true) {
            if (p.val < ancestor.val && q.val < ancestor.val)
                ancestor = ancestor.left;
            else if (p.val > ancestor.val && q.val > ancestor.val)
                ancestor = ancestor.right;
            else break;
        }
        return ancestor;
    }

    //61 旋转链表
    public ListNode rotateRight(ListNode head, int k) {
        if (head == null) return null;
        if (head.next == null) return head;

        ListNode old_tail = head;
        int n;
        for (n = 1; old_tail.next != null; n++)
            old_tail = old_tail.next;
        old_tail.next = head;

        ListNode new_tail = head;
        for (int i = 0; i < n - k % n - 1; i++)
            new_tail = new_tail.next;
        ListNode new_head = new_tail.next;
        new_tail.next = null;

        return new_head;
    }

    //62 不同路径
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++)
            dp[i][0] = 1;
        for (int i = 0; i < n; i++)
            dp[0][i] = 1;
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++)
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
        }
        return dp[m][n];
    }

    //142环形链表
    public ListNode detectCycle(ListNode head) {
        if (head == null) return null;
        ListNode slow = head;
        ListNode fast = head;

        while (fast != null) {
            if (fast.next != null)
                fast = fast.next.next;
            else return null;
            slow = slow.next;

            if (fast == slow) {
                ListNode ptr = head;
                while (ptr != slow) {
                    ptr = ptr.next;
                    slow = slow.next;
                }
                return ptr;
            }
        }
        return null;
    }

    //237 nt题目

    //231 2的幂
    public boolean isPowerOfTwo(int n) {
        Set<Double> set = new HashSet<>();
        for (int i = 0; i < 31; i++) {
            set.add(Math.pow(2, i));
            System.out.println(Math.pow(2, i));
        }
        if (set.contains((double) n)) return true;
        else return false;
    }

    //344 反转字符串
    public void reverseString(char[] s) {
        char c;
        for (int i = 0; i < s.length / 2; i++) {
            c = s[i];
            s[i] = s[s.length - i - 1];
            s[s.length - i - 1] = c;
        }
    }

    //541反转字符串2
    public String reverseStr(String s, int k) {
        char[] cs = s.toCharArray();
        int n = s.length();
        int i = 0;
        while ((i + 2 * k) < n) {
            char c;
            for (int j = 0; j < k / 2; ++j) {
                c = cs[i + j];
                cs[i + j] = cs[i + k - j];
                cs[i + k - j] = c;
            }
            i = i + 2 * k;
        }
        if (n - i < k) {
            char c;
            for (int j = 0; j < (n - i) / 2; ++j) {
                c = cs[i + j];
                cs[i + j] = cs[n - j];
                cs[n - j] = c;
            }
        } else {
            char c;
            for (int j = 0; j < k / 2; ++j) {
                c = cs[i + j];
                cs[i + j] = cs[i + k - j];
                cs[i + k - j] = c;
            }
        }
        return new String(cs);
    }

    //557反转字符串中的单词3
    public String reverseWords(String s) {
        StringBuilder sb = new StringBuilder();
        int length = s.length();
        int i = 0;
        while (i < length) {
            int start = i;
            while (i < length && s.charAt(i) != ' ')
                i++;
            for (int p = start; p < i; p++)
                sb.append(s.charAt(start + i - 1 - p));
            while (i < length && s.charAt(i) == ' ') {
                i++;
                sb.append(' ');
            }
        }
        return sb.toString();
    }

    //43字符串相乘

    //腾讯一笔 3题

    //148排序链表
    public ListNode sortList(ListNode head) {
        if (head == null) return head;
        int length = 0;
        ListNode node = head;
        while (node != null) {
            length++;
            node = node.next;
        }
        ListNode dummyHead = new ListNode(0, head);

        //subLen每次左移一位（即sublen = sublen*2） PS:位运算对CPU来说效率更高
        for (int subLength = 1; subLength < length; subLength <<= 1) {
            ListNode prev = dummyHead, curr = dummyHead.next;
            while (curr != null) {
                ListNode head1 = curr;
                for (int i = 1; i < subLength && curr.next != null; i++)
                    curr = curr.next;
                ListNode head2 = curr.next;
                curr.next = null;
                curr = head2;
                for (int i = 1; i < subLength && curr != null && curr.next != null; i++)
                    curr = curr.next;
                ListNode next = null;
                if (curr != null) {
                    next = curr.next;
                    curr.next = null;
                }
                ListNode merged = merge148(head1, head2);
                prev.next = merged;
                while (prev.next != null)
                    prev = prev.next;
                curr = next;
            }
        }
        return dummyHead.next;

    }

    public ListNode merge148(ListNode head1, ListNode head2) {
        ListNode dummyHead = new ListNode(0);
        ListNode temp = dummyHead, temp1 = head1, temp2 = head2;
        while (temp1 != null && temp2 != null) {

            if (temp1.val <= temp2.val) {
                temp.next = temp1;
                temp1 = temp1.next;
            } else {
                temp.next = temp2;
                temp2 = temp2.next;
            }
            temp = temp.next;
        }

        if (temp1 != null)
            temp.next = temp1;
        else if (temp2 != null)
            temp.next = temp2;
        return dummyHead.next;
    }

    //198 打家劫舍
    public int rob(int[] nums) {
        int n = nums.length;
        int[] dp = new int[n + 1];
        dp[0] = 0;
        dp[1] = nums[0];
        for (int i = 2; i <= n; i++)
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i - 1]);
        return dp[n];
    }

    //213 抢劫2
    public int rob2(int[] nums) {
        int n = nums.length;
        if (n == 0) return 0;
        if (n == 1) return nums[0];
        return Math.max(dp2(Arrays.copyOfRange(nums, 0, n - 1)),
                dp2(Arrays.copyOfRange(nums, 1, n)));
    }

    private int dp2(int[] nums) {
        int n = nums.length;
        int[] dp = new int[n + 1];
        dp[0] = 0;
        dp[1] = nums[0];
        for (int i = 2; i <= n; i++)
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i - 1]);
        return dp[n];
    }

    //124二叉树中的最大路径和
    int maxSum = Integer.MIN_VALUE;

    public int maxPathSum(TreeNode root) {
        maxGain(root);
        return maxSum;
    }

    public int maxGain(TreeNode node) {
        if (node == null) return 0;
        int leftGain = Math.max(maxGain(node.left), 0);
        int rightGain = Math.max(maxGain(node.right), 0);

        int priceNewpath = node.val + leftGain + rightGain;

        maxSum = Math.max(maxSum, priceNewpath);

        return node.val + Math.max(leftGain, rightGain);
    }

    //78子集
    List<Integer> t = new ArrayList<Integer>();
    List<List<Integer>> ans = new ArrayList<>();

    public List<List<Integer>> subsets(int[] nums) {
        dfs3(0, nums);
        return ans;
    }

    public void dfs3(int cur, int[] nums) {
        if (cur == nums.length) {
            ans.add(new ArrayList<Integer>(t));
            return;
        }
        t.add(nums[cur]);
        dfs3(cur + 1, nums);
        t.remove(t.size() - 1);
        dfs3(cur + 1, nums);
    }

    //167 两数之和 II - 输入有序数组
    public int[] twoSum2(int[] numbers, int target) {
        int n = numbers.length;
        int left = 0, right = n - 1;
        int[] ans = new int[2];

        while (left != right) {
            if (numbers[left] + numbers[right] == target)
                return new int[]{left + 1, right + 1};
            else if (numbers[left] + numbers[right] > target)
                right--;
            else if (numbers[left] + numbers[right] < target)
                left++;
        }
        return null;
    }

    //633 平方数之和
    public boolean judgeSquareSum(int c) {
        if (c <= 2) return true;
        int left = 0, right = (int) Math.sqrt((double) c);
        while (left != right) {
            if (left * left + right * right == c)
                return true;
            else if (left * left + right * right > c)
                right--;
            else if (left * left + right * right < c)
                left++;
        }
        return false;
    }

    //345 反转字符串中的元音字母
    public String reverseVowels(String s) {
        char[] c = s.toCharArray();
        int n = s.length();
        int left = 0, right = n - 1;
        while (left < right) {
            while (left < n && !isVowel(c[left]))
                left++;
            while (right >= 0 && !isVowel(c[right]))
                right--;
            if (left >= right) break;
            char temp = c[left];
            c[left] = c[right];
            c[right] = temp;
            left++;
            right--;
        }
        return new String(c);
    }

    private boolean isVowel(char ch) {
        return ch == 'a' || ch == 'e' || ch == 'i' || ch == 'o' || ch == 'u'
                || ch == 'A' || ch == 'E' || ch == 'I' || ch == 'O' || ch == 'U';
    }
}
