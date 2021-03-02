

import javax.swing.tree.TreeNode;
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
        else if (l1.val < l2.val){
            l1.next = mergeTwoLists(l1.next,l2);
            return l1;
        }else{
            l2.next = mergeTwoLists(l1,l2.next);
            return l2;
        }
    }

    //21合并两个有序链表 迭代
    public ListNode mergeTwoLists2(ListNode l1, ListNode l2) {
        ListNode prehead = new ListNode(-1);
        ListNode prev = prehead;
        while(l1!=null && l2!=null){
            if (l1.val>l2.val){
                prev.next = l2;
                l2=l2.next;
            }else {
                prev.next = l1;
                l1 = l1.next;
            }
            prev = prev.next;
        }
        prev.next = l1 ==null ? l2:l1;
        return prehead.next;
    }

}
