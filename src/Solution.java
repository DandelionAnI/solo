import java.util.*;

public class Solution {

    //给定一个非空且只包含非负数的整数数组 nums，数组的度的定义是指数组里任一元素出现频数的最大值。
    //你的任务是在 nums 中找到与 nums 拥有相同大小的度的最短连续子数组，返回其长度。
    public int findShortestSubArry(int[] nums){
        Map<Integer,int[]> map = new HashMap<Integer, int[]>();
        int n = nums.length;
        for(int i =0; i < n; i++){
            if(map.containsKey(nums[i])) {
                map.get(nums[i])[0]++;
                map.get(nums[i])[2] = i;
            } else {
                map.put(nums[i], new int[]{1,i,i});
            }
        }
        int maxNum = 0, minLen = 0;
        for(Map.Entry<Integer,int[]> entry : map.entrySet()){
            int[] arr = entry.getValue() ;
            if(maxNum < arr[0]){
                maxNum = arr[0];
                minLen = arr[2] - arr[1] +1;
            } else if (maxNum == arr[0]){
                if (minLen > arr[2] - arr[1] +1){
                    minLen = arr[2] - arr[1] +1;
                }
            }

        }
        return minLen;
    }


    //1. 两数之和
    public int[] twoSum(int[]nums, int target){
        int[] ans = new int[2];
        for(int i= 0; i<nums.length ; i++){
            for(int j=i+1; j<nums.length; j++){
                if(target == nums[i]+ nums[j]){

                    ans[0]=i;
                    ans[1]=j;
                    return ans;
                }
            }
        }
        return ans;
    }

    //1. 两数之和hash
    public int[] twoSumHash(int[]nums, int target){
        Map<Integer,Integer> hashtable = new HashMap<Integer, Integer>();
        for (int i =0; i<nums.length; i++){
            if (hashtable.containsKey(target - nums[i])){
                return new int[]{hashtable.get(target-nums[i]),i};
            }
            hashtable.put(nums[i],i);
        }
        return new int[0];
    }

    //46 全排列
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        List<Integer> output = new ArrayList<Integer>();
        for(int num : nums){
            output.add(num);
        }

        int n = nums.length;
        backtrack(n,output,res,0);
        return res;
    }

    public void backtrack(int n, List<Integer> output, List<List<Integer>> res, int first){
        // 所有数都填完了
        if(first == n){
            res.add(new ArrayList<Integer>(output));
        }
        for(int i = first; i < n ; i++){
            // 动态维护数组
            Collections.swap(output,first,i);
            // 继续递归填下一个数
            backtrack(n,output,res,first+1);
            // 撤销操作
            Collections.swap(output,first,i);
        }
    }
    //时间复杂度：O(n \times n!)O(n×n!)，其中 nn 为序列的长度。
    //空间复杂度：O(n)O(n)，其中 nn 为序列的长度。

}
