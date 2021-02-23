import java.util.*;

public class test {
    public List<List<Integer>> shownum(int[] nums) {
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
        if (n == first) {
            res.add(new ArrayList<Integer>(output));
        }
        for (int i = 0; i < n; i++) {
            Collections.swap(output,n,first);
            backtrack(n,output,res,first+1);
            Collections.swap(output,n,first);
        }
    }

}
