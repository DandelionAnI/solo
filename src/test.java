import java.util.*;

public class test {

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
                if (Math.abs(target - sum) < Math.abs(target - best))
                    best = sum;
                if (nums[j] + nums[k] > need) {
                    int k0 = k-1;
                    while (k0 > j && nums[k0] == nums[k]) k0--;
                    k = k0;
                } else {
                    int j0 = j+1;
                    while (j0 < k && nums[j0] == nums[j]) j0++;
                    j = j0;
                }

            }
        }
        return best;


    }


}
