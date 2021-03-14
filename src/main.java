import java.util.*;

public class main {
    public static void main(String[] args) {
        int[] ns = {3, 2, 1, 5, 2};
        System.out.println(twiceNumber(ns));
    }

    public static int twiceNumber(int[] nums) {
        int n = nums.length;
        //笨办法，建立数组保存出现次数
        int[] map = new int[n + 1];
        for (int i = 0; i < n; ++i)
            map[nums[i]]++;
        for (int i = 0; i < n; ++i)
            if (map[nums[i]] == 2)
                return nums[i];
        return 0;
    }




















}
