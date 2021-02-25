import java.util.*;

public class test {

    public String longestPalindrome(String s) {
        String ans = "";
        int n = s.length();
        boolean[][] dp = new boolean[n][n];
        for (int len = 0; len < n; len++) {
            for (int i = 0; i + len < n; i++){
                int j = i+len;
                if (len ==0)
                    dp[i][j]=true;
                else if (len ==1)
                    dp[i][j]=(s.charAt(i)==s.charAt(j));
                else
                    dp[i][j]=(s.charAt(i)==s.charAt(j) && dp[i+1][j-1]);
                if (dp[i][j] && len+1 >ans.length())
                ans = s.substring(i,j+1);
            }
        }
        return ans;

    }


}
