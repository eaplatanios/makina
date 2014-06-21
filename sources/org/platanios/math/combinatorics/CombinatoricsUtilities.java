package org.platanios.math.combinatorics;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class CombinatoricsUtilities {
//    public static int factorial(int n) {
//        return (n == 0 || n == 1) ? 1 : n * factorial(n - 1);
//    }

    public static int binomialCoefficient(int n, int k) {
        if ((n == k) || (k == 0)) {
            return 1;
        } else if ((k == 1) || (k == n - 1)) {
            return n;
        } else if (k > n / 2) {
            return binomialCoefficient(n, n - k);
        } else {
            int result = 1;
            int i = n - k + 1;
            for (int j = 1; j <= k; j++) {
                result = result * i / j;
                i++;
            }
            return result;
        }
    }

//    public static List<ArrayList<Integer>> getCombinations(int n, int k) {
//        return getCombinations(n, k, 0);
//    }
//
//    private static List<ArrayList<Integer>> getCombinations(int n, int k, int startIndex) {
//        List<ArrayList<Integer>> indexes;
//
//        if (k == 1) {
//            indexes = new ArrayList<ArrayList<Integer>>();
//            for (int i = startIndex; i < n; i++) {
//                ArrayList<Integer> temp_result = new ArrayList<Integer>();
//                temp_result.add(i);
//                indexes.add(temp_result);
//            }
//        } else {
//            indexes = new ArrayList<ArrayList<Integer>>();
//            for (int i = startIndex; i < n - k + 1; i++) {
//                List<ArrayList<Integer>> inner_indexes = getCombinations(n, k - 1, i + 1);
//                for (ArrayList<Integer> index : inner_indexes) {
//                    index.add(0, i);
//                }
//                indexes.addAll(inner_indexes);
//            }
//        }
//
//        return indexes;
//    }

    public static int[][] getCombinations(int n, int k) {
        return getCombinations(n, k, 0);
    }

    private static int[][] getCombinations(int n, int k, int startIndex) {
        int[][] combinations = new int[binomialCoefficient(n - startIndex, k)][k];
        int combinationIndex = 0;

        if (k == 1) {
            for (int i = startIndex; i < n; i++) {
                combinations[combinationIndex++][0] = i;
            }
        } else {
            for (int i = startIndex; i < n - k + 1; i++) {
                int[][] inner_indexes = getCombinations(n, k - 1, i + 1);
                for (int[] index : inner_indexes) {
                    combinations[combinationIndex][0] = i;
                    System.arraycopy(index, 0, combinations[combinationIndex++], 1, k - 1);
                }
            }
        }

        return combinations;
    }
}
