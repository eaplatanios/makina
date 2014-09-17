package org.platanios.learn.math.combinatorics;

/**
 * A collection of several static methods for performing combinatorics related tasks.
 *
 * @author Emmanouil Antonios Platanios
 */
public class CombinatoricsUtilities {
    // Suppress default constructor for noninstantiability
    private CombinatoricsUtilities() {
        throw new AssertionError();
    }

    /**
     * Computes the binomial coefficient indexed by {@code n} and {@code k}, also known as "{@code n} choose {@code k}".
     * The value of that coefficient represent the total number of ways in which {@code k} items can be picked out of
     * {@code n} possible items, irrespective of the order in which they are picked.
     *
     * @param   n   The total number of items
     * @param   k   The number of items to pick
     * @return      The number of possible combinations
     */
    public static int getBinomialCoefficient(int n, int k) {
        if ((n == k) || (k == 0)) {
            return 1;
        } else if ((k == 1) || (k == n - 1)) {
            return n;
        } else if (k > n / 2) {
            return getBinomialCoefficient(n, n - k);
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

    /**
     * Computes all possible ways of picking {@code k} items out of {@code n} items irrespective of the ordering of
     * those items. The items are represented as all integers between 0 (inclusive) and n (exclusive).
     *
     * @param   n   The total number of items
     * @param   k   The number of items to pick
     * @return      An array containing one integer array for each possible combination of items, where each integer in
     *              that array corresponds to an item
     */
    public static int[][] getCombinations(int n, int k) {
        return getCombinations(n, k, 0);
    }

    /**
     * Computes all possible ways of picking {@code k} items out of {@code n} items irrespective of the ordering of
     * those items and without using items with value &lt; {@code startIndex}. The items are represented as all integers
     * between 0 (inclusive) and n (exclusive). This method is used in the implementation of
     * {@link #getCombinations(int, int)}.
     *
     * @param   n           The total number of items
     * @param   k           The number of items to pick
     * @param   startIndex  Integer specifying which items to use (items with value less than that integer are not used)
     * @return              An array containing one integer array for each possible combination of items, where each
     *                      integer in that array corresponds to an item
     */
    private static int[][] getCombinations(int n, int k, int startIndex) {
        int[][] combinations = new int[getBinomialCoefficient(n - startIndex, k)][k];
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

    /**
     * Computes all possible ways of picking {@code k} integers out of a specified set of integers irrespective of their
     * ordering.
     *
     * @param   index   The set of all possible integers out of which to pick the combinations
     * @param   k       The number of integers to pick
     * @return          An array containing one integer array for each possible combination of integers
     */
    public static int[][] getCombinationsOfIntegers(int[] index, int k) {
        int[][] inner_indexes = getCombinations(index.length, k);
        int[][] keyCombinations = new int[inner_indexes.length][k];
        for (int i = 0; i < inner_indexes.length; i++) {
            for (int j = 0; j < k; j++) {
                keyCombinations[i][j] = index[inner_indexes[i][j]];
            }
        }

        return keyCombinations;
    }
}
