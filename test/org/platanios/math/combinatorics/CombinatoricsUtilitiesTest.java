package org.platanios.math.combinatorics;

import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class CombinatoricsUtilitiesTest {
    @Test
    public void testGetCombinations() {
        int[][] obtainedResult = CombinatoricsUtilities.getCombinations(5, 3);
        int[][] correctResult = new int[][] {
                { 0, 1, 2 },
                { 0, 1, 3 },
                { 0, 1, 4 },
                { 0, 2, 3 },
                { 0, 2, 4 },
                { 0, 3, 4 },
                { 1, 2, 3 },
                { 1, 2, 4 },
                { 1, 3, 4 },
                { 2, 3, 4 }
        };
        Assert.assertArrayEquals(obtainedResult, correctResult);
    }
}
