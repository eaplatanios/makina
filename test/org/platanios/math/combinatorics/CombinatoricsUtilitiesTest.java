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
        List<ArrayList<Integer>> obtainedResult = CombinatoricsUtilities.getCombinations(5, 3);
        List<ArrayList<Integer>> correctResult = new ArrayList<ArrayList<Integer>>();
        correctResult.add(new ArrayList<Integer>(Arrays.asList(0, 1, 2)));
        correctResult.add(new ArrayList<Integer>(Arrays.asList(0, 1, 3)));
        correctResult.add(new ArrayList<Integer>(Arrays.asList(0, 1, 4)));
        correctResult.add(new ArrayList<Integer>(Arrays.asList(0, 2, 3)));
        correctResult.add(new ArrayList<Integer>(Arrays.asList(0, 2, 4)));
        correctResult.add(new ArrayList<Integer>(Arrays.asList(0, 3, 4)));
        correctResult.add(new ArrayList<Integer>(Arrays.asList(1, 2, 3)));
        correctResult.add(new ArrayList<Integer>(Arrays.asList(1, 2, 4)));
        correctResult.add(new ArrayList<Integer>(Arrays.asList(1, 3, 4)));
        correctResult.add(new ArrayList<Integer>(Arrays.asList(2, 3, 4)));
        Assert.assertTrue(obtainedResult.equals(correctResult));
    }
}
