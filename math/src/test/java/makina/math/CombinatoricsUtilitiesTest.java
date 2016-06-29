package makina.math;

import org.junit.Assert;
import org.junit.Test;

/**
 * @author Emmanouil Antonios Platanios
 */
public class CombinatoricsUtilitiesTest {
    @Test
    public void testGetBinomialCoefficient() {
        int actualResult = CombinatoricsUtilities.getBinomialCoefficient(10, 3);
        int expectedResult = 120;
        Assert.assertEquals(expectedResult, actualResult);
    }

    @Test
    public void testGetCombinations() {
        int[][] actualResult = CombinatoricsUtilities.getCombinations(5, 3);
        int[][] expectedResult = new int[][] {
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
        Assert.assertArrayEquals(expectedResult, actualResult);
    }

    @Test
    public void testGetCombinationsOfIntegers() {
        int[][] actualResult = CombinatoricsUtilities.getCombinationsOfIntegers(new int[] { 1, 7, 5, 3, 4 }, 3);
        int[][] expectedResult = new int[][]{
                {1, 7, 5},
                {1, 7, 3},
                {1, 7, 4},
                {1, 5, 3},
                {1, 5, 4},
                {1, 3, 4},
                {7, 5, 3},
                {7, 5, 4},
                {7, 3, 4},
                {5, 3, 4}
        };
        Assert.assertArrayEquals(expectedResult, actualResult);
    }
}
