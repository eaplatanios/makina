package org.platanios.learn.math.matrix;

import org.junit.Assert;
import org.junit.Test;

/**
 * @author Emmanouil Antonios Platanios
 */
public class SingularValueDecompositionTest {
    @Test
    public void testUForNonSingularMatrix() {
        double[][] testMatrixArray = new double[][] {
                {  3.4000, 1.2000,  2.2000 },
                {  0.1000, 7.4000,  0.5000 },
                { -5.3000, 0.4000, -2.1000 }
        };
        SingularValueDecomposition singularValueDecomposition =
                new SingularValueDecomposition(new Matrix(testMatrixArray));
        double[][] actualResultTemp = singularValueDecomposition.getU().getArray();
        double[][] expectedResultTemp = new double [][] {
                { -0.3668, -0.4577,  0.8099 },
                { -0.8923,  0.4194, -0.1670 },
                {  0.2633,  0.7839,  0.5623 }
        };
        double[] actualResult = new double[actualResultTemp.length * actualResultTemp[0].length];
        double[] expectedResult = new double[expectedResultTemp.length * expectedResultTemp[0].length];
        for (int i = 0; i < actualResultTemp.length; i++) {
            for (int j = 0; j < actualResultTemp[0].length; j++) {
                actualResult[i * actualResultTemp.length + j] = actualResultTemp[i][j];
                expectedResult[i * expectedResultTemp.length + j] = expectedResultTemp[i][j];
            }
        }
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-4);
    }

    @Test
    public void testSForNonSingularMatrix() {
        double[][] testMatrixArray = new double[][] {
                {  3.4000, 1.2000,  2.2000 },
                {  0.1000, 7.4000,  0.5000 },
                { -5.3000, 0.4000, -2.1000 }
        };
        SingularValueDecomposition singularValueDecomposition =
                new SingularValueDecomposition(new Matrix(testMatrixArray));
        double[][] actualResultTemp = singularValueDecomposition.getS().getArray();
        double[][] expectedResultTemp = new double [][] {
                { 7.6717,      0,      0 },
                {      0, 6.8071,      0 },
                {      0,      0, 0.5731 }
        };
        double[] actualResult = new double[actualResultTemp.length * actualResultTemp[0].length];
        double[] expectedResult = new double[expectedResultTemp.length * expectedResultTemp[0].length];
        for (int i = 0; i < actualResultTemp.length; i++) {
            for (int j = 0; j < actualResultTemp[0].length; j++) {
                actualResult[i * actualResultTemp.length + j] = actualResultTemp[i][j];
                expectedResult[i * expectedResultTemp.length + j] = expectedResultTemp[i][j];
            }
        }
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-4);
    }

    @Test
    public void testVForNonSingularMatrix() {
        double[][] testMatrixArray = new double[][] {
                {  3.4000, 1.2000,  2.2000 },
                {  0.1000, 7.4000,  0.5000 },
                { -5.3000, 0.4000, -2.1000 }
        };
        SingularValueDecomposition singularValueDecomposition =
                new SingularValueDecomposition(new Matrix(testMatrixArray));
        double[][] actualResultTemp = singularValueDecomposition.getV().getArray();
        double[][] expectedResultTemp = new double [][] {
                { -0.3560, -0.8328, -0.4238 },
                { -0.9043,  0.4213, -0.0682 },
                { -0.2354, -0.3590,  0.9032 }
        };
        double[] actualResult = new double[actualResultTemp.length * actualResultTemp[0].length];
        double[] expectedResult = new double[expectedResultTemp.length * expectedResultTemp[0].length];
        for (int i = 0; i < actualResultTemp.length; i++) {
            for (int j = 0; j < actualResultTemp[0].length; j++) {
                actualResult[i * actualResultTemp.length + j] = actualResultTemp[i][j];
                expectedResult[i * expectedResultTemp.length + j] = expectedResultTemp[i][j];
            }
        }
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-4);
    }

    @Test
    public void testL2NormForNonSingularMatrix() {
        double[][] testMatrixArray = new double[][] {
                {  3.4000, 1.2000,  2.2000 },
                {  0.1000, 7.4000,  0.5000 },
                { -5.3000, 0.4000, -2.1000 }
        };
        SingularValueDecomposition singularValueDecomposition =
                new SingularValueDecomposition(new Matrix(testMatrixArray));
        double actualResult = singularValueDecomposition.computeL2Norm();
        double expectedResult = 7.6717;
        Assert.assertEquals(expectedResult, actualResult, 1e-4);
    }

    @Test
    public void testConditionNumberForNonSingularMatrix() {
        double[][] testMatrixArray = new double[][] {
                {  3.4000, 1.2000,  2.2000 },
                {  0.1000, 7.4000,  0.5000 },
                { -5.3000, 0.4000, -2.1000 }
        };
        SingularValueDecomposition singularValueDecomposition =
                new SingularValueDecomposition(new Matrix(testMatrixArray));
        double actualResult = singularValueDecomposition.computeConditionNumber();
        double expectedResult = 13.3865;
        Assert.assertEquals(expectedResult, actualResult, 1e-4);
    }

    @Test
    public void testEffectiveNumericalRankForNonSingularMatrix() {
        double[][] testMatrixArray = new double[][] {
                {  3.4000, 1.2000,  2.2000 },
                {  0.1000, 7.4000,  0.5000 },
                { -5.3000, 0.4000, -2.1000 }
        };
        SingularValueDecomposition singularValueDecomposition =
                new SingularValueDecomposition(new Matrix(testMatrixArray));
        double actualResult = singularValueDecomposition.computeEffectiveNumericalRank();
        double expectedResult = 3;
        Assert.assertEquals(expectedResult, actualResult, 0);
    }
}
