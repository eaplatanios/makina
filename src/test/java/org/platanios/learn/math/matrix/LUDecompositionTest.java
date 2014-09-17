package org.platanios.learn.math.matrix;

import org.junit.Assert;
import org.junit.Test;

/**
 * @author Emmanouil Antonios Platanios
 */
public class LUDecompositionTest {
    @Test
    public void testLForNonSingularMatrix() {
        double[][] testMatrixArray = new double[][] {
                {  3.4000, 1.2000,  2.2000 },
                {  0.1000, 7.4000,  0.5000 },
                { -5.3000, 0.4000, -2.1000 }
        };
        LUDecomposition luDecomposition = new LUDecomposition(new Matrix(testMatrixArray));
        double[][] actualResultTemp = luDecomposition.getL().getArray();
        double[][] expectedResultTemp = new double [][] {
                {  1.0000,      0,      0 },
                { -0.0189, 1.0000,      0 },
                { -0.6415, 0.1966, 1.0000 }
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
    public void testUForNonSingularMatrix() {
        double[][] testMatrixArray = new double[][] {
                {  3.4000, 1.2000,  2.2000 },
                {  0.1000, 7.4000,  0.5000 },
                { -5.3000, 0.4000, -2.1000 }
        };
        LUDecomposition luDecomposition = new LUDecomposition(new Matrix(testMatrixArray));
        double[][] actualResultTemp = luDecomposition.getU().getArray();
        double[][] expectedResultTemp = new double [][] {
                { -5.3000, 0.4000, -2.1000 },
                {       0, 7.4075,  0.4604 },
                {       0,      0,  0.7623 }
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
    public void testPivotForNonSingularMatrix() {
        double[][] testMatrixArray = new double[][] {
                {  3.4000, 1.2000,  2.2000 },
                {  0.1000, 7.4000,  0.5000 },
                { -5.3000, 0.4000, -2.1000 }
        };
        LUDecomposition luDecomposition = new LUDecomposition(new Matrix(testMatrixArray));
        int[] actualResult = luDecomposition.getPivot();
        int[] expectedResult = new int [] { 2, 1, 0 };
        Assert.assertArrayEquals(expectedResult, actualResult);
    }

    @Test
    public void testDeterminantForNonSingularMatrix() {
        double[][] testMatrixArray = new double[][] {
                {  3.4000, 1.2000,  2.2000 },
                {  0.1000, 7.4000,  0.5000 },
                { -5.3000, 0.4000, -2.1000 }
        };
        LUDecomposition luDecomposition = new LUDecomposition(new Matrix(testMatrixArray));
        double actualResult = luDecomposition.computeDeterminant();
        double expectedResult = 29.9280;
        Assert.assertEquals(expectedResult, actualResult, 1e-4);
    }

    @Test
    public void testIsNonSingularForSingularMatrix() {
        double[][] testMatrixArray = new double[][] {
                {  1, 0, 0 },
                { -2, 0, 0 },
                {  4, 6, 1 }
        };
        LUDecomposition luDecomposition = new LUDecomposition(new Matrix(testMatrixArray));
        Assert.assertEquals(false, luDecomposition.isNonSingular());
    }

    @Test
    public void testVectorSolveForNonSingularMatrix() throws SingularMatrixException {
        double[][] testMatrixArray = new double[][] {
                { 28.4100, 11.4400,  21.1100 },
                { 11.4400,  9.7600,  18.0800 },
                { 21.1100, 18.0800, 106.5300 }
        };
        double[] tempVectorArray = new double[] { 1.12, 3.40, 2.10 };
        LUDecomposition luDecomposition = new LUDecomposition(new Matrix(testMatrixArray));
        double[] actualResult = luDecomposition.solve(new Vector(tempVectorArray)).getArray();
        double[] expectedResult = new double [] { -0.1913, 0.6795, -0.0577 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-4);
    }

    @Test
    public void testMatrixSolveForNonSingularMatrix() throws SingularMatrixException {
        double[][] testMatrixArray1 = new double[][] {
                { 28.4100, 11.4400,  21.1100 },
                { 11.4400,  9.7600,  18.0800 },
                { 21.1100, 18.0800, 106.5300 }
        };
        double[][] tempMatrixArray2 = new double[][] {
                { 1.12,  5.43 },
                { 3.40,  1.20 },
                { 2.10, -2.10 }
        };
        LUDecomposition luDecomposition = new LUDecomposition(new Matrix(testMatrixArray1));
        double[][] actualResultTemp = luDecomposition.solve(new Matrix(tempMatrixArray2)).getArray();
        double[][] expectedResultTemp = new double [][] {
                { -0.1913,  0.2679 },
                {  0.6795, -0.0820 },
                { -0.0577, -0.0589 }
        };
        double[] actualResult = new double[actualResultTemp.length * actualResultTemp[0].length];
        double[] expectedResult = new double[expectedResultTemp.length * expectedResultTemp[0].length];
        for (int i = 0; i < actualResultTemp.length; i++) {
            for (int j = 0; j < actualResultTemp[0].length; j++) {
                actualResult[i * actualResultTemp[0].length + j] = actualResultTemp[i][j];
                expectedResult[i * expectedResultTemp[0].length + j] = expectedResultTemp[i][j];
            }
        }
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-4);
    }
}
