package org.platanios.learn.math.matrix;

import org.junit.Assert;
import org.junit.Test;

/**
 * @author Emmanouil Antonios Platanios
 */
public class CholeskyDecompositionTest {
    @Test
    public void testLForSymmetricAndPositiveDefiniteMatrix1() {
        double[][] testMatrixArray = new double[][] {
                { 1, 1,  1,  1,  1 },
                { 1, 2,  3,  4,  5 },
                { 1, 3,  6, 10, 15 },
                { 1, 4, 10, 20, 35 },
                { 1, 5, 15, 35, 70 }
        };
        CholeskyDecomposition choleskyDecomposition = new CholeskyDecomposition(new Matrix(testMatrixArray));
        double[][] actualResultTemp = choleskyDecomposition.getL().getArray();
        double[][] expectedResultTemp = new double [][] {
                { 1, 0, 0, 0, 0 },
                { 1, 1, 0, 0, 0 },
                { 1, 2, 1, 0, 0 },
                { 1, 3, 3, 1, 0 },
                { 1, 4, 6, 4, 1 }
        };
        double[] actualResult = new double[actualResultTemp.length * actualResultTemp[0].length];
        double[] expectedResult = new double[expectedResultTemp.length * expectedResultTemp[0].length];
        for (int i = 0; i < actualResultTemp.length; i++) {
            for (int j = 0; j < actualResultTemp[0].length; j++) {
                actualResult[i * actualResultTemp.length + j] = actualResultTemp[i][j];
                expectedResult[i * expectedResultTemp.length + j] = expectedResultTemp[i][j];
            }
        }
        Assert.assertArrayEquals(expectedResult, actualResult, 0);
    }

    @Test
    public void testIsSymmetricAndPositiveDefiniteForSymmetricAndPositiveDefiniteMatrix1() {
        double[][] testMatrixArray = new double[][] {
                { 1, 1,  1,  1,  1 },
                { 1, 2,  3,  4,  5 },
                { 1, 3,  6, 10, 15 },
                { 1, 4, 10, 20, 35 },
                { 1, 5, 15, 35, 70 }
        };
        CholeskyDecomposition choleskyDecomposition = new CholeskyDecomposition(new Matrix(testMatrixArray));
        Assert.assertEquals(true, choleskyDecomposition.isSymmetricAndPositiveDefinite());
    }

    @Test
    public void testLForSymmetricAndPositiveDefiniteMatrix2() {
        double[][] testMatrixArray = new double[][] {
                { 28.4100, 11.4400,  21.1100 },
                { 11.4400,  9.7600,  18.0800 },
                { 21.1100, 18.0800, 106.5300 }
        };
        CholeskyDecomposition choleskyDecomposition = new CholeskyDecomposition(new Matrix(testMatrixArray));
        double[][] actualResultTemp = choleskyDecomposition.getL().getArray();
        double[][] expectedResultTemp = new double [][] {
                { 5.3301,      0,      0 },
                { 2.1463, 2.2701,      0 },
                { 3.9605, 4.2199, 8.5462 }
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
    public void testIsSymmetricAndPositiveDefiniteForSymmetricAndPositiveDefiniteMatrix2() {
        double[][] testMatrixArray = new double[][] {
                { 28.4100, 11.4400,  21.1100 },
                { 11.4400,  9.7600,  18.0800 },
                { 21.1100, 18.0800, 106.5300 }
        };
        CholeskyDecomposition choleskyDecomposition = new CholeskyDecomposition(new Matrix(testMatrixArray));
        Assert.assertEquals(true, choleskyDecomposition.isSymmetricAndPositiveDefinite());
    }

    @Test
    public void testIsSymmetricAndPositiveDefiniteForNonSymmetricAndPositiveDefiniteMatrix() {
        double[][] testMatrixArray = new double[][] {
                { 0, 1,  1,  1,  1 },
                { 1, 2,  3,  4,  5 },
                { 1, 3,  6, 10, 15 },
                { 1, 4, 10, 20, 35 },
                { 1, 5, 15, 35, 70 }
        };
        CholeskyDecomposition choleskyDecomposition = new CholeskyDecomposition(new Matrix(testMatrixArray));
        Assert.assertEquals(false, choleskyDecomposition.isSymmetricAndPositiveDefinite());
    }

    @Test
    public void testVectorSolveForSymmetricAndPositiveDefiniteMatrix() {
        double[][] testMatrixArray = new double[][] {
                { 28.4100, 11.4400,  21.1100 },
                { 11.4400,  9.7600,  18.0800 },
                { 21.1100, 18.0800, 106.5300 }
        };
        double[] tempVectorArray = new double[] { 1.12, 3.40, 2.10 };
        CholeskyDecomposition choleskyDecomposition = new CholeskyDecomposition(new Matrix(testMatrixArray));
        double[] actualResult = choleskyDecomposition.solve(new Vector(tempVectorArray)).getArray();
        double[] expectedResult = new double [] { -0.1913, 0.6795, -0.0577 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-4);
    }

    @Test
    public void testMatrixSolveForSymmetricAndPositiveDefiniteMatrix() {
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
        CholeskyDecomposition choleskyDecomposition = new CholeskyDecomposition(new Matrix(testMatrixArray1));
        double[][] actualResultTemp = choleskyDecomposition.solve(new Matrix(tempMatrixArray2)).getArray();
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
