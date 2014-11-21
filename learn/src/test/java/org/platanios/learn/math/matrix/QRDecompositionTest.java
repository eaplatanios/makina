package org.platanios.learn.math.matrix;

import org.junit.Assert;
import org.junit.Test;

/**
 * @author Emmanouil Antonios Platanios
 */
public class QRDecompositionTest {
    @Test
    public void testQForFullRankMatrix() {
        double[][] testMatrixArray = new double[][] {
                {  3.4000, 1.2000,  2.2000 },
                {  0.1000, 7.4000,  0.5000 },
                { -5.3000, 0.4000, -2.1000 }
        };
        QRDecomposition qrDecomposition = new QRDecomposition(new Matrix(testMatrixArray));
        double[][] actualResultTemp = qrDecomposition.getQ().getArray();
        double[][] expectedResultTemp = new double [][] {
                { -0.5399, -0.1292, -0.8318 },
                { -0.0159, -0.9864,  0.1636 },
                {  0.8416, -0.1015, -0.5305 }
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
    public void testRForFullRankMatrix() {
        double[][] testMatrixArray = new double[][] {
                {  3.4000, 1.2000,  2.2000 },
                {  0.1000, 7.4000,  0.5000 },
                { -5.3000, 0.4000, -2.1000 }
        };
        QRDecomposition qrDecomposition = new QRDecomposition(new Matrix(testMatrixArray));
        double[][] actualResultTemp = qrDecomposition.getR().getArray();
        double[][] expectedResultTemp = new double [][] {
                { -6.2976, -0.4287, -2.9630 },
                {       0, -7.4951, -0.5643 },
                {       0,       0, -0.6341 }
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
    public void testIsFullRankForFullRankMatrix() {
        double[][] testMatrixArray = new double[][] {
                {  3.4000, 1.2000,  2.2000 },
                {  0.1000, 7.4000,  0.5000 },
                { -5.3000, 0.4000, -2.1000 }
        };
        QRDecomposition qrDecomposition = new QRDecomposition(new Matrix(testMatrixArray));
        Assert.assertEquals(true, qrDecomposition.isFullRank());
    }

    @Test
    public void testVectorSolveForFullRankMatrix() throws SingularMatrixException {
        double[][] testMatrixArray = new double[][] {
                { 28.4100, 11.4400,  21.1100 },
                { 11.4400,  9.7600,  18.0800 },
                { 21.1100, 18.0800, 106.5300 }
        };
        double[] tempVectorArray = new double[] { 1.12, 3.40, 2.10 };
        QRDecomposition qrDecomposition = new QRDecomposition(new Matrix(testMatrixArray));
        double[] actualResult = qrDecomposition.solve(new DenseVector(tempVectorArray)).getDenseArray();
        double[] expectedResult = new double [] { -0.1913, 0.6795, -0.0577 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-4);
    }

    @Test
    public void testMatrixSolveForFullRankMatrix() throws SingularMatrixException {
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
        QRDecomposition qrDecomposition = new QRDecomposition(new Matrix(testMatrixArray1));
        double[][] actualResultTemp = qrDecomposition.solve(new Matrix(tempMatrixArray2)).getArray();
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
