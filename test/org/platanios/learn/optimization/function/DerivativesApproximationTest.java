package org.platanios.learn.optimization.function;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.junit.Assert;
import org.junit.Test;

/**
 * @author Emmanouil Antonios Platanios
 */
public class DerivativesApproximationTest {
    @Test
    public void testForwardDifferenceGradientApproximation() {
        AbstractFunction function = new RosenbrockFunction();
        DerivativesApproximation derivativesApproximation =
                new DerivativesApproximation(function, DerivativesApproximationMethod.FORWARD_DIFFERENCE);
        RealVector point = new ArrayRealVector(new double[] { -1.2, 1 });
        double[] actualResult = derivativesApproximation.approximateGradient(point).toArray();
        double[] expectedResult = function.getGradient(point).toArray();
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-5);
    }

    @Test
    public void testCentralDifferenceGradientApproximation() {
        AbstractFunction function = new RosenbrockFunction();
        DerivativesApproximation derivativesApproximation =
                new DerivativesApproximation(function, DerivativesApproximationMethod.CENTRAL_DIFFERENCE);
        RealVector point = new ArrayRealVector(new double[] { -1.2, 1 });
        double[] actualResult = derivativesApproximation.approximateGradient(point).toArray();
        double[] expectedResult = function.getGradient(point).toArray();
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-7);
    }

    @Test
    public void testForwardDifferenceHessianApproximation() {
        AbstractFunction function = new RosenbrockFunction();
        DerivativesApproximation derivativesApproximation =
                new DerivativesApproximation(function, DerivativesApproximationMethod.FORWARD_DIFFERENCE);
        RealVector point = new ArrayRealVector(new double[] { -1.2, 1 });
        double[][] actualResultTemp = derivativesApproximation.approximateHessian(point).getData();
        double[][] expectedResultTemp = function.getHessian(point).getData();
        double[] actualResult = new double[actualResultTemp.length * actualResultTemp[0].length];
        double[] expectedResult = new double[expectedResultTemp.length * expectedResultTemp[0].length];
        for (int i = 0; i < actualResultTemp.length; i++) {
            for (int j = 0; j < actualResultTemp[0].length; j++) {
                actualResult[i * actualResultTemp.length + j] = actualResultTemp[i][j];
                expectedResult[i * expectedResultTemp.length + j] = expectedResultTemp[i][j];
            }
        }
        Assert.assertArrayEquals(expectedResult, actualResult, 1e2);
    }

    @Test
    public void testCentralDifferenceHessianApproximation() {
        AbstractFunction function = new RosenbrockFunction();
        DerivativesApproximation derivativesApproximation =
                new DerivativesApproximation(function, DerivativesApproximationMethod.CENTRAL_DIFFERENCE);
        RealVector point = new ArrayRealVector(new double[] { -1.2, 1 });
        double[][] actualResultTemp = derivativesApproximation.approximateHessian(point).getData();
        double[][] expectedResultTemp = function.getHessian(point).getData();
        double[] actualResult = new double[actualResultTemp.length * actualResultTemp[0].length];
        double[] expectedResult = new double[expectedResultTemp.length * expectedResultTemp[0].length];
        for (int i = 0; i < actualResultTemp.length; i++) {
            for (int j = 0; j < actualResultTemp[0].length; j++) {
                actualResult[i * actualResultTemp.length + j] = actualResultTemp[i][j];
                expectedResult[i * expectedResultTemp.length + j] = expectedResultTemp[i][j];
            }
        }
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-3);
    }

    @Test
    public void testForwardDifferenceHessianApproximationGivenGradient() {
        AbstractFunction function = new RosenbrockFunction();
        DerivativesApproximation derivativesApproximation =
                new DerivativesApproximation(function, DerivativesApproximationMethod.FORWARD_DIFFERENCE);
        RealVector point = new ArrayRealVector(new double[] { -1.2, 1 });
        double[][] actualResultTemp = derivativesApproximation.approximateHessianGivenGradient(point).getData();
        double[][] expectedResultTemp = function.getHessian(point).getData();
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
    public void testCentralDifferenceHessianApproximationGivenGradient() {
        AbstractFunction function = new RosenbrockFunction();
        DerivativesApproximation derivativesApproximation =
                new DerivativesApproximation(function, DerivativesApproximationMethod.CENTRAL_DIFFERENCE);
        RealVector point = new ArrayRealVector(new double[] { -1.2, 1 });
        double[][] actualResultTemp = derivativesApproximation.approximateHessianGivenGradient(point).getData();
        double[][] expectedResultTemp = function.getHessian(point).getData();
        double[] actualResult = new double[actualResultTemp.length * actualResultTemp[0].length];
        double[] expectedResult = new double[expectedResultTemp.length * expectedResultTemp[0].length];
        for (int i = 0; i < actualResultTemp.length; i++) {
            for (int j = 0; j < actualResultTemp[0].length; j++) {
                actualResult[i * actualResultTemp.length + j] = actualResultTemp[i][j];
                expectedResult[i * expectedResultTemp.length + j] = expectedResultTemp[i][j];
            }
        }
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-8);
    }

    @Test
    public void testForwardDifferenceHessianVectorProductApproximationGivenGradient() {
        AbstractFunction function = new RosenbrockFunction();
        DerivativesApproximation derivativesApproximation =
                new DerivativesApproximation(function, DerivativesApproximationMethod.FORWARD_DIFFERENCE);
        RealVector point = new ArrayRealVector(new double[] { -1.2, 1 });
        RealVector p = new ArrayRealVector(new double[] { 1.21, 0.53 });
        double[] actualResult =
                derivativesApproximation.approximateHessianVectorProductGivenGradient(point, p).toArray();
        double[] expectedResult = function.getHessian(point).operate(p).toArray();
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-4);
    }

    class RosenbrockFunction extends AbstractFunction {
        public double computeValue(RealVector optimizationVariables) {
            double x1 = optimizationVariables.getEntry(0);
            double x2 = optimizationVariables.getEntry(1);
            return 100 * Math.pow(x2 - Math.pow(x1, 2), 2) + Math.pow(1 - x1, 2);
        }

        public RealVector computeGradient(RealVector optimizationVariables) {
            double x1 = optimizationVariables.getEntry(0);
            double x2 = optimizationVariables.getEntry(1);
            double dx1 = - 400 * (x2 - Math.pow(x1, 2)) * x1 - 2 * (1 - x1);
            double dx2 = 200 * (x2 - Math.pow(x1, 2));
            return new ArrayRealVector(new double[] { dx1, dx2 });
        }

        public RealMatrix computeHessian(RealVector optimizationVariables) {
            double x1 = optimizationVariables.getEntry(0);
            double x2 = optimizationVariables.getEntry(1);
            double dx1x1 = 1200 * Math.pow(x1, 2) - 400 * x2 + 2;
            double dx1x2 = - 400 * x1;
            double dx2x2 = 200;
            return new Array2DRowRealMatrix(new double[][] { { dx1x1, dx1x2 }, { dx1x2, dx2x2 } });
        }
    }
}
