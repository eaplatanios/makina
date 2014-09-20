package org.platanios.learn.optimization.function;

import org.junit.Assert;
import org.junit.Test;
import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.VectorFactory;

/**
 * @author Emmanouil Antonios Platanios
 */
public class DerivativesApproximationTest {
    @Test
    public void testForwardDifferenceGradientApproximation() {
        AbstractFunction function = new RosenbrockFunction();
        DerivativesApproximation derivativesApproximation =
                new DerivativesApproximation(function, DerivativesApproximation.Method.FORWARD_DIFFERENCE);
        Vector point = VectorFactory.buildDense(new double[] { -1.2, 1 });
        double[] actualResult = derivativesApproximation.approximateGradient(point).getDenseArray();
        double[] expectedResult = function.getGradient(point).getDenseArray();
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-5);
    }

    @Test
    public void testCentralDifferenceGradientApproximation() {
        AbstractFunction function = new RosenbrockFunction();
        DerivativesApproximation derivativesApproximation =
                new DerivativesApproximation(function, DerivativesApproximation.Method.CENTRAL_DIFFERENCE);
        Vector point = VectorFactory.buildDense(new double[] { -1.2, 1 });
        double[] actualResult = derivativesApproximation.approximateGradient(point).getDenseArray();
        double[] expectedResult = function.getGradient(point).getDenseArray();
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-7);
    }

    @Test
    public void testForwardDifferenceHessianApproximation() {
        AbstractFunction function = new RosenbrockFunction();
        DerivativesApproximation derivativesApproximation =
                new DerivativesApproximation(function, DerivativesApproximation.Method.FORWARD_DIFFERENCE);
        Vector point = VectorFactory.buildDense(new double[] { -1.2, 1 });
        double[][] actualResultTemp = derivativesApproximation.approximateHessian(point).getArray();
        double[][] expectedResultTemp = function.getHessian(point).getArray();
        double[] actualResult = new double[actualResultTemp.length * actualResultTemp[0].length];
        double[] expectedResult = new double[expectedResultTemp.length * expectedResultTemp[0].length];
        for (int i = 0; i < actualResultTemp.length; i++) {
            for (int j = 0; j < actualResultTemp[0].length; j++) {
                actualResult[i * actualResultTemp[0].length + j] = actualResultTemp[i][j];
                expectedResult[i * expectedResultTemp[0].length + j] = expectedResultTemp[i][j];
            }
        }
        Assert.assertArrayEquals(expectedResult, actualResult, 1e2);
    }

    @Test
    public void testCentralDifferenceHessianApproximation() {
        AbstractFunction function = new RosenbrockFunction();
        DerivativesApproximation derivativesApproximation =
                new DerivativesApproximation(function, DerivativesApproximation.Method.CENTRAL_DIFFERENCE);
        Vector point = VectorFactory.buildDense(new double[] { -1.2, 1 });
        double[][] actualResultTemp = derivativesApproximation.approximateHessian(point).getArray();
        double[][] expectedResultTemp = function.getHessian(point).getArray();
        double[] actualResult = new double[actualResultTemp.length * actualResultTemp[0].length];
        double[] expectedResult = new double[expectedResultTemp.length * expectedResultTemp[0].length];
        for (int i = 0; i < actualResultTemp.length; i++) {
            for (int j = 0; j < actualResultTemp[0].length; j++) {
                actualResult[i * actualResultTemp[0].length + j] = actualResultTemp[i][j];
                expectedResult[i * expectedResultTemp[0].length + j] = expectedResultTemp[i][j];
            }
        }
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-3);
    }

    @Test
    public void testForwardDifferenceHessianApproximationGivenGradient() {
        AbstractFunction function = new RosenbrockFunction();
        DerivativesApproximation derivativesApproximation =
                new DerivativesApproximation(function, DerivativesApproximation.Method.FORWARD_DIFFERENCE);
        Vector point = VectorFactory.buildDense(new double[] { -1.2, 1 });
        double[][] actualResultTemp = derivativesApproximation.approximateHessianGivenGradient(point).getArray();
        double[][] expectedResultTemp = function.getHessian(point).getArray();
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

    @Test
    public void testCentralDifferenceHessianApproximationGivenGradient() {
        AbstractFunction function = new RosenbrockFunction();
        DerivativesApproximation derivativesApproximation =
                new DerivativesApproximation(function, DerivativesApproximation.Method.CENTRAL_DIFFERENCE);
        Vector point = VectorFactory.buildDense(new double[] { -1.2, 1 });
        double[][] actualResultTemp = derivativesApproximation.approximateHessianGivenGradient(point).getArray();
        double[][] expectedResultTemp = function.getHessian(point).getArray();
        double[] actualResult = new double[actualResultTemp.length * actualResultTemp[0].length];
        double[] expectedResult = new double[expectedResultTemp.length * expectedResultTemp[0].length];
        for (int i = 0; i < actualResultTemp.length; i++) {
            for (int j = 0; j < actualResultTemp[0].length; j++) {
                actualResult[i * actualResultTemp[0].length + j] = actualResultTemp[i][j];
                expectedResult[i * expectedResultTemp[0].length + j] = expectedResultTemp[i][j];
            }
        }
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-8);
    }

    @Test
    public void testForwardDifferenceHessianVectorProductApproximationGivenGradient() {
        AbstractFunction function = new RosenbrockFunction();
        DerivativesApproximation derivativesApproximation =
                new DerivativesApproximation(function, DerivativesApproximation.Method.FORWARD_DIFFERENCE);
        Vector point = VectorFactory.buildDense(new double[] { -1.2, 1 });
        Vector p = VectorFactory.buildDense(new double[] { 1.21, 0.53 });
        double[] actualResult =
                derivativesApproximation.approximateHessianVectorProductGivenGradient(point, p).getDenseArray();
        double[] expectedResult = function.getHessian(point).multiply(p).getDenseArray();
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-4);
    }

    class RosenbrockFunction extends AbstractFunction {
        public double computeValue(Vector optimizationVariables) {
            double x1 = optimizationVariables.get(0);
            double x2 = optimizationVariables.get(1);
            return 100 * Math.pow(x2 - Math.pow(x1, 2), 2) + Math.pow(1 - x1, 2);
        }

        public Vector computeGradient(Vector optimizationVariables) {
            double x1 = optimizationVariables.get(0);
            double x2 = optimizationVariables.get(1);
            double dx1 = - 400 * (x2 - Math.pow(x1, 2)) * x1 - 2 * (1 - x1);
            double dx2 = 200 * (x2 - Math.pow(x1, 2));
            return VectorFactory.buildDense(new double[] { dx1, dx2 });
        }

        public Matrix computeHessian(Vector optimizationVariables) {
            double x1 = optimizationVariables.get(0);
            double x2 = optimizationVariables.get(1);
            double dx1x1 = 1200 * Math.pow(x1, 2) - 400 * x2 + 2;
            double dx1x2 = - 400 * x1;
            double dx2x2 = 200;
            return new Matrix(new double[][] { { dx1x1, dx1x2 }, { dx1x2, dx2x2 } });
        }
    }
}
