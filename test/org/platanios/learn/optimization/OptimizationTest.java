package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.junit.Assert;
import org.junit.Test;

/**
 * @author Emmanouil Antonios Platanios
 */
public class OptimizationTest {
    @Test
    public void testSteepestDescentSolver() {
        class QuadraticFunction implements ObjectiveFunctionWithGradient {
            public double computeValue(RealVector optimizationVariables) {
                double x = optimizationVariables.getEntry(0);
                return Math.pow(x, 4) - 3 * Math.pow(x, 3) + 2;
            }

            public RealVector computeGradient(RealVector optimizationVariables) {
                double x = optimizationVariables.getEntry(0);
                return new ArrayRealVector(new double[] { 4 * Math.pow(x, 3) - 9 * Math.pow(x, 2) });
            }
        }

        SteepestDescentSolver sda = new SteepestDescentSolver(new QuadraticFunction(), new double[] { 6 });

        double actualResult = sda.solve().getEntry(0);
        double expectedResult = 9.0 / 4.0;
        Assert.assertEquals(expectedResult, actualResult, 1e-4);
    }

    @Test
    public void testNewtonsMethodSolver() {
        class QuadraticFunction implements ObjectiveFunctionWithGradientAndHessian {
            public double computeValue(RealVector optimizationVariables) {
                double x = optimizationVariables.getEntry(0);
                return Math.pow(x, 4) - 3 * Math.pow(x, 3) + 2;
            }

            public RealVector computeGradient(RealVector optimizationVariables) {
                double x = optimizationVariables.getEntry(0);
                return new ArrayRealVector(new double[] { 4 * Math.pow(x, 3) - 9 * Math.pow(x, 2) });
            }

            public RealMatrix computeHessian(RealVector optimizationVariables) {
                double x = optimizationVariables.getEntry(0);
                return new Array2DRowRealMatrix(new double[][] { { 12 * Math.pow(x, 2) - 18 * x } });
            }
        }

        NewtonsMethodSolver sda = new NewtonsMethodSolver(new QuadraticFunction(), new double[] { 6 });

        double actualResult = sda.solve().getEntry(0);
        double expectedResult = 9.0 / 4.0;
        Assert.assertEquals(expectedResult, actualResult, 1e-4);
    }
}
