package org.platanios.learn.optimization;

import org.junit.Assert;
import org.junit.Test;
import org.platanios.learn.math.linearalgebra.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public class SteepestDescentAlgorithmTest {
    @Test
    public void testSolve() {
        class QuadraticFunction implements ObjectiveFunction {
            public double computeValue(double[] optimizationVariables) {
                double x = optimizationVariables[0];
                return Math.pow(x, 4) - 3 * Math.pow(x, 3) + 2;
            }

            public double[] computeGradient(double[] optimizationVariables) {
                double x = optimizationVariables[0];
                return new double[] { 4 * Math.pow(x, 3) - 9 * Math.pow(x, 2) };
            }

            public double[][] computeHessian(double[] optimizationVariables) {
                return null;
            }
        }

        SteepestDescentAlgorithm sda = new SteepestDescentAlgorithm(new QuadraticFunction(), new double[] { 6 });

        double actualResult = sda.solve()[0];
        double expectedResult = 9.0 / 4.0;
        Assert.assertEquals(expectedResult, actualResult, 1e-5);
    }
}
