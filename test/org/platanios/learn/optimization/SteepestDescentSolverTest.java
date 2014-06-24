package org.platanios.learn.optimization;

import org.junit.Assert;
import org.junit.Test;

/**
 * @author Emmanouil Antonios Platanios
 */
public class SteepestDescentSolverTest {
    @Test
    public void testSolve() {
        class QuadraticFunction implements ObjectiveFunctionWithGradient {
            public double computeValue(double[] optimizationVariables) {
                double x = optimizationVariables[0];
                return Math.pow(x, 4) - 3 * Math.pow(x, 3) + 2;
            }

            public double[] computeGradient(double[] optimizationVariables) {
                double x = optimizationVariables[0];
                return new double[] { 4 * Math.pow(x, 3) - 9 * Math.pow(x, 2) };
            }
        }

        SteepestDescentSolver sda = new SteepestDescentSolver(new QuadraticFunction(), new double[] { 6 });

        double actualResult = sda.solve()[0];
        double expectedResult = 9.0 / 4.0;
        Assert.assertEquals(expectedResult, actualResult, 1e-5);
    }
}
