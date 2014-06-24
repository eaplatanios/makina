package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.junit.Assert;
import org.junit.Test;
import org.platanios.learn.optimization.function.Function;
import org.platanios.learn.optimization.function.QuadraticFunction;

/**
 * @author Emmanouil Antonios Platanios
 */
public class OptimizationTest {
    @Test
    public void testSteepestDescentSolver() {
        System.out.println("General Function SD:\n");
        SteepestDescentSolver steepestDescentSolver =
                new SteepestDescentSolver(new GeneralFunction(), new double[] { 6 });
        double actualResult = steepestDescentSolver.solve().getEntry(0);
        double expectedResult = 9.0 / 4.0;
        Assert.assertEquals(expectedResult, actualResult, 1e-4);

        System.out.println("Quadratic Function SD:\n");
        RealMatrix Q = new Array2DRowRealMatrix(new double[][] { { 2 } });
        RealVector b = new ArrayRealVector(new double[] { 3 });
        steepestDescentSolver = new SteepestDescentSolver(new QuadraticFunction(Q, b), new double[] { 6 });
        actualResult = steepestDescentSolver.solve().getEntry(0);
        expectedResult = - 3.0 / 4.0;
        Assert.assertEquals(expectedResult, actualResult, 1e-4);
    }

    @Test
    public void testNewtonsMethodSolver() {
        System.out.println("General Function Newton's Method:\n");
        NewtonsMethodSolver newtonsMethodSolver = new NewtonsMethodSolver(new GeneralFunction(), new double[] { 6 });
        double actualResult = newtonsMethodSolver.solve().getEntry(0);
        double expectedResult = 9.0 / 4.0;
        Assert.assertEquals(expectedResult, actualResult, 1e-4);

        System.out.println("Quadratic Function Newton's Method:\n");
        RealMatrix Q = new Array2DRowRealMatrix(new double[][] { { 2 } });
        RealVector b = new ArrayRealVector(new double[] { 3 });
        newtonsMethodSolver = new NewtonsMethodSolver(new QuadraticFunction(Q, b), new double[] { 6 });
        actualResult = newtonsMethodSolver.solve().getEntry(0);
        expectedResult = - 3.0 / 4.0;
        Assert.assertEquals(expectedResult, actualResult, 1e-4);
    }

    class GeneralFunction implements Function {
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
}
