package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.junit.Assert;
import org.junit.Test;
import org.platanios.learn.optimization.function.QuadraticFunction;

/**
 * @author Emmanouil Antonios Platanios
 */
public class NewtonSolverTest {
    @Test
    public void testNewtonSolver() {
        System.out.println("Rosenbrock Function Newton Solver:\n");
        NewtonSolver newtonRaphsonSolver = new NewtonSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
        double[] actualResult = newtonRaphsonSolver.solve().toArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);

        System.out.println("Quadratic Function Newton Solver:\n");
        RealMatrix A = new Array2DRowRealMatrix(new double[][] { { 1, 0.5 }, { 0.5, 1 } });
        RealVector b = new ArrayRealVector(new double[] { 1, 2 });
        newtonRaphsonSolver = new NewtonSolver(new QuadraticFunction(A, b), new double[] { 0, 0 });
        actualResult = newtonRaphsonSolver.solve().toArray();
        expectedResult = new double[] { 0, 2 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }
}
