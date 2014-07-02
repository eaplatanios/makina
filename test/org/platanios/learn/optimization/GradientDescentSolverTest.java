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
public class GradientDescentSolverTest {
    @Test
    public void testGradientDescentSolver() {
        System.out.println("Rosenbrock Function Gradient Descent Solver:\n");
        GradientDescentSolver gradientDescentSolver =
                new GradientDescentSolver(new RosenbrockFunction(), new double[] { -1.2, 1 });
        double[] actualResult = gradientDescentSolver.solve().toArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);

        System.out.println("Quadratic Function Gradient Descent Solver:\n");
        RealMatrix A = new Array2DRowRealMatrix(new double[][] { { 1, 0.5 }, { 0.5, 1 } });
        RealVector b = new ArrayRealVector(new double[] { 1, 2 });
        gradientDescentSolver = new GradientDescentSolver(new QuadraticFunction(A, b), new double[] { 0, 0 });
        actualResult = gradientDescentSolver.solve().toArray();
        expectedResult = new double[] { 0, 2 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }
}
