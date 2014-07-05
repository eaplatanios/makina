package org.platanios.learn.optimization;

import org.junit.Assert;
import org.junit.Test;
import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;
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
        double[] actualResult = gradientDescentSolver.solve().getArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);

        System.out.println("Quadratic Function Gradient Descent Solver:\n");
        Matrix A = new Matrix(new double[][] { { 1, 0.5 }, { 0.5, 1 } });
        Vector b = new Vector(new double[] { 1, 2 });
        gradientDescentSolver = new GradientDescentSolver(new QuadraticFunction(A, b), new double[] { 0, 0 });
        actualResult = gradientDescentSolver.solve().getArray();
        expectedResult = new double[] { 0, 2 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }
}
