package org.platanios.learn.optimization;

import org.junit.Assert;
import org.junit.Test;
import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.VectorFactory;
import org.platanios.learn.optimization.function.QuadraticFunction;

/**
 * @author Emmanouil Antonios Platanios
 */
public class GradientDescentSolverTest {
    @Test
    public void testGradientDescentSolver() {
        System.out.println("Rosenbrock Function Gradient Descent Solver:\n");
        GradientDescentSolver gradientDescentSolver =
                new GradientDescentSolver.Builder(new RosenbrockFunction(),
                                                  VectorFactory.buildDense(new double[] { -1.2, 1 })).build();
        double[] actualResult = gradientDescentSolver.solve().getDenseArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);

        System.out.println("Quadratic Function Gradient Descent Solver:\n");
        Matrix A = new Matrix(new double[][] { { 1, 0.5 }, { 0.5, 1 } });
        Vector b = VectorFactory.buildDense(new double[] { 1, 2 });
        gradientDescentSolver =
                new GradientDescentSolver.Builder(new QuadraticFunction(A, b),
                                                  VectorFactory.buildDense(new double[] { 0, 0 })).build();
        actualResult = gradientDescentSolver.solve().getDenseArray();
        expectedResult = new double[] { 0, 2 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }
}
