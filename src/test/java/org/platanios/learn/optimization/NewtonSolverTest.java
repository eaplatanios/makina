package org.platanios.learn.optimization;

import org.junit.Assert;
import org.junit.Test;
import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.VectorFactory;
import org.platanios.learn.math.matrix.VectorType;
import org.platanios.learn.optimization.function.QuadraticFunction;

/**
 * @author Emmanouil Antonios Platanios
 */
public class NewtonSolverTest {
    @Test
    public void testNewtonSolver() {
        System.out.println("Rosenbrock Function Newton Solver:\n");
        NewtonSolver newtonRaphsonSolver =
                new NewtonSolver.Builder(new RosenbrockFunction(), new double[] { -1.2, 1 }).build();
        double[] actualResult = newtonRaphsonSolver.solve().getDenseArray();
        double[] expectedResult = new double[] { 1, 1 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);

        System.out.println("Quadratic Function Newton Solver:\n");
        Matrix A = new Matrix(new double[][] { { 1, 0.5 }, { 0.5, 1 } });
        Vector b = VectorFactory.build(new double[]{1, 2}, VectorType.DENSE);
        newtonRaphsonSolver = new NewtonSolver.Builder(new QuadraticFunction(A, b), new double[] { 0, 0 }).build();
        actualResult = newtonRaphsonSolver.solve().getDenseArray();
        expectedResult = new double[] { 0, 2 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }
}
