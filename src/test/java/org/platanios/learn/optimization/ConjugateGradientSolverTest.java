package org.platanios.learn.optimization;

import org.junit.Assert;
import org.junit.Test;
import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.NonPositiveDefiniteMatrixException;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.VectorFactory;
import org.platanios.learn.optimization.function.QuadraticFunction;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ConjugateGradientSolverTest {
    @Test
    public void testConjugateGradientSolver() throws NonPositiveDefiniteMatrixException {
        System.out.println("Quadratic Function Conjugate Gradient Solver:\n");
        Matrix A = new Matrix(new double[][] { { 1, 0.5 }, { 0.5, 1 } });
        Vector b = VectorFactory.buildDense(new double[]{1, 2});
        ConjugateGradientSolver conjugateGradientSolver =
                new ConjugateGradientSolver.Builder(new QuadraticFunction(A, b), new double[] { 0, 0 })
                .preconditioningMethod(ConjugateGradientSolver.PreconditioningMethod.IDENTITY)
                .build();
        double[] actualResult = conjugateGradientSolver.solve().getDenseArray();
        double[] expectedResult = new double[] { 0, 2 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }
}
