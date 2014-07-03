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
public class ConjugateGradientSolverTest {
    @Test
    public void testConjugateGradientSolver() {
        System.out.println("Quadratic Function Conjugate Gradient Solver:\n");
        RealMatrix A = new Array2DRowRealMatrix(new double[][] { { 1, 0.5 }, { 0.5, 1 } });
        RealVector b = new ArrayRealVector(new double[] { 1, 2 });
        ConjugateGradientSolver conjugateGradientSolver =
                new ConjugateGradientSolver(
                        new QuadraticFunction(A, b),
                        ConjugateGradientSolver.PreconditioningMethod.IDENTITY,
                        new double[] { 0, 0 });
        double[] actualResult = conjugateGradientSolver.solve().toArray();
        double[] expectedResult = new double[] { 0, 2 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-2);
    }
}
