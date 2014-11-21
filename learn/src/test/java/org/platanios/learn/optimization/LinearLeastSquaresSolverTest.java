package org.platanios.learn.optimization;

import org.junit.Assert;
import org.junit.Test;
import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;
import org.platanios.learn.optimization.function.LinearLeastSquaresFunction;

/**
 * @author Emmanouil Antonios Platanios
 */
public class LinearLeastSquaresSolverTest {
    @Test
    public void testCholeskyDecompositionMethod() {
        Matrix J = new Matrix(new double[][] {
                { 1, 1 },
                { 1, 2 },
                { 1, 3 },
                { 1, 4 }
        });
        Vector y = Vectors.dense(new double[]{6, 5, 7, 10});
        LinearLeastSquaresSolver linearLeastSquaresSolver =
                new LinearLeastSquaresSolver.Builder(new LinearLeastSquaresFunction(J, y))
                        .method(LinearLeastSquaresSolver.Method.CHOLESKY_DECOMPOSITION)
                        .build();
        double[] actualResult = linearLeastSquaresSolver.solve().getDenseArray();
        double[] expectedResult = new double[] { 3.5, 1.4 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-8);
    }

    @Test
    public void testQRDecompositionMethod() {
        Matrix J = new Matrix(new double[][] {
                { 1, 1 },
                { 1, 2 },
                { 1, 3 },
                { 1, 4 }
        });
        Vector y = Vectors.dense(new double[]{6, 5, 7, 10});
        LinearLeastSquaresSolver linearLeastSquaresSolver =
                new LinearLeastSquaresSolver.Builder(new LinearLeastSquaresFunction(J, y))
                        .method(LinearLeastSquaresSolver.Method.QR_DECOMPOSITION)
                        .build();
        double[] actualResult = linearLeastSquaresSolver.solve().getDenseArray();
        double[] expectedResult = new double[] { 3.5, 1.4 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-8);
    }

    @Test
    public void testSingularValueDecompositionMethod() {
        Matrix J = new Matrix(new double[][] {
                { 1, 1 },
                { 1, 2 },
                { 1, 3 },
                { 1, 4 }
        });
        Vector y = Vectors.dense(new double[]{6, 5, 7, 10});
        LinearLeastSquaresSolver linearLeastSquaresSolver =
                new LinearLeastSquaresSolver.Builder(new LinearLeastSquaresFunction(J, y))
                        .method(LinearLeastSquaresSolver.Method.SINGULAR_VALUE_DECOMPOSITION)
                        .build();
        double[] actualResult = linearLeastSquaresSolver.solve().getDenseArray();
        double[] expectedResult = new double[] { 3.5, 1.4 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-8);
    }

    @Test
    public void testConjugateGradientMethod() {
        Matrix J = new Matrix(new double[][] {
                { 1, 1 },
                { 1, 2 },
                { 1, 3 },
                { 1, 4 }
        });
        Vector y = Vectors.dense(new double[]{6, 5, 7, 10});
        LinearLeastSquaresSolver linearLeastSquaresSolver =
                new LinearLeastSquaresSolver.Builder(new LinearLeastSquaresFunction(J, y))
                        .method(LinearLeastSquaresSolver.Method.CONJUGATE_GRADIENT)
                        .build();
        double[] actualResult = linearLeastSquaresSolver.solve().getDenseArray();
        double[] expectedResult = new double[] { 3.5, 1.4 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-8);
    }
}
