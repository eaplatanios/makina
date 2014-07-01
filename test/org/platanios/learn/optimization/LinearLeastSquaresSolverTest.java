package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.junit.Assert;
import org.junit.Test;
import org.platanios.learn.optimization.function.LinearLeastSquaresFunction;

/**
 * @author Emmanouil Antonios Platanios
 */
public class LinearLeastSquaresSolverTest {
    @Test
    public void testCholeskyDecompositionMethod() {
        RealMatrix J = new Array2DRowRealMatrix(new double[][] {
                { 1, 1 },
                { 1, 2 },
                { 1, 3 },
                { 1, 4 }
        });
        RealVector y = new ArrayRealVector(new double[] { 6, 5, 7, 10 });
        LinearLeastSquaresSolver linearLeastSquaresSolver =
                new LinearLeastSquaresSolver(new LinearLeastSquaresFunction(J, y));
        linearLeastSquaresSolver.setMethod(LinearLeastSquaresSolver.Method.CHOLESKY_DECOMPOSITION);
        double[] actualResult = linearLeastSquaresSolver.solve().toArray();
        double[] expectedResult = new double[] { 3.5, 1.4 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-8);
    }

    @Test
    public void testQRDecompositionMethod() {
        RealMatrix J = new Array2DRowRealMatrix(new double[][] {
                { 1, 1 },
                { 1, 2 },
                { 1, 3 },
                { 1, 4 }
        });
        RealVector y = new ArrayRealVector(new double[] { 6, 5, 7, 10 });
        LinearLeastSquaresSolver linearLeastSquaresSolver =
                new LinearLeastSquaresSolver(new LinearLeastSquaresFunction(J, y));
        linearLeastSquaresSolver.setMethod(LinearLeastSquaresSolver.Method.QR_DECOMPOSITION);
        double[] actualResult = linearLeastSquaresSolver.solve().toArray();
        double[] expectedResult = new double[] { 3.5, 1.4 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-8);
    }

    @Test
    public void testSingularValueDecompositionMethod() {
        RealMatrix J = new Array2DRowRealMatrix(new double[][] {
                { 1, 1 },
                { 1, 2 },
                { 1, 3 },
                { 1, 4 }
        });
        RealVector y = new ArrayRealVector(new double[] { 6, 5, 7, 10 });
        LinearLeastSquaresSolver linearLeastSquaresSolver =
                new LinearLeastSquaresSolver(new LinearLeastSquaresFunction(J, y));
        linearLeastSquaresSolver.setMethod(LinearLeastSquaresSolver.Method.SINGULAR_VALUE_DECOMPOSITION);
        double[] actualResult = linearLeastSquaresSolver.solve().toArray();
        double[] expectedResult = new double[] { 3.5, 1.4 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-8);
    }

    @Test
    public void testConjugateGradientMethod() {
        RealMatrix J = new Array2DRowRealMatrix(new double[][] {
                { 1, 1 },
                { 1, 2 },
                { 1, 3 },
                { 1, 4 }
        });
        RealVector y = new ArrayRealVector(new double[] { 6, 5, 7, 10 });
        LinearLeastSquaresSolver linearLeastSquaresSolver =
                new LinearLeastSquaresSolver(new LinearLeastSquaresFunction(J, y));
        linearLeastSquaresSolver.setMethod(LinearLeastSquaresSolver.Method.CONJUGATE_GRADIENT);
        double[] actualResult = linearLeastSquaresSolver.solve().toArray();
        double[] expectedResult = new double[] { 3.5, 1.4 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-8);
    }
}
