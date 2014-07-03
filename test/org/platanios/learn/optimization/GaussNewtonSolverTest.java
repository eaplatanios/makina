package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.junit.Assert;
import org.junit.Test;
import org.platanios.learn.optimization.function.AbstractLeastSquaresFunction;

/**
 * @author Emmanouil Antonios Platanios
 */
public class GaussNewtonSolverTest {
    @Test
    public void testGaussNewtonCholeskyDecompositionSolver() {
        System.out.println("Exponential Least Squares Function Gauss-Newton (Cholesky Decomposition) Solver:\n");
        double[] t = { 1, 2, 4, 5, 8 };
        double[] y = { 3.2939, 4.2699, 7.1749, 9.3008, 20.259 };
        GaussNewtonSolver gaussNewtonSolver =
                new GaussNewtonSolver(new ExponentialLeastSquaresFunction(t, y), new double[] { 0, 0 });
        gaussNewtonSolver.setLinearLeastSquaresSubproblemMethod(LinearLeastSquaresSolver.Method.CHOLESKY_DECOMPOSITION);
        double[] actualResult = gaussNewtonSolver.solve().toArray();
        double[] expectedResult = new double[] { 2.5411, 0.2595 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-4);
    }

    @Test
    public void testGaussNewtonQRDecompositionSolver() {
        System.out.println("Exponential Least Squares Function Gauss-Newton (QR Decomposition) Solver:\n");
        double[] t = { 1, 2, 4, 5, 8 };
        double[] y = { 3.2939, 4.2699, 7.1749, 9.3008, 20.259 };
        GaussNewtonSolver gaussNewtonSolver =
                new GaussNewtonSolver(new ExponentialLeastSquaresFunction(t, y), new double[] { 0, 0 });
        gaussNewtonSolver.setLinearLeastSquaresSubproblemMethod(LinearLeastSquaresSolver.Method.QR_DECOMPOSITION);
        double[] actualResult = gaussNewtonSolver.solve().toArray();
        double[] expectedResult = new double[] { 2.5411, 0.2595 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-4);
    }

    @Test
    public void testGaussNewtonSingularValueDecompositionSolver() {
        System.out.println("Exponential Least Squares Function Gauss-Newton (Singular Value Decomposition) Solver:\n");
        double[] t = { 1, 2, 4, 5, 8 };
        double[] y = { 3.2939, 4.2699, 7.1749, 9.3008, 20.259 };
        GaussNewtonSolver gaussNewtonSolver =
                new GaussNewtonSolver(new ExponentialLeastSquaresFunction(t, y), new double[] { 0, 0 });
        gaussNewtonSolver.setLinearLeastSquaresSubproblemMethod(
                LinearLeastSquaresSolver.Method.SINGULAR_VALUE_DECOMPOSITION
        );
        double[] actualResult = gaussNewtonSolver.solve().toArray();
        double[] expectedResult = new double[] { 2.5411, 0.2595 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-4);
    }

    @Test
    public void testGaussNewtonConjugateGradientSolver() {
        System.out.println("Exponential Least Squares Function Gauss-Newton (Conjugate Gradient) Solver:\n");
        double[] t = { 1, 2, 4, 5, 8 };
        double[] y = { 3.2939, 4.2699, 7.1749, 9.3008, 20.259 };
        GaussNewtonSolver gaussNewtonSolver =
                new GaussNewtonSolver(new ExponentialLeastSquaresFunction(t, y), new double[] { 0, 0 });
        gaussNewtonSolver.setLinearLeastSquaresSubproblemMethod(LinearLeastSquaresSolver.Method.CONJUGATE_GRADIENT);
        double[] actualResult = gaussNewtonSolver.solve().toArray();
        double[] expectedResult = new double[] { 2.5411, 0.2595 };
        Assert.assertArrayEquals(expectedResult, actualResult, 1e-4);
    }

    public class ExponentialLeastSquaresFunction extends AbstractLeastSquaresFunction {
        private final double[] t;
        private final double[] y;

        public ExponentialLeastSquaresFunction(double[] t, double[] y) {
            this.t = t;
            this.y = y;
        }

        @Override
        public RealVector computeResiduals(RealVector point) {
            double[] resultArray = new double[t.length];
            for (int i = 0; i < t.length; i++) {
                resultArray[i] = point.getEntry(0) * Math.exp(point.getEntry(1) * t[i]) - y[i];
            }
            return new ArrayRealVector(resultArray);
        }

        @Override
        public RealMatrix computeJacobian(RealVector point) {
            double[][] resultArray = new double[t.length][2];
            for (int i = 0; i < t.length; i++) {
                resultArray[i][0] = Math.exp(point.getEntry(1) * t[i]);
                resultArray[i][1] = point.getEntry(0) * Math.exp(point.getEntry(1) * t[i]) * t[i];
            }
            return new Array2DRowRealMatrix(resultArray);
        }
    }
}
