package org.platanios.learn.optimization;

import org.platanios.learn.math.matrix.*;
import org.platanios.learn.optimization.function.LinearLeastSquaresFunction;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

/**
 * Solves the linear least squares problem of the form: \(min_{x}{\|Jx-y\|^2}\). We assume that
 * \(J\in\mathbb{R}^{m\times n}\) and \(m\geq n\).
 *
 * @author Emmanouil Antonios Platanios
 */
public class LinearLeastSquaresSolver implements Solver {
    private final LinearLeastSquaresFunction objective;

    /** This variable is not final but it can only be changed from within this class (that is, there is no setter). It
     * is only changed if the method selected by the user fails for some reason. */
    private Method method;

    public static class Builder {
        // Required parameters
        private final LinearLeastSquaresFunction objective;

        // Optional parameters - Initialized to default values
        private Method method = Method.SINGULAR_VALUE_DECOMPOSITION;

        public Builder(LinearLeastSquaresFunction objective) {
            this.objective = objective;
        }

        public Builder method(Method method) {
            this.method = method;
            return this;
        }

        public LinearLeastSquaresSolver build() {
            return new LinearLeastSquaresSolver(this);
        }
    }

    private LinearLeastSquaresSolver(Builder builder) {
        objective = builder.objective;
        method = builder.method;
    }

    public Vector solve() {
        Matrix J = objective.getJ();
        Vector y = objective.getY();
        int n = J.getColumnDimension();

        switch(method) {
            case CHOLESKY_DECOMPOSITION:
                    CholeskyDecomposition choleskyDecomposition = new CholeskyDecomposition(J.transpose().multiply(J));
                try {
                    return choleskyDecomposition.solve(J.transpose().multiply(y));
                } catch (NonSymmetricMatrixException e) {
                    System.err.println("WARNING: Non symmetric matrix in linear least squares problem. " +
                                               "Trying the singular value decomposition method instead!");
                    this.method = Method.SINGULAR_VALUE_DECOMPOSITION;
                    return solve();
                } catch (NonPositiveDefiniteMatrixException e) {
                    System.err.println("WARNING: Non positive definite matrix in linear least squares problem. " +
                                               "Trying the singular value decomposition method instead " +
                                               "of the Cholesky decomposition method!");
                    this.method = Method.SINGULAR_VALUE_DECOMPOSITION;
                    return solve();
                }
            case QR_DECOMPOSITION:
                try {
                    QRDecomposition qrDecomposition = new QRDecomposition(J);
                    return qrDecomposition.solve(y);
                } catch (SingularMatrixException e) {
                    System.err.println("WARNING: Rank deficient matrix in linear least squares problem. " +
                                               "Trying the singular value decomposition method instead " +
                                               "of the QR decomposition method!");
                    this.method = Method.SINGULAR_VALUE_DECOMPOSITION;
                    return solve();
                }
            case SINGULAR_VALUE_DECOMPOSITION:
                SingularValueDecomposition singularValueDecomposition = new SingularValueDecomposition(J);
                return singularValueDecomposition.solve(y);
            case CONJUGATE_GRADIENT:
                try {
                    LinearLeastSquaresFunction objective = new LinearLeastSquaresFunction(J, y);
                    ConjugateGradientSolver conjugateGradientSolver =
                            new ConjugateGradientSolver.Builder(objective, new double[n])
                            .preconditioningMethod(
                                    ConjugateGradientSolver.PreconditioningMethod.SYMMETRIC_SUCCESSIVE_OVER_RELAXATION
                            )
                            .build();
                    return conjugateGradientSolver.solve();
                } catch (NonPositiveDefiniteMatrixException e) {
                    System.err.println("WARNING: Non positive definite matrix in linear least squares problem. " +
                                               "Trying the singular value decomposition method instead " +
                                               "of the conjugate gradient method!");
                    this.method = Method.SINGULAR_VALUE_DECOMPOSITION;
                    return solve();
                }
            default:
                throw new NotImplementedException();
        }
    }

    public enum Method {
        /** Compute the solution of the normal equations (that is, \(J^TJx=J^Ty\)) using the Cholesky decomposition. In
         * order to use this method, This \(J^TJ\) is required to be positive definite. Furthermore, this method is
         * often effective in practice, but it has one main disadvantage. The condition number of \(J^TJ\) is equal to
         * the square of the condition number of \(J\) and the relative error in the computed solution is usually
         * proportional to the condition number. Therefore, this method may result in less accurate solutions than those
         * obtained by methods that avoid this squaring of the condition number. In fact, when \(J\) is ill-conditioned
         * this method may even break down because rounding errors may cause small negative values to appear on the
         * diagonal during the factorization process.
         * <br><br>
         * This method is particularly useful and fast when \(m\gg n\) and \(J\) is sparse. */
        CHOLESKY_DECOMPOSITION,
        /** Compute the solution using the QR decomposition. This method does not degrade the conditioning of the
         * problem unnecessarily. The relative error in the final solution is usually proportional to the condition
         * number of \(J\) and not of its square (as in the method that uses the Cholesky decomposition).
         * <br><br>
         * This method is more numerically robust than the method that uses the Cholesky decomposition. */
        QR_DECOMPOSITION,
        /** This method is more robust and provides more information about the sensitivity of the solution to
         * perturbations in the data (that is, either \(J\) or \(y\)) than the other two methods.
         * <br><br>
         * This method is more robust and more reliable than both the other two methods, but it is also more
         * computationally expensive. */
        SINGULAR_VALUE_DECOMPOSITION,
        /** This method uses the (iterative) conjugate gradient numerical optimization solver. It is better than the
         * matrix decomposition based methods when dealing with large-scale problems (in those cases it should be much
         * faster). In this case, the matrix \(J^TJ\) has to be symmetric and positive definite. */
        CONJUGATE_GRADIENT
    }
}
