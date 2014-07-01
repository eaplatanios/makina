package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.*;
import org.platanios.learn.optimization.function.LinearLeastSquaresFunction;
import org.platanios.learn.optimization.function.QuadraticFunction;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

/**
 * Solves the linear least squares problem of the form: \(min_{x}{\|Jx-y\|^2}\). We assume that
 * \(J\in\mathbb{R}^{m\times n}\) and \(m\geq n\).
 *
 * @author Emmanouil Antonios Platanios
 */
public class LinearLeastSquaresSolver implements Solver {
    private final LinearLeastSquaresFunction objective;

    private Method method;

    public LinearLeastSquaresSolver(LinearLeastSquaresFunction objective) {
        this.objective = objective;
        this.method = Method.SINGULAR_VALUE_DECOMPOSITION;
    }

    public RealVector solve() {
        RealMatrix J = objective.getJ();
        RealVector y = objective.getY();
        int n = J.getColumnDimension();

        switch(method) {
            case CHOLESKY_DECOMPOSITION:
                CholeskyDecomposition choleskyDecomposition = new CholeskyDecomposition(J.transpose().multiply(J));
                return choleskyDecomposition.getSolver().solve(J.transpose().operate(y));
            case QR_DECOMPOSITION:
                QRDecomposition qrDecomposition = new QRDecomposition(J);
                return qrDecomposition.getSolver().solve(y);
            case SINGULAR_VALUE_DECOMPOSITION:
                SingularValueDecomposition singularValueDecomposition = new SingularValueDecomposition(J);
                return singularValueDecomposition.getSolver().solve(y);
            case CONJUGATE_GRADIENT:
                QuadraticFunction objective =
                        new QuadraticFunction(J.transpose().multiply(J), J.transpose().operate(y));
                ConjugateGradientSolver conjugateGradientSolver =
                        new ConjugateGradientSolver(objective, new double[n]);
                return conjugateGradientSolver.solve();
            default:
                throw new NotImplementedException();
        }
    }

    public Method getMethod() {
        return method;
    }

    public void setMethod(Method method) {
        this.method = method;
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
         * faster). */
        CONJUGATE_GRADIENT
    }
}