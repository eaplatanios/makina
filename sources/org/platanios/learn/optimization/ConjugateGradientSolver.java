package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.*;
import org.platanios.learn.optimization.function.AbstractFunction;
import org.platanios.learn.optimization.function.LinearLeastSquaresFunction;
import org.platanios.learn.optimization.function.QuadraticFunction;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

/**
 * Matrix \(A\) in this case needs to be symmetric and positive definite. If it is not, then the solver may be able to
 * transform the problem in a way such that the final matrix being used is positive definite. The enumeration
 * {@link org.platanios.learn.optimization.ConjugateGradientSolver.ProblemConversionMethod} contains the currently
 * supported methods to do that. However, none of this methods will work is \(A\) has at least one zero eigenvalue.
 *
 * The biconjugate gradient does not have that requirement about matrix \(A\).
 *
 * @author Emmanouil Antonios Platanios
 */
public class ConjugateGradientSolver extends AbstractIterativeSolver {
    private final PreconditioningMethod preconditioningMethod;
    // TODO: Add constructors to allow changing this method.
    private final ProblemConversionMethod problemConversionMethod;

    private RealMatrix A;
    private RealMatrix oldATranspose; // Used when CONJUGATE_GRADIENT_NORMAL_EQUATION_ERROR is used.
    private RealVector b;
    /** The preconditioner matrix. */
    private RealMatrix preconditionerMatrixInverse;
    private double beta;
    private RealVector currentY;
    private RealVector previousY;
    private double omega = 1;

    public ConjugateGradientSolver(QuadraticFunction objective,
                                   double[] initialPoint) {
        this(objective,
             initialPoint,
             PreconditioningMethod.SYMMETRIC_SUCCESSIVE_OVER_RELAXATION,
             ProblemConversionMethod.CONJUGATE_GRADIENT_NORMAL_EQUATION_RESIDUAL,
             false);
    }

    public ConjugateGradientSolver(LinearLeastSquaresFunction objective,
                                   double[] initialPoint) {
        this(objective,
             initialPoint,
             PreconditioningMethod.SYMMETRIC_SUCCESSIVE_OVER_RELAXATION,
             ProblemConversionMethod.CONJUGATE_GRADIENT_NORMAL_EQUATION_RESIDUAL,
             true);
    }

    public ConjugateGradientSolver(QuadraticFunction objective,
                                   PreconditioningMethod preconditioningMethod,
                                   double[] initialPoint) {
        this(objective,
             initialPoint,
             preconditioningMethod,
             ProblemConversionMethod.CONJUGATE_GRADIENT_NORMAL_EQUATION_RESIDUAL,
             false);
    }

    public ConjugateGradientSolver(LinearLeastSquaresFunction objective,
                                   PreconditioningMethod preconditioningMethod,
                                   double[] initialPoint) {
        this(objective,
             initialPoint,
             preconditioningMethod,
             ProblemConversionMethod.CONJUGATE_GRADIENT_NORMAL_EQUATION_RESIDUAL,
             true);
    }

    public ConjugateGradientSolver(QuadraticFunction objective,
                                   ProblemConversionMethod problemConversionMethod,
                                   double[] initialPoint) {
        this(objective,
             initialPoint,
             PreconditioningMethod.SYMMETRIC_SUCCESSIVE_OVER_RELAXATION,
             problemConversionMethod,
             false);
    }

    public ConjugateGradientSolver(LinearLeastSquaresFunction objective,
                                   ProblemConversionMethod problemConversionMethod,
                                   double[] initialPoint) {
        this(objective,
             initialPoint,
             PreconditioningMethod.SYMMETRIC_SUCCESSIVE_OVER_RELAXATION,
             problemConversionMethod,
             true);
    }

    public ConjugateGradientSolver(QuadraticFunction objective,
                                   PreconditioningMethod preconditioningMethod,
                                   ProblemConversionMethod problemConversionMethod,
                                   double[] initialPoint) {
        this(objective,
             initialPoint,
             preconditioningMethod,
             problemConversionMethod,
             false);
    }

    public ConjugateGradientSolver(LinearLeastSquaresFunction objective,
                                   PreconditioningMethod preconditioningMethod,
                                   ProblemConversionMethod problemConversionMethod,
                                   double[] initialPoint) {
        this(objective,
             initialPoint,
             preconditioningMethod,
             problemConversionMethod,
             true);
    }

    private ConjugateGradientSolver(AbstractFunction objective,
                                    double[] initialPoint,
                                    PreconditioningMethod preconditioningMethod,
                                    ProblemConversionMethod problemConversionMethod,
                                    boolean isLinearLeastSquaresProblem) {
        super(objective, initialPoint);
        this.preconditioningMethod = preconditioningMethod;
        this.problemConversionMethod = problemConversionMethod;
        if (isLinearLeastSquaresProblem) {
            RealMatrix J = ((LinearLeastSquaresFunction) objective).getJ();
            RealVector y = ((LinearLeastSquaresFunction) objective).getY();
            A = J.transpose().multiply(J);
            b = J.transpose().operate(y);
        } else {
            A = ((QuadraticFunction) objective).getA();
            b = ((QuadraticFunction) objective).getB();
        }

        // Check if A is symmetric and positive definite and if it is not make the appropriate changes to the algorithm.
        try {
            CholeskyDecomposition choleskyDecomposition = new CholeskyDecomposition(A);
        } catch (NonSymmetricMatrixException|NonPositiveDefiniteMatrixException e) {
            System.err.println("WARNING: Matrix A is not symmetric. The conjugate gradient normal equation residual " +
                                       "(CGNR) method or the conjugate gradient normal equation error (CGNE) method " +
                                       "will be used, based on the setting specified by the user.");
            switch (problemConversionMethod) {
                case CONJUGATE_GRADIENT_NORMAL_EQUATION_RESIDUAL:
                    b = A.transpose().operate(b);
                    A = A.transpose().multiply(A);
                    break;
                case CONJUGATE_GRADIENT_NORMAL_EQUATION_ERROR:
                    oldATranspose = A.transpose();
                    A = A.multiply(oldATranspose);
                    break;
                default:
                    throw new NotImplementedException();
            }
        }

        currentGradient = A.operate(currentPoint).subtract(b);

        switch (preconditioningMethod) {
            case IDENTITY:
            case JACOBI:
                break;
            case SYMMETRIC_SUCCESSIVE_OVER_RELAXATION:
                double[] diagonal = new double[A.getRowDimension()];
                double[][] lowerDiagonalMatrix = new double[A.getRowDimension()][A.getColumnDimension()];
                for (int i = 0; i < A.getRowDimension(); i++) {
                    diagonal[i] = A.getEntry(i, i);
                    for (int j = 0; j < i; j++) {
                        lowerDiagonalMatrix[i][j] = A.getEntry(i, j);
                    }
                }
                RealMatrix D = MatrixUtils.createRealDiagonalMatrix(diagonal);
                RealMatrix DMinusL = D.subtract((new Array2DRowRealMatrix(lowerDiagonalMatrix)).scalarMultiply(omega));
                preconditionerMatrixInverse =
                        MatrixUtils.inverse(DMinusL.multiply(MatrixUtils.inverse(D)).multiply(DMinusL.transpose()));
                break;
        }

        computePreconditioningSystemSolution();
        currentDirection = currentY.mapMultiply(-1);
    }

    @Override
    public RealVector solve() {
        printHeader();
        while (!checkTerminationConditions()) {
            iterationUpdate();
            currentIteration++;
            printIteration();
        }

        printTerminationMessage();

        if (problemConversionMethod ==
                ProblemConversionMethod.CONJUGATE_GRADIENT_NORMAL_EQUATION_ERROR) {
            currentPoint = oldATranspose.operate(currentPoint);
        }

        return currentPoint;
    }

    @Override
    public void iterationUpdate() {
        previousPoint = currentPoint;
        previousGradient = currentGradient;
        previousDirection = currentDirection;
        previousY = currentY;
        // This procedure can be sped up for the linear least squares case by using Jacobian vector products.
        currentStepSize = previousGradient.dotProduct(previousY)
                / A.preMultiply(previousDirection).dotProduct(previousDirection);
        currentPoint = previousPoint.add(previousDirection.mapMultiply(currentStepSize));
        currentGradient = previousGradient.add(A.operate(previousDirection).mapMultiply(currentStepSize));
        computePreconditioningSystemSolution();
        beta = currentGradient.dotProduct(currentY) / previousGradient.dotProduct(previousY);
        currentDirection = currentY.mapMultiply(-1).add(previousDirection.mapMultiply(beta));
        currentObjectiveValue = objective.getValue(currentPoint);
    }

    private void computePreconditioningSystemSolution() {
        switch (preconditioningMethod) {
            case IDENTITY:
                currentY = currentGradient;
                break;
            case JACOBI:
                double[] tempY = new double[currentGradient.getDimension()];
                for (int i = 0; i < tempY.length; i++) {
                    tempY[i] = currentGradient.getEntry(i) / A.getEntry(i, i);
                }
                currentY = new ArrayRealVector(tempY);
                break;
            case SYMMETRIC_SUCCESSIVE_OVER_RELAXATION:
                currentY = preconditionerMatrixInverse.operate(currentGradient);
                break;
            default:
                throw new NotImplementedException();
        }
    }

    public enum PreconditioningMethod {
        /** Use the identity matrix as the preconditioner matrix. This gives the simple conjugate gradient method (that
         * is, there is no preconditioning). */
        IDENTITY,
        /** Use a diagonal matrix, whose diagonal elements are equal to the diagonal elements of A, as the
         * preconditioner matrix. */
        JACOBI,
        SYMMETRIC_SUCCESSIVE_OVER_RELAXATION
    }

    /**
     * Enumeration of supported methods for converting a non symmetric or non positive definite problem into a symmetric
     * positive definite problem, for solving it using the conjugate gradient solver. The method used by default is the
     * conjugate gradient normal equation residual (CGNR) method.
     * */
    public enum ProblemConversionMethod {
        CONJUGATE_GRADIENT_NORMAL_EQUATION_RESIDUAL,
        CONJUGATE_GRADIENT_NORMAL_EQUATION_ERROR
    }
}
