package org.platanios.learn.optimization;

import org.platanios.learn.math.matrix.*;
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
    // TODO: Add constructors to allow changing this method.
    private final ProblemConversionMethod problemConversionMethod;

    private PreconditioningMethod preconditioningMethod;
    private Matrix A;
    private Matrix oldATranspose; // Used when CONJUGATE_GRADIENT_NORMAL_EQUATION_ERROR is used.
    private Vector b;
    /** The preconditioner matrix. */
    private Matrix preconditionerMatrixInverse;
    private double beta;
    private Vector currentY;
    private Vector previousY;
    private double omega = 1;

    public ConjugateGradientSolver(QuadraticFunction objective,
                                   double[] initialPoint)
            throws NonPositiveDefiniteMatrixException {
        this(objective,
             initialPoint,
             PreconditioningMethod.SYMMETRIC_SUCCESSIVE_OVER_RELAXATION,
             ProblemConversionMethod.CONJUGATE_GRADIENT_NORMAL_EQUATION_RESIDUAL,
             false);
    }

    public ConjugateGradientSolver(LinearLeastSquaresFunction objective,
                                   double[] initialPoint)
            throws NonPositiveDefiniteMatrixException {
        this(objective,
             initialPoint,
             PreconditioningMethod.SYMMETRIC_SUCCESSIVE_OVER_RELAXATION,
             ProblemConversionMethod.CONJUGATE_GRADIENT_NORMAL_EQUATION_RESIDUAL,
             true);
    }

    public ConjugateGradientSolver(QuadraticFunction objective,
                                   PreconditioningMethod preconditioningMethod,
                                   double[] initialPoint)
            throws NonPositiveDefiniteMatrixException {
        this(objective,
             initialPoint,
             preconditioningMethod,
             ProblemConversionMethod.CONJUGATE_GRADIENT_NORMAL_EQUATION_RESIDUAL,
             false);
    }

    public ConjugateGradientSolver(LinearLeastSquaresFunction objective,
                                   PreconditioningMethod preconditioningMethod,
                                   double[] initialPoint)
            throws NonPositiveDefiniteMatrixException {
        this(objective,
             initialPoint,
             preconditioningMethod,
             ProblemConversionMethod.CONJUGATE_GRADIENT_NORMAL_EQUATION_RESIDUAL,
             true);
    }

    public ConjugateGradientSolver(QuadraticFunction objective,
                                   ProblemConversionMethod problemConversionMethod,
                                   double[] initialPoint)
            throws NonPositiveDefiniteMatrixException {
        this(objective,
             initialPoint,
             PreconditioningMethod.SYMMETRIC_SUCCESSIVE_OVER_RELAXATION,
             problemConversionMethod,
             false);
    }

    public ConjugateGradientSolver(LinearLeastSquaresFunction objective,
                                   ProblemConversionMethod problemConversionMethod,
                                   double[] initialPoint)
            throws NonPositiveDefiniteMatrixException {
        this(objective,
             initialPoint,
             PreconditioningMethod.SYMMETRIC_SUCCESSIVE_OVER_RELAXATION,
             problemConversionMethod,
             true);
    }

    public ConjugateGradientSolver(QuadraticFunction objective,
                                   PreconditioningMethod preconditioningMethod,
                                   ProblemConversionMethod problemConversionMethod,
                                   double[] initialPoint)
            throws NonPositiveDefiniteMatrixException {
        this(objective,
             initialPoint,
             preconditioningMethod,
             problemConversionMethod,
             false);
    }

    public ConjugateGradientSolver(LinearLeastSquaresFunction objective,
                                   PreconditioningMethod preconditioningMethod,
                                   ProblemConversionMethod problemConversionMethod,
                                   double[] initialPoint)
            throws NonPositiveDefiniteMatrixException {
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
                                    boolean isLinearLeastSquaresProblem)
            throws NonPositiveDefiniteMatrixException {
        super(objective, initialPoint);
        this.preconditioningMethod = preconditioningMethod;
        this.problemConversionMethod = problemConversionMethod;
        if (isLinearLeastSquaresProblem) {
            Matrix J = ((LinearLeastSquaresFunction) objective).getJ();
            Vector y = ((LinearLeastSquaresFunction) objective).getY();
            A = J.transpose().multiply(J);
            b = J.transpose().multiply(y);
        } else {
            A = ((QuadraticFunction) objective).getA();
            b = ((QuadraticFunction) objective).getB();
        }

        // Check if A is symmetric and positive definite and if it is not make the appropriate changes to the algorithm.
        CholeskyDecomposition choleskyDecomposition = new CholeskyDecomposition(A);
        if (!choleskyDecomposition.isSymmetricAndPositiveDefinite()) {
            System.err.println("WARNING: Matrix A is not symmetric. The conjugate gradient normal equation residual " +
                                       "(CGNR) method or the conjugate gradient normal equation error (CGNE) method " +
                                       "will be used, based on the setting specified by the user.");
            switch (problemConversionMethod) {
                case CONJUGATE_GRADIENT_NORMAL_EQUATION_RESIDUAL:
                    b = A.transpose().multiply(b);
                    A = A.transpose().multiply(A);
                    break;
                case CONJUGATE_GRADIENT_NORMAL_EQUATION_ERROR:
                    oldATranspose = A.transpose();
                    A = A.multiply(oldATranspose);
                    break;
                default:
                    throw new NotImplementedException();
            }

            choleskyDecomposition = new CholeskyDecomposition(A);
            if (!choleskyDecomposition.isSymmetricAndPositiveDefinite()) {
                throw new NonPositiveDefiniteMatrixException(
                        "Non positive definite matrix after trying changing the conjugate gradient problem to avoid it!"
                );
            }
        }

        currentGradient = A.multiply(currentPoint).subtract(b);

        switch (preconditioningMethod) {
            case IDENTITY:
            case JACOBI:
                break;
            case SYMMETRIC_SUCCESSIVE_OVER_RELAXATION:
                double[] diagonal = new double[A.getRowDimension()];
                double[][] lowerDiagonalMatrix = new double[A.getRowDimension()][A.getColumnDimension()];
                for (int i = 0; i < A.getRowDimension(); i++) {
                    diagonal[i] = A.getElement(i, i);
                    for (int j = 0; j < i; j++) {
                        lowerDiagonalMatrix[i][j] = A.getElement(i, j);
                    }
                }
                Matrix D = Matrix.generateDiagonalMatrix(diagonal);
                Matrix DMinusL = D.subtract((new Matrix(lowerDiagonalMatrix)).multiply(omega));
                try {
                    preconditionerMatrixInverse =
                            DMinusL.multiply(D.computeInverse()).multiply(DMinusL.transpose()).computeInverse();
                } catch (SingularMatrixException e) {
                    System.err.println("WARNING: Singular matrix in conjugate gradient problem. " +
                                               "Trying the Jacobi preconditioning method instead of the " +
                                               "symmetric successive over-relaxation preconditioning method!");
                    this.preconditioningMethod = PreconditioningMethod.JACOBI;
                }
                break;
        }

        computePreconditioningSystemSolution();
        currentDirection = currentY.multiply(-1);
    }

    @Override
    public org.platanios.learn.math.matrix.Vector solve() {
        printHeader();
        while (!checkTerminationConditions()) {
            iterationUpdate();
            currentIteration++;
            printIteration();
        }

        printTerminationMessage();

        if (problemConversionMethod ==
                ProblemConversionMethod.CONJUGATE_GRADIENT_NORMAL_EQUATION_ERROR) {
            currentPoint = oldATranspose.multiply(currentPoint);
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
        currentStepSize = previousGradient.innerProduct(previousY)
                / previousDirection.multiply(A).innerProduct(previousDirection);
        currentPoint = previousPoint.add(previousDirection.multiply(currentStepSize));
        currentGradient = previousGradient.add(A.multiply(previousDirection).multiply(currentStepSize));
        computePreconditioningSystemSolution();
        beta = currentGradient.innerProduct(currentY) / previousGradient.innerProduct(previousY);
        currentDirection = currentY.multiply(-1).add(previousDirection.multiply(beta));
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
                    tempY[i] = currentGradient.getElement(i) / A.getElement(i, i);
                }
                currentY = new Vector(tempY);
                break;
            case SYMMETRIC_SUCCESSIVE_OVER_RELAXATION:
                currentY = preconditionerMatrixInverse.multiply(currentGradient);
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
