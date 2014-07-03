package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.*;
import org.platanios.learn.optimization.function.AbstractFunction;
import org.platanios.learn.optimization.function.LinearLeastSquaresFunction;
import org.platanios.learn.optimization.function.QuadraticFunction;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

/**
 * Matrix A in this case needs to be symmetric and positive definite. The biconjugate gradient does not have that
 * requirement.
 *
 * @author Emmanouil Antonios Platanios
 */
public class ConjugateGradientSolver extends AbstractIterativeSolver {
    private final RealMatrix A;
    private final PreconditioningMethod preconditioningMethod;
    private final boolean useJacobianVectorProducts;

    /** The preconditioner matrix. */
    private RealMatrix preconditionerMatrixInverse;
    private double beta;
    private RealVector currentY;
    private RealVector previousY;
    private double omega = 1;

    public ConjugateGradientSolver(QuadraticFunction objective,
                                   double[] initialPoint) {
        this(objective, initialPoint, PreconditioningMethod.SYMMETRIC_SUCCESSIVE_OVER_RELAXATION, false);
    }

    public ConjugateGradientSolver(LinearLeastSquaresFunction objective,
                                   double[] initialPoint) {
        this(objective, initialPoint, PreconditioningMethod.SYMMETRIC_SUCCESSIVE_OVER_RELAXATION, true);
    }

    public ConjugateGradientSolver(QuadraticFunction objective,
                                   PreconditioningMethod preconditioningMethod,
                                   double[] initialPoint) {
        this(objective, initialPoint, preconditioningMethod, false);
    }

    public ConjugateGradientSolver(LinearLeastSquaresFunction objective,
                                   PreconditioningMethod preconditioningMethod,
                                   double[] initialPoint) {
        this(objective, initialPoint, preconditioningMethod, true);
    }

    private ConjugateGradientSolver(AbstractFunction objective,
                                    double[] initialPoint,
                                    PreconditioningMethod preconditioningMethod,
                                    boolean isLinearLeastSquaresProblem) {
        super(objective, initialPoint);
        this.preconditioningMethod = preconditioningMethod;
        this.useJacobianVectorProducts =
                isLinearLeastSquaresProblem && (preconditioningMethod == PreconditioningMethod.IDENTITY);
        if (useJacobianVectorProducts) {
            A = ((LinearLeastSquaresFunction) objective).getJ();
        } else if (isLinearLeastSquaresProblem) {
            RealMatrix J = ((LinearLeastSquaresFunction) objective).getJ();
            A = J.transpose().multiply(J);
        } else {
            A = ((QuadraticFunction) objective).getA();
        }

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
    public void iterationUpdate() {
        previousPoint = currentPoint;
        previousGradient = currentGradient;
        previousDirection = currentDirection;
        previousY = currentY;
        double stepSize = previousGradient.dotProduct(previousY);

        // When we are solving a linear least squares system and we are not using any preconditioning method, then we do
        // not have to perform the matrix multiplication J^TJ and then multiply J^TJ with a vector; we can instead first
        // multiply the first vector by J and then the resulting vector by J^T.
        if (useJacobianVectorProducts) {
            stepSize /= A.transpose().operate(A.operate(previousDirection)).dotProduct(previousDirection);
            currentPoint = previousPoint.add(previousDirection.mapMultiply(stepSize));
            currentGradient =
                    previousGradient.add(A.transpose().operate(A.operate(previousDirection)).mapMultiply(stepSize));
        } else {
            stepSize /= A.preMultiply(previousDirection).dotProduct(previousDirection);
            currentPoint = previousPoint.add(previousDirection.mapMultiply(stepSize));
            currentGradient = previousGradient.add(A.operate(previousDirection).mapMultiply(stepSize));
        }

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
}
