package org.platanios.optimization;

import org.platanios.math.matrix.*;
import org.platanios.optimization.function.LinearLeastSquaresFunction;
import org.platanios.optimization.function.QuadraticFunction;

/**
 * Matrix \(A\) in this case needs to be symmetric and positive definite. If it is not, then the solver may be able to
 * transform the problem in a way such that the final matrix being used is positive definite. The enumeration
 * {@link ConjugateGradientSolver.ProblemConversionMethod} contains the currently
 * supported methods to do that. However, none of this methods will work is \(A\) has at least one zero eigenvalue.
 *
 * The biconjugate gradient does not have that requirement about matrix \(A\).
 *
 * @author Emmanouil Antonios Platanios
 */
public final class ConjugateGradientSolver extends AbstractIterativeSolver {
    private final PreconditioningMethod preconditioningMethod;
    private final Matrix preconditionerMatrixInverse;
    private final ProblemConversionMethod problemConversionMethod;
    private final boolean convertedProblem;
    private final Matrix A;

    private Vector currentDirection;
    private Vector currentY;

    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>>
            extends AbstractIterativeSolver.AbstractBuilder<T> {
        private PreconditioningMethod preconditioningMethod =
                PreconditioningMethod.SYMMETRIC_SUCCESSIVE_OVER_RELAXATION;
        private ProblemConversionMethod problemConversionMethod =
                ProblemConversionMethod.CONJUGATE_GRADIENT_NORMAL_EQUATION_RESIDUAL;
        private double symmetricSuccessiveOverRelaxationOmega = 1;

        public AbstractBuilder(QuadraticFunction objective, Vector initialPoint) {
            super(objective, initialPoint);
        }

        public AbstractBuilder(LinearLeastSquaresFunction objective, Vector initialPoint) {
            super(objective, initialPoint);
        }

        public T preconditioningMethod(PreconditioningMethod preconditioningMethod)
        {
            this.preconditioningMethod = preconditioningMethod;
            return self();
        }

        public T problemConversionMethod(ProblemConversionMethod problemConversionMethod)
        {
            this.problemConversionMethod = problemConversionMethod;
            return self();
        }

        public T symmetricSuccessiveOverRelaxationOmega(double symmetricSuccessiveOverRelaxationOmega) {
            this.symmetricSuccessiveOverRelaxationOmega = symmetricSuccessiveOverRelaxationOmega;
            return self();
        }

        public ConjugateGradientSolver build() {
            try {
                return new ConjugateGradientSolver(this);
            } catch (NonPositiveDefiniteMatrixException e) {
                e.printStackTrace();
                return null;
            }
        }

        // TODO: Have to do something to fix this ugly build method code.
        public ConjugateGradientSolver buildWithChecking()
                throws NonPositiveDefiniteMatrixException {
            return new ConjugateGradientSolver(this);
        }
    }

    public static class Builder extends AbstractBuilder<Builder> {
        public Builder(QuadraticFunction objective,
                       Vector initialPoint) {
            super(objective, initialPoint);
        }

        public Builder(LinearLeastSquaresFunction objective,
                       Vector initialPoint) {
            super(objective, initialPoint);
        }

        @Override
        protected Builder self() {
            return this;
        }
    }

    private ConjugateGradientSolver(AbstractBuilder<?> builder)
            throws NonPositiveDefiniteMatrixException {
        super(builder);
        currentObjectiveValue = objective.getValue(currentPoint);
        problemConversionMethod = builder.problemConversionMethod;

        Matrix temporaryA;
        Vector b;
        if (objective instanceof LinearLeastSquaresFunction) {
            Matrix J = ((LinearLeastSquaresFunction) objective).getJ();
            Vector y = ((LinearLeastSquaresFunction) objective).getY();
            temporaryA = J.transpose().multiply(J);
            b = J.transpose().multiply(y);
        } else {
            temporaryA = ((QuadraticFunction) objective).getA();
            b = ((QuadraticFunction) objective).getB();
        }

        // Check if A is symmetric and positive definite and if it is not make the appropriate changes to the algorithm.
        CholeskyDecomposition choleskyDecomposition = new CholeskyDecomposition(temporaryA);
        if (!choleskyDecomposition.isSymmetricAndPositiveDefinite()) {
            System.err.println("WARNING: Matrix A is not symmetric.");
            convertedProblem = true;
            A = problemConversionMethod.computeNewA(temporaryA);
            b = problemConversionMethod.computeNewB(temporaryA, b);
            choleskyDecomposition = new CholeskyDecomposition(A);
            if (!choleskyDecomposition.isSymmetricAndPositiveDefinite()) {
                throw new NonPositiveDefiniteMatrixException(
                        "Non positive definite matrix after trying changing the conjugate gradient problem to avoid it!"
                );
            }
        } else {
            convertedProblem = false;
            A = temporaryA;
        }

        currentGradient = A.multiply(currentPoint).sub(b);

        // Initialization for the preconditioning method.
        PreconditioningMethod temporaryPreconditioningMethod;
        Matrix temporaryPreconditionerMatrixInverse;
        try {
            temporaryPreconditioningMethod = builder.preconditioningMethod;
            temporaryPreconditionerMatrixInverse = builder.preconditioningMethod.initializeMethod(this, builder);
        } catch (SingularMatrixException e) {
            System.err.println("WARNING: Singular matrix in conjugate gradient problem. " +
                                       "Trying the Jacobi preconditioning method instead of the " +
                                       "symmetric successive over-relaxation preconditioning method!");
            temporaryPreconditioningMethod = PreconditioningMethod.JACOBI;
            temporaryPreconditionerMatrixInverse = null;
        }
        preconditioningMethod = temporaryPreconditioningMethod;
        preconditionerMatrixInverse = temporaryPreconditionerMatrixInverse;

        preconditioningMethod.computePreconditioningSystemSolution(this);
        currentDirection = currentY.mult(-1);
    }

    @Override
    public Vector solve() {
        currentPoint = super.solve();
        if (convertedProblem)
            currentPoint = problemConversionMethod.transformPoint(currentPoint);
        return currentPoint;
    }

    @Override
    public void performIterationUpdates() {
        previousPoint = currentPoint;
        previousGradient = currentGradient;
        Vector previousDirection = currentDirection;
        Vector previousY = currentY;
        // This procedure can be sped up for the linear least squares case by using Jacobian vector products.
        double currentStepSize = previousGradient.inner(previousY)
                / previousDirection.transMult(A).inner(previousDirection);
        currentPoint = previousPoint.add(previousDirection.mult(currentStepSize));
        currentGradient = previousGradient.add(A.multiply(previousDirection).mult(currentStepSize));
        preconditioningMethod.computePreconditioningSystemSolution(this);
        currentDirection = currentY
                .mult(-1)
                .add(previousDirection.mult(currentGradient.inner(currentY) / previousGradient.inner(previousY)));
        if (checkForObjectiveConvergence || logObjectiveValue) {
            previousObjectiveValue = currentObjectiveValue;
            currentObjectiveValue = objective.getValue(currentPoint);
        }
    }

    public enum PreconditioningMethod {
        /** Use the identity matrix as the preconditioner matrix. This gives the simple conjugate gradient method (that
         * is, there is no preconditioning). */
        IDENTITY {
            @Override
            protected Matrix initializeMethod(ConjugateGradientSolver solver,
                                              AbstractBuilder builder) {
                return null;
            }

            @Override
            protected void computePreconditioningSystemSolution(ConjugateGradientSolver solver) {
                solver.currentY = solver.currentGradient;
            }
        },
        /** Use a diagonal matrix, whose diagonal elements are equal to the diagonal elements of A, as the
         * preconditioner matrix. */
        JACOBI {
            @Override
            protected Matrix initializeMethod(ConjugateGradientSolver solver,
                                              AbstractBuilder builder) {
                return null;
            }

            @Override
            protected void computePreconditioningSystemSolution(ConjugateGradientSolver solver) {
                double[] tempY = new double[solver.currentGradient.size()];
                for (int i = 0; i < tempY.length; i++) {
                    tempY[i] = solver.currentGradient.get(i) / solver.A.getElement(i, i);
                }
                solver.currentY = Vectors.dense(tempY);
            }
        },
        SYMMETRIC_SUCCESSIVE_OVER_RELAXATION {
            @Override
            protected Matrix initializeMethod(ConjugateGradientSolver solver,
                                              AbstractBuilder builder) throws SingularMatrixException {
                double[] diagonal = new double[solver.A.getRowDimension()];
                double[][] lowerDiagonalMatrix = new double[solver.A.getRowDimension()][solver.A.getColumnDimension()];
                for (int i = 0; i < solver.A.getRowDimension(); i++) {
                    diagonal[i] = solver.A.getElement(i, i);
                    for (int j = 0; j < i; j++) {
                        lowerDiagonalMatrix[i][j] = solver.A.getElement(i, j);
                    }
                }
                Matrix D = Matrix.diagonal(diagonal);
                Matrix DMinusL = D.subtract(
                        (new Matrix(lowerDiagonalMatrix)).multiply(builder.symmetricSuccessiveOverRelaxationOmega)
                );
                return DMinusL.multiply(D.computeInverse()).multiply(DMinusL.transpose()).computeInverse();
            }

            @Override
            protected void computePreconditioningSystemSolution(ConjugateGradientSolver solver) {
                solver.currentY = solver.preconditionerMatrixInverse.multiply(solver.currentGradient);
            }
        };

        protected abstract Matrix initializeMethod(ConjugateGradientSolver solver,
                                                   AbstractBuilder builder) throws SingularMatrixException;
        protected abstract void computePreconditioningSystemSolution(ConjugateGradientSolver solver);
    }

    /**
     * Enumeration of supported methods for converting a non symmetric or non positive definite problem into a symmetric
     * positive definite problem, for solving it using the conjugate gradient solver. The method used by default is the
     * conjugate gradient normal equation residual (CGNR) method.
     * */
    public enum ProblemConversionMethod {
        CONJUGATE_GRADIENT_NORMAL_EQUATION_RESIDUAL {
            Matrix initialATranspose;

            @Override
            protected Matrix computeNewA(Matrix A) {
                if (initialATranspose == null) {
                    initialATranspose = A.transpose();
                }
                return initialATranspose.multiply(A);
            }

            @Override
            protected Vector computeNewB(Matrix A, Vector b) {
                if (initialATranspose == null) {
                    initialATranspose = A.transpose();
                }
                return initialATranspose.multiply(b);
            }

            @Override
            protected Vector transformPoint(Vector point) {
                return point;
            }
        },
        CONJUGATE_GRADIENT_NORMAL_EQUATION_ERROR {
            Matrix initialATranspose;

            @Override
            protected Matrix computeNewA(Matrix A) {
                if (initialATranspose == null) {
                    initialATranspose = A.transpose();
                }
                return A.multiply(initialATranspose);
            }

            @Override
            protected Vector computeNewB(Matrix A, Vector b) {
                return b;
            }

            @Override
            protected Vector transformPoint(Vector point) {
                return initialATranspose.multiply(point);
            }
        };

        protected abstract Matrix computeNewA(Matrix A);
        protected abstract Vector computeNewB(Matrix A, Vector b);
        protected abstract Vector transformPoint(Vector point);
    }
}
