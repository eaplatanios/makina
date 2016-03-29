package org.platanios.optimization;

import org.platanios.math.matrix.Matrix;
import org.platanios.math.matrix.SingularMatrixException;
import org.platanios.math.matrix.Vector;
import org.platanios.math.matrix.Vectors;
import org.platanios.optimization.constraint.LinearEqualityConstraint;
import org.platanios.optimization.function.AbstractFunction;
import org.platanios.optimization.function.NonSmoothFunctionException;
import org.platanios.optimization.linesearch.StepSizeInitializationMethod;
import org.platanios.optimization.linesearch.StrongWolfeInterpolationLineSearch;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public final class NewtonSolver extends AbstractLineSearchSolver {
    private final LinearEqualityConstraint linearEqualityConstraint;

    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>>
            extends AbstractLineSearchSolver.AbstractBuilder<T> {
        private final List<LinearEqualityConstraint> linearEqualityConstraints = new ArrayList<>();

        public AbstractBuilder(AbstractFunction objective, Vector initialPoint) {
            super(objective, initialPoint);
            // TODO: Figure out why we cannot use exact line search in the case of a quadratic function.
            lineSearch = new StrongWolfeInterpolationLineSearch(objective, 1e-4, 0.9, 1);
            ((StrongWolfeInterpolationLineSearch) lineSearch)
                    .setStepSizeInitializationMethod(StepSizeInitializationMethod.UNIT);
        }

        public T addLinearEqualityConstraint(LinearEqualityConstraint linearEqualityConstraint) {
            if (linearEqualityConstraint != null)
                this.linearEqualityConstraints.add(linearEqualityConstraint);
            return self();
        }

        public T addLinearEqualityConstraints(List<LinearEqualityConstraint> linearEqualityConstraints) {
            this.linearEqualityConstraints.addAll(linearEqualityConstraints);
            return self();
        }

        public NewtonSolver build() {
            return new NewtonSolver(this);
        }
    }

    public static class Builder extends AbstractBuilder<Builder> {
        public Builder(AbstractFunction objective,
                       Vector initialPoint) {
            super(objective, initialPoint);
        }

        @Override
        protected Builder self() {
            return this;
        }
    }

    private NewtonSolver(AbstractBuilder<?> builder) {
        super(builder);
        if (builder.linearEqualityConstraints.size() > 0) {
            LinearEqualityConstraint constraints = builder.linearEqualityConstraints.get(0);
            for (int constraintIndex = 1; constraintIndex < builder.linearEqualityConstraints.size(); constraintIndex++)
                constraints = constraints.append(builder.linearEqualityConstraints.get(constraintIndex));
            linearEqualityConstraint = constraints;
            try {
                currentPoint = linearEqualityConstraint.project(currentPoint);
                currentGradient = objective.getGradient(currentPoint);
            } catch (SingularMatrixException e) {
                throw new IllegalArgumentException("The linear equality constraint matrix is singular.");
            } catch (NonSmoothFunctionException e) {
                throw new IllegalArgumentException("The objective function being optimized is non-smooth.");
            }
            previousPoint = currentPoint;
            previousGradient = currentGradient;
            currentObjectiveValue = objective.getValue(currentPoint);
        } else {
            linearEqualityConstraint = null;
        }
    }

    /**
     * Here, if the Hessian matrix is not positive definite, we modify it so that the bounded modified factorization
     * property holds for it and we have global convergence for Newton's method.
     */
    @Override
    public void updateDirection() {
        try {
            Matrix hessian = objective.getHessian(currentPoint);
            // TODO: Check Hessian for positive definiteness and modify if necessary.
            currentGradient = objective.getGradient(currentPoint);
            if (linearEqualityConstraint == null) {
                currentDirection = hessian.solve(currentGradient).mult(-1);
            } else {
                int numberOfConstraints = linearEqualityConstraint.getA().getRowDimension();
                Matrix linearSystemMatrix = new Matrix(
                        hessian.getRowDimension() + numberOfConstraints,
                        hessian.getColumnDimension() + linearEqualityConstraint.getA().getRowDimension()
                );
                linearSystemMatrix.setSubMatrix(0,
                                                hessian.getRowDimension() - 1,
                                                0,
                                                hessian.getColumnDimension() - 1,
                                                hessian);
                linearSystemMatrix.setSubMatrix(0,
                                                hessian.getRowDimension() - 1,
                                                hessian.getColumnDimension(),
                                                linearSystemMatrix.getColumnDimension() - 1,
                                                linearEqualityConstraint.getA().transpose());
                linearSystemMatrix.setSubMatrix(hessian.getRowDimension(),
                                                linearSystemMatrix.getRowDimension() - 1,
                                                0,
                                                hessian.getColumnDimension() - 1,
                                                linearEqualityConstraint.getA());
                Vector linearSystemVector = Vectors.build(currentGradient.size() + numberOfConstraints,
                                                          currentGradient.type());
                linearSystemVector.set(0,
                                       currentGradient.size() - 1,
                                       currentGradient.mult(-1));
                currentDirection = linearSystemMatrix.solve(linearSystemVector).get(0, currentGradient.size() - 1);
            }
        } catch (SingularMatrixException e) {
            e.printStackTrace();
        } catch (NonSmoothFunctionException ignored) { }
    }

    @Override
    public void updatePoint() {
        currentPoint = previousPoint.add(currentDirection.mult(currentStepSize));
    }
}
