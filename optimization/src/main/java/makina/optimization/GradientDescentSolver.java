package makina.optimization;

import makina.math.matrix.SingularMatrixException;
import makina.math.matrix.Vector;
import makina.math.matrix.Vectors;
import makina.optimization.constraint.LinearEqualityConstraint;
import makina.optimization.function.AbstractFunction;
import makina.optimization.function.NonSmoothFunctionException;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public final class GradientDescentSolver extends AbstractLineSearchSolver {
    private final LinearEqualityConstraint linearEqualityConstraint;
    private final Vector lowerBound;
    private final Vector upperBound;

    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>>
            extends AbstractLineSearchSolver.AbstractBuilder<T> {
        private final List<LinearEqualityConstraint> linearEqualityConstraints = new ArrayList<>();

        private Vector lowerBound = null;
        private Vector upperBound = null;

        public AbstractBuilder(AbstractFunction objective, Vector initialPoint) {
            super(objective, initialPoint);
        }

        public T addLinearEqualityConstraint(LinearEqualityConstraint linearEqualityConstraint) {
            if (lowerBound != null || upperBound != null)
                throw new IllegalStateException("The gradient descent solver can be used with either box " +
                                                        "constraints, or linear equality constraints, but not both.");
            if (linearEqualityConstraint != null)
                linearEqualityConstraints.add(linearEqualityConstraint);
            return self();
        }

        public T addLinearEqualityConstraints(List<LinearEqualityConstraint> linearEqualityConstraints) {
            if (lowerBound != null || upperBound != null)
                throw new IllegalStateException("The gradient descent solver can be used with either box " +
                                                        "constraints, or linear equality constraints, but not both.");
            this.linearEqualityConstraints.addAll(linearEqualityConstraints);
            return self();
        }

        public T lowerBound(double lowerBound) {
            if (linearEqualityConstraints.size() > 0)
                throw new IllegalStateException("The gradient descent solver can be used with either box " +
                                                        "constraints, or linear equality constraints, but not both.");
            this.lowerBound = Vectors.build(1, initialPoint.type());
            this.lowerBound.setAll(lowerBound);
            return self();
        }

        public T lowerBound(Vector lowerBound) {
            if (linearEqualityConstraints.size() > 0)
                throw new IllegalStateException("The gradient descent solver can be used with either box " +
                                                        "constraints, or linear equality constraints, but not both.");
            this.lowerBound = lowerBound;
            return self();
        }

        public T upperBound(double upperBound) {
            if (linearEqualityConstraints.size() > 0)
                throw new IllegalStateException("The gradient descent solver can be used with either box " +
                                                        "constraints, or linear equality constraints, but not both.");
            this.upperBound = Vectors.build(1, initialPoint.type());
            this.upperBound.setAll(upperBound);
            return self();
        }

        public T upperBound(Vector upperBound) {
            if (linearEqualityConstraints.size() > 0)
                throw new IllegalStateException("The gradient descent solver can be used with either box " +
                                                        "constraints, or linear equality constraints, but not both.");
            this.upperBound = upperBound;
            return self();
        }

        public GradientDescentSolver build() {
            return new GradientDescentSolver(this);
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

    private GradientDescentSolver(AbstractBuilder<?> builder) {
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
        lowerBound = builder.lowerBound;
        upperBound = builder.upperBound;
    }

    @Override
    public void updateDirection() {
        currentDirection = currentGradient.mult(-1);
    }

    @Override
    public void updatePoint() {
        currentPoint = previousPoint.add(currentDirection.mult(currentStepSize));
        if (linearEqualityConstraint != null)
            try {
                currentPoint = linearEqualityConstraint.project(currentPoint);
            } catch (SingularMatrixException e) {
                throw new IllegalArgumentException("The linear equality constraint matrix is singular.");
            }
        if (lowerBound != null)
            if (lowerBound.size() > 1)
                currentPoint.maxElementwiseInPlace(lowerBound);
            else
                currentPoint.maxElementwiseInPlace(lowerBound.get(0));
        if (upperBound != null)
            if (upperBound.size() > 1)
                currentPoint.minElementwiseInPlace(upperBound);
            else
                currentPoint.minElementwiseInPlace(upperBound.get(0));
    }
}
