package org.platanios.learn.optimization;

import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;
import org.platanios.learn.optimization.function.AbstractFunction;

/**
 * @author Emmanouil Antonios Platanios
 */
public final class GradientDescentSolver extends AbstractLineSearchSolver {
    private final Vector lowerBound;
    private final Vector upperBound;

    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>>
            extends AbstractLineSearchSolver.AbstractBuilder<T> {
        private Vector lowerBound = null;
        private Vector upperBound = null;

        public AbstractBuilder(AbstractFunction objective, Vector initialPoint) {
            super(objective, initialPoint);
        }

        public T lowerBound(double lowerBound) {
            this.lowerBound = Vectors.build(1, initialPoint.type());
            this.lowerBound.setAll(lowerBound);
            return self();
        }

        public T lowerBound(Vector lowerBound) {
            this.lowerBound = lowerBound;
            return self();
        }

        public T upperBound(double upperBound) {
            this.upperBound = Vectors.build(1, initialPoint.type());
            this.upperBound.setAll(upperBound);
            return self();
        }

        public T upperBound(Vector upperBound) {
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
