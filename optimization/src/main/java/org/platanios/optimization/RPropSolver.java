package org.platanios.optimization;

import org.platanios.math.matrix.Vector;
import org.platanios.math.matrix.Vectors;
import org.platanios.optimization.function.AbstractFunction;

/**
 * @author Emmanouil Antonios Platanios
 */
public final class RPropSolver extends AbstractLineSearchSolver {
    private final double etaMinus;
    private final double etaPlus;
    private final Vector lowerBound;
    private final Vector upperBound;

    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>>
            extends AbstractLineSearchSolver.AbstractBuilder<T> {
        private double etaMinus = 0.5;
        private double etaPlus = 1.2;
        private Vector lowerBound = null;
        private Vector upperBound = null;

        public AbstractBuilder(AbstractFunction objective, Vector initialPoint) {
            super(objective, initialPoint);
        }

        public T etaMinus(double etaMinus) {
            this.etaMinus = etaMinus;
            return self();
        }

        public T etaPlus(double etaPlus) {
            this.etaPlus = etaPlus;
            return self();
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

        public RPropSolver build() {
            return new RPropSolver(this);
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

    private RPropSolver(AbstractBuilder<?> builder) {
        super(builder);
        etaMinus = builder.etaMinus;
        etaPlus = builder.etaPlus;
        lowerBound = builder.lowerBound;
        upperBound = builder.upperBound;
    }

    @Override
    public void updateDirection() {
        if (currentIteration > 1) {
            currentDirection = Vectors.build(currentGradient.size(), currentGradient.type());
            for (Vector.VectorElement element : currentGradient)
                if (Math.signum(element.value()) != Math.signum(previousGradient.get(element.index())))
                    currentDirection.set(element.index(), - etaMinus * currentGradient.get(element.index()));
                else
                    currentDirection.set(element.index(), - etaPlus * currentGradient.get(element.index()));
        } else {
            currentDirection = currentGradient.mult(-1);
        }
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
