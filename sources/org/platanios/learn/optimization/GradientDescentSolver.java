package org.platanios.learn.optimization;

import org.platanios.learn.optimization.function.AbstractFunction;

/**
 * @author Emmanouil Antonios Platanios
 */
public class GradientDescentSolver extends AbstractLineSearchSolver {
    public static class Builder {
        // Required parameters
        private final AbstractFunction objective;
        private final double[] initialPoint;

        public Builder(AbstractFunction objective, double[] initialPoint) {
            this.objective = objective;
            this.initialPoint = initialPoint;
        }

        public GradientDescentSolver build() {
            return new GradientDescentSolver(this);
        }
    }

    public GradientDescentSolver(Builder builder) {
        super(builder.objective, builder.initialPoint);
    }

    @Override
    public void updateDirection() {
        currentGradient = objective.getGradient(currentPoint);
        currentDirection = currentGradient.multiply(-1);
    }

    @Override
    public void updatePoint() {
        currentPoint = currentPoint.add(currentDirection.multiply(currentStepSize));
    }
}
