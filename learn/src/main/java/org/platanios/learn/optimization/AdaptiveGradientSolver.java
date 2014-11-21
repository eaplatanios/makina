package org.platanios.learn.optimization;

import org.platanios.learn.math.matrix.Vectors;
import org.platanios.learn.optimization.function.AbstractStochasticFunction;
import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public final class AdaptiveGradientSolver extends AbstractStochasticIterativeSolver {
    private final double epsilon = Math.sqrt(Double.MIN_VALUE);

    private Vector sumOfGradients;
    private Vector squareRootOfSumOfGradientSquares;

    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>>
            extends AbstractStochasticIterativeSolver.AbstractBuilder<T> {
        public AbstractBuilder(AbstractStochasticFunction objective, Vector initialPoint) {
            super(objective, initialPoint);
        }

        public AdaptiveGradientSolver build() {
            return new AdaptiveGradientSolver(this);
        }
    }

    public static class Builder extends AbstractBuilder<Builder> {
        public Builder(AbstractStochasticFunction objective,
                       Vector initialPoint) {
            super(objective, initialPoint);
        }

        @Override
        protected Builder self() {
            return this;
        }
    }

    private AdaptiveGradientSolver(AbstractBuilder<?> builder) {
        super(builder);

        sumOfGradients = Vectors.build(currentGradient.size(), currentGradient.type());
        squareRootOfSumOfGradientSquares = Vectors.build(currentGradient.size(), currentGradient.type());
    }

    @Override
    public void updateDirection() {
        squareRootOfSumOfGradientSquares.hypotenuseFastInPlace(currentGradient);
        if (useL1Regularization) {
            currentDirection = sumOfGradients
                    .addInPlace(currentGradient)
                    .mapDivElementwise(x -> Math.abs(x) / (currentIteration + 1) <= l1RegularizationWeight ?
                            0.0 : - currentIteration * Math.signum(x)
                                               * (Math.abs(x) / (currentIteration + 1) - l1RegularizationWeight),
                                          squareRootOfSumOfGradientSquares.add(epsilon));
        } else {
            currentDirection = currentGradient.divElementwise(squareRootOfSumOfGradientSquares.add(epsilon)).mult(-1);
        }
    }

    @Override
    public void updatePoint() {
        if (!useL1Regularization) {
            currentPoint = previousPoint.add(currentDirection.mult(currentStepSize));
        } else {
            currentPoint = currentDirection.mult(currentStepSize);
        }
    }
}
