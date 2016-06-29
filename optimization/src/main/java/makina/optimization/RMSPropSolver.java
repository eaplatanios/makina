package makina.optimization;

import makina.math.matrix.Vector;
import makina.math.matrix.Vectors;
import makina.optimization.function.AbstractStochasticFunction;

/**
 * Notice that the x+= update is identical to Adagrad, but the cache variable is a "leaky". Hence, RMSProp still
 * modulates the learning rate of each weight based on the magnitudes of its gradients, which has a beneficial
 * equalizing effect, but unlike Adagrad the updates do not get monotonically smaller.
 *
 * @author Emmanouil Antonios Platanios
 */
public final class RMSPropSolver extends AbstractStochasticIterativeSolver {
    private final double epsilon = Math.sqrt(Double.MIN_VALUE);

    private final double decayRate;

    private Vector sumOfGradients;
    private Vector squareRootOfSumOfGradientSquares;

    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>>
            extends AbstractStochasticIterativeSolver.AbstractBuilder<T> {
        private double decayRate = 0.9;

        public AbstractBuilder(AbstractStochasticFunction objective, Vector initialPoint) {
            super(objective, initialPoint);
        }

        public T decayRate(double decayRate) {
            if (decayRate > 1 || decayRate < 0)
                throw new IllegalArgumentException("The decay rate must lie in [0,1].");
            this.decayRate = decayRate;
            return self();
        }

        public RMSPropSolver build() {
            return new RMSPropSolver(this);
        }
    }

    public static class Builder extends AbstractBuilder<Builder> {
        public Builder(AbstractStochasticFunction objective, Vector initialPoint) {
            super(objective, initialPoint);
        }

        @Override
        protected Builder self() {
            return this;
        }
    }

    protected RMSPropSolver(AbstractBuilder<?> builder) {
        super(builder);
        decayRate = builder.decayRate;
        sumOfGradients = Vectors.build(currentGradient.size(), currentGradient.type());
        squareRootOfSumOfGradientSquares = Vectors.build(currentGradient.size(), currentGradient.type());
    }

    @Override
    public void updateDirection() {
        squareRootOfSumOfGradientSquares
                .multInPlace(Math.sqrt(decayRate))
                .hypotenuseFastInPlace(currentGradient.mult(Math.sqrt(1 - decayRate)));
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

    @Override
    public void handleBoxConstraints() {
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
