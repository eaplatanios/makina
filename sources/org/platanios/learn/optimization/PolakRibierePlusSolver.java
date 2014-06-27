package org.platanios.learn.optimization;

import org.platanios.learn.optimization.function.Function;

/**
 * This solver is more robust than the Polak-Ribiere solver and it also guarantees that the chosen direction is always a
 * descent direction.
 *
 * @author Emmanouil Antonios Platanios
 */
public class PolakRibierePlusSolver extends AbstractNonlinearConjugateGradientSolver {
    public PolakRibierePlusSolver(Function objective,
                                  double[] initialPoint) {
        super(objective, initialPoint);
    }

    public double computeBeta() {
        return Math.max(currentGradient.dotProduct(currentGradient.subtract(previousGradient))
                                / previousGradient.dotProduct(previousGradient), 0);
    }
}
