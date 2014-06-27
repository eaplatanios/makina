package org.platanios.learn.optimization;

import org.platanios.learn.optimization.function.Function;

/**
 * @author Emmanouil Antonios Platanios
 */
public class PolakRibiereSolver extends AbstractNonlinearConjugateGradientSolver {
    public PolakRibiereSolver(Function objective,
                              double[] initialPoint) {
        super(objective, initialPoint);
    }

    public double computeBeta() {
        return currentGradient.dotProduct(currentGradient.subtract(previousGradient))
                / previousGradient.dotProduct(previousGradient);
    }
}
