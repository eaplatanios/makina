package org.platanios.learn.optimization;

import org.platanios.learn.optimization.function.Function;

/**
 * @author Emmanouil Antonios Platanios
 */
public class FletcherReevesSolver extends AbstractNonlinearConjugateGradientSolver {
    public FletcherReevesSolver(Function objective,
                                double[] initialPoint) {
        super(objective, initialPoint);
    }

    public double computeBeta() {
        return currentGradient.dotProduct(currentGradient) / previousGradient.dotProduct(previousGradient);
    }
}
