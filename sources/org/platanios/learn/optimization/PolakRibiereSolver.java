package org.platanios.learn.optimization;

import org.platanios.learn.optimization.function.Function;

/**
 * This solver does not guarantee that the chosen direction is a descent direction. A modification of it that guarantees
 * that is {@link org.platanios.learn.optimization.PolakRibierePlusSolver}, which also appears to be more robust.
 *
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
