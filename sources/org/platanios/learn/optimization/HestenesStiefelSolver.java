package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.RealVector;
import org.platanios.learn.optimization.function.Function;

/**
 * This solver performs very similarly to the Polak-Ribiere solver.
 *
 * @author Emmanouil Antonios Platanios
 */
public class HestenesStiefelSolver extends AbstractNonlinearConjugateGradientSolver {
    public HestenesStiefelSolver(Function objective,
                                 double[] initialPoint) {
        super(objective, initialPoint);
    }

    public double computeBeta() {
        RealVector gradientsDifference = currentGradient.subtract(previousGradient);
        return currentGradient.dotProduct(gradientsDifference) / gradientsDifference.dotProduct(previousDirection);
    }
}
