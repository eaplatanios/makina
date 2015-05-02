package org.platanios.learn.optimization.linesearch;

import org.platanios.learn.math.matrix.Vector;

/**
 * Class for using a fixed step size with the line search optimization algorithms.
 *
 * @author Emmanouil Antonios Platanios
 */
public final class NoLineSearch implements LineSearch {
    /** The fixed step size value to use. */
    private final double stepSize;

    /**
     * Constructs a fixed step size "line search" class for use with the line search optimization algorithms, using the
     * provided step size.
     *
     * @param   stepSize    The fixed step size to use.
     */
    public NoLineSearch(double stepSize) {
        this.stepSize = stepSize;
    }

    /** {@inheritDoc} */
    @Override
    public double computeStepSize(Vector point,
                                  Vector direction,
                                  Vector previousPoint,
                                  Vector previousDirection,
                                  double previousStepSize) {
        return stepSize;
    }
}
