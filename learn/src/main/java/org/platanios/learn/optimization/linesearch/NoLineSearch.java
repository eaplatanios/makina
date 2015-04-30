package org.platanios.learn.optimization.linesearch;

import org.platanios.learn.math.matrix.Vector;

/**
 * Class implementing an exact line search algorithm. It currently requires the objective function to be a quadratic
 * function.
 *
 * @author Emmanouil Antonios Platanios
 */
public final class NoLineSearch implements LineSearch {
    /** The objective function instance. */
    private final double stepSize;

    /**
     * Constructs an exact line search solver for the given objective function instance.
     *
     * @param   stepSize   The objective function instance.
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
