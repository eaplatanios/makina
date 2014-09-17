package org.platanios.learn.optimization.linesearch;

import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.optimization.function.QuadraticFunction;

/**
 * Class implementing an exact line search algorithm. It currently requires the objective function to be a quadratic
 * function.
 *
 * @author Emmanouil Antonios Platanios
 */
public final class ExactLineSearch implements LineSearch {
    /** The objective function instance. */
    private final QuadraticFunction objective;

    /**
     * Constructs an exact line search solver for the given objective function instance.
     *
     * @param   objective   The objective function instance.
     */
    public ExactLineSearch(QuadraticFunction objective) {
        this.objective = objective;
    }

    /** {@inheritDoc} */
    @Override
    public double computeStepSize(Vector point,
                                  Vector direction,
                                  Vector previousPoint,
                                  Vector previousDirection,
                                  double previousStepSize) {
        return -objective.getGradient(point).innerProduct(direction)
                / direction.multiply(objective.getA()).innerProduct(direction);
    }
}
