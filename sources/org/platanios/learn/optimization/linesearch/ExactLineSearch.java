package org.platanios.learn.optimization.linesearch;

import org.apache.commons.math3.linear.RealVector;
import org.platanios.learn.optimization.function.QuadraticFunction;

/**
 * Class implementing an exact line search algorithm. It currently requires the objective function to be a quadratic
 * function.
 *
 * @author Emmanouil Antonios Platanios
 */
public class ExactLineSearch implements LineSearch {
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

    /**
     * {@inheritDoc}
     */
    @Override
    public double computeStepSize(RealVector point,
                                  RealVector direction,
                                  RealVector previousPoint,
                                  RealVector previousDirection,
                                  double previousStepSize) {
        return -objective.getGradient(point).dotProduct(direction)
                / objective.getA().preMultiply(direction).dotProduct(direction);
    }
}
