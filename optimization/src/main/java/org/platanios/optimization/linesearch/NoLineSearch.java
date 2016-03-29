package org.platanios.optimization.linesearch;

import org.platanios.math.matrix.Vector;

/**
 * Class for using a fixed step size with the line search optimization algorithms.
 *
 * @author Emmanouil Antonios Platanios
 */
public final class NoLineSearch implements LineSearch {
    /** The fixed step size value to use. */
    private final double stepSize;
    private final double tau;
    private final double kappa;

    /**
     * Constructs a fixed step size "line search" class for use with the line search optimization algorithms, using the
     * provided step size.
     *
     * @param   stepSize    The fixed step size to use.
     */
    public NoLineSearch(double stepSize) {
        this.stepSize = stepSize;
        this.tau = -1.0;
        this.kappa = -1.0;
    }

    /**
     * Constructs a scaled step size "line search" class for use with the line search optimization algorithms, using the
     * provided step size. The step size is computed as (tau + i + 1) ^ (-kappa), where i is the current iteration
     * number in the optimization algorithm.
     *
     * @param   tau     The tau parameter which much be >=0.
     * @param   kappa   The kappa parameter which must lie in (0.5,1].
     */
    public NoLineSearch(double tau, double kappa) {
        if (tau < 0)
            throw new IllegalArgumentException("The value of the tau parameter (i.e. parameters[0]) must be >= 0.");
        if (kappa <= 0.5 || kappa > 1)
            throw new IllegalArgumentException("The value of the kappa parameter (i.e. parameters[1]) " +
                                                       "must be in the interval (0.5,1].");
        this.stepSize = -1.0;
        this.tau = tau;
        this.kappa = kappa;
    }

    /** {@inheritDoc} */
    @Override
    public double computeStepSize(int iterationNumber,
                                  Vector point,
                                  Vector direction,
                                  Vector previousPoint,
                                  Vector previousDirection,
                                  double previousStepSize) {
        if (stepSize > -1.0)
            return stepSize;
        return Math.pow(tau + iterationNumber + 1, -kappa);
    }
}
