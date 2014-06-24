package org.platanios.learn.optimization;

/**
 * @author Emmanouil Antonios Platanios
 */
public interface LineSearchAlgorithm {
    /**
     * Computes the step size value using the implemented algorithm.
     *
     * @param   currentPoint
     * @param   direction
     * @return
     */
    public double computeStepSize(double[] currentPoint, double[] direction);
}
