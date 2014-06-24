package org.platanios.learn.optimization;

/**
 * An interface specifying the methods that all classes defined as possible objective functions for a numerical
 * optimization problem should implement.
 *
 * @author Emmanouil Antonios Platanios
 */
public interface ObjectiveFunction {
    /**
     * Computes the objective function value and the constraints values at a particular point.
     *
     * @param   optimizationVariables   The point in which to evaluate the objective function and the constraints.
     * @return                          The value of the objective function, evaluated at the given point.
     */
    double computeValue(double[] optimizationVariables);
}
