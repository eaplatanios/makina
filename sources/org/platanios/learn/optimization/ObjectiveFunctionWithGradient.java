package org.platanios.learn.optimization;

/**
 * An interface specifying the methods that all classes defined as possible objective functions for a numerical
 * optimization problem should implement.
 *
 * @author Emmanouil Antonios Platanios
 */
public interface ObjectiveFunctionWithGradient extends ObjectiveFunction {
    /**
     * Computes the first derivatives of the objective function and the constraints at a particular point.
     *
     * @param   optimizationVariables   The point in which to evaluate the derivatives.
     * @return                          The values of the first derivatives of the objective function, evaluated at the
     *                                  given point.
     */
    double[] computeGradient(double[] optimizationVariables);
}
