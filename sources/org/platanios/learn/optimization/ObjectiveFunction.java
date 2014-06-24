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

    /**
     * Computes the first derivatives of the objective function and the constraints at a particular point.
     *
     * @param   optimizationVariables   The point in which to evaluate the derivatives.
     * @return                          The values of the first derivatives of the objective function, evaluated at the
     *                                  given point.
     */
    double[] computeGradient(double[] optimizationVariables);

    /**
     * Computes the Hessian of the objective function at a particular point.
     *
     * @param   optimizationVariables   The point in which to evaluate the Hessian.
     * @return                          The value of the Hessian matrix of the objective function, evaluated at the
     *                                  given point.
     */
    double[][] computeHessian(double[] optimizationVariables);
}
