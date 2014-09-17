package org.platanios.learn.combination.error;

/**
 * An interface specifying the methods that all classes defined as possible objective functions for the numerical
 * optimization problem involved in the error rates estimation process should implement.
 *
 * @author Emmanouil Antonios Platanios
 */
public interface ObjectiveFunction {
    /**
     * Computes the objective value and the constraints values at a particular point.
     *
     * @param   optimizationVariables   The point in which to evaluate the objective function and the constraints.
     * @param   optimizationObjective   The objective value to set for the given point.
     */
    void computeObjective(double[] optimizationVariables,
                          double[] optimizationObjective);

    /**
     * Computes the first derivatives of the objective function and the constraints at a particular point.
     *
     * @param   optimizationVariables           The point in which to evaluate the derivatives.
     * @param   optimizationObjectiveGradients  The objective function gradients vector to modify.
     */
    void computeGradient(double[] optimizationVariables,
                         double[] optimizationObjectiveGradients);

    /**
     * Computes the Hessian of the Lagrangian at a particular point. The constraints in this case are linear and so they
     * do not contribute to the Hessian value.
     *
     * @param   optimizationVariables   The point in which to evaluate the Hessian.
     * @param   optimizationHessian     The Hessian (in sparse/vector form) to modify.
     */
    void computeHessian(double[] optimizationVariables,
                        double[] optimizationHessian);
}
