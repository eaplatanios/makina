package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * An interface specifying the methods that all classes defined as possible objective functions for a numerical
 * optimization problem should implement.
 *
 * @author Emmanouil Antonios Platanios
 */
public interface ObjectiveFunctionWithGradientAndHessian extends ObjectiveFunctionWithGradient {
    /**
     * Computes the Hessian of the objective function at a particular point.
     *
     * @param   optimizationVariables   The point in which to evaluate the Hessian.
     * @return                          The value of the Hessian matrix of the objective function, evaluated at the
     *                                  given point.
     */
    RealMatrix computeHessian(RealVector optimizationVariables);
}
