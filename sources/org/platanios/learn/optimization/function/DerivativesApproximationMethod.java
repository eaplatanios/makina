package org.platanios.learn.optimization.function;

/**
 * TODO: Add support for other methods that involve interpolation or automatic differentiation.
 *
 * @author Emmanouil Antonios Platanios
 */
public enum DerivativesApproximationMethod {
    FORWARD_DIFFERENCE,
    /** Much more accurate than the forward-difference method (O(&epsilon;^2) estimation error instead of O(&epsilon;)). */
    CENTRAL_DIFFERENCE
}
