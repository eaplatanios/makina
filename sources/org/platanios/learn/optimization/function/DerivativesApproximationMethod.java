package org.platanios.learn.optimization.function;

/**
 * @author Emmanouil Antonios Platanios
 */
public enum DerivativesApproximationMethod {
    FORWARD_DIFFERENCE,
    /** Much more accurate than the forward-difference method (O(&epsilon;^2) estimation error instead of O(&epsilon;)). */
    CENTRAL_DIFFERENCE
}
