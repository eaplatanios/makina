package org.platanios.learn.optimization.linesearch;

/**
 * @author Emmanouil Antonios Platanios
 */
public enum StepSizeInitializationMethod {
    CONSTANT,
    UNIT,
    CONSERVE_FIRST_ORDER_CHANGE,
    QUADRATIC_INTERPOLATION,
    MODIFIED_QUADRATIC_INTERPOLATION
}
