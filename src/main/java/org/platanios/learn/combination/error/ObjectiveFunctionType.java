package org.platanios.learn.combination.error;

/**
 * An enumeration of all possible objective function types that are currently supported by our implementation.
 *
 * @author Emmanouil Antonios Platanios
 */
public enum ObjectiveFunctionType {
    /** An objective function that quantifies the dependencies between the error rates of the  different functions. The
     * actual implementation of this objective function types is
     * {@link DependencyObjectiveFunction}. */
    DEPENDENCY,
    SCALED_DEPENDENCY, // TODO: Implement this objective function and add all relevant documentation.
    DEPENDENCY_ACROSS_DOMAINS, // TODO: Implement this objective function and add all relevant documentation.
    SCALED_DEPENDENCY_ACROSS_DOMAINS, // TODO: Implement this objective function and add all relevant documentation.
    L2_NORM, // TODO: Implement this objective function and add all relevant documentation.
    DEPENDENCY_AND_L2_NORM, // TODO: Implement this objective function and add all relevant documentation.
    SCALED_DEPENDENCY_AND_L2_NORM, // TODO: Implement this objective function and add all relevant documentation.
    DEPENDENCY_ACROSS_DOMAINS_AND_L2_NORM, // TODO: Implement this objective function and add all relevant documentation.
    SCALED_DEPENDENCY_ACROSS_DOMAINS_AND_L2_NORM // TODO: Implement this objective function and add all relevant documentation.
}
