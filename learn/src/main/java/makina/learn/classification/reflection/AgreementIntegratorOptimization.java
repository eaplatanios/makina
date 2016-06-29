package makina.learn.classification.reflection;

/**
 * An interface specifying the methods that all classes defined as possible objective problems for the numerical
 * optimization problem involved in the error rates estimation process, should implement. The implementations of this
 * interface typically use a different underlying nonlinear optimization solver.
 *
 * @author Emmanouil Antonios Platanios
 */
interface AgreementIntegratorOptimization {
    /**
     * Solves the numerical optimization problem and returns the error rates estimates in {@code double[]} format.
     *
     * @return  The error rates estimates in a {@code double[]} format.
     */
    double[] solve();
}
