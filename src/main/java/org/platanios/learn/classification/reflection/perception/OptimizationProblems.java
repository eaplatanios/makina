package org.platanios.learn.classification.reflection.perception;

/**
 * Factory class for generating objects representing optimization problems that are used to estimate error rates of
 * several approximations to a single function, by using only the agreement rates of those functions on an unlabeled set
 * of data.
 *
 * @author Emmanouil Antonios Platanios
 */
class OptimizationProblems {
    /**
     * Builds an optimization problem object that uses the IpOpt solver and the provided data.
     *
     * @param   numberOfFunctions       The number of function approximations/classifiers whose error rates we want to
     *                                  estimate.
     * @param   highestOrder            The highest order of agreement rates to consider and equivalently, the highest
     *                                  order of error rates to try and estimate.
     * @param   errorRates              The error rates structure used for this optimization (this structure contains
     *                                  the power set indexing information used to index the error rates over all
     *                                  possible power sets of functions as a simple one-dimensional array).
     * @param   agreementRates          The agreement rates structure used for this optimization (this structure
     *                                  contains the sample agreement rates that are used for defining the equality
     *                                  constraints of the problem).
     * @param   objectiveFunctionType   The type of objective function to minimize (e.g. minimize dependency, scaled
     *                                  dependency, etc.).
     * @return                          An optimization problem object, ready to be solved.
     */
    public static OptimizationProblem build(int numberOfFunctions,
                                            int highestOrder,
                                            ErrorRatesPowerSetVector errorRates,
                                            AgreementRatesPowerSetVector agreementRates,
                                            ObjectiveFunctionType objectiveFunctionType) {
        return build(numberOfFunctions,
                     highestOrder,
                     errorRates,
                     agreementRates,
                     objectiveFunctionType,
                     OptimizationSolverType.IP_OPT);
    }

    /**
     * Builds an optimization problem object that uses the provided solver and the provided data.
     *
     * @param   numberOfFunctions       The number of function approximations/classifiers whose error rates we want to
     *                                  estimate.
     * @param   highestOrder            The highest order of agreement rates to consider and equivalently, the highest
     *                                  order of error rates to try and estimate.
     * @param   errorRates              The error rates structure used for this optimization (this structure contains
     *                                  the power set indexing information used to index the error rates over all
     *                                  possible power sets of functions as a simple one-dimensional array).
     * @param   agreementRates          The agreement rates structure used for this optimization (this structure
     *                                  contains the sample agreement rates that are used for defining the equality
     *                                  constraints of the problem).
     * @param   objectiveFunctionType   The type of objective function to minimize (e.g. minimize dependency, scaled
     *                                  dependency, etc.).
     * @param   optimizationSolverType  The optimization solver type to use for solving this optimization problem.
     * @return                          An optimization problem object, ready to be solved.
     */
    public static OptimizationProblem build(int numberOfFunctions,
                                            int highestOrder,
                                            ErrorRatesPowerSetVector errorRates,
                                            AgreementRatesPowerSetVector agreementRates,
                                            ObjectiveFunctionType objectiveFunctionType,
                                            OptimizationSolverType optimizationSolverType) {
        return optimizationSolverType.build(numberOfFunctions,
                                            highestOrder,
                                            errorRates,
                                            agreementRates,
                                            objectiveFunctionType);
    }
}