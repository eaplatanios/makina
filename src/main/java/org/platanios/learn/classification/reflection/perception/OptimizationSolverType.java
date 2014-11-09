package org.platanios.learn.classification.reflection.perception;

/**
 * An enumeration containing all optimization solver types currently supported.
 *
 * @author Emmanouil Antonios Platanios
 */
public enum OptimizationSolverType {
    /** Uses the IpOpt solver. */
    IP_OPT {
        /** {@inheritDoc} */
        @Override
        public OptimizationProblem build(int numberOfFunctions,
                                         int highestOrder,
                                         ErrorRatesPowerSetVector errorRates,
                                         AgreementRatesPowerSetVector agreementRates,
                                         ObjectiveFunctionType objectiveFunctionType) {
            return new OptimizationProblemIpOpt(numberOfFunctions,
                                                highestOrder,
                                                errorRates,
                                                agreementRates,
                                                objectiveFunctionType);
        }
    },
    /** Uses the KNITRO solver. */
    KNITRO {
        /** {@inheritDoc} */
        @Override
        public OptimizationProblem build(int numberOfFunctions,
                                         int highestOrder,
                                         ErrorRatesPowerSetVector errorRates,
                                         AgreementRatesPowerSetVector agreementRates,
                                         ObjectiveFunctionType objectiveFunctionType) {
            return new OptimizationProblemKNITRO(numberOfFunctions,
                                                 highestOrder,
                                                 errorRates,
                                                 agreementRates,
                                                 objectiveFunctionType);
        }
    };

    /**
     * Builds an optimization problem object that uses the selected solver and the provided data.
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
    public abstract OptimizationProblem build(int numberOfFunctions,
                                              int highestOrder,
                                              ErrorRatesPowerSetVector errorRates,
                                              AgreementRatesPowerSetVector agreementRates,
                                              ObjectiveFunctionType objectiveFunctionType);
}