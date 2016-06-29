package makina.learn.classification.reflection;

/**
 * An enumeration containing all optimization solver types currently supported.
 *
 * @author Emmanouil Antonios Platanios
 */
public enum AgreementIntegratorInternalSolver {
    /** Uses the IpOpt solver. */
    IP_OPT {
        /** {@inheritDoc} */
        @Override
        public AgreementIntegratorOptimization buildOptimizationProblem(int numberOfFunctions,
                                                                        int highestOrder,
                                                                        ErrorRatesPowerSetVector errorRates,
                                                                        AgreementRatesPowerSetVector agreementRates,
                                                                        AgreementIntegratorObjective objective) {
            return new AgreementIntegratorOptimizationIpOpt(numberOfFunctions,
                                                            highestOrder,
                                                            errorRates,
                                                            agreementRates,
                                                            objective);
        }
    },
    /** Uses the KNITRO solver. */
    KNITRO {
        /** {@inheritDoc} */
        @Override
        public AgreementIntegratorOptimization buildOptimizationProblem(int numberOfFunctions,
                                                                        int highestOrder,
                                                                        ErrorRatesPowerSetVector errorRates,
                                                                        AgreementRatesPowerSetVector agreementRates,
                                                                        AgreementIntegratorObjective objective) {
            return new AgreementIntegratorOptimizationKNITRO(numberOfFunctions,
                                                             highestOrder,
                                                             errorRates,
                                                             agreementRates,
                                                             objective);
        }
    };

    /**
     * Builds an optimization problem object that uses the selected solver and the provided data.
     *
     * @param   numberOfFunctions   The number of function approximations/classifiers whose error rates we want to
     *                              estimate.
     * @param   highestOrder        The highest order of agreement rates to consider and equivalently, the highest
     *                              order of error rates to try and estimate.
     * @param   errorRates          The error rates structure used for this optimization (this structure contains the
     *                              power set indexing information used to index the error rates over all possible power
     *                              sets of functions as a simple one-dimensional array).
     * @param   agreementRates      The agreement rates structure used for this optimization (this structure contains
     *                              the sample agreement rates that are used for defining the equality constraints of
     *                              the problem).
     * @param   objective           The type of objective function to minimize (e.g. minimize dependency, scaled
     *                              dependency, etc.).
     * @return                      An optimization problem object, ready to be solved.
     */
    public abstract AgreementIntegratorOptimization buildOptimizationProblem(int numberOfFunctions,
                                                                             int highestOrder,
                                                                             ErrorRatesPowerSetVector errorRates,
                                                                             AgreementRatesPowerSetVector agreementRates,
                                                                             AgreementIntegratorObjective objective);
}