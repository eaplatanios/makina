package org.platanios.learn.classification.reflection.perception;

/**
 * @author Emmanouil Antonios Platanios
 */
public enum OptimizationSolverType {
    IP_OPT {
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
    KNITRO {
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

    public abstract OptimizationProblem build(int numberOfFunctions,
                                              int highestOrder,
                                              ErrorRatesPowerSetVector errorRates,
                                              AgreementRatesPowerSetVector agreementRates,
                                              ObjectiveFunctionType objectiveFunctionType);
}