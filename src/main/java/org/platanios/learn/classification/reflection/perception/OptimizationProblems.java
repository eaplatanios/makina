package org.platanios.learn.classification.reflection.perception;

/**
 * @author Emmanouil Antonios Platanios
 */
class OptimizationProblems {
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
