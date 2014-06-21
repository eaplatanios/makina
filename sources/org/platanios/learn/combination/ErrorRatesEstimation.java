package org.platanios.learn.combination;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ErrorRatesEstimation {
    private DataStructure data;
    private KnitroOptimizationProblem optimizationProblem;

    public ErrorRatesEstimation(DataStructure data) {
        this.data = data;
        this.optimizationProblem = new KnitroOptimizationProblem(data.getNumberOfFunctions(),
                                                                 data.getMaximumOrder(),
                                                                 data.getErrorRates(),
                                                                 data.getAgreementRates());
    }

    public DataStructure solve() {
        data.setErrorRatesValues(optimizationProblem.solve());
        return data;
    }
}
