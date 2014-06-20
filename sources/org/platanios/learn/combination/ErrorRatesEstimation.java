package org.platanios.learn.combination;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ErrorRatesEstimation {
    private ErrorRatesVector errorRates;
    private KNitroOptimizationProblem optimizationProblem;

    public ErrorRatesEstimation(final AgreementRatesVector agreementRates, final int numberOfFunctions, int maximumOrder) {
        this.errorRates = new ErrorRatesVector(numberOfFunctions, maximumOrder);
        this.optimizationProblem = new KNitroOptimizationProblem(numberOfFunctions, maximumOrder, errorRates, agreementRates);
    }

    public ErrorRatesVector solve() {
        errorRates.errorRates = optimizationProblem.solve();
        return errorRates;
    }
}
