package org.platanios.learn.combination;

/**
 * @author Emmanouil Antonios Platanios
 */
public class DataStructure {
    private final int numberOfFunctions;
    private final int maximumOrder;
    private final ErrorRatesVector sampleErrorRates;
    private final AgreementRatesVector agreementRates;
    private final String[] classifierNames;

    private ErrorRatesVector errorRates;

    public DataStructure(int numberOfFunctions,
                         int maximumOrder,
                         ErrorRatesVector errorRates,
                         ErrorRatesVector sampleErrorRates,
                         AgreementRatesVector agreementRates,
                         String[] classifierNames) {
        this.numberOfFunctions = numberOfFunctions;
        this.maximumOrder = maximumOrder;
        this.errorRates = errorRates;
        this.sampleErrorRates = sampleErrorRates;
        this.agreementRates = agreementRates;
        this.classifierNames = classifierNames;
    }

    public int getNumberOfFunctions() {
        return numberOfFunctions;
    }

    public int getMaximumOrder() {
        return maximumOrder;
    }

    public ErrorRatesVector getSampleErrorRates() {
        return sampleErrorRates;
    }

    public AgreementRatesVector getAgreementRates() {
        return agreementRates;
    }

    public String[] getClassifierNames() {
        return classifierNames;
    }

    public ErrorRatesVector getErrorRates() {
        return errorRates;
    }

    public void setErrorRatesValues(double[] errorRatesValues) {
        errorRates.errorRates = errorRatesValues;
    }
}
