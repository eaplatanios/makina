package org.platanios.learn.classification.reflection.perception;

/**
 * A data structure that holds all the data necessary to estimate error rates of functions from unlabeled data. In case
 * labeled data are available this structure also contains the sample error rates of those functions, computed using
 * that data, so that they can be used for evaluation of the error rates estimations method.
 *
 * @author Emmanouil Antonios Platanios
 */
public class EstimationData {
    /** The number of function approximations/classifiers whose error rates we want to estimate. */
    private final int numberOfFunctions;
    /** The highest order of agreement rates considered in the error rates estimation and equivalently, the highest
     * order of error rates that are estimated. */
    private final int highestOrder;
    /** The agreement rates structure used for the error rates estimation (this structure contains the sample agreement
     * rates that are used for defining the equality constraints of the optimization problem involved in the error rates
     * estimation). */
    private final AgreementRatesPowerSetVector agreementRates;
    /** The names of the functions (for example, if the functions correspond to some specific classifier, then they
     * could be named after its name, or after the input features that that classifier is using - if no such names exist
     * for the functions then a simple integer index is assigned to each function as its name). */
    private final String[] functionNames;
    /** The sample error rates which were computed using a set of labeled data. */
    private final ErrorRatesPowerSetVector sampleErrorRates;

    /** The error rates structure used for the error rates estimation (this structure contains the power set indexing
     * information used to index the error rates over all possible power sets of functions as a simple one-dimensional
     * array). */
    private ErrorRatesPowerSetVector errorRates;

    /**
     * Initializes this structure with the provided information. Since no function names are provided, a simple integer
     * index is assigned to each function as its name.
     *
     * @param   numberOfFunctions   The number of function approximations/classifiers whose error rates we want to
     *                              estimate.
     * @param   highestOrder        The highest order of agreement rates considered in the error rates estimation and
     *                              equivalently, the highest order of error rates that are estimated.
     * @param   errorRates          The error rates structure used for the error rates estimation (this structure also
     *                              contains the estimated values after they are computed).
     * @param   agreementRates      The agreement rates structure used for the error rates estimation (this structure
     *                              contains the sample agreement rates that are used for defining the equality
     *                              constraints of the optimization problem involved in the error rates estimation).
     */
    public EstimationData(int numberOfFunctions,
                          int highestOrder,
                          ErrorRatesPowerSetVector errorRates,
                          AgreementRatesPowerSetVector agreementRates) {
        this(numberOfFunctions, highestOrder, errorRates, agreementRates, null, null);
    }

    /**
     * Initializes this structure with the provided information.
     *
     * @param   numberOfFunctions   The number of function approximations/classifiers whose error rates we want to
     *                              estimate.
     * @param   highestOrder        The highest order of agreement rates considered in the error rates estimation and
     *                              equivalently, the highest order of error rates that are estimated.
     * @param   errorRates          The error rates structure used for the error rates estimation (this structure also
     *                              contains the estimated values after they are computed).
     * @param   agreementRates      The agreement rates structure used for the error rates estimation (this structure
     *                              contains the sample agreement rates that are used for defining the equality
     *                              constraints of the optimization problem involved in the error rates estimation).
     * @param   functionNames       The names of the functions.
     */
    public EstimationData(int numberOfFunctions,
                          int highestOrder,
                          ErrorRatesPowerSetVector errorRates,
                          AgreementRatesPowerSetVector agreementRates,
                          String[] functionNames) {
        this(numberOfFunctions, highestOrder, errorRates, agreementRates, functionNames, null);
    }

    /**
     * Initializes this structure with the provided information.
     *
     * @param   numberOfFunctions   The number of function approximations/classifiers whose error rates we want to
     *                              estimate.
     * @param   highestOrder        The highest order of agreement rates considered in the error rates estimation and
     *                              equivalently, the highest order of error rates that are estimated.
     * @param   errorRates          The error rates structure used for the error rates estimation (this structure also
     *                              contains the estimated values after they are computed).
     * @param   agreementRates      The agreement rates structure used for the error rates estimation (this structure
     *                              contains the sample agreement rates that are used for defining the equality
     *                              constraints of the optimization problem involved in the error rates estimation).
     * @param   functionNames       The names of the functions.
     * @param   sampleErrorRates    The sample error rates which were computed using a set of labeled data.
     */
    public EstimationData(int numberOfFunctions,
                          int highestOrder,
                          ErrorRatesPowerSetVector errorRates,
                          AgreementRatesPowerSetVector agreementRates,
                          String[] functionNames,
                          ErrorRatesPowerSetVector sampleErrorRates) {
        this.numberOfFunctions = numberOfFunctions;
        this.highestOrder = highestOrder;
        this.errorRates = errorRates;
        this.agreementRates = agreementRates;
        if (functionNames == null) {
            functionNames = new String[numberOfFunctions];
            for (int i = 1; i <= numberOfFunctions; i++) {
                functionNames[i] = String.valueOf(i);
            }
        }
        this.functionNames = functionNames;
        this.sampleErrorRates = sampleErrorRates;
    }

    /**
     * @return  The number of function approximations/classifiers whose error rates we want to estimate.
     */
    public int getNumberOfFunctions() {
        return numberOfFunctions;
    }

    /**
     * @return  The highest order of agreement rates considered in the error rates estimation and equivalently, the
     * highest order of error rates that are estimated.
     */
    public int getHighestOrder() {
        return highestOrder;
    }

    /**
     * @return  The agreement rates structure used for the error rates estimation (this structure contains the sample
     * agreement rates that are used for defining the equality constraints of the optimization problem involved in the
     * error rates estimation).
     */
    public AgreementRatesPowerSetVector getAgreementRates() {
        return agreementRates;
    }

    /**
     * @return  The names of the functions.
     */
    public String[] getFunctionNames() {
        return functionNames;
    }

    /**
     * @return  The sample error rates which were computed using a set of labeled data.
     */
    public ErrorRatesPowerSetVector getSampleErrorRates() {
        return sampleErrorRates;
    }

    /**
     * @return  The error rates structure used for the error rates estimation (this structure also contains the
     *          estimated values after they are computed).
     */
    public ErrorRatesPowerSetVector getErrorRates() {
        return errorRates;
    }

    /**
     * @param   errorRatesValues    The new error rates values to which the error rates structure array values are set.
     * @return                      The current {@link EstimationData} object, after settings the error rates values.
     */
    public EstimationData setErrorRatesValues(double[] errorRatesValues) {
        errorRates.array = errorRatesValues;
        return this;
    }
}
