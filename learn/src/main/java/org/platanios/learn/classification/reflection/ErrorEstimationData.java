package org.platanios.learn.classification.reflection;

import com.google.common.primitives.Booleans;

import java.util.List;

/**
 * A data structure that holds all the data necessary to estimate error rates of functions from unlabeled data. In case
 * labeled data are available this structure also contains the sample error rates of those functions, computed using
 * that data, so that they can be used for evaluation of the error rates estimations method.
 *
 * @author Emmanouil Antonios Platanios
 */
public class ErrorEstimationData {
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
    /** The sample error rates, which are computed using labeled data. */
    private final ErrorRatesPowerSetVector sampleErrorRates;

    /** The error rates structure used for the error rates estimation (this structure contains the power set indexing
     * information used to index the error rates over all possible power sets of functions as a simple one-dimensional
     * array). */
    private ErrorRatesPowerSetVector errorRates;

    /**
     * A class used to build {@link ErrorEstimationData} objects.
     */
    public static class Builder {
        /** The number of function approximations/classifiers whose error rates we want to estimate. */
        private final int numberOfFunctions;
        /** The highest order of agreement rates considered in the error rates estimation and equivalently, the highest
         * order of error rates that are estimated. */
        private final int highestOrder;
        /** The error rates structure used for the error rates estimation (this structure contains the power set
         * indexing information used to index the error rates over all possible power sets of functions as a simple
         * one-dimensional array). */
        private final ErrorRatesPowerSetVector errorRates;
        /** The agreement rates structure used for the error rates estimation (this structure contains the sample
         * agreement rates that are used for defining the equality constraints of the optimization problem involved in
         * the error rates estimation). */
        private final AgreementRatesPowerSetVector agreementRates;

        /** The names of the functions (for example, if the functions correspond to some specific classifier, then they
         * could be named after its name, or after the input features that that classifier is using - if no such names
         * exist for the functions then a simple integer index is assigned to each function as its name). */
        private String[] functionNames = null;
        /** The sample error rates, which are computed using labeled data. */
        private ErrorRatesPowerSetVector sampleErrorRates = null;

        /**
         * Constructs a builder object for the {@link ErrorEstimationData} class using the provided information. Since
         * no function names are provided, a simple integer index is assigned to each function as its name.
         *
         * @param   numberOfFunctions   The number of function approximations/classifiers whose error rates we want to
         *                              estimate.
         * @param   highestOrder        The highest order of agreement rates considered in the error rates estimation
         *                              and equivalently, the highest order of error rates that are estimated.
         * @param   errorRates          The error rates structure used for the error rates estimation (this structure
         *                              also contains the estimated values after they are computed).
         * @param   agreementRates      The agreement rates structure used for the error rates estimation (this
         *                              structure contains the sample agreement rates that are used for defining the
         *                              equality constraints of the optimization problem involved in the error rates
         *                              estimation).
         */
        public Builder(int numberOfFunctions,
                       int highestOrder,
                       ErrorRatesPowerSetVector errorRates,
                       AgreementRatesPowerSetVector agreementRates) {
            this.numberOfFunctions = numberOfFunctions;
            this.highestOrder = highestOrder;
            this.errorRates = errorRates;
            this.agreementRates = agreementRates;
        }

        /**
         * Constructs a builder object for the {@link ErrorEstimationData} class using the provided information. Since
         * no function names are provided, a simple integer index is assigned to each function as its name.
         *
         * @param   functionOutputs                         A list of sets of outputs of all functions for unknown
         *                                                  inputs.
         * @param   highestOrder                            The highest cardinality of sets of functions to consider,
         *                                                  out of the whole power set, for the error rates and for the
         *                                                  agreement rates.
         * @param   onlyEvenCardinalitySubsetsAgreements    Boolean value indicating whether or not to only consider
         *                                                  sets of even cardinality, out of the whole power set, for
         *                                                  the agreement rates.
         */
        public Builder(List<boolean[]> functionOutputs,
                       int highestOrder,
                       boolean onlyEvenCardinalitySubsetsAgreements) {
            numberOfFunctions = functionOutputs.get(0).length;
            this.highestOrder = highestOrder;
            errorRates = new ErrorRatesPowerSetVector(numberOfFunctions, highestOrder, 0.25);
            boolean[][] functionOutputsArray = functionOutputs.toArray(new boolean[functionOutputs.size()][]);
            agreementRates = new AgreementRatesPowerSetVector(numberOfFunctions,
                                                              highestOrder,
                                                              functionOutputsArray,
                                                              onlyEvenCardinalitySubsetsAgreements);
        }

        /**
         * Constructs a builder object for the {@link ErrorEstimationData} class using the provided information. Since
         * no function names are provided, a simple integer index is assigned to each function as its name.
         *
         * @param   functionOutputs                         A list of sets of outputs of all functions for unknown
         *                                                  inputs.
         * @param   trueLabels                              A list of the true labels for all the inputs for which
         *                                                  function outputs are provided. This list is used for
         *                                                  computing the sample error rates.
         * @param   highestOrder                            The highest cardinality of sets of functions to consider,
         *                                                  out of the whole power set, for the error rates and for the
         *                                                  agreement rates.
         * @param   onlyEvenCardinalitySubsetsAgreements    Boolean value indicating whether or not to only consider
         *                                                  sets of even cardinality, out of the whole power set, for
         *                                                  the agreement rates.
         */
        public Builder(List<boolean[]> functionOutputs,
                       List<Boolean> trueLabels,
                       int highestOrder,
                       boolean onlyEvenCardinalitySubsetsAgreements) {
            numberOfFunctions = functionOutputs.get(0).length;
            this.highestOrder = highestOrder;
            errorRates = new ErrorRatesPowerSetVector(numberOfFunctions, highestOrder, 0.25);
            boolean[][] functionOutputsArray = functionOutputs.toArray(new boolean[functionOutputs.size()][]);
            agreementRates = new AgreementRatesPowerSetVector(numberOfFunctions,
                                                              highestOrder,
                                                              functionOutputsArray,
                                                              onlyEvenCardinalitySubsetsAgreements);
            sampleErrorRates = new ErrorRatesPowerSetVector(numberOfFunctions,
                                                            highestOrder,
                                                            Booleans.toArray(trueLabels),
                                                            functionOutputsArray);
        }

        /**
         * Sets the names of the functions.
         *
         * @param   functionNames   The names of the functions.
         * @return                  The current builder object after setting this property.
         */
        public Builder functionNames(String[] functionNames) {
            this.functionNames = functionNames;
            return this;
        }

        /**
         * Sets the sample error rates.
         *
         * @param   sampleErrorRates    The sample error rates, which were computed using labeled data.
         * @return                      The current builder object after setting this property.
         */
        public Builder sampleErrorRates(ErrorRatesPowerSetVector sampleErrorRates) {
            this.sampleErrorRates = sampleErrorRates;
            return this;
        }

        /**
         * Builds a {@link ErrorEstimationData} object using the current builder object.
         *
         * @return  A {@link ErrorEstimationData} object built using the current builder object.
         */
        public ErrorEstimationData build() {
            return new ErrorEstimationData(this);
        }
    }

    /**
     * Constructs an {@link ErrorEstimationData} object using the data contained in the provided {@link Builder} object.
     *
     * @param   builder The builder object containing the data to be used.
     */
    private ErrorEstimationData(Builder builder) {
        numberOfFunctions = builder.numberOfFunctions;
        highestOrder = builder.highestOrder;
        errorRates = builder.errorRates;
        agreementRates = builder.agreementRates;
        if (builder.functionNames == null) {
            builder.functionNames = new String[numberOfFunctions];
            for (int i = 1; i <= numberOfFunctions; i++)
                builder.functionNames[i - 1] = String.valueOf(i);
        }
        functionNames = builder.functionNames;
        sampleErrorRates = builder.sampleErrorRates;
    }

    /**
     * Gets the number of function approximations/classifiers whose error rates we want to estimate.
     *
     * @return  The number of function approximations/classifiers whose error rates we want to estimate.
     */
    public int getNumberOfFunctions() {
        return numberOfFunctions;
    }

    /**
     * Gets the highest order of agreement rates considered in the error rates estimation and equivalently, the highest
     * order of error rates that are estimated.
     *
     * @return  The highest order of agreement rates considered in the error rates estimation and equivalently, the
     *          highest order of error rates that are estimated.
     */
    public int getHighestOrder() {
        return highestOrder;
    }

    /**
     * Gets the sample agreement rates that are used for defining the equality constraints of the optimization problem
     * involved in the error rates estimation.
     *
     * @return  The sample agreement rates that are used for defining the equality constraints of the optimization
     *          problem involved in the error rates estimation.
     */
    public AgreementRatesPowerSetVector getAgreementRates() {
        return agreementRates;
    }

    /**
     * Gets the names of the functions.
     *
     * @return  The names of the functions.
     */
    public String[] getFunctionNames() {
        return functionNames;
    }

    /**
     * Gets the sample error rates, which were computed using a set of labeled data. This function returns null of we do
     * not have any sample error rates.
     *
     * @return  The sample error rates, which were computed using a set of labeled data.
     */
    public ErrorRatesPowerSetVector getSampleErrorRates() {
        return sampleErrorRates;
    }

    /**
     * Gets the error rates power set vector stored in this data object.
     *
     * @return  The error rates power set vector stored in this data object.
     */
    public ErrorRatesPowerSetVector getErrorRates() {
        return errorRates;
    }

    /**
     * Modifies the error rates power set vector stored in this data object.
     *
     * @param   errorRatesValues    The new error rates values to which the error rates structure array values are set.
     * @return                      The current {@link ErrorEstimationData} object, after settings the error rates
     *                              values.
     */
    public ErrorEstimationData setErrorRatesValues(double[] errorRatesValues) {
        errorRates.array = errorRatesValues;
        return this;
    }
}
