package org.platanios.learn.classification.reflection.perception;

import com.google.common.collect.BiMap;
import com.google.common.primitives.Ints;

import java.util.List;

/**
 * A structure that holds the individual error rates of a set of functions, as well as the joint error rates of all
 * possible subsets of those functions.
 *
 * @author Emmanouil Antonios Platanios
 */
public class ErrorRatesPowerSetVector extends PowerSetVector {
    /**
     * Constructs a power set indexed vector, holding the error rates, stored as a one-dimensional array and initializes
     * the values of those error rates to be 0.25 for the individual function error rates and a computed value for the
     * joint error rates assuming that each function's error rate is independent of the rest (this independence
     * assumption is made only for the initialization of those error rates values and has nothing to do with the actual
     * error rates estimation problem).
     *
     * @param   numberOfFunctions   The total number of functions considered.
     * @param   highestOrder        The highest cardinality of sets of functions to consider, out of the whole power
     *                              set.
     */
    public ErrorRatesPowerSetVector(int numberOfFunctions,
                                    int highestOrder) {
        super(numberOfFunctions, 1, highestOrder);
        initializeValues(0.25);
    }

    /**
     * Constructs a power set indexed vector, holding the error rates, stored as a one-dimensional array and initializes
     * the values of those error rates to be {@code initialValue} for the individual function error rates and a computed
     * value for the joint error rates assuming that each function's error rate is independent of the rest (this
     * independence assumption is made only for the initialization of those error rates values and has nothing to do
     * with the actual error rates estimation problem).
     *
     * @param   numberOfFunctions   The total number of functions considered.
     * @param   highestOrder        The highest cardinality of sets of functions to consider, out of the whole power
     *                              set.
     * @param   initialValue        The initial value for the individual function error rates.
     */
    public ErrorRatesPowerSetVector(int numberOfFunctions,
                                    int highestOrder,
                                    double initialValue) {
        this(numberOfFunctions, highestOrder);
        initializeValues(initialValue);
    }

    /**
     * Constructs a power set indexed vector, holding the error rates, stored as a one-dimensional array and initializes
     * the values of those error rates to the sample error rates computed from the available labeled data samples and
     * the outputs of the functions for each data sample.
     *
     * @param   numberOfFunctions   The total number of functions considered.
     * @param   highestOrder        The highest cardinality of sets of functions to consider, out of the whole power
     *                              set.
     * @param   trueLabels          The true label for each data sample.
     * @param   classifiersOutputs  The output labels of the functions for each data sample (the first dimension of the
     *                              array corresponds to data samples and the second dimension corresponds to the
     *                              functions).
     */
    public ErrorRatesPowerSetVector(int numberOfFunctions,
                                    int highestOrder,
                                    boolean[] trueLabels,
                                    boolean[][] classifiersOutputs) {
        this(numberOfFunctions, highestOrder);
        computeValues(trueLabels, classifiersOutputs);
    }

    /**
     * Initializes the values of the function error rates to be {@code initialValue} for the individual function error
     * rates and a computed value for the joint error rates assuming that each function's error rate is independent of
     * the rest (this independence assumption is made only for the initialization of those error rates values and has
     * nothing to do with the actual error rates estimation problem).
     *
     * @param   initialValue    The initial value for the individual function error rates.
     */
    private void initializeValues(double initialValue) {
        for (BiMap.Entry<List<Integer>, Integer> indexKeyPair : indexKeyMapping.entrySet()) {
            List<Integer> index = indexKeyPair.getKey();
            Integer key = indexKeyPair.getValue();
            if (index.size() == 1) {
                array[key] = initialValue;
            } else {
                array[key] = 1;
                for (Integer i : index)
                    array[key] *= array[indexKeyMapping.get(Ints.asList(i))];
            }
        }
    }

    /**
     * Computes the sample error rates from the available labeled data samples and the outputs of the functions for each
     * data sample and sets the values of the function error rates to those computed values.
     *
     * @param   trueLabels          The true label for each data sample.
     * @param   classifiersOutputs  The output labels of the functions for each data sample (the first dimension of the
     *                              array corresponds to data samples and the second dimension corresponds to the
     *                              functions).
     */
    private void computeValues(boolean[] trueLabels, boolean[][] classifiersOutputs) {
        for (int i = 0; i < trueLabels.length; i++) {
            for (BiMap.Entry<List<Integer>, Integer> entry : indexKeyMapping.entrySet()) {
                boolean equal = true;
                List<Integer> indexes = entry.getKey();
                for (int index : indexes.subList(1, indexes.size()))
                    equal = equal && (classifiersOutputs[i][indexes.get(0)] == classifiersOutputs[i][index]);
                if (equal && (classifiersOutputs[i][indexes.get(0)] != trueLabels[i]))
                    array[entry.getValue()] += 1;
            }
        }
        for (int i = 0; i < array.length; i++)
            array[i] /= classifiersOutputs.length;
    }
}
