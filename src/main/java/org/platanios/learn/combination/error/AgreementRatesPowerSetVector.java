package org.platanios.learn.combination.error;

import com.google.common.collect.BiMap;

import java.util.List;

/**
 * A structure that holds the agreement rates of all possible subsets of a set of functions of cardinality greater than
 * or equal to 2.
 *
 * @author Emmanouil Antonios Platanios
 */
public class AgreementRatesPowerSetVector extends PowerSetVector {
    /**
     * Constructs a power set indexed vector, holding the agreement rates, stored as a one-dimensional array and
     * initializes the values of those agreement rates to the sample agreement rates computed from the observed outputs
     * of the functions for a set of data samples, where only subsets of functions of even cardinality are
     * considered. That is because the agreement rates of a subset of functions with odd cardinality can be written in
     * terms of the agreement rates of all subsets of functions of even cardinality less than the original subset
     * cardinality.
     *
     * @param   numberOfFunctions   The total number of functions considered.
     * @param   highestOrder        The highest cardinality of sets of functions to consider, out of the whole power
     *                              set.
     * @param   classifiersOutputs  The output labels of the functions for each data sample (the first dimension of the
     *                              array corresponds to data samples and the second dimension corresponds to the
     *                              functions).
     */
    public AgreementRatesPowerSetVector(int numberOfFunctions,
                                        int highestOrder,
                                        boolean[][] classifiersOutputs) {
        super(numberOfFunctions, 2, highestOrder, true);
        computeValues(classifiersOutputs);
    }

    /**
     * Constructs a power set indexed vector, holding the agreement rates, stored as a one-dimensional array and
     * initializes the values of those agreement rates to the sample agreement rates computed from the observed outputs
     * of the functions for a set of data samples, where an option is given on whether or not to only consider sets of
     * even cardinality within the power set.
     *
     * @param   numberOfFunctions           The total number of functions considered.
     * @param   highestOrder                The highest cardinality of sets of functions to consider, out of the whole
     *                                      power set.
     * @param   classifiersOutputs          The output labels of the functions for each data sample (the first dimension
     *                                      of the array corresponds to data samples and the second dimension
     *                                      corresponds to the functions).
     * @param onlyEvenCardinalitySubsets    Boolean value indicating whether or not to only consider sets of even
     *                                      cardinality, out of the whole power set.
     */
    public AgreementRatesPowerSetVector(int numberOfFunctions,
                                        int highestOrder,
                                        boolean[][] classifiersOutputs,
                                        boolean onlyEvenCardinalitySubsets) {
        super(numberOfFunctions, 2, highestOrder, onlyEvenCardinalitySubsets);
        computeValues(classifiersOutputs);
    }

    /**
     * Computes the sample agreement rates from the observed outputs of the functions for a set of data samples and sets
     * the values of the function agreement rates to those computed values.
     *
     * @param   classifiersOutputs  The output labels of the functions for each data sample (the first dimension of the
     *                              array corresponds to data samples and the second dimension corresponds to the
     *                              functions).
     */
    private void computeValues(boolean[][] classifiersOutputs) {
        for (boolean[] classifiersOutput : classifiersOutputs) {
            for (BiMap.Entry<List<Integer>, Integer> entry : indexKeyMapping.entrySet()) {
                boolean equal = true;
                List<Integer> indexes = entry.getKey();
                for (int index : indexes.subList(1, indexes.size())) {
                    equal = equal && (classifiersOutput[indexes.get(0)] == classifiersOutput[index]);
                }
                if (equal) {
                    array[entry.getValue()] += 1;
                }
            }
        }
        for (int i = 0; i < length; i++) {
            array[i] /= classifiersOutputs.length;
        }
    }
}
