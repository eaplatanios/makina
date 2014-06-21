package org.platanios.learn.combination.error;

import com.google.common.collect.BiMap;

import java.util.*;

/**
 * @author Emmanouil Antonios Platanios
 */
public class AgreementRatesVector extends PowerSetVector {
    public AgreementRatesVector(int numberOfFunctions, int maximumOrder, boolean[][] observations) {
        super(numberOfFunctions, 2, maximumOrder, true);
        computeValues(observations);
    }

    public AgreementRatesVector(int numberOfFunctions, int maximumOrder, boolean[][] observations, boolean onlyEvenCardinalitySubsets) {
        super(numberOfFunctions, 2, maximumOrder, onlyEvenCardinalitySubsets);
        computeValues(observations);
    }

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
