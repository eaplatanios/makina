package org.platanios.learn.combination.error;

import com.google.common.collect.BiMap;
import com.google.common.primitives.Ints;

import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ErrorRatesVector extends PowerSetVector {
    public ErrorRatesVector(int numberOfFunctions, int maximumOrder) {
        super(numberOfFunctions, 1, maximumOrder);
        initializeValues(0.25);
    }

    public ErrorRatesVector(int numberOfFunctions, int maximumOrder, double initialValue) {
        this(numberOfFunctions, maximumOrder);
        initializeValues(initialValue);
    }

    public ErrorRatesVector(int numberOfFunctions, int maximumOrder, boolean[] trueLabels, boolean[][] classifiersOutputs) {
        this(numberOfFunctions, maximumOrder);
        computeValues(trueLabels, classifiersOutputs);
    }

    private void initializeValues(double initialValue) {
        for (BiMap.Entry<List<Integer>, Integer> indexKeyPair : indexKeyMapping.entrySet()) {
            List<Integer> index = indexKeyPair.getKey();
            Integer key = indexKeyPair.getValue();
            if (index.size() == 1) {
                array[key] = initialValue;
            } else {
                array[key] = 1;
                for (Integer i : index) {
                    array[key] *= array[indexKeyMapping.get(Ints.asList(i))];
                }
            }
        }
    }

    private void computeValues(boolean[] trueLabels, boolean[][] classifiersOutputs) {
        for (int i = 0; i < trueLabels.length; i++) {
            for (BiMap.Entry<List<Integer>, Integer> entry : indexKeyMapping.entrySet()) {
                boolean equal = true;
                List<Integer> indexes = entry.getKey();
                for (int index : indexes.subList(1, indexes.size())) {
                    equal = equal && (classifiersOutputs[i][indexes.get(0)] == classifiersOutputs[i][index]);
                }
                if (equal && (classifiersOutputs[i][indexes.get(0)] != trueLabels[i])) {
                    array[entry.getValue()] += 1;
                }
            }
        }
        for (int i = 0; i < array.length; i++) {
            array[i] /= classifiersOutputs.length;
        }
    }
}
