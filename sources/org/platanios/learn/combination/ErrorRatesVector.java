package org.platanios.learn.combination;

import com.google.common.collect.BiMap;
import com.google.common.collect.ImmutableBiMap;
import com.google.common.primitives.Ints;
import org.platanios.math.combinatorics.CombinatoricsUtilities;

import java.util.*;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ErrorRatesVector {
    public double[] errorRates;
    public final BiMap<List<Integer>, Integer> indexKeyMapping;

    private int numberOfFunctions;
    private int maximumOrder;
    private int errorRatesLength;

    public ErrorRatesVector(int numberOfFunctions, int maximumOrder) {
        this.numberOfFunctions = numberOfFunctions;
        this.maximumOrder = maximumOrder;
        errorRatesLength = numberOfFunctions;
        for (int m = 2; m <= maximumOrder; m++) {
            errorRatesLength += CombinatoricsUtilities.binomialCoefficient(numberOfFunctions, m);
        }
        errorRates = new double[errorRatesLength];
        indexKeyMapping = createIndexKeyMappingBuilder().build();
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

    private ImmutableBiMap.Builder<List<Integer>, Integer> createIndexKeyMappingBuilder() {
        ImmutableBiMap.Builder<List<Integer>, Integer> indexKeyMappingBuilder = new ImmutableBiMap.Builder<List<Integer>, Integer>();

        for (int i = 0; i < numberOfFunctions; i++)
        {
            indexKeyMappingBuilder.put(Ints.asList(i), i);
        }

        int offset = numberOfFunctions;
        for (int m = 2; m <= maximumOrder; m++) {
            int[][] indexes = CombinatoricsUtilities.getCombinations(numberOfFunctions, m);
            for (int i = 0; i < indexes.length; i++) {
                indexKeyMappingBuilder.put(Ints.asList(indexes[i]), offset + i);
            }
            offset += CombinatoricsUtilities.binomialCoefficient(numberOfFunctions, m);
        }

        return indexKeyMappingBuilder;
    }

    private void initializeValues(double initialValue) {
        for (BiMap.Entry<List<Integer>, Integer> indexKeyPair : indexKeyMapping.entrySet()) {
            List<Integer> index = indexKeyPair.getKey();
            Integer key = indexKeyPair.getValue();
            if (index.size() == 1) {
                errorRates[key] = initialValue;
            } else {
                errorRates[key] = 1;
                for (Integer i : index) {
                    errorRates[key] *= errorRates[indexKeyMapping.get(Ints.asList(i))];
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
                    errorRates[entry.getValue()] += 1;
                }
            }
        }
        for (int i = 0; i < errorRates.length; i++) {
            errorRates[i] /= classifiersOutputs.length;
        }
    }

    public int getLength() {
        return errorRatesLength;
    }
}
