package org.platanios.learn.combination;

import org.platanios.math.combinatorics.CombinatoricsUtilities;

import java.util.*;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ErrorRatesVector {
    public double[] errorRates;

    private int numberOfFunctions;
    private int maximumOrder;
    private int errorRatesLength;
    private Map<ArrayList<Integer>, Integer> indexToKeyMapping;
    private Map<Integer, ArrayList<Integer>> keyToIndexMapping;

    public ErrorRatesVector(int numberOfFunctions, int maximumOrder) {
        this.numberOfFunctions = numberOfFunctions;
        this.maximumOrder = maximumOrder;
        errorRatesLength = numberOfFunctions;
        for (int m = 2; m <= maximumOrder; m++) {
            errorRatesLength += CombinatoricsUtilities.binomialCoefficient(numberOfFunctions, m);
        }
        errorRates = new double[errorRatesLength];

        createMappings();
        initializeValues(0.25);
    }

    public ErrorRatesVector(int numberOfFunctions, int maximumOrder, double initialValue) {
        this(numberOfFunctions, maximumOrder);
        initializeValues(initialValue);
    }

    private void createMappings() {
        indexToKeyMapping = new LinkedHashMap<ArrayList<Integer>, Integer>();
        keyToIndexMapping = new LinkedHashMap<Integer, ArrayList<Integer>>();

        for (int i = 0; i < numberOfFunctions; i++)
        {
            indexToKeyMapping.put(new ArrayList<Integer>(Arrays.asList(i)), i);
            keyToIndexMapping.put(i, new ArrayList<Integer>(Arrays.asList(i)));
        }

        int offset = numberOfFunctions;

        for (int m = 2; m <= maximumOrder; m++) {
            List<ArrayList<Integer>> indexes = CombinatoricsUtilities.getCombinations(numberOfFunctions, m);
            for (int i = 0; i < indexes.size(); i++) {
                indexToKeyMapping.put(indexes.get(i), offset + i);
                keyToIndexMapping.put(offset + i, indexes.get(i));
            }
            offset += CombinatoricsUtilities.binomialCoefficient(numberOfFunctions, m);
        }
    }

    private void initializeValues(double initialValue) {
        for (int i = 0; i < numberOfFunctions; i++)
        {
            errorRates[i] = initialValue;
        }

        for (int m = 2; m <= maximumOrder; m++) {
            List<ArrayList<Integer>> indexes = CombinatoricsUtilities.getCombinations(numberOfFunctions, m);
            for (int i = 0; i < indexes.size(); i++) {
                errorRates[indexToKeyMapping.get(indexes.get(i))] = 1;
                for (int index : indexes.get(i)) {
                    errorRates[indexToKeyMapping.get(indexes.get(i))] *= errorRates[indexToKeyMapping.get(new ArrayList<Integer>(Arrays.asList(index)))];
                }
            }
        }
    }

    public Map<ArrayList<Integer>, Integer> getIndexToKeyMapping() {
        return indexToKeyMapping;
    }

    public Map<Integer, ArrayList<Integer>> getKeyToIndexMapping() {
        return keyToIndexMapping;
    }

    public int getLength() {
        return errorRatesLength;
    }
}
