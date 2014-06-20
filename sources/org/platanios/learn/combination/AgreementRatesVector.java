package org.platanios.learn.combination;

import org.platanios.math.combinatorics.CombinatoricsUtilities;

import java.util.*;

/**
 * @author Emmanouil Antonios Platanios
 */
public class AgreementRatesVector {
    public double[] agreementRates;

    private int numberOfFunctions;
    private int maximumOrder;
    private int agreementRatesLength;
    private Map<ArrayList<Integer>, Integer> indexToKeyMapping;
    private Map<Integer, ArrayList<Integer>> keyToIndexMapping;

    public AgreementRatesVector(int numberOfFunctions, int maximumOrder, boolean[][] observations) {
        this.numberOfFunctions = numberOfFunctions;
        this.maximumOrder = maximumOrder;
        agreementRatesLength = 0;
        for (int m = 2; m <= maximumOrder; m++) {
//            if (m % 2 == 0) {
                agreementRatesLength += CombinatoricsUtilities.binomialCoefficient(numberOfFunctions, m);
//            }
        }
        agreementRates = new double[agreementRatesLength];

        createMappings();
        computeValues(observations);
    }

    private void createMappings() {
        indexToKeyMapping = new LinkedHashMap<ArrayList<Integer>, Integer>();
        keyToIndexMapping = new LinkedHashMap<Integer, ArrayList<Integer>>();

        int offset = 0;

        for (int m = 2; m <= maximumOrder; m++) {
//            if (m % 2 == 0) {
                List<ArrayList<Integer>> indexes = CombinatoricsUtilities.getCombinations(numberOfFunctions, m);
                for (int i = 0; i < indexes.size(); i++) {
                    indexToKeyMapping.put(indexes.get(i), offset + i);
                    keyToIndexMapping.put(offset + i, indexes.get(i));
                }
                offset += CombinatoricsUtilities.binomialCoefficient(numberOfFunctions, m);
//            }
        }
    }

    private void computeValues(boolean[][] observations) {
        for (boolean[] observation : observations) {
            for (Map.Entry<ArrayList<Integer>, Integer> entry : indexToKeyMapping.entrySet()) {
                boolean equal = true;
                ArrayList<Integer> indexes = entry.getKey();
                for (int index : indexes.subList(1, indexes.size())) {
                    equal = equal && (observation[indexes.get(0)] == observation[index]);
                }
                if (equal) {
                    agreementRates[entry.getValue()] += 1;
                }
            }
        }
        for (int i = 0; i < agreementRatesLength; i++) {
            agreementRates[i] /= observations.length;
        }
    }

    public Map<ArrayList<Integer>, Integer> getIndexToKeyMapping() {
        return indexToKeyMapping;
    }

    public Map<Integer, ArrayList<Integer>> getKeyToIndexMapping() {
        return keyToIndexMapping;
    }

    public int getLength() {
        return agreementRatesLength;
    }
}
