package org.platanios.learn.combination;

import com.google.common.collect.BiMap;
import com.google.common.collect.ImmutableBiMap;
import com.google.common.primitives.Ints;
import org.platanios.math.combinatorics.CombinatoricsUtilities;

import java.util.*;

/**
 * @author Emmanouil Antonios Platanios
 */
public class AgreementRatesVector {
    public double[] agreementRates;
    public final BiMap<List<Integer>, Integer> indexKeyMapping;

    private int numberOfFunctions;
    private int maximumOrder;
    private int agreementRatesLength;
    private boolean onlyEvenCardinalitySubsets;

    public AgreementRatesVector(int numberOfFunctions, int maximumOrder, boolean[][] observations) {
        this(numberOfFunctions, maximumOrder, observations, true);
    }

    public AgreementRatesVector(int numberOfFunctions, int maximumOrder, boolean[][] observations, boolean onlyEvenCardinalitySubsets) {
        this.numberOfFunctions = numberOfFunctions;
        this.maximumOrder = maximumOrder;
        this.onlyEvenCardinalitySubsets = onlyEvenCardinalitySubsets;
        agreementRatesLength = 0;
        for (int m = 2; m <= maximumOrder; m++) {
            if ((m % 2 == 0) || (m % 2 != 0 && !onlyEvenCardinalitySubsets)) {
                agreementRatesLength += CombinatoricsUtilities.binomialCoefficient(numberOfFunctions, m);
            }
        }
        agreementRates = new double[agreementRatesLength];
        indexKeyMapping = createIndexKeyMappingBuilder().build();
        computeValues(observations);
    }

    private ImmutableBiMap.Builder<List<Integer>, Integer> createIndexKeyMappingBuilder() {
        ImmutableBiMap.Builder<List<Integer>, Integer> indexKeyMappingBuilder = new ImmutableBiMap.Builder<List<Integer>, Integer>();

        int offset = 0;
        for (int m = 2; m <= maximumOrder; m++) {
            if ((m % 2 == 0) || (m % 2 != 0 && !onlyEvenCardinalitySubsets)) {
                int[][] indexes = CombinatoricsUtilities.getCombinations(numberOfFunctions, m);
                for (int i = 0; i < indexes.length; i++) {
                    indexKeyMappingBuilder.put(Ints.asList(indexes[i]), offset + i);
                }
                offset += CombinatoricsUtilities.binomialCoefficient(numberOfFunctions, m);
            }
        }

        return indexKeyMappingBuilder;
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
                    agreementRates[entry.getValue()] += 1;
                }
            }
        }
        for (int i = 0; i < agreementRatesLength; i++) {
            agreementRates[i] /= classifiersOutputs.length;
        }
    }

    public int getLength() {
        return agreementRatesLength;
    }
}
