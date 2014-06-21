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
        indexKeyMapping = createIndexKeyMappingBuilder().build();
        computeValues(observations);
    }

    private ImmutableBiMap.Builder<List<Integer>, Integer> createIndexKeyMappingBuilder() {
        ImmutableBiMap.Builder<List<Integer>, Integer> indexKeyMappingBuilder = new ImmutableBiMap.Builder<List<Integer>, Integer>();

        int offset = 0;
        for (int m = 2; m <= maximumOrder; m++) {
//            if (m % 2 == 0) {
            int[][] indexes = CombinatoricsUtilities.getCombinations(numberOfFunctions, m);
            for (int i = 0; i < indexes.length; i++) {
                indexKeyMappingBuilder.put(Ints.asList(indexes[i]), offset + i);
            }
            offset += CombinatoricsUtilities.binomialCoefficient(numberOfFunctions, m);
//            }
        }

        return indexKeyMappingBuilder;
    }

    private void computeValues(boolean[][] observations) {
        for (boolean[] observation : observations) {
            for (BiMap.Entry<List<Integer>, Integer> entry : indexKeyMapping.entrySet()) {
                boolean equal = true;
                List<Integer> indexes = entry.getKey();
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

    public int getLength() {
        return agreementRatesLength;
    }
}
