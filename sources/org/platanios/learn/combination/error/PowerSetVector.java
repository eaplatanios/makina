package org.platanios.learn.combination.error;

import com.google.common.collect.BiMap;
import com.google.common.collect.ImmutableBiMap;
import com.google.common.primitives.Ints;
import org.platanios.math.combinatorics.CombinatoricsUtilities;

import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
abstract class PowerSetVector {
    public double[] array;
    public final BiMap<List<Integer>, Integer> indexKeyMapping;
    public final int length;
    private final int highestIndex;
    private final int lowestOrder;
    private final int highestOrder;
    private final boolean onlyEvenCardinalitySubsets;

    public PowerSetVector(int highestIndex, int lowestOrder, int highestOrder) {
        this(highestIndex, lowestOrder, highestOrder, false);
    }

    public PowerSetVector(int highestIndex, int lowestOrder, int highestOrder, boolean onlyEvenCardinalitySubsets) {
        this.highestIndex = highestIndex;
        this.lowestOrder = lowestOrder;
        this.highestOrder = highestOrder;
        this.onlyEvenCardinalitySubsets = onlyEvenCardinalitySubsets;
        int length = 0;
        for (int m = this.lowestOrder; m <= highestOrder; m++) {
            if ((m % 2 == 0) || (m % 2 != 0 && !onlyEvenCardinalitySubsets)) {
                length += CombinatoricsUtilities.binomialCoefficient(highestIndex, m);
            }
        }
        this.length = length;
        array = new double[this.length];
        indexKeyMapping = createIndexKeyMappingBuilder().build();
    }

    private ImmutableBiMap.Builder<List<Integer>, Integer> createIndexKeyMappingBuilder() {
        ImmutableBiMap.Builder<List<Integer>, Integer> indexKeyMappingBuilder = new ImmutableBiMap.Builder<List<Integer>, Integer>();

        int offset = 0;
        for (int m = this.lowestOrder; m <= highestOrder; m++) {
            if ((m % 2 == 0) || (m % 2 != 0 && !onlyEvenCardinalitySubsets)) {
                int[][] indexes = CombinatoricsUtilities.getCombinations(highestIndex, m);
                for (int i = 0; i < indexes.length; i++) {
                    indexKeyMappingBuilder.put(Ints.asList(indexes[i]), offset + i);
                }
                offset += CombinatoricsUtilities.binomialCoefficient(highestIndex, m);
            }
        }

        return indexKeyMappingBuilder;
    }
}
