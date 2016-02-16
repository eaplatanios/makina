package org.platanios.learn.classification.reflection;

import com.google.common.collect.BiMap;
import com.google.common.collect.ImmutableBiMap;
import com.google.common.primitives.Ints;
import org.platanios.learn.math.combinatorics.CombinatoricsUtilities;

import java.util.List;

/**
 * A general implementation for storing elements indexed by a power set of integers (for example, indexed by: [0], [1],
 * [2], [0,1], [0,2], [1,2] and [0,1,2]) as one-dimensional arrays and providing ways of accessing those elements.
 *
 * @author Emmanouil Antonios Platanios
 */
abstract class PowerSetVector {
    /** The total number of elements (that is equivalent to the length of the one-dimensional array storing all those
     * elements). */
    public final int length;
    /** A mapping between the power set indexing of those elements and their one-dimensional array indexing. */
    public final BiMap<List<Integer>, Integer> indexKeyMapping;
    /** The number of elements of the whole set, out of which the power set is constructed. */
    private final int highestIndex;
    /** The smallest cardinality of sets to consider, out of the whole power set. */
    private final int lowestOrder;
    /** The highest cardinality of sets to consider, out of the whole power set. */
    private final int highestOrder;
    /** Boolean computeValue indicating whether or not to only consider sets of even cardinality, out of the whole power set.
     * That is useful when considering the agreement rates constraints in the error rates estimation problem, where the
     * agreement rates of a subset of functions with odd cardinality can be written in terms of the agreement rates of
     * all subsets of functions of even cardinality less than the original subset cardinality. */
    private final boolean onlyEvenCardinalitySubsets;

    /** The array that contains the stored elements. */
    public double[] array;  // TODO: Change that to using a generic type and move the whole class to a different package

    /**
     * Constructs a power set indexed vector stored as a one-dimensional array. By default, all possible index sets,
     * within the cardinality limits selected, are considered.
     *
     * @param   highestIndex    The number of elements of the whole set, out of which the power set is constructed.
     * @param   lowestOrder     The smallest cardinality of sets to consider, out of the whole power set.
     * @param   highestOrder    The highest cardinality of sets to consider, out of the whole power set.
     */
    public PowerSetVector(int highestIndex, int lowestOrder, int highestOrder) {
        this(highestIndex, lowestOrder, highestOrder, false);
    }

    /**
     * Constructs a power set indexed vector stored as a one-dimensional array, where an option is given on whether or
     * not to only consider sets of even cardinality within the power set.
     *
     * @param   highestIndex                The number of elements of the whole set, out of which the power set is
     *                                      constructed.
     * @param   lowestOrder                 The smallest cardinality of sets to consider, out of the whole power set.
     * @param   highestOrder                The highest cardinality of sets to consider, out of the whole power set.
     * @param   onlyEvenCardinalitySubsets  Boolean computeValue indicating whether or not to only consider sets of even
     *                                      cardinality, out of the whole power set.
     */
    public PowerSetVector(int highestIndex, int lowestOrder, int highestOrder, boolean onlyEvenCardinalitySubsets) {
        this.highestIndex = highestIndex;
        this.lowestOrder = lowestOrder;
        this.highestOrder = highestOrder;
        this.onlyEvenCardinalitySubsets = onlyEvenCardinalitySubsets;
        int length = 0;
        for (int m = this.lowestOrder; m <= highestOrder; m++)
            if ((m % 2 == 0) || (m % 2 != 0 && !onlyEvenCardinalitySubsets))
                length += CombinatoricsUtilities.getBinomialCoefficient(highestIndex, m);
        this.length = length;
        array = new double[this.length];
        indexKeyMapping = createIndexKeyMapping();
    }

    /**
     * Builds the mapping between the power set indexes and the one-dimensional indexes of the power set vector
     * elements.
     *
     * @return  A {@link com.google.common.collect.BiMap} containing the mapping between the power set indexes and the
     *          one-dimensional indexes of the power set vector elements.
     */
    private ImmutableBiMap<List<Integer>, Integer> createIndexKeyMapping() {
        ImmutableBiMap.Builder<List<Integer>, Integer> indexKeyMappingBuilder = new ImmutableBiMap.Builder<>();
        int offset = 0;
        for (int m = this.lowestOrder; m <= highestOrder; m++) {
            if ((m % 2 == 0) || (m % 2 != 0 && !onlyEvenCardinalitySubsets)) {
                int[][] indexes = CombinatoricsUtilities.getCombinations(highestIndex, m);
                for (int i = 0; i < indexes.length; i++)
                    indexKeyMappingBuilder.put(Ints.asList(indexes[i]), offset + i);
                offset += CombinatoricsUtilities.getBinomialCoefficient(highestIndex, m);
            }
        }
        return indexKeyMappingBuilder.build();
    }
}
