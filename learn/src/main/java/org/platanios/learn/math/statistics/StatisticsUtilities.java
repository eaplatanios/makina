package org.platanios.learn.math.statistics;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * A collection of static methods performing some common statistics-related operations.
 *
 * @author Emmanouil Antonios Platanios
 */
public class StatisticsUtilities {
    private static Random random = new Random();

    // Suppress default constructor for noninstantiability
    private StatisticsUtilities() {
        throw new AssertionError();
    }

    /**
     * Sets the seed for the random number generator that is used by the static methods of this class.
     *
     * @param   seed    The seed value to use.
     */
    private static void setSeed(long seed) {
        random = new Random(seed);
    }

    /**
     * Shuffles the order of the elements in the provided array using Knuth's shuffle algorithm.
     *
     * @param   array   The array whose elements are shuffled.
     */
    public static <T> void shuffle(T[] array) {
        for (int i = array.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            T tempValue = array[j];
            array[j] = array[i];
            array[i] = tempValue;
        }
    }

    /**
     * Shuffles the order of the elements in the provided list using Knuth's shuffle algorithm.
     *
     * @param   list    The list whose elements are shuffled.
     */
    public static <T> void shuffle(List<T> list) {
        for (int i = list.size() - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            T tempValue = list.get(j);
            list.set(j, list.get(i));
            list.set(i, tempValue);
        }
    }

    /**
     * Randomly samples a specified number of elements from an array without replacement. Note that this method modifies
     * the provided array (i.e., it shuffles the order of its elements).
     *
     * @param   array           The array from which we randomly sample elements.
     * @param   numberOfSamples The number of elements to sample from the given list.
     * @return                  A new array containing the sampled elements.
     */
    public static <T> T[] sampleWithoutReplacement(T[] array, int numberOfSamples) {
        shuffle(array);
        return Arrays.copyOfRange(array, 0, numberOfSamples);
    }

    /**
     * Randomly samples a specified number of elements from a list without replacement. Note that this method modifies
     * the provided list (i.e., it shuffles the order of its elements).
     *
     * @param   list            The list from which we randomly sample elements.
     * @param   numberOfSamples The number of elements to sample from the given list.
     * @return                  A new {@link java.util.ArrayList} containing the sampled elements.
     */
    public static <T> List<T> sampleWithoutReplacement(List<T> list, int numberOfSamples) {
        shuffle(list);
        return new ArrayList<>(list.subList(0, numberOfSamples));
    }
}
