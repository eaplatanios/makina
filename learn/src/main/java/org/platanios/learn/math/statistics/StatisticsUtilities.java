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
    public static void setSeed(long seed) {
        random = new Random(seed);
    }

    /**
     * Shuffles the order of the elements in the provided array using Knuth's shuffle algorithm.
     *
     * @param   array   The array whose elements are shuffled.
     */
    public static <T> void shuffle(T[] array) {
        shuffle(array, random);
    }

    /**
     * Shuffles the order of the elements in the provided array using Knuth's shuffle algorithm.
     *
     * @param   array   The array whose elements are shuffled.
     * @param   random  The random number generator to use for the shuffling.
     */
    public static <T> void shuffle(T[] array, Random random) {
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
        shuffle(list, random);
    }

    /**
     * Shuffles the order of the elements in the provided list using Knuth's shuffle algorithm.
     *
     * @param   list    The list whose elements are shuffled.
     * @param   random  The random number generator to use for the shuffling.
     */
    public static <T> void shuffle(List<T> list, Random random) {
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
        return sampleWithoutReplacement(array, numberOfSamples, random);
    }
    	
    /**
     * Randomly samples a specified number of elements from an array without replacement. Note that this method modifies
     * the provided array (i.e., it shuffles the order of its elements).
     *
     * @param   array           The array from which we randomly sample elements.
     * @param   numberOfSamples The number of elements to sample from the given list.
     * @param   random          The random number generator to use for the shuffling.
     * @return                  A new array containing the sampled elements.
     */
    public static <T> T[] sampleWithoutReplacement(T[] array, int numberOfSamples, Random random) {
        shuffle(array, random);
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
        return sampleWithoutReplacement(list, numberOfSamples, random);
    }

    /**
     * Randomly samples a specified number of elements from a list without replacement. Note that this method modifies
     * the provided list (i.e., it shuffles the order of its elements).
     *
     * @param   list            The list from which we randomly sample elements.
     * @param   numberOfSamples The number of elements to sample from the given list.
     * @param   random          The random number generator to use for the shuffling.
     * @return                  A new {@link java.util.ArrayList} containing the sampled elements.
     */
    public static <T> List<T> sampleWithoutReplacement(List<T> list, int numberOfSamples, Random random) {
        shuffle(list, random);
        return new ArrayList<>(list.subList(0, numberOfSamples));
    }

    public static int[] sampleWithReplacement(int[] array, double[] probabilities, int numberOfSamples) {
        if (array.length != probabilities.length)
            throw new IllegalArgumentException("The length of the array with the elements must match the length " +
                                                       "of the array with the probabilities of those elements!");

        double[] cumulativeDensityFunction = new double[probabilities.length];
        cumulativeDensityFunction[0] = probabilities[0];
        for (int index = 1; index < probabilities.length; index++)
            cumulativeDensityFunction[index] = cumulativeDensityFunction[index - 1] + probabilities[index];
        int[] resultArray = new int[numberOfSamples];
        for (int index = 0; index < numberOfSamples; index++) {
            double number = random.nextDouble();
            int elementIndex = Arrays.binarySearch(cumulativeDensityFunction, number);
            if (elementIndex < 0) {
                elementIndex = -elementIndex - 1;
                if (elementIndex == probabilities.length)
                    elementIndex--;
            }
            resultArray[index] = array[elementIndex];
        }
        return resultArray;
    }
}
