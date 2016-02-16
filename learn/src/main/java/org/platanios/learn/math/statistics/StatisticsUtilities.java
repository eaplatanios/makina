package org.platanios.learn.math.statistics;

import org.platanios.learn.math.matrix.Vector;

import java.util.*;

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

    public static double mean(double[] values) {
        double mean = 0;
        for (double value : values)
            mean += value;
        return mean / values.length;
    }

    public static double mean(List<Double> values) {
        double mean = 0;
        for (double value : values)
            mean += value;
        return mean / values.size();
    }

    public static double mean(Vector vector) {
        return vector.sum() / vector.size();
    }

    public static double variance(double[] values) {
        double mean = mean(values);
        double variance = 0;
        for (double value : values)
            variance += Math.pow(value - mean, 2);
        return variance / values.length;
    }

    public static double variance(List<Double> values) {
        double mean = mean(values);
        double variance = 0;
        for (double value : values)
            variance += Math.pow(value - mean, 2);
        return variance / values.size();
    }

    public static double variance(Vector vector) {
        double mean = mean(vector);
        final double[] variance = { 0 };
        vector.iterator().forEachRemaining(element -> variance[0] += Math.pow(element.value() - mean, 2));
        return variance[0] / vector.size();
    }

    public static double standardDeviation(double[] values) {
        return Math.sqrt(variance(values));
    }

    public static double standardDeviation(List<Double> values) {
        return Math.sqrt(variance(values));
    }

    public static double standardDeviation(Vector vector) {
        return Math.sqrt(variance(vector));
    }

    /**
     * Sets the seed for the random number generator that is used by the static methods of this class.
     *
     * @param   seed    The seed computeValue to use.
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

    /**
     * Randomly samples a specified number of elements from an integer array without replacement, while assigning a
     * probability proportional to the provided weight for each element of that array. This method does not modify the
     * provided array.
     *
     * @param   array           The array from which we randomly sample elements.
     * @param   weights         The weights to use for each element in the provided array.
     * @param   numberOfSamples The number of elements to sample from the given array.
     * @return                  A new integer array containing the sampled elements.
     * @throws  IllegalArgumentException    The length of the array with the elements must match the length of the array
     *                                      with the weights of those elements.
     */
    public static int[] sampleWithoutReplacement(int[] array, double[] weights, int numberOfSamples) {
        if (array.length != weights.length)
            throw new IllegalArgumentException("The length of the array with the elements must match the length " +
                                                       "of the array with the weights of those elements!");

        PriorityQueue<WeightedElement> priorityQueue = new PriorityQueue<>(
                array.length,
                (element1, element2) -> (int) Math.signum(element2.weight - element1.weight)
        );
        for (int index = 0; index < array.length; index++)
            priorityQueue.add(new WeightedElement(array[index], Math.pow(random.nextDouble(), 1 / weights[index])));
        int[] resultArray = new int[numberOfSamples];
        for (int index = 0; index < numberOfSamples; index++)
            resultArray[index] = priorityQueue.poll().element;
        return resultArray;
    }

    /**
     * Randomly samples a specified number of elements from an integer array with replacement, while assigning a
     * probability proportional to the provided weight for each element of that array. This method does not modify the
     * provided array.
     *
     * @param   array           The array from which we randomly sample elements.
     * @param   weights         The weights to use for each element in the provided array.
     * @param   numberOfSamples The number of elements to sample from the given array.
     * @return                  A new integer array containing the sampled elements.
     * @throws  IllegalArgumentException    The length of the array with the elements must match the length of the array
     *                                      with the weights of those elements.
     */
    public static int[] sampleWithReplacement(int[] array, double[] weights, int numberOfSamples) {
        if (array.length != weights.length)
            throw new IllegalArgumentException("The length of the array with the elements must match the length " +
                                                       "of the array with the weights of those elements!");

        double weightsSum = weights[0];
        for (int index = 1; index < weights.length; index++)
            weightsSum += weights[index];
        double[] cumulativeDensityFunction = new double[weights.length];
        cumulativeDensityFunction[0] = weights[0] / weightsSum;
        for (int index = 1; index < weights.length; index++)
            cumulativeDensityFunction[index] = cumulativeDensityFunction[index - 1] + (weights[index] / weightsSum);
        int[] resultArray = new int[numberOfSamples];
        for (int index = 0; index < numberOfSamples; index++) {
            double number = random.nextDouble();
            int elementIndex = Arrays.binarySearch(cumulativeDensityFunction, number);
            if (elementIndex < 0) {
                elementIndex = -elementIndex - 1;
                if (elementIndex == weights.length)
                    elementIndex--;
            }
            resultArray[index] = array[elementIndex];
        }
        return resultArray;
    }

    private static class WeightedElement {
        private int element;
        private double weight;

        private WeightedElement(int element, double weight) {
            this.element = element;
            this.weight = weight;
        }
    }
}
