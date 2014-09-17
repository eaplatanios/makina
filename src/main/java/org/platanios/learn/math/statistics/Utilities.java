package org.platanios.learn.math.statistics;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class Utilities {
    // Suppress default constructor for noninstantiability
    private Utilities() {
        throw new AssertionError();
    }

    /**
     * Randomly samples a specified number of elements from a list without replacement.
     *
     * @param   list            The list from which we randomly sample elements.
     * @param   numberOfSamples The number of elements to sample from the given list.
     * @return                  A new {@link java.util.ArrayList} containing the sampled elements.
     */
    public static <T> List<T> sampleWithoutReplacement(List<T> list, int numberOfSamples) {
        Collections.shuffle(list);
        return new ArrayList<>(list.subList(0, numberOfSamples));
    }
}
