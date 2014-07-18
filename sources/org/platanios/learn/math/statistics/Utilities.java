package org.platanios.learn.math.statistics;

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

    public static <T> List<T> sample(List<T> list, int numberOfSamples) {
        Collections.shuffle(list);
        return list.subList(0, numberOfSamples);
    }
}
