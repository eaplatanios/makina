package org.platanios.learn;

import org.apache.commons.lang3.ArrayUtils;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * @author Emmanouil Antonios Platanios
 */
public class Utilities {

    // Suppress default constructor for noninstantiability
    private Utilities() {
        throw new AssertionError();
    }

    public static int[] union(int[] a, int[] b) {
        Set<Integer> set = new HashSet<>(Arrays.asList(ArrayUtils.toObject(a)));
        set.addAll(Arrays.asList(ArrayUtils.toObject(b)));
        return ArrayUtils.toPrimitive(set.toArray(new Integer[set.size()]));
    }
}
