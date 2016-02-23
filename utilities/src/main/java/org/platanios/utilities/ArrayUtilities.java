package org.platanios.utilities;

import java.util.Arrays;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ArrayUtilities {
    public static <T> T[] append(T[] array, T element) {
        final int length = array.length;
        array = Arrays.copyOf(array, length + 1);
        array[length] = element;
        return array;
    }

    public static <T> boolean contains(final T[] array, final T value) {
        if (value == null) {
            for (final T element : array)
                if (element == null)
                    return true;
        } else {
            for (final T element : array)
                if (element == value || value.equals(element))
                    return true;
        }
        return false;
    }

    public static boolean equals(double[] array1, double[] array2, double tolerance) {
        if (array1.length != array2.length)
            return false;
        for (int i = 0; i < array1.length; i++)
            if (Math.abs(array1[i] - array2[i]) > tolerance)
                return false;
        return true;
    }
}
