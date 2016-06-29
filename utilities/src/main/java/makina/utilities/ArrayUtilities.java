package makina.utilities;

import java.lang.reflect.Array;
import java.util.Arrays;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ArrayUtilities {
    public static double sum(double[] array) {
        double sum = 0.0;
        for (double element : array)
            sum += element;
        return sum;
    }

    public static <T> T[] append(T[] array, T element) {
        final int length = array.length;
        array = Arrays.copyOf(array, length + 1);
        array[length] = element;
        return array;
    }

    public static void copy(Object sourceArray, Object destinationArray) {
        if (sourceArray.getClass().isArray() && destinationArray.getClass().isArray()) {
            for (int i = 0; i < Array.getLength(sourceArray); i++) {
                if (Array.get(sourceArray, i) != null && Array.get(sourceArray, i).getClass().isArray())
                    copy(Array.get(sourceArray, i), Array.get(destinationArray, i));
                else
                    Array.set(destinationArray, i, Array.get(sourceArray, i));
            }
        } else {
            throw new IllegalArgumentException("The provided arguments are not arrays.");
        }
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
