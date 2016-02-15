package org.platanios.utilities;

/**
 * @author Emmanouil Antonios Platanios
 */
public class Arrays {
    public static boolean equals(double[] array1, double[] array2, double tolerance) {
        if (array1.length != array2.length)
            return false;
        for (int i = 0; i < array1.length; i++)
            if (Math.abs(array1[i] - array2[i]) > tolerance)
                return false;
        return true;
    }
}
