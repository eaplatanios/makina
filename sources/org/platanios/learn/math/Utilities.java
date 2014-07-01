package org.platanios.learn.math;

/**
 * @author Emmanouil Antonios Platanios
 */
public class Utilities {
    public static double calculateMachineEpsilonDouble() {
        double epsilon = 1;
        while (1 + epsilon / 2 > 1.0) {
            epsilon /= 2;
        }
        return epsilon;
    }
}
