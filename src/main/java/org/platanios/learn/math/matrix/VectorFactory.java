package org.platanios.learn.math.matrix;

/**
 * @author Emmanouil Antonios Platanios
 */
public class VectorFactory {
    public static Vector createVector(int size) {
        return VectorType.DENSE.createVector(size, 0);
    }

    public static Vector createVector(int size, VectorType type) {
        return type.createVector(size, 0);
    }

    public static Vector createVector(int size, double initialValue, VectorType type) {
        return type.createVector(size, initialValue);
    }
}
