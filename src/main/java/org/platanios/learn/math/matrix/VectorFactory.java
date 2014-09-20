package org.platanios.learn.math.matrix;

import java.util.HashMap;

/**
 * This class provides several static methods to build vectors of different types and initialize them in various ways.
 *
 * @author Emmanouil Antonios Platanios
 */
public class VectorFactory {
    /**
     * Builds a vector of the given size and type and fills it with zeros.
     *
     * @param   size    The size of the vector.
     * @param   type    The type of vector to build.
     * @return          The new vector.
     */
    public static Vector build(int size, VectorType type) {
        return type.buildVector(size);
    }

    /**
     * Builds a dense vector of the given size and fills it with zeros.
     *
     * @param   size    The size of the vector.
     * @return          The new vector.
     */
    public static Vector buildDense(int size) {
        return new DenseVector(size);
    }

    /**
     * Builds a dense vector of the given size and fills it with the provided value.
     *
     * @param   size    The size of the vector.
     * @param   value   The value with which to fill the vector.
     * @return          The new vector.
     */
    public static Vector buildDense(int size, double value) {
        return new DenseVector(size, value);
    }

    /**
     * Builds a dense vector from a one-dimensional array.
     *
     * @param   elements    One-dimensional array of doubles.
     * @return              The new vector.
     */
    public static Vector buildDense(double[] elements) {
        return new DenseVector(elements);
    }

    /**
     * Builds a sparse vector of the given size from a hash map.
     *
     * @param   size        The size of the vector.
     * @param   elements    Hash map containing the indexes of elements as keys and the values of the corresponding
     *                      elements as values.
     * @return              The new vector.
     */
    public static Vector buildSparse(int size, HashMap<Integer, Double> elements) {
        return new SparseVector(size, elements);
    }
}
