package org.platanios.learn.math.matrix;

/**
 * This class provides several static methods to build vectors of different types and initialize them in various ways.
 *
 * @author Emmanouil Antonios Platanios
 */
public class VectorFactory {
    /**
     * Builds a dense vector of the given size and fills it with zeros.
     *
     * @param   size    The size of the vector.
     * @return          The new vector.
     */
    public static Vector build(int size) {
        return VectorType.DENSE.buildVector(size, 0);
    }

    /**
     * Builds a dense vector of the given size and fills it with the provided value.
     *
     * @param   size    The size of the vector.
     * @param   value   The value with which to fill the vector.
     * @return          The new vector.
     */
    public static Vector build(int size, double value) {
        return VectorType.DENSE.buildVector(size, value);
    }

    /**
     * Builds a dense vector from a one-dimensional array.
     *
     * @param   elements    One-dimensional array of doubles.
     * @return              The new vector.
     */
    public static Vector build(double[] elements) {
        return VectorType.DENSE.buildVector(elements);
    }

    /**
     * Builds a vector of the given size and type and fills it with zeros.
     *
     * @param   size    The size of the vector.
     * @param   type    The type of vector to build.
     * @return          The new vector.
     */
    public static Vector build(int size, VectorType type) {
        return type.buildVector(size, 0);
    }

    /**
     * Builds a vector of the given size and type and fills it with the provided value.
     *
     * @param   size    The size of the vector.
     * @param   value   The value with which to fill the vector.
     * @param   type    The type of vector to build.
     * @return          The new vector.
     */
    public static Vector build(int size, double value, VectorType type) {
        return type.buildVector(size, value);
    }

    /**
     * Builds a vector of the given type from a one-dimensional array.
     * TODO: Talk about how this method handles sparse vectors (i.e., does it threshold the values?).
     *
     * @param   elements    One-dimensional array of values with which to fill the vector.
     * @param   type        The type of vector to build.
     * @return              The new vector.
     */
    public static Vector build(double[] elements, VectorType type) {
        return type.buildVector(elements);
    }
}
