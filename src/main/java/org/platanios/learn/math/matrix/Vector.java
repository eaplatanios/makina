package org.platanios.learn.math.matrix;

import java.util.function.Function;

/**
 * @author Emmanouil Antonios Platanios
 */
public interface Vector {
    /**
     * Gets the type of this vector (i.e., dense, sparse, etc.).
     *
     * @return  The type of this vector.
     */
    VectorType getType();

    /**
     * Copies this vector. // TODO: Switch this to a static factory or copy constructor.
     *
     * @return  A copy of this vector.
     */
    Vector copy();

    /**
     * Gets a dense array representation of this vector. This array is completely separate from the inner representation
     * used by the vector implementation.
     *
     * @return  A dense array representation of this vector.
     */
    double[] getDenseArray();

    /**
     * Gets the dimension of this vector.
     *
     * @return  The dimension of this vector.
     */
    int getDimension();

    /**
     * Gets the value of the vector element at the provided index.
     *
     * @param   index   The index of the element.
     * @return          The value of the element at the provided index.
     */
    double get(int index);

    /**
     * Gets a sub-vector of this vector.
     *
     * @param   initialIndex    The initial index.
     * @param   finalIndex      The final index.
     * @return                  The sub-vector corresponding to the provided indexes.
     *
     * @exception   ArrayIndexOutOfBoundsException  Some or all of the provided sub-vector indexes are out of bounds.
     */
    Vector get(int initialIndex, int finalIndex);

    /**
     * Gets a sub-vector of this vector.
     *
     * @param   indexes The indexes of the elements of this vector to be included in the returned sub-vector.
     * @return          The sub-vector corresponding to the provided indexes.
     *
     * @exception   ArrayIndexOutOfBoundsException  Some or all of the provided sub-vector indexes are out of bounds.
     */
    Vector get(int[] indexes);

    /**
     * Sets the value of the vector element at the provided index to the provided value.
     *
     * @param   index   The index of the element.
     * @param   value   The value to which to set the element at the provided index.
     */
    void set(int index, double value);

    /**
     * Sets a sub-vector of this vector to the provided vector values.
     *
     * @param   initialIndex    The initial index.
     * @param   finalIndex      The final index.
     * @param   vector          The vector to whose values we set the values of the specified sub-vector of this vector.
     *
     * @exception   ArrayIndexOutOfBoundsException  Some or all of the provided vector indexes are out of bounds.
     */
    void set(int initialIndex, int finalIndex, Vector vector);

    /**
     * Sets a sub-vector of this matrix to the provided vector values.
     *
     * @param   indexes     The indexes of the elements of this vector to be changed to values of the elements of the
     *                      provided sub-vector.
     * @param   vector      The vector to whose values we set the values of the specified sub-vector of this vector.
     *
     * @exception   ArrayIndexOutOfBoundsException  Some or all of the provided vector indexes are out of bounds.
     */
    void set(int[] indexes, Vector vector);

    /**
     * Sets the value of all of the vector elements to the provided value.
     *
     * @param   value   The value to which to set the elements of this vector.
     */
    void setAll(double value);

    /**
     * Gets the maximum value of all elements in this vector.
     *
     * @return  The maximum value of all elements in this vector.
     */
    double max();

    /**
     * Gets the minimum value of all elements in this vector.
     *
     * @return  The minimum value of all elements in this vector.
     */
    double min();

    /**
     * Computes and returns the sum of all elements in this vector.
     *
     * @return  The sum of all elements in this vector.
     */
    double sum();

    /**
     * Computes the specified norm of this vector.
     *
     * @return  The specified norm of this vector.
     */
    double norm(VectorNorm normType);

    /**
     * Computes the result of applying the supplied function element-wise to the current vector and returns it in a new
     * vector.
     *
     * @param   function    The function to apply to the current vector element-wise.
     * @return              A new vector holding the result of the operation.
     */
    Vector computeFunctionResult(Function<Double, Double> function);

    /**
     * Adds a scalar to all entries of the current vector and returns the result in a new vector.
     *
     * @param   scalar  The scalar to add to entries of the current vector.
     * @return          A new vector holding the result of the addition.
     */
    Vector add(double scalar);

    /**
     * Adds another vector to the current vector and returns the result in a new vector.
     *
     * @param   vector  The vector to add to the current vector.
     * @return          A new vector holding the result of the addition.
     */
    Vector add(Vector vector);

    /**
     * Adds a scalar to all entries of the current vector and replaces the current vector with the result.
     *
     * @param   scalar  The scalar to add to entries of the current vector.
     */
    void addEquals(double scalar);

    /**
     * Adds another vector to the current vector and replaces the current vector with the result.
     *
     * @param   vector  The vector to add to the current vector.
     */
    void addEquals(Vector vector);

    /**
     * Subtracts a scalar from all entries of the current vector and returns the result in a new vector.
     *
     * @param   scalar  The scalar to subtract from all entries of the current vector.
     * @return          A new vector holding the result of the subtraction.
     */
    Vector subtract(double scalar);

    /**
     * Subtracts another vector from the current vector and returns the result in a new vector.
     *
     * @param   vector  The vector to subtract from the current vector.
     * @return          A new vector holding the result of the subtraction.
     */
    Vector subtract(Vector vector);

    /**
     * Subtracts a scalar from all entries of the current vector and replaces the current vector with the result.
     *
     * @param   scalar  The scalar to subtract from all entries of the current vector.
     */
    void subtractEquals(double scalar);

    /**
     * Subtracts another vector from the current vector and replaces the current vector with the result.
     *
     * @param   vector  The vector to subtract from the current vector.
     */
    void subtractEquals(Vector vector);

    /**
     * Multiplies another vector with the current vector element-wise and returns the result in a new vector.
     *
     * @param   vector  The vector to multiply with the current vector element-wise.
     * @return          A new vector holding the result of the multiplication.
     */
    Vector multiplyElementwise(Vector vector);

    /**
     * Multiplies another vector with the current vector element-wise and replaces the current vector with the result.
     *
     * @param   vector  The vector to multiply with the current vector element-wise.
     */
    void multiplyElementwiseEquals(Vector vector);

    /**
     * Divides another vector with the current vector element-wise and returns the result in a new vector.
     *
     * @param   vector  The vector to divide with the current vector element-wise.
     * @return          A new vector holding the result of the division.
     */
    Vector divideElementwise(Vector vector);

    /**
     * Divides another vector with the current vector element-wise and replaces the current vector with the result.
     *
     * @param   vector  The vector to divide with the current vector element-wise.
     */
    void divideElementwiseEquals(Vector vector);

    /**
     * Multiplies the current vector with a scalar and returns the result in a new vector.
     *
     * @param   scalar  The scalar with which to multiply the current vector.
     * @return          A new vector holding the result of the multiplication.
     */
    Vector multiply(double scalar);

    /**
     * Multiplies the current vector with a scalar and replaces the current vector with the result.
     *
     * @param   scalar  The scalar with which to multiply the current vector.
     */
    void multiplyEquals(double scalar);

    /**
     * Divides the current vector with a scalar and returns the result in a new vector.
     *
     * @param   scalar  The scalar with which to divide the current vector.
     * @return          A new vector holding the result of the division.
     */
    Vector divide(double scalar);

    /**
     * Divides the current vector with a scalar and replaces the current vector with the result.
     *
     * @param   scalar  The scalar with which to divide the current vector.
     */
    void divideEquals(double scalar);

    /**
     * Computes the inner product (also known as the dot product) between the current vector and another vector.
     *
     * @param   vector  The vector used to compute the inner product with the current vector.
     * @return          The resulting inner product value.
     */
    double innerProduct(Vector vector);

    /**
     * Computes the outer product between the current vector and another vector and returns the result in a new matrix.
     *
     * @param   vector  The vector used to compute the outer product with the current vector.
     * @return          A new matrix containing the result of the outer product operation.
     */
    Matrix outerProduct(Vector vector);

    /**
     * Computes the product of this vector with a matrix and returns the result in a new vector.
     *
     * @param   matrix  The matrix with which to multiply the current vector.
     * @return          A new vector holding the result of the multiplication.
     */
    Vector multiply(Matrix matrix);
}
