package org.platanios.learn.math.matrix;

/**
 * Implements a class representing vectors and supporting operations related to vectors. Vectors are stored in an
 * internal one-dimensional array.
 *
 * @author Emmanouil Antonios Platanios
 */
public class Vector {
    /** The dimension of the vector. */
    private final int dimension;

    /** Array for internal storage of the vector elements. */
    private double[] array;

    //region Constructors
    /**
     * Constructs a vector with the given dimension and fills it with zeros.
     *
     * @param   dimension   The dimension of the vector.
     */
    public Vector(int dimension) {
        this.dimension = dimension;
        array = new double[dimension];
    }

    /**
     * Constructs a vector with the given dimensions and fills it with the given value.
     *
     * @param   dimension   The dimension of the vector.
     * @param   value       The value with which to fill the vector.
     */
    public Vector(int dimension, double value) {
        this.dimension = dimension;
        array = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            array[i] = value;
        }
    }

    /**
     * Constructs a vector from a one-dimensional array.
     *
     * @param   array   One-dimensional array of doubles.
     */
    public Vector(double[] array) {
        dimension = array.length;
        this.array = array;
    }
    //endregion

    //region Getters, Setters and Other Such Methods
    /**
     * Copies this vector.
     *
     * @return  A copy of this vector.
     */
    public Vector copy() {
        Vector resultVector = new Vector(dimension);
        double[] resultVectorArray = resultVector.getArray();
        System.arraycopy(array, 0, resultVectorArray, 0, dimension);
        return resultVector;
    }

    /**
     * Copies this vector to a new matrix structure.
     *
     * @return  A copy of this vector represented as a new matrix.
     */
    public Matrix copyAsMatrix() {
        return new Matrix(array, array.length);
    }

    /**
     * Gets a pointer to the internal one-dimensional array.
     *
     * @return  A pointer to the internal one-dimensional array.
     */
    public double[] getArray() {
        return array;
    }

    /**
     * Copies the internal one-dimensional array.
     *
     * @return  A copy of the internal one-dimensional array.
     */
    public double[] getArrayCopy() {
        double[] resultArray = new double[dimension];
        System.arraycopy(array, 0, resultArray, 0, dimension);
        return resultArray;
    }

    /**
     * Gets the dimension of this vector.
     *
     * @return  The dimension of this vector.
     */
    public int getDimension() {
        return dimension;
    }

    /**
     * Gets the value of the vector element at the provided index.
     *
     * @param   index   The index of the element.
     * @return          The value of the element at the provided index.
     */
    public double getElement(int index) {
        return array[index]; // TODO: Check for the index values (i.e. out of bounds).
    }

    /**
     * Sets the value of the vector element at the provided index to the provided value.
     *
     * @param   index   The index of the element.
     * @param   value   The value to which to set the element at the provided index.
     */
    public void setElement(int index, double value) {
        array[index] = value;
    }

    /**
     * Gets a sub-vector of this vector.
     *
     * @param   initialIndex    The initial index.
     * @param   finalIndex      The final index.
     * @return                  The sub-vector corresponding to the provided indexes.
     *
     * @exception   ArrayIndexOutOfBoundsException  Some or all of the provided sub-vector indexes are out of bounds.
     */
    public Vector getSubVector(int initialIndex, int finalIndex) {
        Vector resultVector = new Vector(finalIndex - initialIndex + 1);
        double[] resultVectorArray = resultVector.getArray();
        try {
            System.arraycopy(array, initialIndex, resultVectorArray, 0, finalIndex + 1 - initialIndex);
        } catch(ArrayIndexOutOfBoundsException e) {
            throw new ArrayIndexOutOfBoundsException(
                    "Some or all of the provided sub-vector indexes are out of bounds."
            );
        }
        return resultVector;
    }

    /**
     * Gets a sub-vector of this vector.
     *
     * @param   indexes The indexes of the elements of this vector to be included in the returned sub-vector.
     * @return          The sub-vector corresponding to the provided indexes.
     *
     * @exception   ArrayIndexOutOfBoundsException  Some or all of the provided sub-vector indexes are out of bounds.
     */
    public Vector getSubVector(int[] indexes) {
        Vector resultVector = new Vector(indexes.length);
        double[] resultVectorArray = resultVector.getArray();
        try {
            for (int i = 0; i < indexes.length; i++) {
                resultVectorArray[i] = array[indexes[i]];
            }
        } catch(ArrayIndexOutOfBoundsException e) {
            throw new ArrayIndexOutOfBoundsException(
                    "Some or all of the provided sub-vector indexes are out of bounds."
            );
        }
        return resultVector;
    }

    /**
     * Sets a sub-vector of this vector to the provided vector values.
     *
     * @param   initialIndex    The initial index.
     * @param   finalIndex      The final index.
     * @param   vector          The vector to whose values we set the values of the specified sub-vector of this vector.
     *
     * @exception   ArrayIndexOutOfBoundsException  Some or all of the provided vector indexes are out of bounds.
     */
    public void setSubVector(int initialIndex, int finalIndex, Vector vector) {
        try {
            for (int i = initialIndex; i <= finalIndex; i++) {
                array[i] = vector.getElement(i - initialIndex);
            }
        } catch(ArrayIndexOutOfBoundsException e) {
            throw new ArrayIndexOutOfBoundsException(
                    "Some or all of the provided vector indexes are out of bounds."
            );
        }
    }

    /**
     * Sets a sub-vector of this matrix to the provided vector values.
     *
     * @param   indexes     The indexes of the elements of this vector to be changed to values of the elements of the
     *                      provided sub-vector.
     * @param   vector      The vector to whose values we set the values of the specified sub-vector of this vector.
     *
     * @exception   ArrayIndexOutOfBoundsException  Some or all of the provided vector indexes are out of bounds.
     */
    public void setSubVector(int[] indexes, Vector vector) {
        try {
            for (int i = 0; i < indexes.length; i++) {
                array[indexes[i]] = vector.getElement(i);
            }
        } catch(ArrayIndexOutOfBoundsException e) {
            throw new ArrayIndexOutOfBoundsException(
                    "Some or all of the provided vector indexes are out of bounds."
            );
        }
    }
    //endregion

    //region Norm Computations
    /**
     * Computes the \(L_1\) norm of this vector. Denoting this vector by \(\boldsymbol{x}\in\mathbb{R}^{n}\), its
     * element at index \(i\) by \(x_i\) and its \(L_1\) norm by \(\|\boldsymbol{x}\|_1\), we have that:
     * \[\|\boldsymbol{x}\|_1=\sum_{i=1}^n{\left|x_i\right|}.\]
     *
     * @return  The \(L_1\) norm of this vector.
     */
    public double computeL1Norm() {
        double l1Norm = 0;
        for (int i = 0; i < dimension; i++) {
            l1Norm += array[i];
        }
        return l1Norm;
    }

    /**
     * Computes the \(L_2\) norm of this vector. Denoting this vector by \(\boldsymbol{x}\in\mathbb{R}^{n}\), its
     * element at index \(i\) by \(x_i\) and its \(L_2\) norm by \(\|\boldsymbol{x}\|_2\), we have that:
     * \[\|\boldsymbol{x}\|_2=\sqrt{\sum_{i=1}^n{x_i^2}}.\]
     *
     * @return  The \(L_2\) norm of this vector.
     */
    public double computeL2Norm() {
        double l2Norm = 0;
        for (int i = 0; i < dimension; i++) {
            l2Norm += array[i] * array[i];
        }
        return Math.sqrt(l2Norm);
    }

    /**
     * Computes the \(L_\infty\) norm of this vector. Denoting this vector by \(\boldsymbol{x}\in\mathbb{R}^{n}\), its
     * element at index \(i\) by \(x_i\) and its \(L_\infty\) norm by \(\|\boldsymbol{x}\|_\infty\), we have that:
     * \[\|\boldsymbol{x}\|_\infty=\max_{1\leq i\leq n}{\left|x_i\right|}.\]
     *
     * @return  The \(L_\infty\) norm of this vector.
     */
    public double computeLInfinityNorm() {
        double lInfinityNorm = 0;
        for (int i = 0; i < dimension; i++) {
            lInfinityNorm = Math.max(lInfinityNorm, array[i]);
        }
        return Math.sqrt(lInfinityNorm);
    }
    //endregion

    //region Element-wise Operations
    /**
     * Adds another vector to the current vector and returns the result in a new vector.
     *
     * @param   vector  The vector to add to the current vector.
     * @return          A new vector holding the result of the addition.
     */
    public Vector add(Vector vector) {
        checkVectorDimensions(vector);
        Vector resultVector = new Vector(dimension);
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < dimension; i++) {
            resultVectorArray[i] = array[i] + vector.array[i];
        }
        return resultVector;
    }

    /**
     * Adds another vector to the current vector and replaces the current vector with the result.
     *
     * @param   vector  The vector to add to the current vector.
     */
    public void addEquals(Vector vector) {
        checkVectorDimensions(vector);
        for (int i = 0; i < dimension; i++) {
            array[i] += vector.array[i];
        }
    }

    /**
     * Subtracts another vector from the current vector and returns the result in a new vector.
     *
     * @param   vector  The vector to subtract from the current vector.
     * @return          A new vector holding the result of the subtraction.
     */
    public Vector subtract(Vector vector) {
        checkVectorDimensions(vector);
        Vector resultVector = new Vector(dimension);
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < dimension; i++) {
            resultVectorArray[i] = array[i] - vector.array[i];
        }
        return resultVector;
    }

    /**
     * Subtracts another vector from the current vector and replaces the current vector with the result.
     *
     * @param   vector  The vector to subtract from the current vector.
     */
    public void subtractEquals(Vector vector) {
        checkVectorDimensions(vector);
        for (int i = 0; i < dimension; i++) {
            array[i] -= vector.array[i];
        }
    }

    /**
     * Multiplies another vector with the current vector element-wise and returns the result in a new vector.
     *
     * @param   vector  The vector to multiply with the current vector element-wise.
     * @return          A new vector holding the result of the multiplication.
     */
    public Vector multiplyElementwise(Vector vector) {
        checkVectorDimensions(vector);
        Vector resultVector = new Vector(dimension);
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < dimension; i++) {
            resultVectorArray[i] = array[i] * vector.array[i];
        }
        return resultVector;
    }

    /**
     * Multiplies another vector with the current vector element-wise and replaces the current vector with the result.
     *
     * @param   vector  The vector to multiply with the current vector element-wise.
     */
    public void multiplyElementwiseEquals(Vector vector) {
        checkVectorDimensions(vector);
        for (int i = 0; i < dimension; i++) {
            array[i] *= vector.array[i];
        }
    }

    /**
     * Divides another vector with the current vector element-wise and returns the result in a new vector.
     *
     * @param   vector  The vector to divide with the current vector element-wise.
     * @return          A new vector holding the result of the division.
     */
    public Vector divideElementwise(Vector vector) {
        checkVectorDimensions(vector);
        Vector resultVector = new Vector(dimension);
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < dimension; i++) {
            resultVectorArray[i] = array[i] / vector.array[i];
        }
        return resultVector;
    }

    /**
     * Divides another vector with the current vector element-wise and replaces the current vector with the result.
     *
     * @param   vector  The vector to divide with the current vector element-wise.
     */
    public void divideElementwiseEquals(Vector vector) {
        checkVectorDimensions(vector);
        for (int i = 0; i < dimension; i++) {
            array[i] /= vector.array[i];
        }
    }
    //endregion

    /**
     * Multiplies the current vector with a scalar and returns the result in a new vector.
     *
     * @param   scalar  The scalar with which to multiply the current vector.
     * @return          A new vector holding the result of the multiplication.
     */
    public Vector multiply(double scalar) {
        Vector resultVector = new Vector(dimension);
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < dimension; i++) {
            resultVectorArray[i] = array[i] * scalar;
        }
        return resultVector;
    }

    /**
     * Multiplies the current vector with a scalar and replaces the current vector with the result.
     *
     * @param   scalar  The scalar with which to multiply the current vector.
     */
    public void multiplyEquals(double scalar) {
        for (int i = 0; i < dimension; i++) {
            array[i] *= scalar;
        }
    }

    /**
     * Divides the current vector with a scalar and returns the result in a new vector.
     *
     * @param   scalar  The scalar with which to divide the current vector.
     * @return          A new vector holding the result of the division.
     */
    public Vector divide(double scalar) {
        Vector resultVector = new Vector(dimension);
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < dimension; i++) {
            resultVectorArray[i] = array[i] / scalar;
        }
        return resultVector;
    }

    /**
     * Divides the current vector with a scalar and replaces the current vector with the result.
     *
     * @param   scalar  The scalar with which to divide the current vector.
     */
    public void divideEquals(double scalar) {
        for (int i = 0; i < dimension; i++) {
            array[i] /= scalar;
        }
    }

    /**
     * Computes the inner product (also known as the dot product) between the current vector and another vector.
     *
     * @param   vector  The vector used to compute the inner product with the current vector.
     * @return          The resulting inner product value.
     */
    public double innerProduct(Vector vector) {
        checkVectorDimensions(vector);
        double dotProduct = 0;
        for (int i = 0; i < dimension; i++) {
            dotProduct += array[i] * vector.getElement(i);
        }
        return dotProduct;
    }

    /**
     * Computes the outer product between the current vector and another vector and returns the result in a new matrix.
     *
     * @param   vector  The vector used to compute the outer product with the current vector.
     * @return          A new matrix containing the result of the outer product operation.
     */
    public Matrix outerProduct(Vector vector) {
        checkVectorDimensions(vector);
        Matrix resultMatrix = new Matrix(dimension, dimension);
        double[][] resultMatrixArray = resultMatrix.getArray();
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                resultMatrixArray[i][j] = array[i] * vector.getElement(j);
            }
        }
        return resultMatrix;
    }

    /**
     * Computes the product of this vector with a matrix and returns the result in a new vector.
     *
     * @param   matrix  The matrix with which to multiply the current vector.
     * @return          A new vector holding the result of the multiplication.
     */
    public Vector multiply(Matrix matrix) {
        if (matrix.getRowDimension() != dimension) {
            throw new IllegalArgumentException(
                    "The row dimension of the matrix must agree with the dimension of the vector."
            );
        }
        Vector resultVector = new Vector(matrix.getColumnDimension());
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < dimension; j++) {
                resultVectorArray[i] += array[j] * matrix.getElement(i, j);
            }
        }
        return resultVector;
    }

    //region Special Matrix "Constructors"
    /**
     * Constructs and returns a vector with the provided dimension, filled with ones.
     *
     * @param   dimension   The dimension of the vector.
     * @return              A vector with the provided dimension, filled with ones.
     */
    public static Vector generateOnesVector(int dimension) {
        Vector onesVector = new Vector(dimension);
        double[] onesVectorArray = onesVector.getArray();
        for (int i = 0; i < dimension; i++) {
            onesVectorArray[i] = 1.0;
        }
        return onesVector;
    }

    /**
     * Constructs and returns a vector with the provided dimension, filled with random values ranging from {@code 0.0}
     * to {@code 1.0}.
     *
     * @param   dimension   The dimension of the random vector.
     * @return              A vector with the provided dimension, filled with random values ranging from {@code 0.0} to
     * {@code 1.0}.
     */
    public static Vector generateRandomVector(int dimension) {
        Vector randomVector = new Vector(dimension);
        double[] randomVectorArray = randomVector.getArray();
        for (int i = 0; i < dimension; i++) {
            randomVectorArray[i] = Math.random();
        }
        return randomVector;
    }
    //endregion

    /**
     * Checks whether the provided vector has the same dimension as this vector. If the dimensions of the two vectors do
     * not agree an exception is thrown.
     *
     * @param   vector  The vector whose dimension to check.
     *
     * @exception   IllegalArgumentException    Vector dimensions must agree.
     */
    private void checkVectorDimensions(Vector vector) {
        if (vector.dimension != dimension) {
            throw new IllegalArgumentException("Vector dimensions must agree.");
        }
    }
}
