package org.platanios.learn.math.matrix;

import java.util.function.Function;

/**
 * Implements a class representing dense vectors and supporting operations related to them. The dense vectors are stored
 * in an internal one-dimensional array.
 *
 * @author Emmanouil Antonios Platanios
 */
public class DenseVector extends Vector {
    /** The dimension of the vector. */
    private final int dimension;

    /** Array for internal storage of the vector elements. */
    private double[] array;

    /**
     * Constructs a vector with the given dimension and fills it with zeros.
     *
     * @param   dimension   The dimension of the vector.
     */
    public DenseVector(int dimension) {
        this.dimension = dimension;
        array = new double[dimension];
    }

    /**
     * Constructs a vector with the given dimensions and fills it with the given value.
     *
     * @param   dimension   The dimension of the vector.
     * @param   value       The value with which to fill the vector.
     */
    public DenseVector(int dimension, double value) {
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
    public DenseVector(double[] array) {
        dimension = array.length;
        this.array = array;
    }

    /** {@inheritDoc} */
    @Override
    public VectorType type() {
        return VectorType.DENSE;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector copy() {
        DenseVector resultVector = new DenseVector(dimension);
        double[] resultVectorArray = resultVector.getArray();
        System.arraycopy(array, 0, resultVectorArray, 0, dimension);
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public double[] getDenseArray() {
        double[] resultArray = new double[dimension];
        System.arraycopy(array, 0, resultArray, 0, dimension);
        return resultArray;
    }

    /** {@inheritDoc} */
    @Override
    public int size() {
        return dimension;
    }

    /** {@inheritDoc} */
    @Override
    public double get(int index) {
        try {
            return array[index];
        } catch(ArrayIndexOutOfBoundsException e) {
            throw new ArrayIndexOutOfBoundsException("The provided index is out of bounds.");
        }
    }

    /** {@inheritDoc} */
    @Override
    public Vector get(int initialIndex, int finalIndex) {
        DenseVector resultVector = new DenseVector(finalIndex - initialIndex + 1);
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

    /** {@inheritDoc} */
    @Override
    public Vector get(int[] indexes) {
        DenseVector resultVector = new DenseVector(indexes.length);
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

    /** {@inheritDoc} */
    @Override
    public void set(int index, double value) {
        try {
            array[index] = value;
        } catch(ArrayIndexOutOfBoundsException e) {
            throw new ArrayIndexOutOfBoundsException("The provided index is out of bounds.");
        }
    }

    /** {@inheritDoc} */
    @Override
    public void set(int initialIndex, int finalIndex, Vector vector) {
        try {
            for (int i = initialIndex; i <= finalIndex; i++) {
                array[i] = vector.get(i - initialIndex);
            }
        } catch(ArrayIndexOutOfBoundsException e) {
            throw new ArrayIndexOutOfBoundsException(
                    "Some or all of the provided vector indexes are out of bounds."
            );
        }
    }

    /** {@inheritDoc} */
    @Override
    public void set(int[] indexes, Vector vector) {
        try {
            for (int i = 0; i < indexes.length; i++) {
                array[indexes[i]] = vector.get(i);
            }
        } catch(ArrayIndexOutOfBoundsException e) {
            throw new ArrayIndexOutOfBoundsException(
                    "Some or all of the provided vector indexes are out of bounds."
            );
        }
    }

    /** {@inheritDoc} */
    @Override
    public void setAll(double value) {
        for (int i = 0; i < dimension; i++) {
            array[i] = value;
        }
    }

    /** {@inheritDoc} */
    @Override
    public double max() {
        double maxValue = array[0];
        for (int i = 1; i < dimension; i++) {
            maxValue = Math.max(maxValue, array[i]);
        }
        return maxValue;
    }

    /** {@inheritDoc} */
    @Override
    public double min() {
        double minValue = array[0];
        for (int i = 1; i < dimension; i++) {
            minValue = Math.min(minValue, array[i]);
        }
        return minValue;
    }

    /** {@inheritDoc} */
    @Override
    public double sum() {
        double sum = array[0];
        for (int i = 1; i < dimension; i++) {
            sum += array[i];
        }
        return sum;
    }

    /** {@inheritDoc} */
    @Override
    public double norm(VectorNorm normType) {
        return normType.compute(array);
    }

    /** {@inheritDoc} */
    @Override
    public Vector map(Function<Double, Double> function) {
        DenseVector resultVector = new DenseVector(dimension);
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < dimension; i++) {
            resultVectorArray[i] = function.apply(array[i]);
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public Vector add(double scalar) {
        DenseVector resultVector = new DenseVector(dimension);
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < dimension; i++) {
            resultVectorArray[i] = array[i] + scalar;
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public Vector add(Vector vector) {
        checkVectorDimensions(vector);
        DenseVector resultVector = new DenseVector(dimension);
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < dimension; i++) {
            resultVectorArray[i] = array[i] + vector.get(i);
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public Vector addEquals(double scalar) {
        for (int i = 0; i < dimension; i++) {
            array[i] += scalar;
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public Vector addEquals(Vector vector) {
        checkVectorDimensions(vector);
        for (int i = 0; i < dimension; i++) {
            array[i] += vector.get(i);
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public Vector subtract(double scalar) {
        DenseVector resultVector = new DenseVector(dimension);
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < dimension; i++) {
            resultVectorArray[i] = array[i] - scalar;
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public Vector subtract(Vector vector) {
        checkVectorDimensions(vector);
        DenseVector resultVector = new DenseVector(dimension);
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < dimension; i++) {
            resultVectorArray[i] = array[i] - vector.get(i);
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public Vector subtractEquals(double scalar) {
        for (int i = 0; i < dimension; i++) {
            array[i] -= scalar;
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public Vector subtractEquals(Vector vector) {
        checkVectorDimensions(vector);
        for (int i = 0; i < dimension; i++) {
            array[i] -= vector.get(i);
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public Vector multiplyElementwise(Vector vector) {
        checkVectorDimensions(vector);
        DenseVector resultVector = new DenseVector(dimension);
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < dimension; i++) {
            resultVectorArray[i] = array[i] * vector.get(i);
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public Vector multiplyElementwiseEquals(Vector vector) {
        checkVectorDimensions(vector);
        for (int i = 0; i < dimension; i++) {
            array[i] *= vector.get(i);
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public Vector divideElementwise(Vector vector) {
        checkVectorDimensions(vector);
        DenseVector resultVector = new DenseVector(dimension);
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < dimension; i++) {
            resultVectorArray[i] = array[i] / vector.get(i);
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public Vector divideElementwiseEquals(Vector vector) {
        checkVectorDimensions(vector);
        for (int i = 0; i < dimension; i++) {
            array[i] /= vector.get(i);
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public Vector multiply(double scalar) {
        DenseVector resultVector = new DenseVector(dimension);
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < dimension; i++) {
            resultVectorArray[i] = array[i] * scalar;
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public Vector multiplyEquals(double scalar) {
        for (int i = 0; i < dimension; i++) {
            array[i] *= scalar;
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public Vector divide(double scalar) {
        DenseVector resultVector = new DenseVector(dimension);
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < dimension; i++) {
            resultVectorArray[i] = array[i] / scalar;
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public Vector divideEquals(double scalar) {
        for (int i = 0; i < dimension; i++) {
            array[i] /= scalar;
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public double inner(Vector vector) {
        checkVectorDimensions(vector);
        double dotProduct = 0;
        for (int i = 0; i < dimension; i++) {
            dotProduct += array[i] * vector.get(i);
        }
        return dotProduct;
    }

    /** {@inheritDoc} */
    @Override
    public Matrix outer(Vector vector) {
        checkVectorDimensions(vector);
        Matrix resultMatrix = new Matrix(dimension, dimension);
        double[][] resultMatrixArray = resultMatrix.getArray();
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                resultMatrixArray[i][j] = array[i] * vector.get(j);
            }
        }
        return resultMatrix;
    }

    /** {@inheritDoc} */
    @Override
    public Vector multiply(Matrix matrix) {
        if (matrix.getRowDimension() != dimension) {
            throw new IllegalArgumentException(
                    "The row dimension of the matrix must agree with the dimension of the vector."
            );
        }
        DenseVector resultVector = new DenseVector(matrix.getColumnDimension());
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < dimension; j++) {
                resultVectorArray[i] += array[j] * matrix.getElement(i, j);
            }
        }
        return resultVector;
    }

    /**
     * Constructs and returns a vector with the provided dimension, filled with ones.
     *
     * @param   dimension   The dimension of the vector.
     * @return              A vector with the provided dimension, filled with ones.
     */
    public static Vector generateOnesVector(int dimension) {
        DenseVector onesVector = new DenseVector(dimension);
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
        DenseVector randomVector = new DenseVector(dimension);
        double[] randomVectorArray = randomVector.getArray();
        for (int i = 0; i < dimension; i++) {
            randomVectorArray[i] = Math.random();
        }
        return randomVector;
    }

    /**
     * Gets a pointer to the internal one-dimensional array.
     *
     * @return  A pointer to the internal one-dimensional array.
     */
    private double[] getArray() {
        return array;
    }

    /**
     * Checks whether the provided vector has the same dimension as this vector. If the dimensions of the two vectors do
     * not agree an exception is thrown.
     *
     * @param   vector  The vector whose dimension to check.
     *
     * @exception   IllegalArgumentException    Vector dimensions must agree.
     */
    private void checkVectorDimensions(Vector vector) {
        if (vector.size() != dimension) {
            throw new IllegalArgumentException("Vector dimensions must agree.");
        }
    }
}
