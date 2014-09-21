package org.platanios.learn.math.matrix;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.function.Function;

/**
 * Implements a class representing sparse vectors and supporting operations related to them. The sparse vectors are
 * stored in an internal hash map.
 * TODO: Add toDenseVector() method (or appropriate constructors).
 *
 * @author Emmanouil Antonios Platanios
 */
public class SparseVector extends Vector {
    /** The smallest value allowed in this vector. Values smaller than this are assumed to be equal to zero. */
    private final double epsilon = Math.sqrt(Double.MIN_VALUE);
    /** The size which the internal hash map uses as its initial capacity. */
    private final int initialSize = 128;
    /** The size of the vector. */
    private final int size;

    /** Hash map for internal storage of the vector elements. */
    private HashMap<Integer, Double> hashMap;

    /**
     * Constructs a sparse vector of the given size and fills it with zeros.
     *
     * @param   size    The size of the vector.
     */
    protected SparseVector(int size) {
        this.size = size;
        hashMap = new HashMap<>(initialSize);
    }

    /**
     * Constructs a sparse vector of the given size from a hash map.
     *
     * @param   size        The size of the vector.
     * @param   elements    Hash map containing the indexes of elements as keys and the values of the corresponding
     *                      elements as values.
     */
    protected SparseVector(int size, HashMap<Integer, Double> elements) {
        this.size = size;
        hashMap = new HashMap<>(initialSize);
        hashMap.putAll(elements);
    }

    /** {@inheritDoc} */
    @Override
    public VectorType type() {
        return VectorType.SPARSE;
    }

    /** {@inheritDoc} */
    @Override
    public Vector copy() {
        return new SparseVector(size, hashMap);
    }

    /** {@inheritDoc} */
    @Override
    public double[] getDenseArray() {
        double[] resultArray = new double[size];
        for (int index : hashMap.keySet()) {
            resultArray[index] = hashMap.get(index);
        }
        return resultArray;
    }

    /** {@inheritDoc} */
    @Override
    public int size() {
        return size;
    }

    /** {@inheritDoc} */
    @Override
    public double get(int index) {
        if (index < 0 || index >= size) {
            throw new IllegalArgumentException(
                    "The provided index must be between 0 (inclusive) and the size of the vector (exclusive)."
            );
        }
        Double value = hashMap.get(index);
        return value == null ? 0 : value;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector get(int initialIndex, int finalIndex) {
        if (initialIndex < 0 || initialIndex >= size || finalIndex < 0 || finalIndex >= size) {
            throw new IllegalArgumentException(
                    "The provided indexes must be between 0 (inclusive) and the size of the vector (exclusive)."
            );
        }
        if (initialIndex > finalIndex) {
            throw new IllegalArgumentException("The initial index must be smaller or equal to the final index.");
        }
        SparseVector resultVector = new SparseVector(finalIndex - initialIndex + 1);
        for (int i = initialIndex; i <= finalIndex; i++) {
            Double value = hashMap.get(i);
            resultVector.set(i - initialIndex, value == null ? 0 : value);
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector get(int[] indexes) {
        SparseVector resultVector = new SparseVector(indexes.length);
        for (int i = 0; i < indexes.length; i++) {
            if (i < 0 || i >= size) {
                throw new IllegalArgumentException(
                        "The provided indexes must be between 0 (inclusive) and the size of the vector (exclusive)."
                );
            }
            Double value = hashMap.get(indexes[i]);
            resultVector.set(i, value == null ? 0 : value);
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public void set(int index, double value) {
        if (index < 0 || index >= size) {
            throw new IllegalArgumentException(
                    "The provided index must be between 0 (inclusive) and the size of the vector (exclusive)."
            );
        }
        if (Math.abs(value) >= epsilon) {
            hashMap.put(index, value);
        } else {
            hashMap.remove(index);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void set(int initialIndex, int finalIndex, Vector vector) {
        if (initialIndex < 0 || initialIndex >= size || finalIndex < 0 || finalIndex >= size) {
            throw new IllegalArgumentException(
                    "The provided indexes must be between 0 (inclusive) and the size of the vector (exclusive)."
            );
        }
        if (initialIndex > finalIndex) {
            throw new IllegalArgumentException("The initial index must be smaller or equal to the final index");
        }
        for (int i = initialIndex; i <= finalIndex; i++) {
            double value = vector.get(i - initialIndex);
            if (Math.abs(value) >= epsilon) {
                hashMap.put(i, value);
            } else {
                hashMap.remove(i);
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public void set(int[] indexes, Vector vector) {
        for (int i = 0; i < indexes.length; i++) {
            if (indexes[i] < 0 || indexes[i] >= size) {
                throw new IllegalArgumentException(
                        "The provided indexes must be between 0 (inclusive) and the size of the vector (exclusive)."
                );
            }
            double value = vector.get(i);
            if (Math.abs(value) >= epsilon) {
                hashMap.put(indexes[i], value);
            } else {
                hashMap.remove(indexes[i]);
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public void setAll(double value) {
        if (Math.abs(value) >= epsilon) {
            for (int i = 0; i < size; i++) {
                hashMap.put(i, value);
            }
        } else {
            hashMap = new HashMap<>(initialSize); // Use new hash map instead of the clear() method so that the memory is freed.
        }
    }

    /** {@inheritDoc} */
    @Override
    public double max() {
        return hashMap.values().stream().mapToDouble(Double::doubleValue).max().getAsDouble();
    }

    /** {@inheritDoc} */
    @Override
    public double min() {
        return hashMap.values().stream().mapToDouble(Double::doubleValue).min().getAsDouble();
    }

    /** {@inheritDoc} */
    @Override
    public double sum() {
        return hashMap.values().stream().mapToDouble(Double::doubleValue).sum();
    }

    /** {@inheritDoc} */
    @Override
    public double norm(VectorNorm normType) {
        return normType.compute(hashMap.values());
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector map(Function<Double, Double> function) {
        SparseVector resultVector = new SparseVector(size, hashMap); // TODO: What happens when the function is applied to zeros?
        for (int key : resultVector.hashMap.keySet()) {
            resultVector.set(key, function.apply(resultVector.get(key)));
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector add(double scalar) {
        SparseVector resultVector = new SparseVector(size, hashMap);
        for (int i = 0; i < size; i++) {
            resultVector.set(i, this.get(i) + scalar);
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector addInPlace(double scalar) {
        for (int i = 0; i < size; i++) {
            this.set(i, this.get(i) + scalar);
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector add(Vector vector) {
        checkVectorSize(vector);
        SparseVector resultVector = new SparseVector(size, hashMap);
        if (vector.type() != VectorType.SPARSE) {
            for (int i = 0; i < size; i++) {
                resultVector.set(i, this.get(i) + vector.get(i));
            }
        } else {
            List<Integer> keysUnion = new ArrayList<>(hashMap.keySet());
            keysUnion.addAll(((SparseVector) vector).hashMap.keySet());
            for (int key : keysUnion) {
                resultVector.set(key, this.get(key) + vector.get(key));
            }
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector addInPlace(Vector vector) {
        checkVectorSize(vector);
        if (vector.type() != VectorType.SPARSE) {
            for (int i = 0; i < size; i++) {
                this.set(i, this.get(i) + vector.get(i));
            }
        } else {
            List<Integer> keysUnion = new ArrayList<>(hashMap.keySet());
            keysUnion.addAll(((SparseVector) vector).hashMap.keySet());
            for (int key : keysUnion) {
                this.set(key, this.get(key) + vector.get(key));
            }
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector sub(double scalar) {
        SparseVector resultVector = new SparseVector(size, hashMap);
        for (int i = 0; i < size; i++) {
            resultVector.set(i, this.get(i) - scalar);
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector subInPlace(double scalar) {
        for (int i = 0; i < size; i++) {
            this.set(i, this.get(i) - scalar);
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector sub(Vector vector) {
        checkVectorSize(vector);
        SparseVector resultVector = new SparseVector(size, hashMap);
        if (vector.type() != VectorType.SPARSE) {
            for (int i = 0; i < size; i++) {
                resultVector.set(i, this.get(i) - vector.get(i));
            }
        } else {
            List<Integer> keysUnion = new ArrayList<>(hashMap.keySet());
            keysUnion.addAll(((SparseVector) vector).hashMap.keySet());
            for (int key : keysUnion) {
                resultVector.set(key, this.get(key) - vector.get(key));
            }
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector subInPlace(Vector vector) {
        checkVectorSize(vector);
        if (vector.type() != VectorType.SPARSE) {
            for (int i = 0; i < size; i++) {
                this.set(i, this.get(i) - vector.get(i));
            }
        } else {
            List<Integer> keysUnion = new ArrayList<>(hashMap.keySet());
            keysUnion.addAll(((SparseVector) vector).hashMap.keySet());
            for (int key : keysUnion) {
                this.set(key, this.get(key) - vector.get(key));
            }
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector multElementwise(Vector vector) {
        checkVectorSize(vector);
        SparseVector resultVector = new SparseVector(size, hashMap);
        if (vector.type() != VectorType.SPARSE) {
            for (int key : hashMap.keySet()) {
                resultVector.set(key, hashMap.get(key) * vector.get(key));
            }
        } else {
            if (size <= vector.size()) {
                hashMap.keySet().stream().filter(((SparseVector) vector).hashMap::containsKey).forEach(
                        key -> resultVector.set(key, hashMap.get(key) * vector.get(key))
                );
            } else {
                ((SparseVector) vector).hashMap.keySet().stream().filter(hashMap::containsKey).forEach(
                        key -> resultVector.set(key, hashMap.get(key) * vector.get(key))
                );
            }
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector multElementwiseInPlace(Vector vector) {
        checkVectorSize(vector);
        if (vector.type() != VectorType.SPARSE) {
            for (int key : hashMap.keySet()) {
                this.set(key, hashMap.get(key) * vector.get(key));
            }
        } else {
            if (size <= vector.size()) {
                hashMap.keySet().stream().filter(((SparseVector) vector).hashMap::containsKey).forEach(
                        key -> this.set(key, hashMap.get(key) * vector.get(key))
                );
            } else {
                ((SparseVector) vector).hashMap.keySet().stream().filter(hashMap::containsKey).forEach(
                        key -> this.set(key, hashMap.get(key) * vector.get(key))
                );
            }
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector divElementwise(Vector vector) {
        checkVectorSize(vector); // TODO: Need to check whether any element of vector is zero.
        SparseVector resultVector = new SparseVector(size, hashMap);
        for (int key : hashMap.keySet()) {
            resultVector.set(key, hashMap.get(key) / vector.get(key));
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector divElementwiseInPlace(Vector vector) {
        checkVectorSize(vector); // TODO: Need to check whether any element of vector is zero.
        for (int key : hashMap.keySet()) {
            this.set(key, hashMap.get(key) / vector.get(key));
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector mult(double scalar) {
        SparseVector resultVector = new SparseVector(size, hashMap);
        for (int key : hashMap.keySet()) {
            resultVector.set(key, hashMap.get(key) * scalar);
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector multInPlace(double scalar) {
        for (int key : hashMap.keySet()) {
            this.set(key, hashMap.get(key) * scalar);
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector div(double scalar) {
        SparseVector resultVector = new SparseVector(size, hashMap);
        for (int key : hashMap.keySet()) {
            resultVector.set(key, hashMap.get(key) / scalar);
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector divInPlace(double scalar) {
        for (int key : hashMap.keySet()) {
            this.set(key, hashMap.get(key) / scalar);
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector saxpy(double scalar, Vector vector) {
        checkVectorSize(vector);
        SparseVector resultVector = new SparseVector(size, hashMap);
        if (vector.type() != VectorType.SPARSE) {
            for (int i = 0; i < size; i++) {
                resultVector.set(i, this.get(i) + scalar * vector.get(i));
            }
        } else {
            List<Integer> keysUnion = new ArrayList<>(hashMap.keySet());
            keysUnion.addAll(((SparseVector) vector).hashMap.keySet());
            for (int key : keysUnion) {
                resultVector.set(key, this.get(key) + scalar * vector.get(key));
            }
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector saxpyInPlace(double scalar, Vector vector) {
        checkVectorSize(vector);
        if (vector.type() != VectorType.SPARSE) {
            for (int i = 0; i < size; i++) {
                this.set(i, this.get(i) + scalar * vector.get(i));
            }
        } else {
            List<Integer> keysUnion = new ArrayList<>(hashMap.keySet());
            keysUnion.addAll(((SparseVector) vector).hashMap.keySet());
            for (int key : keysUnion) {
                this.set(key, this.get(key) + scalar * vector.get(key));
            }
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public double inner(Vector vector) {
        checkVectorSize(vector);
        if (vector.type() != VectorType.SPARSE) {
            double result = 0;
            for (int key : hashMap.keySet()) {
                result += hashMap.get(key) * vector.get(key);
            }
            return result;
        } else {
            final double[] result = { 0 }; // Use an array because all variables in a lambda formula must be final.
            if (size <= vector.size()) {
                hashMap.keySet().stream().filter(((SparseVector) vector).hashMap::containsKey).forEach(
                        key -> result[0] += hashMap.get(key) * vector.get(key)
                );
            } else {
                ((SparseVector) vector).hashMap.keySet().stream().filter(hashMap::containsKey).forEach(
                        key -> result[0] += hashMap.get(key) * vector.get(key)
                );
            }
            return result[0];
        }
    }

    /** {@inheritDoc} */
    @Override
    public Matrix outer(Vector vector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Vector gaxpy(Matrix matrix, Vector vector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Vector gaxpyInPlace(Matrix matrix, Vector vector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Vector transMult(Matrix matrix) {
        return null;
    }
}
