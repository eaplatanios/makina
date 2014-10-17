package org.platanios.learn.math.matrix;

import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Arrays;
import java.util.Map;
import java.util.TreeMap;
import java.util.function.Function;

/**
 * Implements a class representing sparse vectors and supporting operations related to them. The sparse vectors are
 * stored in an internal hash map.
 * TODO: Add toDenseVector() method (or appropriate constructors).
 * TODO: Make this implementation faster by iterating over key-value pairs instead of iterating over keys and then retrieving values.
 * TODO: Make unsafe versions of all the methods.
 *
 * Performs binary search to retrieve elements within the vector.
 *
 * @author Emmanouil Antonios Platanios
 */
public class SparseVector extends Vector {
    /** The size of the vector. */
    private final int size;

    private int numberOfNonzeroEntries;
    /** Integer array for internal storage of the indexes of the non-zero vector elements. This array is always ordered
     * and is "parallel" to the {@link #values} array. */
    private int[] indexes;
    /** Double array for internal storage of the values of the non-zero vector elements. This array is always "parallel"
     * to the {@link #indexes} array. */
    private double[] values;

    public static class Builder {
        private int size;
        private int numberOfNonzeroEntries;
        private boolean usingMap = false;
        private int[] indexes;
        private double[] values;
        private Map<Integer, Double> vectorElements;

        public Builder(int size) {
            this.size = size;
            numberOfNonzeroEntries = 0;
            indexes = new int[0];
            values = new double[0];
        }

        public Builder(int size, Map<Integer, Double> vectorElements) {
            this.size = size;
            numberOfNonzeroEntries = vectorElements.size();
            usingMap = true;
            this.vectorElements = new TreeMap<>();
            this.vectorElements.putAll(vectorElements);
        }

        public Builder(int size, int[] indexes, double[] values) {
            if (indexes.length != values.length)
                throw new IllegalArgumentException("The indexes array and the values array must have the same length");

            this.size = size;
            numberOfNonzeroEntries = indexes.length;
            this.indexes = Arrays.copyOf(indexes, indexes.length);
            this.values = Arrays.copyOf(values, values.length);
        }

        /**
         * Constructs a sparse vector from the contents of the provided input stream. Note that the contents of the stream
         * must have been written using the {@link #writeToStream(java.io.ObjectOutputStream)} function of this class in
         * order to be compatible with this constructor. If the contents are not compatible, then an
         * {@link java.io.IOException} might be thrown, or the constructed vector might be corrupted in some way.
         *
         * The indexes must be ordered.
         *
         * @param   inputStream The input stream to read the contents of this vector from.
         * @throws  IOException
         */
        public Builder(ObjectInputStream inputStream) throws IOException {
            size = inputStream.readInt();
            numberOfNonzeroEntries = inputStream.readInt();
            indexes = new int[numberOfNonzeroEntries];
            values = new double[numberOfNonzeroEntries];
            for (int i = 0; i < numberOfNonzeroEntries; i++) {
                indexes[i] = inputStream.readInt();
                values[i] = inputStream.readDouble();
            }
        }

        public SparseVector build() {
            return new SparseVector(this);
        }
    }

    private SparseVector(Builder builder) {
        size = builder.size;
        numberOfNonzeroEntries = builder.numberOfNonzeroEntries;
        if (builder.usingMap) {
            indexes = new int[numberOfNonzeroEntries];
            values = new double[numberOfNonzeroEntries];
            int i = 0;
            for (int key : builder.vectorElements.keySet()) {
                indexes[i] = key;
                values[i] = builder.vectorElements.get(key);
                i++;
            }
        } else {
            indexes = builder.indexes;
            values = builder.values;
        }
    }

    /** {@inheritDoc} */
    @Override
    public VectorType type() {
        return VectorType.SPARSE;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector copy() {
        return new Builder(size, indexes, values).build();
    }

    /** {@inheritDoc} */
    @Override
    public double[] getDenseArray() {
        double[] resultArray = new double[size];
        for (int i = 0; i < numberOfNonzeroEntries; i++)
            resultArray[indexes[i]] = values[i];
        return resultArray;
    }

    /** {@inheritDoc} */
    @Override
    public int size() {
        return size;
    }

    /** {@inheritDoc} */
    @Override
    public int cardinality() {
        return numberOfNonzeroEntries;
    }

    /** {@inheritDoc} */
    @Override
    public double get(int index) {
        if (index < 0 || index >= size) {
            throw new IllegalArgumentException(
                    "The provided index must be between 0 (inclusive) and the size of the vector (exclusive)."
            );
        }
        int valueIndex = Arrays.binarySearch(indexes, index);
        if (valueIndex >= 0) {
            return values[valueIndex];
        } else {
            return 0;
        }
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
        int startIndex = Arrays.binarySearch(indexes, initialIndex);
        if (startIndex < 0) {
            startIndex = -startIndex - 1;
            if (startIndex < 0)
                startIndex = 0;
        }
        int endIndex = Arrays.binarySearch(indexes, finalIndex);
        if (endIndex < 0) {
            endIndex = -endIndex - 2;
        }
        return new Builder(finalIndex - initialIndex + 1,
                           Arrays.copyOfRange(indexes, startIndex, endIndex + 1),
                           Arrays.copyOfRange(values, startIndex, endIndex + 1)).build();
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector get(int[] indexes) { // TODO: It may be possible to make this method faster.
        Map<Integer, Double> elements = new TreeMap<>();
        for (int i = 0; i < indexes.length; i++) {
            if (i < 0 || i >= size) {
                throw new IllegalArgumentException(
                        "The provided indexes must be between 0 (inclusive) and the size of the vector (exclusive)."
                );
            }
            elements.put(i, get(indexes[i]));
        }
        return new Builder(indexes.length, elements).build();
    }

    /** {@inheritDoc} */
    @Override
    public void set(int index, double value) { // TODO: Not working correctly for some reason.
        if (index < 0 || index >= size) {
            throw new IllegalArgumentException(
                    "The provided index must be between 0 (inclusive) and the size of the vector (exclusive)."
            );
        }
        int foundIndex = Arrays.binarySearch(indexes, index);
        if (foundIndex >= 0) {
            values[indexes[foundIndex]] = value;
        } else {
            foundIndex = - foundIndex - 1;
            numberOfNonzeroEntries++;
            int[] newIndexes = new int[numberOfNonzeroEntries];
            double[] newValues = new double[numberOfNonzeroEntries];
            for (int i = 0; i < numberOfNonzeroEntries; i++) {
                if (i < foundIndex) {
                    newIndexes[i] = indexes[i];
                    newValues[i] = values[i];
                } else if (i == foundIndex) {
                    newIndexes[i] = index;
                    newValues[i] = value;
                } else {
                    newIndexes[i] = indexes[i - 1];
                    newValues[i] = values[i - 1];
                }
            }
            indexes = newIndexes;
            values = newValues;
        }
    }

    public void setFromZero(int index, double value) {
        if (index < 0 || index >= size) {
            throw new IllegalArgumentException(
                    "The provided index must be between 0 (inclusive) and the size of the vector (exclusive)."
            );
        }
        numberOfNonzeroEntries++;
        int[] newIndexes = new int[numberOfNonzeroEntries];
        double[] newValues = new double[numberOfNonzeroEntries];
        boolean passedIndex = false;
        for (int i = 0; i < numberOfNonzeroEntries; i++) {
            if (indexes[i] < index) {
                newIndexes[i] = indexes[i];
                newValues[i] = values[i];
            } else {
                if (!passedIndex) {
                    newIndexes[i] = index;
                    newValues[i] = value;
                    passedIndex = true;
                } else {
                    newIndexes[i] = indexes[i - 1];
                    newValues[i] = values[i - 1];
                }
            }
        }
        indexes = newIndexes;
        values = newValues;
    }

    // TODO: Create a compact() method to get rid of zero elements.

    /** {@inheritDoc} */
    @Override
    public void set(int initialIndex, int finalIndex, Vector vector) { // TODO: It may be possible to make this method faster.
        if (initialIndex < 0 || initialIndex >= size || finalIndex < 0 || finalIndex >= size) {
            throw new IllegalArgumentException(
                    "The provided indexes must be between 0 (inclusive) and the size of the vector (exclusive)."
            );
        }
        if (initialIndex > finalIndex) {
            throw new IllegalArgumentException("The initial index must be smaller or equal to the final index");
        }
        for (int i = initialIndex; i <= finalIndex; i++) {
            set(i, vector.get(i - initialIndex));
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
            set(indexes[i], vector.get(i));
        }
    }

    /** {@inheritDoc} */
    @Override
    public void setAll(double value) {
        if (Math.abs(value) >= epsilon) {
            numberOfNonzeroEntries = size;
            indexes = new int[numberOfNonzeroEntries];
            values = new double[numberOfNonzeroEntries];
            for (int i = 0; i < numberOfNonzeroEntries; i++) {
                indexes[i] = 0;
                values[i] = value;
            }
        } else {
            numberOfNonzeroEntries = 0;
            indexes = new int[0];
            values = new double[0];
        }
    }

    /** {@inheritDoc} */
    @Override
    public double max() {
        double maxValue = values[0];
        for (int i = 1; i < numberOfNonzeroEntries; i++) {
            maxValue = Math.max(maxValue, values[i]);
        }
        return maxValue;
    }

    /** {@inheritDoc} */
    @Override
    public double min() {
        double minValue = values[0];
        for (int i = 1; i < numberOfNonzeroEntries; i++) {
            minValue = Math.min(minValue, values[i]);
        }
        return minValue;
    }

    /** {@inheritDoc} */
    @Override
    public double sum() {
        double sum = values[0];
        for (int i = 1; i < numberOfNonzeroEntries; i++) {
            sum += values[i];
        }
        return sum;
    }

    /** {@inheritDoc} */
    @Override
    public double norm(VectorNorm normType) {
        return normType.compute(values);
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector map(Function<Double, Double> function) { // TODO: What happens when the function is applied to zeros? Maybe store the "zero" value somewhere and modify that.
        SparseVector resultVector = new Builder(size, indexes, values).build();
        for (int i = 0; i < numberOfNonzeroEntries; i++) {
            resultVector.values[i] = function.apply(resultVector.values[i]);
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector add(double scalar) {
        SparseVector resultVector = new Builder(size, indexes, values).build();
        for (int i = 0; i < numberOfNonzeroEntries; i++) {
            resultVector.values[i] += scalar;
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector addInPlace(double scalar) {
        for (int i = 0; i < numberOfNonzeroEntries; i++) {
            values[i] += scalar;
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector add(Vector vector) {
        checkVectorSize(vector);
        SparseVector resultVector;
        if (vector.type() == VectorType.SPARSE) {
            int[] newIndexes = new int[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            double[] newValues = new double[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            int currentIndex = 0;
            int vector1Index = 0;
            int vector2Index = 0;
            while (true) {
                if (numberOfNonzeroEntries > 0
                        && vector1Index < numberOfNonzeroEntries
                        && ((SparseVector) vector).numberOfNonzeroEntries > 0
                        && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                    if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                        newIndexes[currentIndex] = indexes[vector1Index];
                        newValues[currentIndex] = values[vector1Index];
                        currentIndex++;
                        vector1Index++;
                    } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                        newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                        newValues[currentIndex] = ((SparseVector) vector).values[vector2Index];
                        currentIndex++;
                        vector2Index++;
                    } else {
                        newIndexes[currentIndex] = indexes[vector1Index];
                        newValues[currentIndex] = values[vector1Index] + ((SparseVector) vector).values[vector2Index];
                        currentIndex++;
                        vector1Index++;
                        vector2Index++;
                    }
                } else if (numberOfNonzeroEntries > 0 && vector1Index < numberOfNonzeroEntries) {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] = values[vector1Index];
                    currentIndex++;
                    vector1Index++;
                } else if (((SparseVector) vector).numberOfNonzeroEntries > 0
                        && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                    newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                    newValues[currentIndex] = ((SparseVector) vector).values[vector2Index];
                    currentIndex++;
                    vector2Index++;
                } else {
                    break;
                }
            }
            resultVector = new Builder(size,
                                       Arrays.copyOfRange(newIndexes, 0, currentIndex),
                                       Arrays.copyOfRange(newValues, 0, currentIndex)).build();
        } else {
            throw new NotImplementedException();
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector addInPlace(Vector vector) {
        checkVectorSize(vector);
        if (vector.type() == VectorType.SPARSE) {
            int[] newIndexes = new int[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            double[] newValues = new double[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            int currentIndex = 0;
            int vector1Index = 0;
            int vector2Index = 0;
            while (true) {
                if (numberOfNonzeroEntries > 0
                        && vector1Index < numberOfNonzeroEntries
                        && ((SparseVector) vector).numberOfNonzeroEntries > 0
                        && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                    if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                        newIndexes[currentIndex] = indexes[vector1Index];
                        newValues[currentIndex] = values[vector1Index];
                        currentIndex++;
                        vector1Index++;
                    } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                        newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                        newValues[currentIndex] = ((SparseVector) vector).values[vector2Index];
                        currentIndex++;
                        vector2Index++;
                    } else {
                        newIndexes[currentIndex] = indexes[vector1Index];
                        newValues[currentIndex] = values[vector1Index] + ((SparseVector) vector).values[vector2Index];
                        currentIndex++;
                        vector1Index++;
                        vector2Index++;
                    }
                } else if (numberOfNonzeroEntries > 0 && vector1Index < numberOfNonzeroEntries) {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] = values[vector1Index];
                    currentIndex++;
                    vector1Index++;
                } else if (((SparseVector) vector).numberOfNonzeroEntries > 0
                        && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                    newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                    newValues[currentIndex] = ((SparseVector) vector).values[vector2Index];
                    currentIndex++;
                    vector2Index++;
                } else {
                    break;
                }
            }
            indexes = Arrays.copyOfRange(newIndexes, 0, currentIndex);
            values = Arrays.copyOfRange(newValues, 0, currentIndex);
            numberOfNonzeroEntries = currentIndex;
        } else {
            throw new NotImplementedException();
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector sub(double scalar) {
        SparseVector resultVector = new Builder(size, indexes, values).build();
        for (int i = 0; i < numberOfNonzeroEntries; i++) {
            resultVector.values[i] -= scalar;
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector subInPlace(double scalar) {
        for (int i = 0; i < numberOfNonzeroEntries; i++) {
            values[i] -= scalar;
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector sub(Vector vector) {
        checkVectorSize(vector);
        SparseVector resultVector;
        if (vector.type() == VectorType.SPARSE) {
            int[] newIndexes = new int[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            double[] newValues = new double[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            int currentIndex = 0;
            int vector1Index = 0;
            int vector2Index = 0;
            while (true) {
                if (numberOfNonzeroEntries > 0
                        && vector1Index < numberOfNonzeroEntries
                        && ((SparseVector) vector).numberOfNonzeroEntries > 0
                        && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                    if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                        newIndexes[currentIndex] = indexes[vector1Index];
                        newValues[currentIndex] = values[vector1Index];
                        currentIndex++;
                        vector1Index++;
                    } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                        newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                        newValues[currentIndex] = - ((SparseVector) vector).values[vector2Index];
                        currentIndex++;
                        vector2Index++;
                    } else {
                        newIndexes[currentIndex] = indexes[vector1Index];
                        newValues[currentIndex] = values[vector1Index] - ((SparseVector) vector).values[vector2Index];
                        currentIndex++;
                        vector1Index++;
                        vector2Index++;
                    }
                } else if (numberOfNonzeroEntries > 0 && vector1Index < numberOfNonzeroEntries) {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] = values[vector1Index];
                    currentIndex++;
                    vector1Index++;
                } else if (((SparseVector) vector).numberOfNonzeroEntries > 0
                        && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                    newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                    newValues[currentIndex] = - ((SparseVector) vector).values[vector2Index];
                    currentIndex++;
                    vector2Index++;
                } else {
                    break;
                }
            }
            resultVector = new Builder(size,
                                       Arrays.copyOfRange(newIndexes, 0, currentIndex),
                                       Arrays.copyOfRange(newValues, 0, currentIndex)).build();
        } else {
            throw new NotImplementedException();
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector subInPlace(Vector vector) {
        checkVectorSize(vector);
        if (vector.type() == VectorType.SPARSE) {
            int[] newIndexes = new int[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            double[] newValues = new double[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            int currentIndex = 0;
            int vector1Index = 0;
            int vector2Index = 0;
            while (true) {
                if (numberOfNonzeroEntries > 0
                        && vector1Index < numberOfNonzeroEntries
                        && ((SparseVector) vector).numberOfNonzeroEntries > 0
                        && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                    if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                        newIndexes[currentIndex] = indexes[vector1Index];
                        newValues[currentIndex] = values[vector1Index];
                        currentIndex++;
                        vector1Index++;
                    } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                        newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                        newValues[currentIndex] = - ((SparseVector) vector).values[vector2Index];
                        currentIndex++;
                        vector2Index++;
                    } else {
                        newIndexes[currentIndex] = indexes[vector1Index];
                        newValues[currentIndex] = values[vector1Index] - ((SparseVector) vector).values[vector2Index];
                        currentIndex++;
                        vector1Index++;
                        vector2Index++;
                    }
                } else if (numberOfNonzeroEntries > 0 && vector1Index < numberOfNonzeroEntries) {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] = values[vector1Index];
                    currentIndex++;
                    vector1Index++;
                } else if (((SparseVector) vector).numberOfNonzeroEntries > 0
                        && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                    newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                    newValues[currentIndex] = - ((SparseVector) vector).values[vector2Index];
                    currentIndex++;
                    vector2Index++;
                } else {
                    break;
                }
            }
            indexes = Arrays.copyOfRange(newIndexes, 0, currentIndex);
            values = Arrays.copyOfRange(newValues, 0, currentIndex);
            numberOfNonzeroEntries = currentIndex;
        } else {
            throw new NotImplementedException();
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector multElementwise(Vector vector) {
        checkVectorSize(vector);
        SparseVector resultVector;
        if (vector.type() == VectorType.SPARSE) {
            int[] newIndexes = new int[numberOfNonzeroEntries];
            double[] newValues = new double[numberOfNonzeroEntries];
            int currentIndex = 0;
            int vector1Index = 0;
            int vector2Index = 0;
            while (true) {
                if (numberOfNonzeroEntries > 0
                        && vector1Index < numberOfNonzeroEntries
                        && ((SparseVector) vector).numberOfNonzeroEntries > 0
                        && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                    if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                        vector1Index++;
                    } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                        vector2Index++;
                    } else {
                        newIndexes[currentIndex] = indexes[vector1Index];
                        newValues[currentIndex] = values[vector1Index] * ((SparseVector) vector).values[vector2Index];
                        currentIndex++;
                        vector1Index++;
                        vector2Index++;
                    }
                } else {
                    break;
                }
            }
            resultVector = new Builder(size,
                                       Arrays.copyOfRange(newIndexes, 0, currentIndex),
                                       Arrays.copyOfRange(newValues, 0, currentIndex)).build();
        } else {
            throw new NotImplementedException();
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector multElementwiseInPlace(Vector vector) {
        checkVectorSize(vector);
        if (vector.type() == VectorType.SPARSE) {
            int[] newIndexes = new int[numberOfNonzeroEntries];
            double[] newValues = new double[numberOfNonzeroEntries];
            int currentIndex = 0;
            int vector1Index = 0;
            int vector2Index = 0;
            while (true) {
                if (numberOfNonzeroEntries > 0
                        && vector1Index < numberOfNonzeroEntries
                        && ((SparseVector) vector).numberOfNonzeroEntries > 0
                        && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                    if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                        vector1Index++;
                    } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                        vector2Index++;
                    } else {
                        newIndexes[currentIndex] = indexes[vector1Index];
                        newValues[currentIndex] = values[vector1Index] * ((SparseVector) vector).values[vector2Index];
                        currentIndex++;
                        vector1Index++;
                        vector2Index++;
                    }
                } else {
                    break;
                }
            }
            indexes = Arrays.copyOfRange(newIndexes, 0, currentIndex);
            values = Arrays.copyOfRange(newValues, 0, currentIndex);
            numberOfNonzeroEntries = currentIndex;
        } else {
            throw new NotImplementedException();
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector divElementwise(Vector vector) {
        checkVectorSize(vector);
        SparseVector resultVector;
        if (vector.type() == VectorType.SPARSE) {
            int[] newIndexes = new int[numberOfNonzeroEntries];
            double[] newValues = new double[numberOfNonzeroEntries];
            int currentIndex = 0;
            int vector1Index = 0;
            int vector2Index = 0;
            while (true) {
                if (numberOfNonzeroEntries > 0
                        && vector1Index < numberOfNonzeroEntries
                        && ((SparseVector) vector).numberOfNonzeroEntries > 0
                        && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                    if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                        vector1Index++;
                    } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                        vector2Index++;
                    } else {
                        newIndexes[currentIndex] = indexes[vector1Index];
                        newValues[currentIndex] = values[vector1Index] / ((SparseVector) vector).values[vector2Index];
                        currentIndex++;
                        vector1Index++;
                        vector2Index++;
                    }
                } else {
                    break;
                }
            }
            resultVector = new Builder(size,
                                       Arrays.copyOfRange(newIndexes, 0, currentIndex),
                                       Arrays.copyOfRange(newValues, 0, currentIndex)).build();
        } else {
            throw new NotImplementedException();
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector divElementwiseInPlace(Vector vector) {
        checkVectorSize(vector);
        if (vector.type() == VectorType.SPARSE) {
            int[] newIndexes = new int[numberOfNonzeroEntries];
            double[] newValues = new double[numberOfNonzeroEntries];
            int currentIndex = 0;
            int vector1Index = 0;
            int vector2Index = 0;
            while (true) {
                if (numberOfNonzeroEntries > 0
                        && vector1Index < numberOfNonzeroEntries
                        && ((SparseVector) vector).numberOfNonzeroEntries > 0
                        && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                    if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                        vector1Index++;
                    } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                        vector2Index++;
                    } else {
                        newIndexes[currentIndex] = indexes[vector1Index];
                        newValues[currentIndex] = values[vector1Index] / ((SparseVector) vector).values[vector2Index];
                        currentIndex++;
                        vector1Index++;
                        vector2Index++;
                    }
                } else {
                    break;
                }
            }
            indexes = Arrays.copyOfRange(newIndexes, 0, currentIndex);
            values = Arrays.copyOfRange(newValues, 0, currentIndex);
            numberOfNonzeroEntries = currentIndex;
        } else {
            throw new NotImplementedException();
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector mult(double scalar) {
        SparseVector resultVector = new Builder(size, indexes, values).build();
        for (int i = 0; i < numberOfNonzeroEntries; i++) {
            resultVector.values[i] *= scalar;
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector multInPlace(double scalar) {
        for (int i = 0; i < numberOfNonzeroEntries; i++) {
            values[i] *= scalar;
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector div(double scalar) {
        SparseVector resultVector = new Builder(size, indexes, values).build();
        for (int i = 0; i < numberOfNonzeroEntries; i++) {
            resultVector.values[i] /= scalar;
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector divInPlace(double scalar) {
        for (int i = 0; i < numberOfNonzeroEntries; i++) {
            values[i] /= scalar;
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector saxpy(double scalar, Vector vector) {
        checkVectorSize(vector);
        SparseVector resultVector;
        if (vector.type() == VectorType.SPARSE) {
            int[] newIndexes = new int[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            double[] newValues = new double[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            int currentIndex = 0;
            int vector1Index = 0;
            int vector2Index = 0;
            while (true) {
                if (numberOfNonzeroEntries > 0
                        && vector1Index < numberOfNonzeroEntries
                        && ((SparseVector) vector).numberOfNonzeroEntries > 0
                        && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                    if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                        newIndexes[currentIndex] = indexes[vector1Index];
                        newValues[currentIndex] = values[vector1Index];
                        currentIndex++;
                        vector1Index++;
                    } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                        newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                        newValues[currentIndex] = scalar * ((SparseVector) vector).values[vector2Index];
                        currentIndex++;
                        vector2Index++;
                    } else {
                        newIndexes[currentIndex] = indexes[vector1Index];
                        newValues[currentIndex] = values[vector1Index] + scalar * ((SparseVector) vector).values[vector2Index];
                        currentIndex++;
                        vector1Index++;
                        vector2Index++;
                    }
                } else if (numberOfNonzeroEntries > 0 && vector1Index < numberOfNonzeroEntries) {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] = values[vector1Index];
                    currentIndex++;
                    vector1Index++;
                } else if (((SparseVector) vector).numberOfNonzeroEntries > 0
                        && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                    newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                    newValues[currentIndex] = scalar * ((SparseVector) vector).values[vector2Index];
                    currentIndex++;
                    vector2Index++;
                } else {
                    break;
                }
            }
            resultVector = new Builder(size,
                                       Arrays.copyOfRange(newIndexes, 0, currentIndex),
                                       Arrays.copyOfRange(newValues, 0, currentIndex)).build();
        } else {
            throw new NotImplementedException();
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector saxpyInPlace(double scalar, Vector vector) {
        checkVectorSize(vector);
        if (vector.type() == VectorType.SPARSE) {
            int[] newIndexes = new int[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            double[] newValues = new double[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            int currentIndex = 0;
            int vector1Index = 0;
            int vector2Index = 0;
            while (true) {
                if (numberOfNonzeroEntries > 0
                        && vector1Index < numberOfNonzeroEntries
                        && ((SparseVector) vector).numberOfNonzeroEntries > 0
                        && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                    if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                        newIndexes[currentIndex] = indexes[vector1Index];
                        newValues[currentIndex] = values[vector1Index];
                        currentIndex++;
                        vector1Index++;
                    } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                        newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                        newValues[currentIndex] = scalar * ((SparseVector) vector).values[vector2Index];
                        currentIndex++;
                        vector2Index++;
                    } else {
                        newIndexes[currentIndex] = indexes[vector1Index];
                        newValues[currentIndex] = values[vector1Index] + scalar * ((SparseVector) vector).values[vector2Index];
                        currentIndex++;
                        vector1Index++;
                        vector2Index++;
                    }
                } else if (numberOfNonzeroEntries > 0 && vector1Index < numberOfNonzeroEntries) {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] = values[vector1Index];
                    currentIndex++;
                    vector1Index++;
                } else if (((SparseVector) vector).numberOfNonzeroEntries > 0
                        && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                    newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                    newValues[currentIndex] = scalar * ((SparseVector) vector).values[vector2Index];
                    currentIndex++;
                    vector2Index++;
                } else {
                    break;
                }
            }
            indexes = Arrays.copyOfRange(newIndexes, 0, currentIndex);
            values = Arrays.copyOfRange(newValues, 0, currentIndex);
            numberOfNonzeroEntries = currentIndex;
        } else {
            throw new NotImplementedException();
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public double inner(Vector vector) {
        checkVectorSize(vector);
        double result = 0;
        if (numberOfNonzeroEntries > 0) {
            if (vector.type() == VectorType.SPARSE) {
                int vector1Index = 0;
                int vector2Index = 0;
                while (true) {
                    if (numberOfNonzeroEntries > 0
                            && vector1Index < numberOfNonzeroEntries
                            && ((SparseVector) vector).numberOfNonzeroEntries > 0
                            && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                        if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                            vector1Index++;
                        } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                            vector2Index++;
                        } else {
                            result += values[vector1Index] * ((SparseVector) vector).values[vector2Index];
                            vector1Index++;
                            vector2Index++;
                        }
                    } else {
                        break;
                    }
                }
            } else {
                throw new NotImplementedException();
            }
        }
        return result;
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

    /** {@inheritDoc} */
    @Override
    public void writeToStream(ObjectOutputStream outputStream) throws IOException {
        outputStream.writeInt(size);
        outputStream.writeInt(numberOfNonzeroEntries);
        for (int i = 0; i < numberOfNonzeroEntries; i++) {
            outputStream.writeInt(indexes[i]);
            outputStream.writeDouble(values[i]);
        }
    }

    /** {@inheritDoc} */
    @Override
    public boolean equals(Object object) {
        if (!(object instanceof SparseVector))
            return false;
        if (object == this)
            return true;

        SparseVector other = (SparseVector) object;
        return new EqualsBuilder()
                .append(size, other.size)
                .append(indexes, other.indexes)
                .append(values, other.values)
                .isEquals();
    }

    /** {@inheritDoc} */
    @Override
    public int hashCode() {
        return new HashCodeBuilder(17, 31) // Two randomly chosen prime numbers.
                .append(size)
                .append(indexes)
                .append(values)
                .toHashCode();
    }
}
