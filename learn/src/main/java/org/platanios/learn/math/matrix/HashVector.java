package org.platanios.learn.math.matrix;

import cern.colt.list.IntArrayList;
import cern.colt.map.OpenIntDoubleHashMap;

import org.platanios.learn.math.matrix.SparseVector.SparseVectorIterator;
import org.platanios.learn.math.matrix.Vector.VectorElement;
import org.platanios.learn.serialization.UnsafeSerializationUtilities;

import java.io.IOException;
import java.io.InputStream;
import java.io.InvalidObjectException;
import java.io.OutputStream;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * Implements a class representing sparse vectors and supporting operations related to them. The sparse vectors are
 * stored in an internal hash map.
 * TODO: Add toDenseVector() method (or appropriate constructors).
 * TODO: Make this implementation faster by iterating over key-value pairs instead of iterating over keys and then retrieving values.
 * TODO: Add Builder class and remove constructors.
 * TODO: Serialization can become much faster and lower memory "heavy" by storing pairs of indexes and values sequentially, instead of all indexes and then all values.
 * TODO: Serialization is currently broken for this class.
 *
 * @author Emmanouil Antonios Platanios
 */
public class HashVector extends Vector {
    /** The size which the internal hash map uses as its initial capacity. */
    protected int initialSize = 128;
    /** The size of the vector. */
    protected int size;

    /** Hash map for internal storage of the vector elements. */
    protected OpenIntDoubleHashMap hashMap;

    /**
     * Constructs a sparse vector of the given size and fills it with zeros.
     *
     * @param   size    The size of the vector.
     */
    protected HashVector(int size) {
        this.size = size;
        hashMap = new OpenIntDoubleHashMap(initialSize);
    }

    /**
     * Constructs a sparse vector of the given size from a hash map.
     *
     * @param   size        The size of the vector.
     * @param   elements    Hash map containing the indexes of elements as keys and the values of the corresponding
     *                      elements as values.
     */
    protected HashVector(int size, OpenIntDoubleHashMap elements) {
        this.size = size;
        hashMap = (OpenIntDoubleHashMap) elements.copy();
    }

    /** {@inheritDoc} */
    @Override
    public VectorType type() {
        return VectorType.SPARSE;
    }

    /** {@inheritDoc} */
    @Override
    public HashVector copy() {
        return new HashVector(size, hashMap);
    }

    /** {@inheritDoc} */
    @Override
    public double[] getDenseArray() {
        double[] resultArray = new double[size];
        hashMap.forEachPair((key, value) -> { resultArray[key] = value; return true; });
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
        return hashMap.keys().size();
    }

    /** {@inheritDoc} */
    @Override
    public double get(int index) {
        if (index < 0 || index >= size) {
            throw new IllegalArgumentException(
                    "The provided index must be between 0 (inclusive) and the size of the vector (exclusive)."
            );
        }
        return hashMap.get(index);
    }

    /** {@inheritDoc} */
    @Override
    public HashVector get(int initialIndex, int finalIndex) {
        if (initialIndex < 0 || initialIndex >= size || finalIndex < 0 || finalIndex >= size) {
            throw new IllegalArgumentException(
                    "The provided indexes must be between 0 (inclusive) and the size of the vector (exclusive)."
            );
        }
        if (initialIndex > finalIndex) {
            throw new IllegalArgumentException("The initial index must be smaller or equal to the final index.");
        }
        HashVector resultVector = new HashVector(finalIndex - initialIndex + 1);
        for (int i = initialIndex; i <= finalIndex; i++) {
            resultVector.set(i - initialIndex, hashMap.get(i));
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public HashVector get(int... indexes) {
        HashVector resultVector = new HashVector(indexes.length);
        for (int i = 0; i < indexes.length; i++) {
            if (i < 0 || i >= size) {
                throw new IllegalArgumentException(
                        "The provided indexes must be between 0 (inclusive) and the size of the vector (exclusive)."
                );
            }
            resultVector.set(i, hashMap.get(indexes[i]));
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
            hashMap.removeKey(index);
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
                hashMap.removeKey(i);
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
                hashMap.removeKey(indexes[i]);
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
            hashMap.clear();
        }
    }

    /** {@inheritDoc} */
    @Override
    public double max() {
        double[] values = hashMap.values().elements();
        double maxValue = values[0];
        for (int i = 1; i < size; i++) {
            maxValue = Math.max(maxValue, values[i]);
        }
        return maxValue;
    }

    /** {@inheritDoc} */
    @Override
    public double min() {
        double[] values = hashMap.values().elements();
        double minValue = values[0];
        for (int i = 1; i < size; i++) {
            minValue = Math.min(minValue, values[i]);
        }
        return minValue;
    }

    /** {@inheritDoc} */
    @Override
    public double sum() {
        double[] values = hashMap.values().elements();
        double sum = values[0];
        for (int i = 1; i < size; i++) {
            sum += values[i];
        }
        return sum;
    }

    /** {@inheritDoc} */
    @Override
    public double norm(VectorNorm normType) {
        return normType.compute(hashMap.values().elements());
    }

    /** {@inheritDoc} */
    @Override
    public HashVector add(double scalar) {
        HashVector resultVector = new HashVector(size, hashMap);
        resultVector.hashMap.assign(element -> element + scalar);
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public HashVector addInPlace(double scalar) {
        hashMap.assign(element -> element + scalar);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public HashVector add(Vector vector) {
        checkVectorSize(vector);
        HashVector resultVector = new HashVector(size, hashMap);
        if (vector.type() != VectorType.SPARSE) {
            for (int i = 0; i < size; i++) {
                resultVector.set(i, this.get(i) + vector.get(i));
            }
        } else {
            IntArrayList keysUnion = new IntArrayList(hashMap.keys().elements());
            keysUnion.addAllOf(((HashVector) vector).hashMap.keys());
            for (int key : keysUnion.elements()) {
                resultVector.set(key, this.get(key) + vector.get(key));
            }
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public HashVector addInPlace(Vector vector) {
        checkVectorSize(vector);
        if (vector.type() != VectorType.SPARSE) {
            for (int i = 0; i < size; i++) {
                this.set(i, this.get(i) + vector.get(i));
            }
        } else {
            IntArrayList keysUnion = new IntArrayList(hashMap.keys().elements());
            keysUnion.addAllOf(((HashVector) vector).hashMap.keys());
            for (int key : keysUnion.elements()) {
                this.set(key, this.get(key) + vector.get(key));
            }
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public HashVector sub(double scalar) {
        HashVector resultVector = new HashVector(size, hashMap);
        resultVector.hashMap.assign(element -> element - scalar);
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public HashVector subInPlace(double scalar) {
        hashMap.assign(element -> element - scalar);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public HashVector sub(Vector vector) {
        checkVectorSize(vector);
        HashVector resultVector = new HashVector(size, hashMap);
        if (vector.type() != VectorType.SPARSE) {
            for (int i = 0; i < size; i++) {
                resultVector.set(i, this.get(i) - vector.get(i));
            }
        } else {
            IntArrayList keysUnion = new IntArrayList(hashMap.keys().elements());
            keysUnion.addAllOf(((HashVector) vector).hashMap.keys());
            for (int key : keysUnion.elements()) {
                resultVector.set(key, this.get(key) - vector.get(key));
            }
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public HashVector subInPlace(Vector vector) {
        checkVectorSize(vector);
        if (vector.type() != VectorType.SPARSE) {
            for (int i = 0; i < size; i++) {
                this.set(i, this.get(i) - vector.get(i));
            }
        } else {
            IntArrayList keysUnion = new IntArrayList(hashMap.keys().elements());
            keysUnion.addAllOf(((HashVector) vector).hashMap.keys());
            for (int key : keysUnion.elements()) {
                this.set(key, this.get(key) - vector.get(key));
            }
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public HashVector multElementwise(Vector vector) {
        checkVectorSize(vector);
        HashVector resultVector = new HashVector(size);
        if (vector.type() != VectorType.SPARSE) {
            for (int key : hashMap.keys().elements()) {
                resultVector.set(key, hashMap.get(key) * vector.get(key));
            }
        } else {
            if (hashMap.keys().size() <= ((HashVector) vector).hashMap.keys().size()) {
                for (int key : hashMap.keys().elements()) {
                    resultVector.set(key, hashMap.get(key) * vector.get(key));
                }
            } else {
                for (int key : ((HashVector) vector).hashMap.keys().elements()) {
                    resultVector.set(key, hashMap.get(key) * vector.get(key));
                }
            }
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public HashVector multElementwiseInPlace(Vector vector) {
        checkVectorSize(vector);
        if (vector.type() != VectorType.SPARSE) {
            for (int key : hashMap.keys().elements()) {
                this.set(key, hashMap.get(key) * vector.get(key));
            }
        } else {
            if (this.cardinality() <= vector.cardinality()) {
                for (int key : hashMap.keys().elements()) {
                    this.set(key, hashMap.get(key) * vector.get(key));
                }
            } else {
                for (int key : ((HashVector) vector).hashMap.keys().elements()) {
                    this.set(key, hashMap.get(key) * vector.get(key));
                }
            }
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public HashVector divElementwise(Vector vector) {
        checkVectorSize(vector); // TODO: Need to check whether any element of vector is zero.
        HashVector resultVector = new HashVector(size, hashMap);
        for (int key : hashMap.keys().elements()) {
            resultVector.set(key, hashMap.get(key) / vector.get(key));
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public HashVector divElementwiseInPlace(Vector vector) {
        checkVectorSize(vector); // TODO: Need to check whether any element of vector is zero.
        for (int key : hashMap.keys().elements()) {
            this.set(key, hashMap.get(key) / vector.get(key));
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public HashVector mult(double scalar) {
        HashVector resultVector = new HashVector(size, hashMap);
        resultVector.hashMap.assign(element -> element * scalar);
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public HashVector multInPlace(double scalar) {
        hashMap.assign(element -> element * scalar);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public HashVector div(double scalar) {
        HashVector resultVector = new HashVector(size, hashMap);
        resultVector.hashMap.assign(element -> element / scalar);
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public HashVector divInPlace(double scalar) {
        hashMap.assign(element -> element / scalar);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public HashVector saxpy(double scalar, Vector vector) {
        checkVectorSize(vector);
        HashVector resultVector = new HashVector(size, hashMap);
        if (vector.type() != VectorType.SPARSE) {
            for (int i = 0; i < size; i++) {
                resultVector.set(i, this.get(i) + scalar * vector.get(i));
            }
        } else {
            IntArrayList keysUnion = new IntArrayList(hashMap.keys().elements());
            keysUnion.addAllOf(((HashVector) vector).hashMap.keys());
            for (int key : keysUnion.elements()) {
                resultVector.set(key, this.get(key) + scalar * vector.get(key));
            }
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public HashVector saxpyInPlace(double scalar, Vector vector) {
        checkVectorSize(vector);
        if (vector.type() != VectorType.SPARSE) {
            for (int i = 0; i < size; i++) {
                this.set(i, this.get(i) + scalar * vector.get(i));
            }
        } else {
            IntArrayList keysUnion = new IntArrayList(hashMap.keys().elements());
            keysUnion.addAllOf(((HashVector) vector).hashMap.keys());
            for (int key : keysUnion.elements()) {
                this.set(key, this.get(key) + scalar * vector.get(key));
            }
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector saxpyPlusConstant(double scalar, Vector vector) {
        throw new UnsupportedOperationException();
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector saxpyPlusConstantInPlace(double scalar, Vector vector) {
        throw new UnsupportedOperationException();
    }

    /** {@inheritDoc} */
    @Override
    public double inner(Vector vector) {
        checkVectorSize(vector);
        double result = 0;
        if (vector.type() != VectorType.SPARSE) {
            for (int key : hashMap.keys().elements()) {
                result += hashMap.get(key) * vector.get(key);
            }
        } else {
            if (this.cardinality() <= vector.cardinality()) {
                for (int key : hashMap.keys().elements()) {
                    result += hashMap.get(key) * vector.get(key);
                }
            } else {
                for (int key : ((HashVector) vector).hashMap.keys().elements()) {
                    result += hashMap.get(key) * vector.get(key);
                }
            }
        }
        return result;
    }

    /** {@inheritDoc} */
    @Override
    public double innerPlusConstant(Vector vector) {
        throw new UnsupportedOperationException();
    }

    /** {@inheritDoc} */
    @Override
    public HashVector hypotenuse(Vector vector) {
        throw new UnsupportedOperationException();
    }

    /** {@inheritDoc} */
    @Override
    public HashVector hypotenuseInPlace(Vector vector) {
        throw new UnsupportedOperationException();
    }

    /** {@inheritDoc} */
    @Override
    public HashVector hypotenuseFast(Vector vector) {
        throw new UnsupportedOperationException();
    }

    /** {@inheritDoc} */
    @Override
    public HashVector hypotenuseFastInPlace(Vector vector) {
        throw new UnsupportedOperationException();
    }

    /** {@inheritDoc} */
    @Override
    public HashVector map(Function<Double, Double> function) {
        HashVector resultVector = new HashVector(size, hashMap); // TODO: What happens when the function is applied to zeros?
        resultVector.hashMap.assign(function::apply);
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public HashVector mapInPlace(Function<Double, Double> function) { // TODO: What happens when the function is applied to zeros?
        hashMap.assign(function::apply);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public HashVector mapBiFunction(BiFunction<Double, Double, Double> function, Vector vector) {
        throw new UnsupportedOperationException();
    }

    /** {@inheritDoc} */
    @Override
    public HashVector mapBiFunctionInPlace(BiFunction<Double, Double, Double> function, Vector vector) {
        throw new UnsupportedOperationException();
    }

    /** {@inheritDoc} */
    @Override
    public HashVector mapAdd(Function<Double, Double> function, Vector vector) {
        throw new UnsupportedOperationException();
    }

    /** {@inheritDoc} */
    @Override
    public HashVector mapAddInPlace(Function<Double, Double> function, Vector vector) {
        throw new UnsupportedOperationException();
    }

    /** {@inheritDoc} */
    @Override
    public HashVector mapSub(Function<Double, Double> function, Vector vector) {
        throw new UnsupportedOperationException();
    }

    /** {@inheritDoc} */
    @Override
    public HashVector mapSubInPlace(Function<Double, Double> function, Vector vector) {
        throw new UnsupportedOperationException();
    }

    /** {@inheritDoc} */
    @Override
    public HashVector mapMultElementwise(Function<Double, Double> function, Vector vector) {
        throw new UnsupportedOperationException();
    }

    /** {@inheritDoc} */
    @Override
    public HashVector mapMultElementwiseInPlace(Function<Double, Double> function, Vector vector) {
        throw new UnsupportedOperationException();
    }

    /** {@inheritDoc} */
    @Override
    public HashVector mapDivElementwise(Function<Double, Double> function, Vector vector) {
        throw new UnsupportedOperationException();
    }

    /** {@inheritDoc} */
    @Override
    public HashVector mapDivElementwiseInPlace(Function<Double, Double> function, Vector vector) {
        throw new UnsupportedOperationException();
    }

    /** {@inheritDoc} */
    @Override
    public Matrix outer(Vector vector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public HashVector gaxpy(Matrix matrix, Vector vector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public HashVector gaxpyInPlace(Matrix matrix, Vector vector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public HashVector transMult(Matrix matrix) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public HashVector prepend(double value) {
        throw new UnsupportedOperationException();
    }

    /** {@inheritDoc} */
    @Override
    public HashVector append(double value) {
        throw new UnsupportedOperationException();
    }

    /** {@inheritDoc} */
    @Override
    public boolean equals(Object object) { // TODO: Fix this! It is currently implemented in a very "hacky" way.
        if (!(object instanceof Vector))
            return false;
        if (object == this)
            return true;

        HashVector that = (HashVector) object;
        SparseVector thisSparse = new SparseVector(size,
                                                   hashMap.keys().elements(),
                                                   hashMap.values().elements());
        SparseVector thatSparse = new SparseVector(that.size,
                                                   that.hashMap.keys().elements(),
                                                   that.hashMap.values().elements());
        return thisSparse.equals(thatSparse);
    }

    /** {@inheritDoc} */
    @Override
    public void write(OutputStream outputStream, boolean includeType) throws IOException { // TODO: This is currently not working.
        if (includeType)
            UnsafeSerializationUtilities.writeInt(outputStream, type().ordinal());
        UnsafeSerializationUtilities.writeInt(outputStream, size);
        UnsafeSerializationUtilities.writeInt(outputStream, hashMap.keys().size());
        UnsafeSerializationUtilities.writeIntArray(outputStream, hashMap.keys().elements());
        UnsafeSerializationUtilities.writeDoubleArray(outputStream, hashMap.values().elements());
    }
    
    /**
     * Deserializes the hash vector stored in the provided input stream and returns it.
     *
     * @param   inputStream Input stream from which the dense vector will be "read".
     * @return              The hash vector obtained from the provided input stream.
     * @throws  IOException
     */
    public static HashVector read(InputStream inputStream, boolean includeType) throws IOException {
        if (includeType) {
            VectorType vectorType = VectorType.values()[UnsafeSerializationUtilities.readInt(inputStream)];
            if (vectorType != VectorType.HASH)
                throw new InvalidObjectException("The stored vector is of type " + vectorType.name() + "!");
        }
        int size = UnsafeSerializationUtilities.readInt(inputStream);
        int numberOfNonzeroEntries = UnsafeSerializationUtilities.readInt(inputStream);
        int[] indexes = UnsafeSerializationUtilities.readIntArray(inputStream,
                                                                  numberOfNonzeroEntries,
                                                                  Math.min(numberOfNonzeroEntries, 1024 * 1024));
        double[] values = UnsafeSerializationUtilities.readDoubleArray(inputStream,
                                                                       numberOfNonzeroEntries,
                                                                       Math.min(numberOfNonzeroEntries, 1024 * 1024));
        HashVector vector = new HashVector(size);
        vector.hashMap = new OpenIntDoubleHashMap(Integer.highestOneBit(numberOfNonzeroEntries) << 1);
        for (int i = 0; i < numberOfNonzeroEntries; i++)
            vector.hashMap.put(indexes[i], values[i]);
        return vector;
    }

    /** {@inheritDoc} */
    @Override
    public InputStream getEncoder(boolean includeType) {
        return new Encoder(includeType);
    }
    
    @Override
	public Iterator<VectorElement> iterator() {
		throw new UnsupportedOperationException();
	}

    /**
     * Encoder class for hash vectors. This class extends the Java {@link InputStream} class and can be used to copy
     * hash vector instances into other locations (e.g., in a database). Note that this encoder uses the underlying
     * vector and so, if that vector is changed, the output of this encoder might be changed and even become corrupt.
     *
     * The hash vector is serialized in the following way: (i) the size of the vector is encoded first, (ii) the
     * number of nonzero elements in the sparse vector is encoded next, (iii) the elements of an underlying array
     * holding the indexes of the nonzero elements are encoded next, in the order in which they appear in the array, and
     * (iv) the elements of an underlying array holding the values of the nonzero elements are encoded finally, in the
     * order in which they appear in the array.
     */
    protected class Encoder extends InputStream {
        /** A pointer in memory used to represent the current position while serializing different fields of the object
         * that is being serialized. */
        long position;
        /** The largest value that the {@link #position} pointer can take (i.e., this pointer represents the end of the
         * field that is currently being serialized, in memory). */
        long endPosition;
        /** The current state of the encoder, representing which field of the object is currently being serialized. */
        EncoderState state;

        /** The memory address offset of {@link #size} from the base address of the hash vector instance that is being
         * encoded. */
        final long sizeFieldOffset;
        /** The memory address offset of {@link #numberOfNonzeroEntries} from the base address of the sparse vector
         * instance that is being encoded. */
        final long numberOfNonzeroEntriesFieldOffset;
        /** The number of nonzero elements in the hash vector being encoded. */
        final int numberOfNonzeroEntries;
        /** An array holding the indexes of the nonzero elements of the hash vector being encoded. This array is parallel
         * to the {@link #values} array. */
        final int[] indexes;
        /** An array holding the values of the nonzero elements of the hash vector being encoded. This array is parallel
         * to the {@link #indexes} array. */
        final double[] values;
        /** The {@link VectorType} ordinal number of the type of the vector being encoded
         * (i.e., {@link VectorType#HASH}). */
        final int type;
        /** Boolean value indicating whether or not to also encode the type of the current vector
         * (i.e., {@link VectorType#HASH}). */
        final boolean includeType;

        /** Constructs an encoder object from the current vector. */
        public Encoder(boolean includeType) {
            long typeFieldOffset;
            try {
                sizeFieldOffset = UNSAFE.objectFieldOffset(HashVector.class.getDeclaredField("size"));
                numberOfNonzeroEntriesFieldOffset =
                        UNSAFE.objectFieldOffset(Encoder.class.getDeclaredField("numberOfNonzeroEntries"));
                typeFieldOffset = UNSAFE.objectFieldOffset(Encoder.class.getDeclaredField("type"));
            } catch (NoSuchFieldException e) {
                throw new RuntimeException(e);
            }
            if (!includeType) {
                position = sizeFieldOffset;
                endPosition = sizeFieldOffset + 4;
                state = EncoderState.SIZE;
            } else {
                position = typeFieldOffset;
                endPosition = typeFieldOffset + 4;
                state = EncoderState.TYPE;
            }
            indexes = hashMap.keys().elements();
            values = hashMap.values().elements();
            numberOfNonzeroEntries = indexes.length;
            type = VectorType.SPARSE.ordinal();
            this.includeType = includeType;
        }

        /** {@inheritDoc} */
        @Override
        public int read() {
            switch(state) {
                case TYPE:
                    if (position == endPosition) {
                        position = sizeFieldOffset;
                        endPosition = sizeFieldOffset + 4;
                        state = EncoderState.SIZE;
                    } else {
                        return UNSAFE.getByte(this, position++);
                    }
                case SIZE:
                    if (position == endPosition) {
                        position = numberOfNonzeroEntriesFieldOffset;
                        endPosition = numberOfNonzeroEntriesFieldOffset + 4;
                        state = EncoderState.NUMBER_OF_NONZERO_ENTRIES;
                    } else {
                        return UNSAFE.getByte(HashVector.this, position++);
                    }
                case NUMBER_OF_NONZERO_ENTRIES:
                    if (position == endPosition) {
                        position = INT_ARRAY_OFFSET;
                        endPosition = INT_ARRAY_OFFSET + (numberOfNonzeroEntries << 2);
                        state = EncoderState.INDEXES;
                    } else {
                        return UNSAFE.getByte(this, position++);
                    }
                case INDEXES:
                    if (position == endPosition) {
                        position = DOUBLE_ARRAY_OFFSET;
                        endPosition = DOUBLE_ARRAY_OFFSET + (numberOfNonzeroEntries << 3);
                        state = EncoderState.VALUES;
                    } else {
                        return UNSAFE.getByte(indexes, position++);
                    }
                case VALUES:
                    if (position == endPosition)
                        return -1;
                    else
                        return UNSAFE.getByte(values, position++);
            }
            return -1;
        }

        /** {@inheritDoc} */
        @Override
        public int read(byte[] destination) {
            return read(destination, 0, destination.length);
        }

        /** {@inheritDoc} */
        @Override
        public int read(byte destination[], int offset, int length) {
            if (destination == null)
                throw new NullPointerException();
            if (offset < 0 || length < 0 || length > destination.length - offset)
                throw new IndexOutOfBoundsException();
            int bytesRead;
            switch(state) {
                case TYPE:
                    bytesRead = readBytes(this, destination, offset, length);
                    if (bytesRead != -1) {
                        return bytesRead;
                    } else {
                        position = sizeFieldOffset;
                        endPosition = numberOfNonzeroEntriesFieldOffset + 4;
                        state = EncoderState.SIZE;
                    }
                case SIZE:
                    bytesRead = readBytes(HashVector.this, destination, offset, length);
                    if (bytesRead != -1) {
                        return bytesRead;
                    } else {
                        position = numberOfNonzeroEntriesFieldOffset;
                        endPosition = numberOfNonzeroEntriesFieldOffset + 4;
                        state = EncoderState.NUMBER_OF_NONZERO_ENTRIES;
                    }
                case NUMBER_OF_NONZERO_ENTRIES:
                    bytesRead = readBytes(this, destination, offset, length);
                    if (bytesRead != -1) {
                        return bytesRead;
                    } else {
                        position = INT_ARRAY_OFFSET;
                        endPosition = INT_ARRAY_OFFSET + (numberOfNonzeroEntries << 2);
                        state = EncoderState.INDEXES;
                    }
                case INDEXES:
                    bytesRead = readBytes(indexes, destination, offset, length);
                    if (bytesRead != -1) {
                        return bytesRead;
                    } else {
                        position = DOUBLE_ARRAY_OFFSET;
                        endPosition = DOUBLE_ARRAY_OFFSET + (numberOfNonzeroEntries << 3);
                        state = EncoderState.VALUES;
                    }
                case VALUES:
                    return readBytes(values, destination, offset, length);
            }
            return -1;
        }

        /**
         * Reads up to {@code length} bytes of data from the input stream into an array of bytes and returns the number
         * of bytes read.
         *
         * @param   source      The source object to read data from.
         * @param   destination The byte array to write data into.
         * @param   offset      The memory address offset into the destination byte array, to start writing data from.
         * @param   length      The maximum number of bytes to read from {@code source} and write to
         *                      {@code destination}.
         * @return              The number of bytes read from {@code source} and written to {@code destination}.
         */
        private int readBytes(Object source, byte[] destination, int offset, int length) {
            long numberOfBytesToRead = Math.min(endPosition - position, length);
            if (numberOfBytesToRead > 0) {
                UNSAFE.copyMemory(source,
                                  position,
                                  destination,
                                  BYTE_ARRAY_OFFSET + offset,
                                  numberOfBytesToRead);
                position += numberOfBytesToRead;
                return (int) numberOfBytesToRead;
            } else {
                return -1;
            }
        }
    }

    /** Enumeration containing the possible encoder states used within the {@link Encoder} class. */
    private enum EncoderState {
        /** Represents the state the encoder is in, while encoding the type of the sparse vector. */
        TYPE,
        /** Represents the state the encoder is in, while encoding the size of the hash vector. */
        SIZE,
        /** Represents the state the encoder is in, while encoding the number of nonzero entries of the hash vector. */
        NUMBER_OF_NONZERO_ENTRIES,
        /** Represents the state the encoder is in, while encoding the underlying indexes array of the hash vector. */
        INDEXES,
        /** Represents the state the encoder is in, while encoding the underlying values array of the hash vector. */
        VALUES
    }
}
