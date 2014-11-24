package org.platanios.learn.serialization;

import sun.misc.Unsafe;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.reflect.Field;

/**
 * A class containing static methods that are used for reading and writing data from and to streams, using direct memory
 * manipulation (i.e., unsafe code) in order to achieve fast execution time.
 *
 * @author Emmanouil Antonios Platanios
 */
public class UnsafeSerializationUtilities {
    /** Obtain a singleton instance of the {@link sun.misc.Unsafe} class for use within the unsafe serialization
     * mechanism. */
    private static final Unsafe UNSAFE;
    static
    {
        try
        {
            Field field = Unsafe.class.getDeclaredField("theUnsafe");
            field.setAccessible(true);
            UNSAFE = (Unsafe)field.get(null);
        }
        catch (Exception e)
        {
            throw new RuntimeException(e);
        }
    }
    /** The offset, in bytes, between the base memory address of a byte array and its first element. */
    protected static final long BYTE_ARRAY_OFFSET = UNSAFE.arrayBaseOffset(byte[].class);
    /** The offset, in bytes, between the base memory address of a character array and its first element. */
    protected static final long CHAR_ARRAY_OFFSET = UNSAFE.arrayBaseOffset(char[].class);
    /** The offset, in bytes, between the base memory address of a integer array and its first element. */
    protected static final long INT_ARRAY_OFFSET = UNSAFE.arrayBaseOffset(int[].class);
    /** The offset, in bytes, between the base memory address of a double array and its first element. */
    protected static final long DOUBLE_ARRAY_OFFSET = UNSAFE.arrayBaseOffset(double[].class);

    /**
     * Writes the provided integer to the provided output stream byte by byte.
     *
     * @param   outputStream    The output stream to write the provided integer to.
     * @param   value           The integer to write to the provided output stream.
     * @throws  IOException
     */
    public static void writeInt(OutputStream outputStream, int value) throws IOException {
        byte[] buffer = new byte[4];
        UNSAFE.putInt(buffer, BYTE_ARRAY_OFFSET, value);
        outputStream.write(buffer);
    }

    /**
     * Reads an integer from the provided input stream byte by byte and returns it.
     *
     * @param   inputStream     The input stream to read the integer from.
     * @return                  The integer read from the provided input stream.
     * @throws  IOException
     */
    public static int readInt(InputStream inputStream) throws IOException {
        long bytesToRead = 4;
        byte[] buffer = new byte[4];
        while (bytesToRead > 0) {
            int bytesRead = inputStream.read(buffer, 0, 4);
            if (bytesRead == -1 && bytesToRead > 0)
                throw new RuntimeException();
            bytesToRead -= bytesRead;
        }
        return UNSAFE.getInt(buffer, BYTE_ARRAY_OFFSET);
    }

    /**
     * Writes the provided double to the provided output stream byte by byte.
     *
     * @param   outputStream    The output stream to write the provided integer to.
     * @param   value           The double to write to the provided output stream.
     * @throws  IOException
     */
    public static void writeDouble(OutputStream outputStream, double value) throws IOException {
        byte[] buffer = new byte[8];
        UNSAFE.putDouble(buffer, BYTE_ARRAY_OFFSET, value);
        outputStream.write(buffer);
    }

    /**
     * Reads an double from the provided input stream byte by byte and returns it.
     *
     * @param   inputStream     The input stream to read the integer from.
     * @return                  The double read from the provided input stream.
     * @throws  IOException
     */
    public static double readDouble(InputStream inputStream) throws IOException {
        long bytesToRead = 8;
        byte[] buffer = new byte[8];
        while (bytesToRead > 0) {
            int bytesRead = inputStream.read(buffer, 0, 8);
            if (bytesRead == -1 && bytesToRead > 0)
                throw new RuntimeException();
            bytesToRead -= bytesRead;
        }
        return UNSAFE.getDouble(buffer, BYTE_ARRAY_OFFSET);
    }

    /**
     * Writes the provided integer array to the provided output stream byte by byte. Note that the size of the array is
     * not stored and so the user of this method must make sure to store that as well somewhere, so that the array can
     * be serialized later on.
     *
     * @param   outputStream    The output stream to write the provided integer array to.
     * @param   array           The integer array to write to the provided output stream.
     * @throws  IOException
     */
    public static void writeIntArray(OutputStream outputStream, int[] array) throws IOException {
        byte[] buffer = new byte[array.length << 2];
        UNSAFE.copyMemory(array, INT_ARRAY_OFFSET, buffer, BYTE_ARRAY_OFFSET, array.length << 2);
        outputStream.write(buffer);
    }

    /**
     * Reads an integer array of a given size from the provided input stream byte by byte and returns it.
     *
     * @param   inputStream     The input stream to read the integer array from.
     * @param   size            The size of the integer array to read from the provided input stream.
     * @param   bufferSize      The buffer size to use for the buffer used during the data reading.
     * @return                  The integer array read from the provided input stream.
     * @throws  IOException
     */
    public static int[] readIntArray(InputStream inputStream, int size, int bufferSize) throws IOException {
        int[] array = new int[size];
        long position = INT_ARRAY_OFFSET;
        long bytesToRead = size << 2;
        byte[] buffer = new byte[bufferSize];
        while (bytesToRead > 0) {
            int bytesRead = inputStream.read(buffer, 0, (int) Math.min(bufferSize, bytesToRead));
            if (bytesRead == -1 && bytesToRead > 0)
                throw new RuntimeException();
            UNSAFE.copyMemory(buffer, BYTE_ARRAY_OFFSET, array, position, bytesRead);
            position += bytesRead;
            bytesToRead -= bytesRead;
        }
        return array;
    }

    /**
     * Writes the provided double array to the provided output stream byte by byte. Note that the size of the array is
     * not stored and so the user of this method must make sure to store that as well somewhere, so that the array can
     * be serialized later on.
     *
     * @param   outputStream    The output stream to write the provided double array to.
     * @param   array           The double array to write to the provided output stream.
     * @throws  IOException
     */
    public static void writeDoubleArray(OutputStream outputStream, double[] array) throws IOException {
        byte[] buffer = new byte[array.length << 3];
        UNSAFE.copyMemory(array, DOUBLE_ARRAY_OFFSET, buffer, BYTE_ARRAY_OFFSET, array.length << 3);
        outputStream.write(buffer);
    }

    /**
     * Reads a double array of a given size from the provided input stream byte by byte and returns it.
     *
     * @param   inputStream     The input stream to read the double array from.
     * @param   size            The size of the double array to read from the provided input stream.
     * @param   bufferSize      The buffer size to use for the buffer used during the data reading.
     * @return                  The double array read from the provided input stream.
     * @throws  IOException
     */
    public static double[] readDoubleArray(InputStream inputStream, int size, int bufferSize) throws IOException {
        double[] array = new double[size];
        long position = DOUBLE_ARRAY_OFFSET;
        long bytesToRead = size << 3;
        byte[] buffer = new byte[bufferSize];
        while (bytesToRead > 0) {
            int bytesRead = inputStream.read(buffer, 0, (int) Math.min(bufferSize, bytesToRead));
            if (bytesRead == -1 && bytesToRead > 0)
                throw new RuntimeException();
            UNSAFE.copyMemory(buffer, BYTE_ARRAY_OFFSET, array, position, bytesRead);
            position += bytesRead;
            bytesToRead -= bytesRead;
        }
        return array;
    }

    /**
     * Writes the provided string to the provided output stream byte by byte.
     *
     * // TODO: This method can probably be made faster by using unsafe code. I just wasn't sure how to do that yet.
     *
     * @param   outputStream    The output stream to write the provided double array to.
     * @param   string          The string to write to the provided output stream.
     * @throws  IOException
     */
    public static void writeString(OutputStream outputStream, String string) throws IOException {
        byte[] array = string.getBytes();
        writeInt(outputStream, array.length);
        outputStream.write(array);
    }

    /**
     * Reads a string from the provided input stream byte by byte and returns it.
     *
     * @param   inputStream     The input stream to read the string from.
     * @param   bufferSize      The buffer size to use for the buffer used during the data reading.
     * @return                  The string read from the provided input stream.
     * @throws  IOException
     */
    public static String readString(InputStream inputStream, int bufferSize) throws IOException {
        int size = readInt(inputStream);
        byte[] array = new byte[size];
        long position = BYTE_ARRAY_OFFSET;
        long bytesToRead = size;
        byte[] buffer = new byte[bufferSize];
        while (bytesToRead > 0) {
            int bytesRead = inputStream.read(buffer, 0, (int) Math.min(bufferSize, bytesToRead));
            if (bytesRead == -1 && bytesToRead > 0)
                throw new RuntimeException();
            UNSAFE.copyMemory(buffer, BYTE_ARRAY_OFFSET, array, position, bytesRead);
            position += bytesRead;
            bytesToRead -= bytesRead;
        }
        return new String(array);
    }
}
