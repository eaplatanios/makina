package org.platanios.learn.utilities;

import sun.misc.Unsafe;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.reflect.Field;

/**
 * @author Emmanouil Antonios Platanios
 */
public class UnsafeSerializationUtilities {
    private static final Unsafe unsafe;
    static
    {
        try
        {
            Field field = Unsafe.class.getDeclaredField("theUnsafe");
            field.setAccessible(true);
            unsafe = (Unsafe)field.get(null);
        }
        catch (Exception e)
        {
            throw new RuntimeException(e);
        }
    }
    private static final long byteArrayOffset = unsafe.arrayBaseOffset(byte[].class);
    private static final long intArrayOffset = unsafe.arrayBaseOffset(int[].class);
    private static final long doubleArrayOffset = unsafe.arrayBaseOffset(double[].class);

    public static void writeInt(OutputStream outputStream, int value) throws IOException {
        byte[] buffer = new byte[4];
        unsafe.putInt(buffer, byteArrayOffset, value);
        outputStream.write(buffer);
    }

    public static int readInt(InputStream inputStream) throws IOException {
        long bytesToRead = 4;
        byte[] buffer = new byte[4];
        while (bytesToRead > 0) {
            int bytesRead = inputStream.read(buffer, 0, 4);
            if (bytesRead == -1 && bytesToRead > 0)
                throw new RuntimeException();
            bytesToRead -= bytesRead;
        }
        return unsafe.getInt(buffer, byteArrayOffset);
    }

    public static void writeIntArray(OutputStream outputStream, int[] array) throws IOException {
        byte[] buffer = new byte[array.length << 2];
        unsafe.copyMemory(array, intArrayOffset, buffer, byteArrayOffset, array.length << 2);
        outputStream.write(buffer);
    }

    public static int[] readIntArray(InputStream inputStream, int size, int bufferSize) throws IOException {
        int[] array = new int[size];
        long position = intArrayOffset;
        long bytesToRead = size << 2;
        byte[] buffer = new byte[bufferSize];
        while (bytesToRead > 0) {
            int bytesRead = inputStream.read(buffer, 0, (int) Math.min(bufferSize, bytesToRead));
            if (bytesRead == -1 && bytesToRead > 0)
                throw new RuntimeException();
            unsafe.copyMemory(buffer, byteArrayOffset, array, position, bytesRead);
            position += bytesRead;
            bytesToRead -= bytesRead;
        }
        return array;
    }

    public static void writeDoubleArray(OutputStream outputStream, double[] array) throws IOException {
        byte[] buffer = new byte[array.length << 3];
        unsafe.copyMemory(array, doubleArrayOffset, buffer, byteArrayOffset, array.length << 3);
        outputStream.write(buffer);
    }

    public static double[] readDoubleArray(InputStream inputStream, int size, int bufferSize) throws IOException {
        double[] array = new double[size];
        long position = doubleArrayOffset;
        long bytesToRead = size << 3;
        byte[] buffer = new byte[bufferSize];
        while (bytesToRead > 0) {
            int bytesRead = inputStream.read(buffer, 0, (int) Math.min(bufferSize, bytesToRead));
            if (bytesRead == -1 && bytesToRead > 0)
                throw new RuntimeException();
            unsafe.copyMemory(buffer, byteArrayOffset, array, position, bytesRead);
            position += bytesRead;
            bytesToRead -= bytesRead;
        }
        return array;
    }
}
