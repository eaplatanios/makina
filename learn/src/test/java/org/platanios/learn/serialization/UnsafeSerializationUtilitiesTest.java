package org.platanios.learn.serialization;

import org.junit.Assert;
import org.junit.Test;
import org.platanios.utilities.UnsafeSerializationUtilities;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Random;

/**
 * @author Emmanouil Antonios Platanios
 */
public class UnsafeSerializationUtilitiesTest {
    @Test
    public void testBooleanSerialization() {
        try {
            ByteArrayOutputStream outputStream = new ByteArrayOutputStream(4);
            UnsafeSerializationUtilities.writeBoolean(outputStream, false);
            UnsafeSerializationUtilities.writeBoolean(outputStream, true);
            UnsafeSerializationUtilities.writeBoolean(outputStream, true);
            UnsafeSerializationUtilities.writeBoolean(outputStream, true);
            outputStream.close();
            ByteArrayInputStream inputStream = new ByteArrayInputStream(outputStream.toByteArray());
            Assert.assertEquals(false, UnsafeSerializationUtilities.readBoolean(inputStream));
            Assert.assertEquals(true, UnsafeSerializationUtilities.readBoolean(inputStream));
            Assert.assertEquals(true, UnsafeSerializationUtilities.readBoolean(inputStream));
            Assert.assertEquals(true, UnsafeSerializationUtilities.readBoolean(inputStream));
        } catch (IOException e) {
            Assert.fail(e.getMessage());
        }
    }

    @Test
    public void testIntSerialization() {
        try {
            Random random = new Random();
            int value = random.nextInt();
            int expectedValue = value;
            ByteArrayOutputStream outputStream = new ByteArrayOutputStream(4);
            UnsafeSerializationUtilities.writeInt(outputStream, value);
            outputStream.close();
            ByteArrayInputStream inputStream = new ByteArrayInputStream(outputStream.toByteArray());
            int actualValue = UnsafeSerializationUtilities.readInt(inputStream);
            Assert.assertEquals(expectedValue, actualValue);
        } catch (IOException e) {
            Assert.fail(e.getMessage());
        }
    }

    @Test
    public void testDoubleSerialization() {
        try {
            double value = Math.random();
            double expectedValue = value;
            ByteArrayOutputStream outputStream = new ByteArrayOutputStream(4);
            UnsafeSerializationUtilities.writeDouble(outputStream, value);
            outputStream.close();
            ByteArrayInputStream inputStream = new ByteArrayInputStream(outputStream.toByteArray());
            double actualValue = UnsafeSerializationUtilities.readDouble(inputStream);
            Assert.assertEquals(expectedValue, actualValue, 0);
        } catch (IOException e) {
            Assert.fail(e.getMessage());
        }
    }

    @Test
    public void testIntArraySerialization() {
        try {
            Random random = new Random();
            int[] array = new int[] { random.nextInt(), random.nextInt(), random.nextInt(), random.nextInt() };
            int[] expectedArray = new int[array.length];
            System.arraycopy(array, 0, expectedArray, 0, array.length);
            ByteArrayOutputStream outputStream = new ByteArrayOutputStream(array.length << 2);
            UnsafeSerializationUtilities.writeIntArray(outputStream, array);
            outputStream.close();
            ByteArrayInputStream inputStream = new ByteArrayInputStream(outputStream.toByteArray());
            int[] actualArray = UnsafeSerializationUtilities.readIntArray(inputStream, array.length, array.length << 2);
            Assert.assertArrayEquals(expectedArray, actualArray);
        } catch (IOException e) {
            Assert.fail(e.getMessage());
        }
    }

    @Test
    public void testDoubleArraySerialization() {
        try {
            double[] array = new double[] { Math.random(), Math.random(), Math.random(), Math.random() };
            double[] expectedArray = new double[array.length];
            System.arraycopy(array, 0, expectedArray, 0, array.length);
            ByteArrayOutputStream outputStream = new ByteArrayOutputStream(array.length << 3);
            UnsafeSerializationUtilities.writeDoubleArray(outputStream, array);
            outputStream.close();
            ByteArrayInputStream inputStream = new ByteArrayInputStream(outputStream.toByteArray());
            double[] actualArray =
                    UnsafeSerializationUtilities.readDoubleArray(inputStream, array.length, array.length << 3);
            Assert.assertArrayEquals(expectedArray, actualArray, 0);
        } catch (IOException e) {
            Assert.fail(e.getMessage());
        }
    }

    @Test
    public void testStringSerialization() {
        try {
            String string = "Testing unsafe string serialization! Including some greek characters: για να δούμε!";
            String expectedString = new String(string);
            ByteArrayOutputStream outputStream = new ByteArrayOutputStream(4 + (string.length() << 1));
            UnsafeSerializationUtilities.writeString(outputStream, string);
            outputStream.close();
            ByteArrayInputStream inputStream = new ByteArrayInputStream(outputStream.toByteArray());
            String actualString = UnsafeSerializationUtilities.readString(inputStream, 4 + (string.length() << 1));
            Assert.assertEquals(expectedString, actualString);
        } catch (IOException e) {
            Assert.fail(e.getMessage());
        }
    }
}
