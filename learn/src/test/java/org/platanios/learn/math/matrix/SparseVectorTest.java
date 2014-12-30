package org.platanios.learn.math.matrix;

import org.junit.Assert;
import org.junit.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;

/**
 * @author Emmanouil Antonios Platanios
 */
public class SparseVectorTest {
    @Test
    public void testEquals() {
        int vectorSize = 1000;
        SparseVector vector1 = new SparseVector(vectorSize,
                                                new int[] { 1, 5, 8, 35, 56 },
                                                new double[] { 0.53, 0.32, 0.91, 0.05, 0 });
        SparseVector vector2 = new SparseVector(vectorSize,
                                                new int[] { 1, 5, 8, 35 },
                                                new double[] { 0.53, 0.32, 0.91, 0.05 });
        SparseVector vector3 = new SparseVector(vectorSize,
                                                new int[] { 3, 5, 8, 35, 56 },
                                                new double[] { 0.53, 0.32, 0.91, 0.05, 0 });
        SparseVector vector4 = new SparseVector(2 * vectorSize,
                                                new int[] { 1, 5, 8, 35, 56 },
                                                new double[] { 0.53, 0.32, 0.91, 0.05, 0 });
        Assert.assertTrue(vector1.equals(vector2));
        Assert.assertTrue(!vector1.equals(vector3));
        Assert.assertTrue(!vector1.equals(vector4));
        Assert.assertTrue(!vector2.equals(vector3));
        Assert.assertTrue(!vector2.equals(vector4));
        Assert.assertTrue(!vector3.equals(vector4));
    }

    @Test
    public void testSerialization() {
        int vectorSize = 1000;
        try {
            // Test for when we do not store the vector type
            ByteArrayOutputStream outputStream = new ByteArrayOutputStream(4 + (vectorSize << 3));
            SparseVector vector = new SparseVector(vectorSize,
                                                   new int[] { 1, 5, 8, 35 },
                                                   new double[] { 0.53, 0.32, 0.91, 0.05 });
            SparseVector expectedVector = vector.copy();
            vector.write(outputStream, false);
            outputStream.close();
            ByteArrayInputStream inputStream = new ByteArrayInputStream(outputStream.toByteArray());
            SparseVector actualVector = SparseVector.read(inputStream, false);
            inputStream.close();
            Assert.assertTrue(expectedVector.equals(actualVector));
            // Test for when we store the vector type
            outputStream = new ByteArrayOutputStream(4 + (vectorSize << 3));
            vector = new SparseVector(vectorSize,
                                      new int[] { 1, 5, 8, 35 },
                                      new double[] { 0.53, 0.32, 0.91, 0.05 });
            vector.write(outputStream, true);
            outputStream.close();
            inputStream = new ByteArrayInputStream(outputStream.toByteArray());
            actualVector = SparseVector.read(inputStream, true);
            inputStream.close();
            Assert.assertTrue(expectedVector.equals(actualVector));
            // Test for when the indexes and the values arrays have a bigger length than the number of non-zero elements
            outputStream = new ByteArrayOutputStream(4 + (vectorSize << 3));
            vector = new SparseVector(vectorSize,
                                      new int[] { 1, 5, 8, 35, 0, 0, 0, 0 },
                                      new double[] { 0.53, 0.32, 0.91, 0.05, 0, 0, 0, 0 });
            vector.numberOfNonzeroEntries = 4;
            vector.write(outputStream, true);
            outputStream.close();
            inputStream = new ByteArrayInputStream(outputStream.toByteArray());
            actualVector = SparseVector.read(inputStream, true);
            inputStream.close();
            Assert.assertTrue(expectedVector.equals(actualVector));
        } catch (IOException e) {
            Assert.fail(e.getMessage());
        }
    }

    @Test
    public void testEncoder() {
        int vectorSize = 1000;
        try {
            // Test for when we do not store the vector type
            SparseVector vector = new SparseVector(vectorSize,
                                                   new int[] { 1, 5, 8, 35 },
                                                   new double[] { 0.53, 0.32, 0.91, 0.05 });
            SparseVector expectedVector = vector.copy();
            InputStream inputStream = vector.getEncoder(false);
            SparseVector actualVector = SparseVector.read(inputStream, false);
            inputStream.close();
            Assert.assertTrue(expectedVector.equals(actualVector));
            // Test for when we store the vector type
            vector = new SparseVector(vectorSize,
                                      new int[] { 1, 5, 8, 35 },
                                      new double[] { 0.53, 0.32, 0.91, 0.05 });
            expectedVector = vector.copy();
            inputStream = vector.getEncoder(true);
            actualVector = SparseVector.read(inputStream, true);
            inputStream.close();
            Assert.assertTrue(expectedVector.equals(actualVector));
            // Test for when the indexes and the values arrays have a bigger length than the number of non-zero elements
            vector = new SparseVector(vectorSize,
                                      new int[] { 1, 5, 8, 35, 0, 0, 0, 0 },
                                      new double[] { 0.53, 0.32, 0.91, 0.05, 0, 0, 0, 0 });
            vector.numberOfNonzeroEntries = 4;
            expectedVector = vector.copy();
            inputStream = vector.getEncoder(true);
            actualVector = SparseVector.read(inputStream, true);
            inputStream.close();
            Assert.assertTrue(expectedVector.equals(actualVector));
        } catch (IOException e) {
            Assert.fail(e.getMessage());
        }
    }

    @Test
    public void testPrepend() {
        int vectorSize = 1000;
        SparseVector actualVector = new SparseVector(vectorSize,
                                                     new int[] { 1, 5, 8, 35, 56 },
                                                     new double[] { 0.53, 0.32, 0.91, 0.05, 0 });
        SparseVector expectedVector = new SparseVector(vectorSize + 1,
                                                       new int[] { 0, 2, 6, 9, 36, 57 },
                                                       new double[] { 5.4, 0.53, 0.32, 0.91, 0.05, 0 });
        actualVector.prepend(5.4);
        Assert.assertTrue(expectedVector.equals(actualVector));
        expectedVector = new SparseVector(vectorSize + 2,
                                          new int[] { 0, 1, 3, 7, 10, 37, 58 },
                                          new double[] { 3.2, 5.4, 0.53, 0.32, 0.91, 0.05, 0 });
        actualVector.prepend(3.2);
        Assert.assertTrue(expectedVector.equals(actualVector));
    }

    @Test
    public void testAppend() {
        int vectorSize = 1000;
        SparseVector actualVector = new SparseVector(vectorSize,
                                                     new int[] { 1, 5, 8, 35, 56 },
                                                     new double[] { 0.53, 0.32, 0.91, 0.05, 0 });
        SparseVector expectedVector = new SparseVector(vectorSize + 1,
                                                       new int[] { 1, 5, 8, 35, 56, 1000 },
                                                       new double[] { 0.53, 0.32, 0.91, 0.05, 0, 5.4 });
        actualVector.append(5.4);
        Assert.assertTrue(expectedVector.equals(actualVector));
        expectedVector = new SparseVector(vectorSize + 2,
                                          new int[] { 1, 5, 8, 35, 56, 1000, 1001 },
                                          new double[] { 0.53, 0.32, 0.91, 0.05, 0, 5.4, 3.2 });
        actualVector.append(3.2);
        Assert.assertTrue(expectedVector.equals(actualVector));
    }
}
