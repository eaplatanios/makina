package org.platanios.learn.math.matrix;

import org.junit.Assert;
import org.junit.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;

/**
 * @author Emmanouil Antonios Platanios
 */
public class SparseVectorTest {
    @Test
    public void testEquals() {
        SparseVector vector1 = new SparseVector(1000,
                                                new int[] { 1, 5, 8, 35, 56 },
                                                new double[] { 0.53, 0.32, 0.91, 0.05, 0 });
        SparseVector vector2 = new SparseVector(1000,
                                                new int[] { 1, 5, 8, 35 },
                                                new double[] { 0.53, 0.32, 0.91, 0.05 });
        SparseVector vector3 = new SparseVector(1000,
                                                new int[] { 3, 5, 8, 35, 56 },
                                                new double[] { 0.53, 0.32, 0.91, 0.05, 0 });
        SparseVector vector4 = new SparseVector(2000,
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
            ByteArrayOutputStream outputStream = new ByteArrayOutputStream(4 + (vectorSize << 3));
            SparseVector vector = new SparseVector(1000,
                                                   new int[] { 1, 5, 8, 35 },
                                                   new double[] { 0.53, 0.32, 0.91, 0.05 });
            SparseVector expectedVector = vector.copy();
            vector.write(outputStream);
            outputStream.close();
            ByteArrayInputStream inputStream = new ByteArrayInputStream(outputStream.toByteArray());
            SparseVector actualVector = SparseVector.read(inputStream);
            inputStream.close();
            Assert.assertTrue(expectedVector.equals(actualVector));
        } catch (IOException e) {
            Assert.fail(e.getMessage());
        }
    }
}
