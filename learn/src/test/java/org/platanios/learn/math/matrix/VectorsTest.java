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
public class VectorsTest {
    @Test
    public void testBuildDenseVectorFromInputStream() {
        int vectorSize = 1000;
        try {
            // Test for vector written using write(OutputStream)
            ByteArrayOutputStream outputStream = new ByteArrayOutputStream(4 + (vectorSize << 3));
            DenseVector vector = DenseVector.generateRandomVector(vectorSize);
            DenseVector expectedVector = vector.copy();
            vector.write(outputStream, true);
            outputStream.close();
            ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(outputStream.toByteArray());
            Vector actualVector = Vectors.build(byteArrayInputStream);
            byteArrayInputStream.close();
            Assert.assertTrue(expectedVector.equals(actualVector));
            // Test for vector written using its encoder
            vector = DenseVector.generateRandomVector(vectorSize);
            expectedVector = vector.copy();
            InputStream inputStream = vector.getEncoder(true);
            actualVector = Vectors.build(inputStream);
            inputStream.close();
            Assert.assertTrue(expectedVector.equals(actualVector));
        } catch (IOException e) {
            Assert.fail(e.getMessage());
        }
    }

    @Test
    public void testBuildSparseVectorFromInputStream() {
        int vectorSize = 1000;
        try {
            // Test for vector written using write(OutputStream)
            ByteArrayOutputStream outputStream = new ByteArrayOutputStream(4 + (vectorSize << 3));
            SparseVector vector = new SparseVector(vectorSize,
                                                   new int[] { 1, 5, 8, 35 },
                                                   new double[] { 0.53, 0.32, 0.91, 0.05 });
            SparseVector expectedVector = vector.copy();
            vector.write(outputStream, true);
            outputStream.close();
            ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(outputStream.toByteArray());
            Vector actualVector = Vectors.build(byteArrayInputStream);
            byteArrayInputStream.close();
            Assert.assertTrue(expectedVector.equals(actualVector));
            // Test for vector written using its encoder
            vector = new SparseVector(vectorSize,
                                      new int[] { 1, 5, 8, 35 },
                                      new double[] { 0.53, 0.32, 0.91, 0.05 });
            InputStream inputStream = vector.getEncoder(true);
            actualVector = Vectors.build(inputStream);
            inputStream.close();
            Assert.assertTrue(expectedVector.equals(actualVector));
        } catch (IOException e) {
            Assert.fail(e.getMessage());
        }
    }
}
