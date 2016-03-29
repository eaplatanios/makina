package org.platanios.math.matrix;

import cern.colt.map.OpenIntDoubleHashMap;
import org.junit.Assert;
import org.junit.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;

/**
 * @author Emmanouil Antonios Platanios
 */
public class HashVectorTest {
    @Test
    public void testEquals() {
        throw new UnsupportedOperationException();
    }

    @Test
    public void testSerialization() {
        int vectorSize = 1000;
        try {
            OpenIntDoubleHashMap elements = new OpenIntDoubleHashMap(4);
            elements.put(1, 0.53);
            elements.put(5, 0.32);
            elements.put(8, 0.91);
            elements.put(35, 0.05);
            // Test for when we do not store the vector type
            ByteArrayOutputStream outputStream = new ByteArrayOutputStream(4 + (vectorSize << 3));
            HashVector vector = new HashVector(vectorSize, elements);
            HashVector expectedVector = vector.copy();
            vector.write(outputStream, false);
            outputStream.close();
            ByteArrayInputStream inputStream = new ByteArrayInputStream(outputStream.toByteArray());
            HashVector actualVector = HashVector.read(inputStream, false);
            inputStream.close();
            Assert.assertTrue(expectedVector.equals(actualVector));
            // Test for when we store the vector type
            outputStream = new ByteArrayOutputStream(4 + (vectorSize << 3));
            vector = new HashVector(vectorSize, elements);
            vector.write(outputStream, true);
            outputStream.close();
            inputStream = new ByteArrayInputStream(outputStream.toByteArray());
            actualVector = HashVector.read(inputStream, true);
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
            OpenIntDoubleHashMap elements = new OpenIntDoubleHashMap(4);
            elements.put(1, 0.53);
            elements.put(5, 0.32);
            elements.put(8, 0.91);
            elements.put(35, 0.05);
            // Test for when we do not store the vector type
            HashVector vector = new HashVector(vectorSize, elements);
            HashVector expectedVector = vector.copy();
            InputStream inputStream = vector.getEncoder(false);
            HashVector actualVector = HashVector.read(inputStream, false);
            inputStream.close();
            Assert.assertTrue(expectedVector.equals(actualVector));
            // Test for when we store the vector type
            vector = new HashVector(vectorSize, elements);
            inputStream = vector.getEncoder(true);
            actualVector = HashVector.read(inputStream, true);
            inputStream.close();
            Assert.assertTrue(expectedVector.equals(actualVector));
        } catch (IOException e) {
            Assert.fail(e.getMessage());
        }
    }
}
