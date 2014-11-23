package org.platanios.learn.math.matrix;

import cern.colt.map.OpenIntDoubleHashMap;
import org.junit.Assert;
import org.junit.Test;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

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
        throw new NotImplementedException();
    }

    @Test
    public void testSerialization() {
        int vectorSize = 1000;
        try {
            ByteArrayOutputStream outputStream = new ByteArrayOutputStream(4 + (vectorSize << 3));
            OpenIntDoubleHashMap elements = new OpenIntDoubleHashMap(4);
            elements.put(1, 0.53);
            elements.put(5, 0.32);
            elements.put(8, 0.91);
            elements.put(35, 0.05);
            HashVector vector = new HashVector(vectorSize, elements);
            HashVector expectedVector = vector.copy();
            vector.write(outputStream);
            outputStream.close();
            ByteArrayInputStream inputStream = new ByteArrayInputStream(outputStream.toByteArray());
            HashVector actualVector = HashVector.read(inputStream);
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
            HashVector vector = new HashVector(vectorSize, elements);
            HashVector expectedVector = vector.copy();
            InputStream inputStream = vector.getEncoder();
            HashVector actualVector = HashVector.read(inputStream);
            inputStream.close();
            Assert.assertTrue(expectedVector.equals(actualVector));
        } catch (IOException e) {
            Assert.fail(e.getMessage());
        }
    }
}
