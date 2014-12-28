package org.platanios.learn.data;

import org.junit.Assert;
import org.junit.Test;
import org.platanios.learn.math.matrix.DenseVector;

/**
 * @author Emmanouil Antonios Platanios
 */
public class DataSetUsingFeatureMapTest {
    @Test
    public void testAddingAndGettingDataInstances() {
        DenseVector appleView1Vector = DenseVector.generateOnesVector(10);
        DenseVector appleView2Vector = DenseVector.generateRandomVector(20);
        DenseVector orangeView0Vector = DenseVector.generateRandomVector(15);
        DenseVector orangeView1Vector = DenseVector.generateRandomVector(5);
        FeatureMap<DenseVector> featureMap = new FeatureMapInMemory<>(3);
        featureMap.addFeatureMappings("apple", appleView1Vector, 1);
        featureMap.addFeatureMappings("apple", appleView2Vector, 2);
        featureMap.addFeatureMappings("orange", orangeView0Vector, 0);
        featureMap.addFeatureMappings("orange", orangeView1Vector, 1);

        DataSetUsingFeatureMap<DenseVector, DataInstance<DenseVector>> dataSet =
                new DataSetUsingFeatureMap<>(featureMap, 1);
        dataSet.add(new DataInstance<>("apple", appleView1Vector));
        dataSet.add(new DataInstance<>("orange", null));
        DataInstance<DenseVector> appleDataInstance = dataSet.get(0);
        DataInstance<DenseVector> orangeDataInstance = dataSet.get(1);

        Assert.assertTrue(appleDataInstance.name().equals("apple"));
        Assert.assertTrue(featureMap.getFeatureVector("apple", 1).equals(appleDataInstance.features()));
        Assert.assertTrue(orangeDataInstance.name().equals("orange"));
        Assert.assertTrue(featureMap.getFeatureVector("orange", 1).equals(orangeDataInstance.features()));
    }

    @Test
    public void testAddingAndGettingLabeledDataInstances() {
        DenseVector appleView1Vector = DenseVector.generateOnesVector(10);
        DenseVector appleView2Vector = DenseVector.generateRandomVector(20);
        DenseVector orangeView0Vector = DenseVector.generateRandomVector(15);
        DenseVector orangeView1Vector = DenseVector.generateRandomVector(5);
        FeatureMap<DenseVector> featureMap = new FeatureMapInMemory<>(3);
        featureMap.addFeatureMappings("apple", appleView1Vector, 1);
        featureMap.addFeatureMappings("apple", appleView2Vector, 2);
        featureMap.addFeatureMappings("orange", orangeView0Vector, 0);
        featureMap.addFeatureMappings("orange", orangeView1Vector, 1);

        DataSetUsingFeatureMap<DenseVector, LabeledDataInstance<DenseVector, Integer>> dataSet =
                new DataSetUsingFeatureMap<>(featureMap, 1);
        dataSet.add(new LabeledDataInstance<>("apple", appleView1Vector, 1, null));
        dataSet.add(new LabeledDataInstance<>("orange", null, 2, null));
        LabeledDataInstance<DenseVector, Integer> appleDataInstance = dataSet.get(0);
        LabeledDataInstance<DenseVector, Integer> orangeDataInstance = dataSet.get(1);

        Assert.assertTrue(appleDataInstance.name().equals("apple"));
        Assert.assertTrue(featureMap.getFeatureVector("apple", 1).equals(appleDataInstance.features()));
        Assert.assertTrue(appleDataInstance.label() == 1);
        Assert.assertTrue(appleDataInstance.source() == null);
        Assert.assertTrue(orangeDataInstance.name().equals("orange"));
        Assert.assertTrue(featureMap.getFeatureVector("orange", 1).equals(orangeDataInstance.features()));
        Assert.assertTrue(orangeDataInstance.label() == 2);
        Assert.assertTrue(orangeDataInstance.source() == null);
    }
}
