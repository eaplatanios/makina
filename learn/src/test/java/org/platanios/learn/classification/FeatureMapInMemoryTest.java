package org.platanios.learn.classification;

import org.junit.Assert;
import org.junit.Test;
import org.platanios.learn.math.matrix.DenseVector;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @author Emmanouil Antonios Platanios
 */
public class FeatureMapInMemoryTest {
    @Test
    public void testAddingAndGettingSingleNameSingleViewFeatureVectors() {
        DenseVector appleView1Vector = DenseVector.generateOnesVector(10);
        DenseVector appleView2Vector = DenseVector.generateRandomVector(20);
        DenseVector orangeView0Vector = DenseVector.generateRandomVector(15);
        DenseVector orangeView1Vector = DenseVector.generateRandomVector(5);
        FeatureMap<DenseVector> featureMap = new FeatureMapInMemory<>(3);
        featureMap.addSingleViewFeatureMappings("apple", appleView1Vector, 1);
        featureMap.addSingleViewFeatureMappings("apple", appleView2Vector, 2);
        featureMap.addSingleViewFeatureMappings("orange", orangeView0Vector, 0);
        featureMap.addSingleViewFeatureMappings("orange", orangeView1Vector, 1);
        Assert.assertTrue(featureMap.getSingleViewFeatureVector("apple", 0) == null);
        Assert.assertTrue(featureMap.getSingleViewFeatureVector("apple", 1).equals(appleView1Vector));
        Assert.assertTrue(featureMap.getSingleViewFeatureVector("apple", 2).equals(appleView2Vector));
        Assert.assertTrue(featureMap.getSingleViewFeatureVector("orange", 0).equals(orangeView0Vector));
        Assert.assertTrue(featureMap.getSingleViewFeatureVector("orange", 1).equals(orangeView1Vector));
        Assert.assertTrue(featureMap.getSingleViewFeatureVector("orange", 2) == null);
    }

    @Test
    public void testAddingAndGettingMultipleNameSingleViewFeatureVectors() {
        DenseVector appleView1Vector = DenseVector.generateOnesVector(10);
        DenseVector appleView2Vector = DenseVector.generateRandomVector(20);
        DenseVector orangeView0Vector = DenseVector.generateRandomVector(15);
        DenseVector orangeView1Vector = DenseVector.generateRandomVector(5);
        Map<String, DenseVector> view0FeatureMappings = new HashMap<>();
        view0FeatureMappings.put("orange", orangeView0Vector);
        Map<String, DenseVector> view1FeatureMappings = new HashMap<>();
        view1FeatureMappings.put("apple", appleView1Vector);
        view1FeatureMappings.put("orange", orangeView1Vector);
        Map<String, DenseVector> view2FeatureMappings = new HashMap<>();
        view2FeatureMappings.put("apple", appleView2Vector);
        FeatureMap<DenseVector> featureMap = new FeatureMapInMemory<>(3);
        featureMap.addSingleViewFeatureMappings(view1FeatureMappings, 1);
        featureMap.addSingleViewFeatureMappings(view2FeatureMappings, 2);
        featureMap.addSingleViewFeatureMappings(view0FeatureMappings, 0);
        Assert.assertTrue(featureMap.getSingleViewFeatureVector("apple", 0) == null);
        Assert.assertTrue(featureMap.getSingleViewFeatureVector("apple", 1).equals(appleView1Vector));
        Assert.assertTrue(featureMap.getSingleViewFeatureVector("apple", 2).equals(appleView2Vector));
        Assert.assertTrue(featureMap.getSingleViewFeatureVector("orange", 0).equals(orangeView0Vector));
        Assert.assertTrue(featureMap.getSingleViewFeatureVector("orange", 1).equals(orangeView1Vector));
        Assert.assertTrue(featureMap.getSingleViewFeatureVector("orange", 2) == null);
    }

    @Test
    public void testAddingAndGettingSingleNameAllViewFeatureVectors() {
        DenseVector appleView1Vector = DenseVector.generateOnesVector(10);
        DenseVector appleView2Vector = DenseVector.generateRandomVector(20);
        DenseVector orangeView0Vector = DenseVector.generateRandomVector(15);
        DenseVector orangeView1Vector = DenseVector.generateRandomVector(5);
        List<DenseVector> appleFeatures = new ArrayList<>(3);
        appleFeatures.add(null);
        appleFeatures.add(appleView1Vector);
        appleFeatures.add(appleView2Vector);
        List<DenseVector> orangeFeatures = new ArrayList<>(3);
        orangeFeatures.add(orangeView0Vector);
        orangeFeatures.add(orangeView1Vector);
        orangeFeatures.add(null);
        FeatureMap<DenseVector> featureMap = new FeatureMapInMemory<>(3);
        featureMap.addFeatureMappings("apple", appleFeatures);
        featureMap.addFeatureMappings("orange", orangeFeatures);
        Assert.assertTrue(featureMap.getSingleViewFeatureVector("apple", 0) == null);
        Assert.assertTrue(featureMap.getSingleViewFeatureVector("apple", 1).equals(appleView1Vector));
        Assert.assertTrue(featureMap.getSingleViewFeatureVector("apple", 2).equals(appleView2Vector));
        Assert.assertTrue(featureMap.getSingleViewFeatureVector("orange", 0).equals(orangeView0Vector));
        Assert.assertTrue(featureMap.getSingleViewFeatureVector("orange", 1).equals(orangeView1Vector));
        Assert.assertTrue(featureMap.getSingleViewFeatureVector("orange", 2) == null);
    }

    @Test
    public void testAddingAndGettingMultipleNameAllViewFeatureVectors() {
        DenseVector appleView1Vector = DenseVector.generateOnesVector(10);
        DenseVector appleView2Vector = DenseVector.generateRandomVector(20);
        DenseVector orangeView0Vector = DenseVector.generateRandomVector(15);
        DenseVector orangeView1Vector = DenseVector.generateRandomVector(5);
        List<DenseVector> appleFeatures = new ArrayList<>(3);
        appleFeatures.add(null);
        appleFeatures.add(appleView1Vector);
        appleFeatures.add(appleView2Vector);
        List<DenseVector> orangeFeatures = new ArrayList<>(3);
        orangeFeatures.add(orangeView0Vector);
        orangeFeatures.add(orangeView1Vector);
        orangeFeatures.add(null);
        Map<String, List<DenseVector>> featureMappings = new HashMap<>();
        featureMappings.put("apple", appleFeatures);
        featureMappings.put("orange", orangeFeatures);
        FeatureMap<DenseVector> featureMap = new FeatureMapInMemory<>(3);
        featureMap.addFeatureMappings(featureMappings);
        Assert.assertTrue(featureMap.getSingleViewFeatureVector("apple", 0) == null);
        Assert.assertTrue(featureMap.getSingleViewFeatureVector("apple", 1).equals(appleView1Vector));
        Assert.assertTrue(featureMap.getSingleViewFeatureVector("apple", 2).equals(appleView2Vector));
        Assert.assertTrue(featureMap.getSingleViewFeatureVector("orange", 0).equals(orangeView0Vector));
        Assert.assertTrue(featureMap.getSingleViewFeatureVector("orange", 1).equals(orangeView1Vector));
        Assert.assertTrue(featureMap.getSingleViewFeatureVector("orange", 2) == null);
    }
}
