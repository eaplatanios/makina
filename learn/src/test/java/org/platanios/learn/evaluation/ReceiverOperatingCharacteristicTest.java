package org.platanios.learn.evaluation;

import org.junit.Test;
import org.platanios.learn.classification.Utilities;
import org.platanios.learn.data.DataSet;
import org.platanios.learn.data.PredictedDataInstance;
import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ReceiverOperatingCharacteristicTest {
    @Test
    public void testPlotCurvesAllPoints() {
        String filename = "/Users/Anthony/Development/Data Sets/Classification/covtype.binary.scale.txt";
        DataSet<PredictedDataInstance<Vector, Double>> dataSet = Utilities.parseCovTypeDataFromFile(filename, false);
        ReceiverOperatingCharacteristic<Vector, Double> roc = new ReceiverOperatingCharacteristic<>();
        roc.addResult("Test Curve #1", dataSet.subSet(0, 1000), prediction -> prediction.features().sum() > 5.8);
        roc.addResult("Test Curve #2", dataSet.subSet(0, 1000), prediction -> prediction.features().sum() > 6.2);
        roc.addResult("Test Curve #3", dataSet.subSet(0, 1000), prediction -> prediction.features().sum() > 6.4);
        roc.plotCurves();
    }

    @Test
    public void testPlotCurves100Points() {
        String filename = "/Users/Anthony/Development/Data Sets/Classification/covtype.binary.scale.txt";
        DataSet<PredictedDataInstance<Vector, Double>> dataSet = Utilities.parseCovTypeDataFromFile(filename, false);
        ReceiverOperatingCharacteristic<Vector, Double> roc = new ReceiverOperatingCharacteristic<>(100);
        roc.addResult("Test Curve #1", dataSet.subSet(0, 1000), prediction -> prediction.features().sum() > 5.8);
        roc.addResult("Test Curve #2", dataSet.subSet(0, 1000), prediction -> prediction.features().sum() > 6.2);
        roc.addResult("Test Curve #3", dataSet.subSet(0, 1000), prediction -> prediction.features().sum() > 6.4);
        roc.plotCurves();
    }
}
