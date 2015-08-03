package org.platanios.learn.evaluation;

import org.junit.Test;
import org.platanios.learn.classification.Utilities;
import org.platanios.learn.data.DataSet;
import org.platanios.learn.data.PredictedDataInstance;
import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public class PrecisionRecallTest {
    @Test
    public void testPlotCurvesAllPoints() {
        String filename = "/Users/Anthony/Development/Data Sets/Classification/covtype.binary.scale.txt";
        DataSet<PredictedDataInstance<Vector, Double>> dataSet = Utilities.parseCovTypeDataFromFile(filename, false);
        PrecisionRecall<Vector, Double> pr = new PrecisionRecall<>();
        pr.addResult("Test Curve #1", dataSet.subSet(0, 1000), prediction -> prediction.features().sum() > 5.8);
        pr.addResult("Test Curve #2", dataSet.subSet(0, 1000), prediction -> prediction.features().sum() > 6.2);
        pr.addResult("Test Curve #3", dataSet.subSet(0, 1000), prediction -> prediction.features().sum() > 6.4);
        pr.plotCurves();
    }

    @Test
    public void testPlotCurves100Points() {
        String filename = "/Users/Anthony/Development/Data Sets/Classification/covtype.binary.scale.txt";
        DataSet<PredictedDataInstance<Vector, Double>> dataSet = Utilities.parseCovTypeDataFromFile(filename, false);
        PrecisionRecall<Vector, Double> pr = new PrecisionRecall<>(100);
        pr.addResult("Test Curve #1", dataSet.subSet(0, 1000), prediction -> prediction.features().sum() > 5.8);
        pr.addResult("Test Curve #2", dataSet.subSet(0, 1000), prediction -> prediction.features().sum() > 6.2);
        pr.addResult("Test Curve #3", dataSet.subSet(0, 1000), prediction -> prediction.features().sum() > 6.4);
        pr.plotCurves();
    }
}
