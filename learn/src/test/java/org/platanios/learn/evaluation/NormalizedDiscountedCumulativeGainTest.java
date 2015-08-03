package org.platanios.learn.evaluation;

import org.junit.Test;
import org.platanios.learn.classification.Utilities;
import org.platanios.learn.data.DataSet;
import org.platanios.learn.data.PredictedDataInstance;
import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public class NormalizedDiscountedCumulativeGainTest {
    @Test
    public void testPlotCurvesUsingOriginalFormulation() {
        String filename = "/Users/Anthony/Development/Data Sets/Classification/covtype.binary.scale.txt";
        DataSet<PredictedDataInstance<Vector, Double>> dataSet = Utilities.parseCovTypeDataFromFile(filename, false);
        NormalizedDiscountedCumulativeGain<Vector, Double> ndcg = new NormalizedDiscountedCumulativeGain<>(false);
        ndcg.addResult("Test Curve #1", dataSet.subSet(0, 1000), prediction -> prediction.features().sum() > 5.8);
        ndcg.addResult("Test Curve #2", dataSet.subSet(0, 1000), prediction -> prediction.features().sum() > 6.2);
        ndcg.addResult("Test Curve #3", dataSet.subSet(0, 1000), prediction -> prediction.features().sum() > 6.4);
        ndcg.plotCurves();
    }

    @Test
    public void testPlotCurvesUsingAlternativeFormulation() {
        String filename = "/Users/Anthony/Development/Data Sets/Classification/covtype.binary.scale.txt";
        DataSet<PredictedDataInstance<Vector, Double>> dataSet = Utilities.parseCovTypeDataFromFile(filename, false);
        NormalizedDiscountedCumulativeGain<Vector, Double> ndcg = new NormalizedDiscountedCumulativeGain<>(true);
        ndcg.addResult("Test Curve #1", dataSet.subSet(0, 1000), prediction -> prediction.features().sum() > 5.8);
        ndcg.addResult("Test Curve #2", dataSet.subSet(0, 1000), prediction -> prediction.features().sum() > 6.2);
        ndcg.addResult("Test Curve #3", dataSet.subSet(0, 1000), prediction -> prediction.features().sum() > 6.4);
        ndcg.plotCurves();
    }
}
