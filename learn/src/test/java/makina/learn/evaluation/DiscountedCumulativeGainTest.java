//package makina.learn.evaluation;
//
//import makina.learn.classification.Utilities;
//import makina.learn.data.DataSet;
//import makina.learn.data.PredictedDataInstance;
//import makina.math.matrix.Vector;
//import org.junit.Test;
//
///**
// * @author Emmanouil Antonios Platanios
// */
//public class DiscountedCumulativeGainTest {
//    @Test
//    public void testPlotCurvesUsingOriginalFormulation() {
//        String filename = "/Users/Anthony/Development/Data Sets/Classification/covtype.binary.scale.txt";
//        DataSet<PredictedDataInstance<Vector, Double>> dataSet = Utilities.parseCovTypeDataFromFile(filename, false);
//        DiscountedCumulativeGain<Vector, Double> dcg = new DiscountedCumulativeGain<>(false);
//        dcg.addResult("Test Curve #1", dataSet.subSet(0, 1000), prediction -> prediction.features().sum() > 5.8);
//        dcg.addResult("Test Curve #2", dataSet.subSet(0, 1000), prediction -> prediction.features().sum() > 6.2);
//        dcg.addResult("Test Curve #3", dataSet.subSet(0, 1000), prediction -> prediction.features().sum() > 6.4);
//        dcg.plotCurves();
//    }
//
//    @Test
//    public void testPlotCurvesUsingAlternativeFormulation() {
//        String filename = "/Users/Anthony/Development/Data Sets/Classification/covtype.binary.scale.txt";
//        DataSet<PredictedDataInstance<Vector, Double>> dataSet = Utilities.parseCovTypeDataFromFile(filename, false);
//        DiscountedCumulativeGain<Vector, Double> dcg = new DiscountedCumulativeGain<>(true);
//        dcg.addResult("Test Curve #1", dataSet.subSet(0, 1000), prediction -> prediction.features().sum() > 5.8);
//        dcg.addResult("Test Curve #2", dataSet.subSet(0, 1000), prediction -> prediction.features().sum() > 6.2);
//        dcg.addResult("Test Curve #3", dataSet.subSet(0, 1000), prediction -> prediction.features().sum() > 6.4);
//        dcg.plotCurves();
//    }
//}
