package org.platanios.learn.combination.error;

import org.junit.Assert;
import org.junit.Test;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ErrorRatesEstimationTest {
//    @Test
//    public void testAgreementRatesPowerSetVector() {
//        String filename = ErrorRatesEstimationTest.class.getResource("/animal.csv").getPath();
//        String separator = ",";
//        int maximumOrder = 4;
//        double[] classificationThresholds = new double[] { 0.05, 0.05, 0.1, 0.05 };
//        EstimationData data = DataPreprocessing.parseLabeledDataFromCSVFile(filename, separator, classificationThresholds, maximumOrder, false);
//
//        double[] actualResult = data.getAgreementRates().array;
//        double[] expectedResult = new double[]{
//                0.646698499975884,
//                0.698548208170549,
//                0.575362947957363,
//                0.693290888921044,
//                0.583803598128587,
//                0.607678580041480,
//                0.519268798533739,
//                0.402932523030917,
//                0.440794868084696,
//                0.442386533545555,
//                0.337481309988907
//        };
//        Assert.assertArrayEquals(expectedResult, actualResult, 1E-15);
//    }
//
//    @Test
//    public void testErrorRatesEstimationVector() {
//        String filename = ErrorRatesEstimationTest.class.getResource("/animal.csv").getPath();
////        filename = "/Users/Anthony/Development/GitHub/org.platanios.org.platanios.learn/data/combination/error/brain/input/region_1.csv";
//        String separator = ",";
//        int highestOrder = 4;
//        double[] classificationThresholds = new double[] { 0.05, 0.05, 0.1, 0.05 };
////        classificationThresholds = new double[] { 0.5 };
//        EstimationData data = DataPreprocessing.parseLabeledDataFromCSVFile(filename,
//                                                                            separator,
//                                                                            classificationThresholds,
//                                                                            highestOrder,
//                                                                            true);
//        ErrorRatesEstimation ere = new ErrorRatesEstimation(data);
//        data = ere.solve();
//
//        double[] obtainedErrorRates = data.getErrorRates().array;
//        double[] sampleErrorRates = data.getSampleErrorRates().array;
//
//        double mad = 0.0;
//        double mad_ind = 0.0;
//
//        for (int i = 0; i < obtainedErrorRates.length; i++) {
//            mad += Math.abs(sampleErrorRates[i] - obtainedErrorRates[i]);
//            if (i < data.getNumberOfFunctions()) {
//                mad_ind += Math.abs(sampleErrorRates[i] - obtainedErrorRates[i]);
//            }
//        }
//
//        mad /= sampleErrorRates.length;
//        mad_ind /= data.getNumberOfFunctions();
//
//        Assert.assertEquals(mad, 0.023921770444994158, 1E-18);
//        Assert.assertEquals(mad_ind, 0.041488317430976, 1E-18);
//    }
}
