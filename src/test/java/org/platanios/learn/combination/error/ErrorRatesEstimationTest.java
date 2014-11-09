package org.platanios.learn.combination.error;

import org.junit.Assert;
import org.junit.Test;
import org.platanios.learn.classification.reflection.perception.DataPreprocessing;
import org.platanios.learn.classification.reflection.perception.ErrorRatesEstimation;
import org.platanios.learn.classification.reflection.perception.EstimationData;
import org.platanios.learn.classification.reflection.perception.OptimizationSolverType;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ErrorRatesEstimationTest {
    @Test
    public void testAgreementRatesPowerSetVector() {
        String filename = "/Users/Anthony/Development/GitHub/learn/data/combination/error/nell/input/animal.csv";
        String separator = ",";
        int maximumOrder = 4;
        double[] classificationThresholds = new double[] { 0.05, 0.05, 0.1, 0.05 };
        EstimationData data = DataPreprocessing.parseLabeledDataFromCSVFile(filename, separator, classificationThresholds, maximumOrder, false);
        double[] actualResult = data.getAgreementRates().array;
        double[] expectedResult = new double[]{
                0.646698499975884,
                0.698548208170549,
                0.575362947957363,
                0.693290888921044,
                0.583803598128587,
                0.607678580041480,
                0.519268798533739,
                0.402932523030917,
                0.440794868084696,
                0.442386533545555,
                0.337481309988907
        };
        Assert.assertArrayEquals(expectedResult, actualResult, 1E-15);
    }

    @Test
    public void testErrorRatesEstimationVector() {
        String filename = "/Users/Anthony/Development/GitHub/learn/data/combination/error/nell/input/animal.csv";
//        filename = "/Users/Anthony/Development/GitHub/learn/data/combination/error/brain/input/region_1.csv";
        String separator = ",";
        int highestOrder = 4;
        double[] classificationThresholds = new double[] { 0.05, 0.05, 0.1, 0.05 };
//        classificationThresholds = new double[] { 0.5 };
        EstimationData data = DataPreprocessing.parseLabeledDataFromCSVFile(filename,
                                                                            separator,
                                                                            classificationThresholds,
                                                                            highestOrder,
                                                                            true);
        ErrorRatesEstimation ere = new ErrorRatesEstimation.Builder(data)
                .optimizationSolverType(OptimizationSolverType.IP_OPT)
                .build();
        data = ere.solve();

        double[] obtainedErrorRates = data.getErrorRates().array;
        double[] sampleErrorRates = data.getSampleErrorRates().array;

        double mad = 0.0;
        double mad_ind = 0.0;

        for (int i = 0; i < obtainedErrorRates.length; i++) {
            mad += Math.abs(sampleErrorRates[i] - obtainedErrorRates[i]);
            if (i < data.getNumberOfFunctions()) {
                mad_ind += Math.abs(sampleErrorRates[i] - obtainedErrorRates[i]);
            }
        }

        mad /= sampleErrorRates.length;
        mad_ind /= data.getNumberOfFunctions();

        Assert.assertEquals(0.022264846242268578, mad, 1E-10);
        Assert.assertEquals(0.03873716293662867, mad_ind, 1E-10);
    }
}
