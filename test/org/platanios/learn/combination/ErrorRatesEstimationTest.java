package org.platanios.learn.combination;

import org.junit.Assert;
import org.junit.Test;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ErrorRatesEstimationTest {
    @Test
    public void testErrorRatesEstimationVector() {
        String filename = ErrorRatesEstimationTest.class.getResource("resources/animal.csv").getPath();
        String separator = ",";
        int maximumOrder = 4;
        double[] classificationThresholds = new double[] { 0.05, 0.05, 0.1, 0.05 };
        DataStructure data = DataPreprocessing.parseLabeledDataFromCSVFile(filename, separator, classificationThresholds, maximumOrder);
        ErrorRatesEstimation ere = new ErrorRatesEstimation(data);
        data = ere.solve();

        double[] obtainedErrorRates = data.getErrorRates().errorRates;
        double[] sampleErrorRates = data.getSampleErrorRates().errorRates;

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

        Assert.assertEquals(mad, 0.023921770444994158, 1E-18);
        Assert.assertEquals(mad_ind, 0.041488317430976, 1E-18);
    }
}
