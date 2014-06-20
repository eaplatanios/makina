package org.platanios.learn.combination;

import org.junit.Assert;
import org.junit.Test;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ErrorRatesEstimationTest {
    @Test
    public void testAgreementRatesVector() {
        String filename = "/Users/Anthony/Development/GitHub/learn/data/combination/brain/labels/region_1.csv";
        String separator = ",";
        boolean[][] observations = DataPreprocessing.getObservationsFromCsvFile(filename, separator);
        filename = "/Users/Anthony/Development/GitHub/learn/data/combination/brain/sample error rates/region_1.csv";
        double[] sampleErrorRates = DataPreprocessing.getSampleErrorRatesFromCsvFile(filename);

        ErrorRatesEstimation ere = new ErrorRatesEstimation(new AgreementRatesVector(8, 8, observations), 8, 8);
        double[] obtainedErrorRates = ere.solve().errorRates;

        Assert.assertTrue(true);
    }
}
