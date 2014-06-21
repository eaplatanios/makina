package org.platanios.learn.combination;

import org.junit.Assert;
import org.junit.Test;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ErrorRatesEstimationTest {
    @Test
    public void testErrorRatesEstimationVector() {
        String filename = ErrorRatesEstimationTest.class.getResource("resources/food_labels.csv").getPath();
        String separator = ",";
        boolean[][] observations = DataPreprocessing.getObservationsFromCsvFile(filename, separator);
        filename = ErrorRatesEstimationTest.class.getResource("resources/food_sample_error_rates.csv").getPath();
        double[] sampleErrorRates = DataPreprocessing.getSampleErrorRatesFromCsvFile(filename);

        ErrorRatesEstimation ere = new ErrorRatesEstimation(new AgreementRatesVector(4, 4, observations, true), 4, 4);
        double[] obtainedErrorRates = ere.solve().errorRates;

        double mad = 0.0;
        double mad_ind = 0.0;

        for (int i = 0; i < obtainedErrorRates.length; i++) {
            mad += Math.abs(sampleErrorRates[i] - obtainedErrorRates[i]);
            if (i < observations[0].length) {
                mad_ind += Math.abs(sampleErrorRates[i] - obtainedErrorRates[i]);
            }
        }

        mad /= sampleErrorRates.length;
        mad_ind /= observations[0].length;

        Assert.assertEquals(mad, 0.0038168724169213237, 1E-18);
        Assert.assertEquals(mad_ind, 0.008144776534951827, 1E-18);
    }
}
