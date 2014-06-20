package org.platanios.learn.combination;

import org.junit.Assert;
import org.junit.Test;

/**
 * @author Emmanouil Antonios Platanios
 */
public class AgreementRatesVectorTest {
    @Test
    public void testAgreementRatesVector() {
        String filename = "C:\\Users\\Anthony\\Documents\\GitHub\\learn\\data\\combination\\nell\\labels\\animal.csv";
        String separator = ",";
        boolean[][] observations = DataPreprocessing.getObservationsFromCsvFile(filename, separator);
        double[] obtainedAgreementRates = new AgreementRatesVector(4, 4, observations).agreementRates;
        double[] correctAgreementRates = new double[] { 0.74, 0.66, 0.62, 0.80, 0.52, 0.60, 0.60, 0.44, 0.44, 0.46, 0.40 };
        Assert.assertArrayEquals(obtainedAgreementRates, correctAgreementRates, 0);
    }
}
