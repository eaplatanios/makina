package org.platanios.learn.combination;

import org.junit.Assert;
import org.junit.Test;

/**
 * @author Emmanouil Antonios Platanios
 */
public class AgreementRatesVectorTest {
    @Test
    public void testAgreementRatesVector() {
        String filename = ErrorRatesEstimationTest.class.getResource("resources/animal.csv").getPath();
        String separator = ",";
        int maximumOrder = 4;
        double[] classificationThresholds = new double[] { 0.05, 0.05, 0.1, 0.05 };
        DataStructure data = DataPreprocessing.parseLabeledDataFromCSVFile(filename, separator, classificationThresholds, maximumOrder, false);
        double[] obtainedAgreementRates = data.getAgreementRates().agreementRates;
        double[] correctAgreementRates = new double[]{
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
        Assert.assertArrayEquals(obtainedAgreementRates, correctAgreementRates, 1E-15);
    }
}
