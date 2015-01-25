package org.platanios.learn.classification.reflection;

import org.junit.Assert;
import org.junit.Test;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ErrorEstimationTest {
    @Test
    public void testAgreementRatesPowerSetVector() {
        String filename = "/Users/Anthony/Development/GitHub/learn/learn/data/combination/error/nell/input/animal.csv";
        String separator = ",";
        int maximumOrder = 4;
        double[] classificationThresholds = new double[] { 0.05, 0.05, 0.1, 0.05 };
        ErrorEstimationData data = parseLabeledDataFromCSVFile(filename,
                                                               separator,
                                                               classificationThresholds,
                                                               maximumOrder,
                                                               false);
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
        String filename = "/Users/Anthony/Development/GitHub/learn/learn/data/combination/error/nell/input/animal.csv";
//        filename = "/Users/Anthony/Development/GitHub/learn/learn/data/combination/error/brain/input/region_1.csv";
        String separator = ",";
        int highestOrder = 4;
        double[] classificationThresholds = new double[] { 0.05, 0.05, 0.1, 0.05 };
//        classificationThresholds = new double[] { 0.5 };
        ErrorEstimationData data = parseLabeledDataFromCSVFile(filename,
                                                               separator,
                                                               classificationThresholds,
                                                               highestOrder,
                                                               true);
        ErrorEstimation ere = new ErrorEstimation.Builder(data)
                .optimizationSolverType(ErrorEstimationInternalSolver.IP_OPT)
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

    public static ErrorEstimationData parseLabeledDataFromCSVFile(String filename, String separator) {
        return parseLabeledDataFromCSVFile(filename, separator, null, -1, true);
    }

    public static ErrorEstimationData parseLabeledDataFromCSVFile(String filename, String separator, double classificationThreshold) {
        return parseLabeledDataFromCSVFile(filename, separator, new double[] { classificationThreshold }, -1, true);
    }

    public static ErrorEstimationData parseLabeledDataFromCSVFile(String filename, String separator, double[] classificationThresholds) {
        return parseLabeledDataFromCSVFile(filename, separator, classificationThresholds, -1, true);
    }

    public static ErrorEstimationData parseLabeledDataFromCSVFile(String filename, String separator, double[] classificationThresholds, int maximumOrder) {
        return parseLabeledDataFromCSVFile(filename, separator, classificationThresholds, maximumOrder, true);
    }

    public static ErrorEstimationData parseLabeledDataFromCSVFile(String filename, String separator, boolean onlyEvenCardinalitySubsetsAgreements) {
        return parseLabeledDataFromCSVFile(filename, separator, null, -1, onlyEvenCardinalitySubsetsAgreements);
    }

    public static ErrorEstimationData parseLabeledDataFromCSVFile(String filename, String separator, double classificationThreshold, boolean onlyEvenCardinalitySubsetsAgreements) {
        return parseLabeledDataFromCSVFile(filename, separator, new double[] { classificationThreshold }, -1, onlyEvenCardinalitySubsetsAgreements);
    }

    public static ErrorEstimationData parseLabeledDataFromCSVFile(String filename, String separator, double[] classificationThresholds, boolean onlyEvenCardinalitySubsetsAgreements) {
        return parseLabeledDataFromCSVFile(filename, separator, classificationThresholds, -1, onlyEvenCardinalitySubsetsAgreements);
    }

    public static ErrorEstimationData parseLabeledDataFromCSVFile(String filename,
                                                                  String separator,
                                                                  double[] classificationThresholds,
                                                                  int highestOrder,
                                                                  boolean onlyEvenCardinalitySubsetsAgreements) {
        int numberOfFunctions = 0;

        BufferedReader br = null;
        String line;
        String[] classifiersNames = null;
        List<boolean[]> classifiersOutputsList = new ArrayList<>();
        List<Boolean> trueLabelsList = new ArrayList<Boolean>();

        try {
            br = new BufferedReader(new FileReader(filename));
            line = br.readLine();
            classifiersNames = line.split(separator);
            numberOfFunctions = classifiersNames.length - 1;
            while ((line = br.readLine()) != null) {
                String[] outputs = line.split(separator);
                trueLabelsList.add(!outputs[0].equals("0"));
                boolean[] booleanOutputs = new boolean[outputs.length - 1];
                for (int i = 1; i < outputs.length; i++) {
                    if (classificationThresholds == null) {
                        booleanOutputs[i - 1] = Double.parseDouble(outputs[i]) >= 0.5;
                    } else if (classificationThresholds.length == 1) {
                        booleanOutputs[i - 1] = Double.parseDouble(outputs[i]) >= classificationThresholds[0];
                    } else {
                        booleanOutputs[i - 1] = Double.parseDouble(outputs[i]) >= classificationThresholds[i - 1];
                    }
                }
                classifiersOutputsList.add(booleanOutputs);
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        if (highestOrder == -1) {
            highestOrder = numberOfFunctions;
        }

        return new ErrorEstimationData.Builder(classifiersOutputsList,
                                               trueLabelsList,
                                               highestOrder,
                                               onlyEvenCardinalitySubsetsAgreements)
                .functionNames(classifiersNames)
                .build();
    }
}
