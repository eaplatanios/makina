package org.platanios.learn.combination;

import com.google.common.primitives.Booleans;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class DataPreprocessing {
    public static DataStructure parseLabeledDataFromCSVFile(String filename, String separator) {
        return parseLabeledDataFromCSVFile(filename, separator, null, -1, true);
    }

    public static DataStructure parseLabeledDataFromCSVFile(String filename, String separator, double classificationThreshold) {
        return parseLabeledDataFromCSVFile(filename, separator, new double[] { classificationThreshold }, -1, true);
    }

    public static DataStructure parseLabeledDataFromCSVFile(String filename, String separator, double[] classificationThresholds) {
        return parseLabeledDataFromCSVFile(filename, separator, classificationThresholds, -1, true);
    }

    public static DataStructure parseLabeledDataFromCSVFile(String filename, String separator, double[] classificationThresholds, int maximumOrder) {
        return parseLabeledDataFromCSVFile(filename, separator, classificationThresholds, maximumOrder, true);
    }

    public static DataStructure parseLabeledDataFromCSVFile(String filename, String separator, boolean onlyEvenCardinalitySubsetsAgreements) {
        return parseLabeledDataFromCSVFile(filename, separator, null, -1, onlyEvenCardinalitySubsetsAgreements);
    }

    public static DataStructure parseLabeledDataFromCSVFile(String filename, String separator, double classificationThreshold, boolean onlyEvenCardinalitySubsetsAgreements) {
        return parseLabeledDataFromCSVFile(filename, separator, new double[] { classificationThreshold }, -1, onlyEvenCardinalitySubsetsAgreements);
    }

    public static DataStructure parseLabeledDataFromCSVFile(String filename, String separator, double[] classificationThresholds, boolean onlyEvenCardinalitySubsetsAgreements) {
        return parseLabeledDataFromCSVFile(filename, separator, classificationThresholds, -1, onlyEvenCardinalitySubsetsAgreements);
    }

    public static DataStructure parseLabeledDataFromCSVFile(String filename, String separator, double[] classificationThresholds, int maximumOrder, boolean onlyEvenCardinalitySubsetsAgreements) {
        int numberOfFunctions = 0;

        BufferedReader br = null;
        String line;
        String[] classifiersNames = null;
        List<boolean[]> classifiersOutputsList = new ArrayList<boolean[]>();
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
        } catch (FileNotFoundException e) {
            e.printStackTrace();
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

        if (maximumOrder == -1) {
            maximumOrder = numberOfFunctions;
        }

        boolean[][] classifiersOutputs = classifiersOutputsList.toArray(new boolean[classifiersOutputsList.size()][]);
        boolean[] trueLabels = Booleans.toArray(trueLabelsList);

        ErrorRatesVector errorRates = new ErrorRatesVector(numberOfFunctions, maximumOrder, 0.25);
        ErrorRatesVector sampleErrorRates = new ErrorRatesVector(numberOfFunctions, maximumOrder, trueLabels, classifiersOutputs);
        AgreementRatesVector agreementRates = new AgreementRatesVector(numberOfFunctions, maximumOrder, classifiersOutputs, onlyEvenCardinalitySubsetsAgreements);

        return new DataStructure(numberOfFunctions, maximumOrder, errorRates, sampleErrorRates, agreementRates, classifiersNames);
    }
}
