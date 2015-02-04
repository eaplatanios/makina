package org.platanios.experiment;

import org.platanios.learn.classification.reflection.ErrorEstimationSimpleGraphicalModel;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ErrorEstimationGraphicalModelExperiment {
    public static void main(String[] args) {
        String filename = "/Users/Anthony/Development/GitHub/org.platanios/learn/data/combination/error/nell/input/";
//        filename = "/Users/Anthony/Development/GitHub/org.platanios/learn/data/combination/error/brain/input/";
        String separator = ",";
        double[] classificationThresholds = new double[] { 0.05, 0.05, 0.1, 0.05 };
//        classificationThresholds = new double[] { 0.5 };
        List<boolean[][]> functionOutputs = new ArrayList<>();
        List<boolean[]> trueLabels = new ArrayList<>();
        for (File file : new File(filename).listFiles()) {
            if (file.isFile()) {
                DomainData data = parseLabeledDataFromCSVFile(file,
                                                              separator,
                                                              classificationThresholds);
                functionOutputs.add(data.functionOutputs);
                trueLabels.add(data.trueLabels);
//                if (functionOutputs.size() == 2)
//                    break;
            }
        }
        ErrorEstimationSimpleGraphicalModel eegm = new ErrorEstimationSimpleGraphicalModel(functionOutputs, 100);
        eegm.performGibbsSampling();
        double[][] labelMeans = eegm.getLabelMeans();
        double[][] labelVariances = eegm.getLabelVariances();
        double[][] errorRatesMeans = eegm.getErrorRatesMeans();
        double[][] errorRatesVariances = eegm.getErrorRatesVariances();
        double[] combinedErrorRate = new double[functionOutputs.size()];
        double combinedErrorRateMean = 0;
        double[] mad = new double[functionOutputs.size()];
        double madMean = 0;
        for (int p = 0; p < functionOutputs.size(); p++) {
            combinedErrorRate[p] = 0;
            double[] realErrorRates = new double[errorRatesMeans[p].length];
            for (int i = 0; i < trueLabels.get(p).length; i++) {
                combinedErrorRate[p] += ((labelMeans[p][i] >= 0.5) != trueLabels.get(p)[i]) ? 1 : 0;
                for (int j = 0; j < errorRatesMeans[p].length; j++)
                    realErrorRates[j] += (functionOutputs.get(p)[i][j] != trueLabels.get(p)[i]) ? 1 : 0;
            }
            combinedErrorRate[p] /= trueLabels.get(p).length;
            combinedErrorRateMean += combinedErrorRate[p];
            mad[p] = 0;
            for (int j = 0; j < errorRatesMeans[p].length; j++) {
                realErrorRates[j] /= trueLabels.get(p).length;
                mad[p] += Math.abs(errorRatesMeans[p][j] - realErrorRates[j]);
            }
            mad[p] /= errorRatesMeans[p].length;
            madMean += mad[p];
            System.out.println("DOMAIN #" + p + ":");
            System.out.println("===================================================");
            System.out.print("Real error rates:\t\t");
            for (int j = 0; j < realErrorRates.length; j++)
                if (j != realErrorRates.length - 1)
                    System.out.print(realErrorRates[j] + ", ");
                else
                    System.out.print(realErrorRates[j] + "\n");
            System.out.print("Error rates means:\t\t");
            for (int j = 0; j < errorRatesMeans[p].length; j++)
                if (j != errorRatesMeans[p].length - 1)
                    System.out.print(errorRatesMeans[p][j] + ", ");
                else
                    System.out.print(errorRatesMeans[p][j] + "\n");
            System.out.print("Error rates variances:\t");
            for (int j = 0; j < errorRatesVariances[p].length; j++)
                if (j != errorRatesVariances[p].length - 1)
                    System.out.print(errorRatesVariances[p][j] + ", ");
                else
                    System.out.print(errorRatesVariances[p][j] + "\n");
            System.out.println("---------------------------------------------------");
            System.out.println("Combined labels error rate:\t" + combinedErrorRate[p]);
            System.out.println("MAD:\t\t\t\t\t\t" + mad[p]);
            System.out.println("---------------------------------------------------");
        }
        combinedErrorRateMean /= functionOutputs.size();
        madMean /= functionOutputs.size();
        System.out.println("===================================================");
        System.out.println("Combined Labels Error Rate Mean:\t" + combinedErrorRateMean);
        System.out.println("MAD Mean:\t\t\t\t\t\t" + madMean);
        System.out.println("===================================================");
    }

    public static DomainData parseLabeledDataFromCSVFile(
            File file,
            String separator,
            double[] classificationThresholds
    ) {
        BufferedReader br = null;
        String line;
        List<boolean[]> classifiersOutputsList = new ArrayList<>();
        List<Boolean> trueLabelsList = new ArrayList<>();

        try {
            br = new BufferedReader(new FileReader(file));
            br.readLine();
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

        boolean[] trueLabels = new boolean[trueLabelsList.size()];
        for (int i = 0; i < trueLabels.length; i++)
            trueLabels[i] = trueLabelsList.get(i);

        return new DomainData(classifiersOutputsList.toArray(new boolean[classifiersOutputsList.size()][]), trueLabels);
    }

    private static class DomainData {
        private boolean[][] functionOutputs;
        private boolean[] trueLabels;

        protected DomainData(boolean[][] functionOutputs, boolean[] trueLabels) {
            this.functionOutputs = functionOutputs;
            this.trueLabels = trueLabels;
        }

        protected boolean[][] getFunctionOutputs() {
            return functionOutputs;
        }

        protected boolean[] getTrueLabels() {
            return trueLabels;
        }
    }
}
