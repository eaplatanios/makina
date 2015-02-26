package org.platanios.experiment;

import org.platanios.learn.classification.reflection.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ErrorEstimationGraphicalModelExperiment {
    public static void main(String[] args) {
        int errorEstimationMethod = Integer.parseInt(args[0]);
//        manyThresholdsExperiment(errorEstimationMethod);
        String filename = "/Users/Anthony/Development/GitHub/org.platanios/learn/src/test/resources/org/platanios/learn/classification/reflection/nell/input/";
//        filename = "/Users/Anthony/Development/GitHub/org.platanios/learn/src/test/resources/org/platanios/learn/classification/reflection/brain/input/";
        String separator = ",";
        double[] classificationThresholds = new double[] { 0.1, 0.1, 0.1, 0.1 };
//        classificationThresholds = new double[] { 0.5 };
        List<boolean[][]> functionOutputs = new ArrayList<>();
        List<boolean[]> trueLabels = new ArrayList<>();
        for (File file : new File(filename).listFiles()) {
            if (file.isFile()) {
                DomainData data = parseLabeledDataFromCSVFile(file,
                                                              separator,
                                                              classificationThresholds,
                                                              1);
                functionOutputs.add(data.functionOutputs);
                trueLabels.add(data.trueLabels);
            }
        }
        double[][] errorRates = new double[functionOutputs.size()][];
        switch (errorEstimationMethod) {
            case 0:
                for (int p = 0; p < functionOutputs.size(); p++) {
                    int numberOfFunctions = functionOutputs.get(p)[0].length;
                    ErrorEstimationData errorEstimationData = new ErrorEstimationData.Builder(
                            Arrays.asList(functionOutputs.get(p)),
                            2, //functionOutputs.get(0)[0].length,
                            true).build();
                    ErrorEstimation errorEstimation = new ErrorEstimation.Builder(errorEstimationData)
                            .optimizationSolverType(ErrorEstimationInternalSolver.IP_OPT)
                            .build();
                    double[] allErrorRates = errorEstimation.solve().getErrorRates().array;
                    errorRates[p] = new double[numberOfFunctions];
                    System.arraycopy(allErrorRates, 0, errorRates[p], 0, numberOfFunctions);
                }
                break;
            case 1:
                ErrorEstimationSimpleGraphicalModel eesgm = new ErrorEstimationSimpleGraphicalModel(functionOutputs, 20000, 100);
                eesgm.performGibbsSampling();
                errorRates = eesgm.getErrorRatesMeans();
//                double[][] labelMeans = eegm.getLabelMeans();
                break;
            case 2:
                ErrorEstimationDomainsDPGraphicalModel eeddpgm = new ErrorEstimationDomainsDPGraphicalModel(functionOutputs, 10000, 100, 0.1);
                eeddpgm.performGibbsSampling();
                errorRates = eeddpgm.getErrorRatesMeans();
//                double[][] labelMeans = eegm.getLabelMeans();
                break;
            case 3:
                ErrorEstimationDomainsDPMixedGraphicalModel eeddpmgm = new ErrorEstimationDomainsDPMixedGraphicalModel(functionOutputs, 100000, 100, 0.1);
                eeddpmgm.performGibbsSampling();
                errorRates = eeddpmgm.getErrorRatesMeans();
//                double[][] labelMeans = eegm.getLabelMeans();
                break;
            case 4:
                ErrorEstimationDomainsDPFinalGraphicalModel eeddpfgm = new ErrorEstimationDomainsDPFinalGraphicalModel(functionOutputs, 20000, 100, 0.1);
                eeddpfgm.performGibbsSampling();
                errorRates = eeddpfgm.getErrorRatesMeans();
//                double[][] labelMeans = eegm.getLabelMeans();
                break;
            case 5:
                ErrorEstimationDomainsDPGraphicalModelComplicated eeddpgmc = new ErrorEstimationDomainsDPGraphicalModelComplicated(functionOutputs, 1000, null);
                eeddpgmc.performGibbsSampling();
                errorRates = eeddpgmc.getErrorRatesMeans();
//                double[][] labelMeans = eegm.getLabelMeans();
        }
//        double[] combinedErrorRate = new double[functionOutputs.size()];
//        double combinedErrorRateMean = 0;
        double[] mad = new double[functionOutputs.size()];
        double madMean = 0;
        for (int p = 0; p < functionOutputs.size(); p++) {
//            combinedErrorRate[p] = 0;
            double[] realErrorRates = new double[errorRates[p].length];
            for (int i = 0; i < trueLabels.get(p).length; i++) {
//                combinedErrorRate[p] += ((labelMeans[p][i] >= 0.5) != trueLabels.get(p)[i]) ? 1 : 0;
                for (int j = 0; j < errorRates[p].length; j++)
                    realErrorRates[j] += (functionOutputs.get(p)[i][j] != trueLabels.get(p)[i]) ? 1 : 0;
            }
//            combinedErrorRate[p] /= trueLabels.get(p).length;
//            combinedErrorRateMean += combinedErrorRate[p];
            mad[p] = 0;
            for (int j = 0; j < errorRates[p].length; j++) {
                realErrorRates[j] /= trueLabels.get(p).length;
                mad[p] += Math.abs(errorRates[p][j] - realErrorRates[j]);
            }
            mad[p] /= errorRates[p].length;
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
            for (int j = 0; j < errorRates[p].length; j++)
                if (j != errorRates[p].length - 1)
                    System.out.print(errorRates[p][j] + ", ");
                else
                    System.out.print(errorRates[p][j] + "\n");
//            System.out.print("Error rates variances:\t");
//            for (int j = 0; j < errorRatesVariances[p].length; j++)
//                if (j != errorRatesVariances[p].length - 1)
//                    System.out.print(errorRatesVariances[p][j] + ", ");
//                else
//                    System.out.print(errorRatesVariances[p][j] + "\n");
//            System.out.println("---------------------------------------------------");
//            System.out.println("Combined labels error rate:\t" + combinedErrorRate[p]);
//            System.out.println("MAD:\t\t\t\t\t\t" + mad[p]);
//            System.out.println("---------------------------------------------------");
        }
//        combinedErrorRateMean /= functionOutputs.size();
        madMean /= functionOutputs.size();
        System.out.println("===================================================");
//        System.out.println("Combined Labels Error Rate Mean:\t" + combinedErrorRateMean);
        System.out.println("MAD Mean:\t\t\t\t\t\t" + madMean);
        System.out.println("===================================================");
    }

    public static void manyThresholdsExperiment(int errorEstimationMethod) {
        String filename = "/Users/Anthony/Development/GitHub/org.platanios/learn/src/test/resources/org/platanios/learn/classification/reflection/nell/input/";
//        filename = "/Users/Anthony/Development/GitHub/org.platanios/learn/src/test/resources/org/platanios/learn/classification/reflection/brain/input/";
        String separator = ",";
        double[] classificationThresholds = new double[] { 0.01, 0.02, 0.03, 0.04, 0.05 };
//        classificationThresholds = new double[] { 0.5 };
        double[] madMeans = new double[classificationThresholds.length];
        double madMean = 0;
        for (int t = 0; t < classificationThresholds.length; t++) {
            List<boolean[][]> functionOutputs = new ArrayList<>();
            List<boolean[]> trueLabels = new ArrayList<>();
            for (File file : new File(filename).listFiles()) {
                if (file.isFile()) {
                    DomainData data = parseLabeledDataFromCSVFile(file,
                                                                  separator,
                                                                  new double[] { classificationThresholds[t] },
                                                                  1);
                    functionOutputs.add(data.functionOutputs);
                    trueLabels.add(data.trueLabels);
                }
            }
            double[][] errorRates = new double[functionOutputs.size()][];
            switch (errorEstimationMethod) {
                case 0:
                    for (int p = 0; p < functionOutputs.size(); p++) {
                        int numberOfFunctions = functionOutputs.get(p)[0].length;
                        ErrorEstimationData errorEstimationData = new ErrorEstimationData.Builder(
                                Arrays.asList(functionOutputs.get(p)),
                                numberOfFunctions,
                                true).build();
                        ErrorEstimation errorEstimation = new ErrorEstimation.Builder(errorEstimationData)
                                .optimizationSolverType(ErrorEstimationInternalSolver.IP_OPT)
                                .build();
                        double[] allErrorRates = errorEstimation.solve().getErrorRates().array;
                        errorRates[p] = new double[numberOfFunctions];
                        System.arraycopy(allErrorRates, 0, errorRates[p], 0, numberOfFunctions);
                    }
                    break;
                case 1:
                    ErrorEstimationSimpleGraphicalModel eesgm = new ErrorEstimationSimpleGraphicalModel(functionOutputs, 1000, 10);
                    eesgm.performGibbsSampling();
                    errorRates = eesgm.getErrorRatesMeans();
                    break;
                case 2:
                    ErrorEstimationDomainsDPGraphicalModelComplicated eeddpgm = new ErrorEstimationDomainsDPGraphicalModelComplicated(functionOutputs, 1000, null);
                    eeddpgm.performGibbsSampling();
                    errorRates = eeddpgm.getErrorRatesMeans();
            }
            double[] mad = new double[functionOutputs.size()];

            for (int p = 0; p < functionOutputs.size(); p++) {
                double[] realErrorRates = new double[errorRates[p].length];
                for (int i = 0; i < trueLabels.get(p).length; i++) {
                    for (int j = 0; j < errorRates[p].length; j++)
                        realErrorRates[j] += (functionOutputs.get(p)[i][j] != trueLabels.get(p)[i]) ? 1 : 0;
                }
                mad[p] = 0;
                for (int j = 0; j < errorRates[p].length; j++) {
                    realErrorRates[j] /= trueLabels.get(p).length;
                    mad[p] += Math.abs(errorRates[p][j] - realErrorRates[j]);
                }
                mad[p] /= errorRates[p].length;
                madMeans[t] += mad[p];
            }
            madMeans[t] /= functionOutputs.size();
            madMean += madMeans[t];
        }
        madMean /= madMeans.length;
        System.out.println("===================================================");
        System.out.println("MAD Mean:\t\t\t\t\t\t" + madMean);
        System.out.println("===================================================");
    }

    public static DomainData parseLabeledDataFromCSVFile(
            File file,
            String separator,
            double[] classificationThresholds,
            int subSampling
    ) {
        BufferedReader br = null;
        String line;
        List<boolean[]> classifiersOutputsList = new ArrayList<>();
        List<Boolean> trueLabelsList = new ArrayList<>();

        try {
            br = new BufferedReader(new FileReader(file));
            br.readLine();
            int numberOfSamplesRead = 0;
            while ((line = br.readLine()) != null) {
                if (numberOfSamplesRead % subSampling == 0) {
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
                numberOfSamplesRead++;
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
