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
public class ErrorEstimationLabels {
    public static void main(String[] args) {
        String filename = "/Users/Anthony/Development/GitHub/org.platanios/learn/src/test/resources/org/platanios/learn/classification/reflection/nell/input/";
        filename = "/Users/Anthony/Development/GitHub/org.platanios/learn/src/test/resources/org/platanios/learn/classification/reflection/brain/input/";
        String separator = ",";
        double[] classificationThresholds = new double[] { 0.1, 0.1, 0.1, 0.1 };
        classificationThresholds = new double[] { 0.5 };
        List<boolean[][]> functionOutputs = new ArrayList<>();
        List<boolean[]> trueLabels = new ArrayList<>();
        for (File file : new File(filename).listFiles()) {
            if (file.isFile()) {
                DomainData data = parseLabeledDataFromCSVFile(file,
                                                              separator,
                                                              classificationThresholds,
                                                              10);
                functionOutputs.add(data.functionOutputs);
                trueLabels.add(data.trueLabels);
            }
        }
        ErrorEstimationMethod[] errorEstimationMethods = new ErrorEstimationMethod[] {
                ErrorEstimationMethod.AR_2,
                ErrorEstimationMethod.SIMPLE_GM,
                ErrorEstimationMethod.DOMAINS_DP_GM,
                ErrorEstimationMethod.PAIRS_DP_GM
        };
        double[] alphaValues = new double[]{
                1e-5,
                1e-4,
                1e-3,
                1e-2,
                1e-1,
                1e0,
                1e1,
                1e2,
                1e3,
                1e4,
                1e5
        };
        runExperiments(errorEstimationMethods, alphaValues, functionOutputs, trueLabels);
    }

    public static void runExperiments(ErrorEstimationMethod[] errorEstimationMethods, double[] alphaValues, List<boolean[][]> functionOutputs, List<boolean[]> trueLabels) {
        // Combine the labels using a simple majority vote
        boolean[][] labels = new boolean[functionOutputs.size()][];
        for (int p = 0; p < functionOutputs.size(); p++) {
            labels[p] = new boolean[functionOutputs.get(p).length];
            for (int i = 0; i < functionOutputs.get(p).length; i++) {
                double labelsSum = 0;
                for (int j = 0; j < functionOutputs.get(p)[i].length; j++) {
                    labelsSum += (functionOutputs.get(p)[i][j] ? 1 : 0);
                }
                labels[p][i] = labelsSum / functionOutputs.get(p)[i].length >= 0.5;
            }
        }
        double[] labelsErrorRate = new double[functionOutputs.size()];
        double labelsErrorRateMean = 0;
        for (int p = 0; p < functionOutputs.size(); p++) {
            labelsErrorRate[p] = 0;
            for (int i = 0; i < trueLabels.get(p).length; i++)
                labelsErrorRate[p] += (labels[p][i] != trueLabels.get(p)[i]) ? 1 : 0;
            labelsErrorRate[p] /= trueLabels.get(p).length;
            labelsErrorRateMean += labelsErrorRate[p];
        }
        labelsErrorRateMean /= functionOutputs.size();
        System.out.println("Simple Majority Vote Labels Error Rate Mean: " + labelsErrorRateMean);
        Arrays.asList(errorEstimationMethods).parallelStream().forEach(method -> {
            if (method != ErrorEstimationMethod.AR_2 && method != ErrorEstimationMethod.AR_N && method != ErrorEstimationMethod.SIMPLE_GM) {
                for (double alpha : alphaValues) {
                    Results results = runExperiment(method, functionOutputs, trueLabels, alpha);
                    System.out.println(method + "\t-\tα = " + alpha
                                               + "\t-\tError Rates MAD Mean: " + results.getErrorRatesMAD()
                                               + "\t-\tLabels Error Rate Mean: " + results.getLabelsMeanErrorRate());
                }
            } else {
                Results results = runExperiment(method, functionOutputs, trueLabels, 0);
                System.out.println(method
                                           + "\t-\tError Rates MAD Mean: " + results.getErrorRatesMAD()
                                           + "\t-\tLabels Error Rate Mean: " + results.getLabelsMeanErrorRate());
            }
        });
    }

    public static Results runExperiment(ErrorEstimationMethod method, List<boolean[][]> functionOutputs, List<boolean[]> trueLabels, double alpha) {
        double[][] errorRates = new double[functionOutputs.size()][];
        boolean[][] labels = new boolean[functionOutputs.size()][];
        switch (method) {
            case AR_2:
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

                    // Combine the labels using a weighted majority vote
                    labels[p] = new boolean[functionOutputs.get(p).length];
                    for (int i = 0; i < functionOutputs.get(p).length; i++) {
                        double labelsSum = 0;
                        double errorRatesSum = 0;
                        for (int j = 0; j < functionOutputs.get(p)[i].length; j++) {
                            labelsSum += (1 - errorRates[p][j]) * (functionOutputs.get(p)[i][j] ? 1 : 0);
                            errorRatesSum += (1 - errorRates[p][j]);
                        }
                        labels[p][i] = labelsSum / errorRatesSum >= 0.5;
                    }
                }
                break;
            case AR_N:
                for (int p = 0; p < functionOutputs.size(); p++) {
                    int numberOfFunctions = functionOutputs.get(p)[0].length;
                    ErrorEstimationData errorEstimationData = new ErrorEstimationData.Builder(
                            Arrays.asList(functionOutputs.get(p)),
                            functionOutputs.get(0)[0].length,
                            true).build();
                    ErrorEstimation errorEstimation = new ErrorEstimation.Builder(errorEstimationData)
                            .optimizationSolverType(ErrorEstimationInternalSolver.IP_OPT)
                            .build();
                    double[] allErrorRates = errorEstimation.solve().getErrorRates().array;
                    errorRates[p] = new double[numberOfFunctions];
                    System.arraycopy(allErrorRates, 0, errorRates[p], 0, numberOfFunctions);

                    // Combine the labels using a weighted majority vote
                    labels[p] = new boolean[functionOutputs.get(p).length];
                    for (int i = 0; i < functionOutputs.get(p).length; i++) {
                        double labelsSum = 0;
                        double errorRatesSum = 0;
                        for (int j = 0; j < functionOutputs.get(p)[i].length; j++) {
                            labelsSum += (1 - errorRates[p][j]) * (functionOutputs.get(p)[i][j] ? 1 : 0);
                            errorRatesSum += (1 - errorRates[p][j]);
                        }
                        labels[p][i] = labelsSum / errorRatesSum >= 0.5;
                    }
                }
                break;
            case SIMPLE_GM:
                ErrorEstimationSimpleGraphicalModel eesgm = new ErrorEstimationSimpleGraphicalModel(functionOutputs, 20000, 10);
                eesgm.performGibbsSampling();
                errorRates = eesgm.getErrorRatesMeans();

                double[][] labelMeansEesgm = eesgm.getLabelMeans();
                for (int p = 0; p < functionOutputs.size(); p++) {
                    labels[p] = new boolean[functionOutputs.get(p).length];
                    for (int i = 0; i < functionOutputs.get(p).length; i++) {
                        labels[p][i] = labelMeansEesgm[p][i] >= 0.5;
                    }
                }
                break;
            case DOMAINS_DP_GM:
                ErrorEstimationDomainsDPGraphicalModel eeddpgm = new ErrorEstimationDomainsDPGraphicalModel(functionOutputs, 20000, 10, alpha);
                eeddpgm.performGibbsSampling();
                errorRates = eeddpgm.getErrorRatesMeans();

                double[][] labelMeansEeddpgm = eeddpgm.getLabelMeans();
                for (int p = 0; p < functionOutputs.size(); p++) {
                    labels[p] = new boolean[functionOutputs.get(p).length];
                    for (int i = 0; i < functionOutputs.get(p).length; i++) {
                        labels[p][i] = labelMeansEeddpgm[p][i] >= 0.5;
                    }
                }
                break;
            case PAIRS_DP_GM:
                ErrorEstimationDomainsDPMixedGraphicalModel eeddpmgm = new ErrorEstimationDomainsDPMixedGraphicalModel(functionOutputs, 20000, 10, alpha);
                eeddpmgm.performGibbsSampling();
                errorRates = eeddpmgm.getErrorRatesMeans();

                double[][] labelMeansEeddpmgm = eeddpmgm.getLabelMeans();
                for (int p = 0; p < functionOutputs.size(); p++) {
                    labels[p] = new boolean[functionOutputs.get(p).length];
                    for (int i = 0; i < functionOutputs.get(p).length; i++) {
                        labels[p][i] = labelMeansEeddpmgm[p][i] >= 0.5;
                    }
                }
                break;
            case DOMAINS_PER_CLASSIFIER_DP_GM:
                ErrorEstimationDomainsDPFinalGraphicalModel eeddpfgm = new ErrorEstimationDomainsDPFinalGraphicalModel(functionOutputs, 20000, 10, alpha);
                eeddpfgm.performGibbsSampling();
                errorRates = eeddpfgm.getErrorRatesMeans();

                double[][] labelMeansEeddpfgm = eeddpfgm.getLabelMeans();
                for (int p = 0; p < functionOutputs.size(); p++) {
                    labels[p] = new boolean[functionOutputs.get(p).length];
                    for (int i = 0; i < functionOutputs.get(p).length; i++) {
                        labels[p][i] = labelMeansEeddpfgm[p][i] >= 0.5;
                    }
                }
                break;
            case COMPLICATED_DP_GM:
                ErrorEstimationDomainsDPGraphicalModelComplicated eeddpgmc = new ErrorEstimationDomainsDPGraphicalModelComplicated(functionOutputs, 1000, null);
                eeddpgmc.performGibbsSampling();
                errorRates = eeddpgmc.getErrorRatesMeans();

                double[][] labelMeansEeddpgmc = eeddpgmc.getLabelMeans();
                for (int p = 0; p < functionOutputs.size(); p++) {
                    labels[p] = new boolean[functionOutputs.get(p).length];
                    for (int i = 0; i < functionOutputs.get(p).length; i++) {
                        labels[p][i] = labelMeansEeddpgmc[p][i] >= 0.5;
                    }
                }
        }
        double[] errorRatesMAD = new double[functionOutputs.size()];
        double errorRatesMADMean = 0;
        double[] labelsErrorRate = new double[functionOutputs.size()];
        double labelsErrorRateMean = 0;
        for (int p = 0; p < functionOutputs.size(); p++) {
            labelsErrorRate[p] = 0;
            double[] realErrorRates = new double[errorRates[p].length];
            for (int i = 0; i < trueLabels.get(p).length; i++) {
                labelsErrorRate[p] += (labels[p][i] != trueLabels.get(p)[i]) ? 1 : 0;
                for (int j = 0; j < errorRates[p].length; j++)
                    realErrorRates[j] += (functionOutputs.get(p)[i][j] != trueLabels.get(p)[i]) ? 1 : 0;
            }
            labelsErrorRate[p] /= trueLabels.get(p).length;
            labelsErrorRateMean += labelsErrorRate[p];
            errorRatesMAD[p] = 0;
            for (int j = 0; j < errorRates[p].length; j++) {
                realErrorRates[j] /= trueLabels.get(p).length;
                errorRatesMAD[p] += Math.abs(errorRates[p][j] - realErrorRates[j]);
            }
            errorRatesMAD[p] /= errorRates[p].length;
            errorRatesMADMean += errorRatesMAD[p];
        }
        errorRatesMADMean /= functionOutputs.size();
        labelsErrorRateMean /= functionOutputs.size();
        return new Results(errorRatesMADMean, labelsErrorRateMean);
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

    private static class Results {
        private double errorRatesMAD;
        private double labelsMeanErrorRate;

        protected Results(double errorRatesMAD, double labelsMeanErrorRate) {
            this.errorRatesMAD = errorRatesMAD;
            this.labelsMeanErrorRate = labelsMeanErrorRate;
        }

        protected double getErrorRatesMAD() {
            return errorRatesMAD;
        }

        protected double getLabelsMeanErrorRate() {
            return labelsMeanErrorRate;
        }
    }

    private static enum ErrorEstimationMethod {
        AR_2,
        AR_N,
        SIMPLE_GM,
        DOMAINS_DP_GM,
        PAIRS_DP_GM,
        DOMAINS_PER_CLASSIFIER_DP_GM,
        COMPLICATED_DP_GM
    }
}
