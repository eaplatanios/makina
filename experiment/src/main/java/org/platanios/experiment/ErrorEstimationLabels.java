package org.platanios.experiment;

import org.platanios.learn.classification.reflection.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ErrorEstimationLabels {
    public static void main(String[] args) {
        String filename = "/Users/Anthony/Development/GitHub/org.platanios/learn/src/test/resources/org/platanios/learn/classification/reflection/nell/input/";
//        filename = "/Users/Anthony/Development/GitHub/org.platanios/learn/src/test/resources/org/platanios/learn/classification/reflection/brain/input/";
        String separator = ",";
        double[] classificationThresholds = new double[] { 0.05, 0.05, 0.05, 0.05 };
//        classificationThresholds = new double[] { 0.5 };
        List<boolean[][]> functionOutputs = new ArrayList<>();
        List<boolean[]> trueLabels = new ArrayList<>();
        for (File file : new File(filename).listFiles()) {
            if (file.isFile()) {
                DomainData data = parseLabeledDataFromCSVFile(file,
                                                              separator,
                                                              classificationThresholds,
                                                              1,
                                                              false);
                functionOutputs.add(data.functionOutputs);
                trueLabels.add(data.trueLabels);
            }
        }
        ErrorEstimationMethod[] errorEstimationMethods = new ErrorEstimationMethod[] {
//                ErrorEstimationMethod.AR_2,
//                ErrorEstimationMethod.AR_N,
//                ErrorEstimationMethod.SIMPLE_GM,
                ErrorEstimationMethod.DOMAINS_FAST_DP_GM,
//                ErrorEstimationMethod.PAIRS_FAST_DP_GM,
            ErrorEstimationMethod.HDP_GM,
//                ErrorEstimationMethod.DOMAINS_PER_CLASSIFIER_DP_GM
        };
        Double[] alphaValues = new Double[]{
                1e-3,
                1e-2,
                1e-1,
                1e0,
                1e1,
                1e2,
                1e3,
                1e4,
                1e5,
//                1e6,
//                1e7,
//                1e8,
//                1e9,
//                1e10,
//                1e11,
//                1e12,
//                1e13,
//                1e14,
//                1e15,
//                1e16
        };
        Double[] gammaValues = new Double[] {
                1e-3,
                1e-2,
                1e-1,
                1e0,
                1e1,
                1e2,
                1e3,
//                1e4,
//                1e5,
        };
        runExperiments(errorEstimationMethods, alphaValues, gammaValues, functionOutputs, trueLabels);
    }

    public static void runExperiments(ErrorEstimationMethod[] errorEstimationMethods, Double[] alphaValues, Double[] gammaValues, List<boolean[][]> functionOutputs, List<boolean[]> trueLabels) {
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
                Arrays.asList(alphaValues).parallelStream().forEach(alpha -> {
                    if (method == ErrorEstimationMethod.HDP_GM) {
                        Arrays.asList(gammaValues).parallelStream().forEach(gamma -> {
                            Results results = runExperiment(method, functionOutputs, trueLabels, alpha, gamma);
                            System.out.println(method + "\t-\tγ = " + gamma
                                                       + "\t-\tα = " + alpha
                                                       + "\t-\tError Rates MAD Mean: " + results.getErrorRatesMAD()
                                                       + "\t-\tLabels Error Rate Mean: " + results.getLabelsMeanErrorRate()
                                                       + "\t-\tNumber of Clusters: " + results.getNumberOfClusters());
                        });
                    } else {
                        Results results = runExperiment(method, functionOutputs, trueLabels, alpha, 0);
                        System.out.println(method + "\t-\tα = " + alpha
                                                   + "\t-\tError Rates MAD Mean: " + results.getErrorRatesMAD()
                                                   + "\t-\tLabels Error Rate Mean: " + results.getLabelsMeanErrorRate()
                                                   + "\t-\tNumber of Clusters: " + results.getNumberOfClusters());
                    }
                });
            } else {
                Results results = runExperiment(method, functionOutputs, trueLabels, 0, 0);
                System.out.println(method
                                           + "\t-\tError Rates MAD Mean: " + results.getErrorRatesMAD()
                                           + "\t-\tLabels Error Rate Mean: " + results.getLabelsMeanErrorRate());
            }
        });
    }

    public static Results runExperiment(ErrorEstimationMethod method, List<boolean[][]> functionOutputs, List<boolean[]> trueLabels, double alpha, double gamma) {
        double[][] errorRates = new double[functionOutputs.size()][];
        boolean[][] labels = new boolean[functionOutputs.size()][];
        int numberOfClusters = 1;
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
                ErrorEstimationSimpleGraphicalModel eesgm = new ErrorEstimationSimpleGraphicalModel(functionOutputs, 100000, 100);
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
                ErrorEstimationDomainsDPGraphicalModel eeddpgm = new ErrorEstimationDomainsDPGraphicalModel(functionOutputs, 100000, 100, alpha);
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
            case DOMAINS_FAST_DP_GM:
                ErrorEstimationDomainsFastDPGraphicalModel eedfdpgm = new ErrorEstimationDomainsFastDPGraphicalModel(functionOutputs, 20000, 10, alpha);
                eedfdpgm.performGibbsSampling();
                errorRates = eedfdpgm.getErrorRatesMeans();
                numberOfClusters = eedfdpgm.numberOfClusters;

                double[][] labelMeansEedfdpgm = eedfdpgm.getLabelMeans();
                for (int p = 0; p < functionOutputs.size(); p++) {
                    labels[p] = new boolean[functionOutputs.get(p).length];
                    for (int i = 0; i < functionOutputs.get(p).length; i++) {
                        labels[p][i] = labelMeansEedfdpgm[p][i] >= 0.5;
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
            case PAIRS_FAST_DP_GM:
                ErrorEstimationDomainsFastDPMixedGraphicalModel eedfdpmgm = new ErrorEstimationDomainsFastDPMixedGraphicalModel(functionOutputs, 20000, 10, alpha);
                eedfdpmgm.performGibbsSampling();
                errorRates = eedfdpmgm.getErrorRatesMeans();

                double[][] labelMeansEedfdpmgm = eedfdpmgm.getLabelMeans();
                for (int p = 0; p < functionOutputs.size(); p++) {
                    labels[p] = new boolean[functionOutputs.get(p).length];
                    for (int i = 0; i < functionOutputs.get(p).length; i++) {
                        labels[p][i] = labelMeansEedfdpmgm[p][i] >= 0.5;
                    }
                }
                break;
            case HDP_GM:
                ErrorEstimationDomainsFastHDPMixedGraphicalModel eedfhdpmgm = new ErrorEstimationDomainsFastHDPMixedGraphicalModel(functionOutputs, 20000, 10, alpha, gamma);
                eedfhdpmgm.performGibbsSampling();
                errorRates = eedfhdpmgm.getErrorRatesMeans();
                numberOfClusters = eedfhdpmgm.numberOfClusters;

                double[][] labelMeansEedfhdpmgm = eedfhdpmgm.getLabelMeans();
                for (int p = 0; p < functionOutputs.size(); p++) {
                    labels[p] = new boolean[functionOutputs.get(p).length];
                    for (int i = 0; i < functionOutputs.get(p).length; i++) {
                        labels[p][i] = labelMeansEedfhdpmgm[p][i] >= 0.5;
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
            case DOMAINS_PER_CLASSIFIER_FAST_DP_GM:
                ErrorEstimationDomainsFastDPFinalGraphicalModel eedfdpfgm = new ErrorEstimationDomainsFastDPFinalGraphicalModel(functionOutputs, 10000, 10, alpha);
                eedfdpfgm.performGibbsSampling();
                errorRates = eedfdpfgm.getErrorRatesMeans();

                double[][] labelMeansEedfdpfgm = eedfdpfgm.getLabelMeans();
                for (int p = 0; p < functionOutputs.size(); p++) {
                    labels[p] = new boolean[functionOutputs.get(p).length];
                    for (int i = 0; i < functionOutputs.get(p).length; i++) {
                        labels[p][i] = labelMeansEedfdpfgm[p][i] >= 0.5;
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
        return new Results(errorRatesMADMean, labelsErrorRateMean, numberOfClusters);
    }

    public static DomainData parseLabeledDataFromCSVFile(
            File file,
            String separator,
            double[] classificationThresholds,
            int subSampling,
            boolean evaluationData
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

        Random random = new Random();
        List<boolean[]> evaluationClassifiersOutputsList = new ArrayList<>();
        List<Boolean> evaluationTrueLabelsList = new ArrayList<>();

        if (evaluationData) {
            for (int sample = 0; sample < classifiersOutputsList.size(); sample++) {
                double uniform = random.nextDouble();
                if (uniform > 1 / classifiersOutputsList.size()) {
                    evaluationClassifiersOutputsList.add(classifiersOutputsList.get(sample - evaluationClassifiersOutputsList.size()));
                    evaluationTrueLabelsList.add(trueLabelsList.get(sample - evaluationTrueLabelsList.size()));
                    classifiersOutputsList.remove(sample);
                    trueLabelsList.remove(sample);
                }
                if (evaluationClassifiersOutputsList.size() > classifiersOutputsList.size() / 9)
                    break;
            }
        }

        boolean[] trueLabels = new boolean[trueLabelsList.size()];
        for (int i = 0; i < trueLabels.length; i++)
            trueLabels[i] = trueLabelsList.get(i);
        boolean[] evaluationTrueLabels = new boolean[evaluationTrueLabelsList.size()];
        for (int i = 0; i < evaluationTrueLabels.length; i++)
            evaluationTrueLabels[i] = evaluationTrueLabelsList.get(i);

        return new DomainData(classifiersOutputsList.toArray(new boolean[classifiersOutputsList.size()][]),
                              trueLabels,
                              evaluationClassifiersOutputsList.toArray(new boolean[evaluationClassifiersOutputsList.size()][]),
                              evaluationTrueLabels);
    }

    private static class DomainData {
        private boolean[][] functionOutputs;
        private boolean[] trueLabels;
        private boolean[][] evaluationFunctionOutputs;
        private boolean[] evaluationTrueLabels;

        protected DomainData(boolean[][] functionOutputs,
                             boolean[] trueLabels,
                             boolean[][] evaluationFunctionOutputs,
                             boolean[] evaluationTrueLabels) {
            this.functionOutputs = functionOutputs;
            this.trueLabels = trueLabels;
            this.evaluationFunctionOutputs = evaluationFunctionOutputs;
            this.evaluationTrueLabels = evaluationTrueLabels;
        }

        protected boolean[][] getFunctionOutputs() {
            return functionOutputs;
        }

        protected boolean[] getTrueLabels() {
            return trueLabels;
        }

        protected boolean[][] getEvaluationFunctionOutputs() {
            return evaluationFunctionOutputs;
        }

        protected boolean[] getEvaluationTrueLabels() {
            return evaluationTrueLabels;
        }
    }

    private static class Results {
        private double errorRatesMAD;
        private double labelsMeanErrorRate;
        private int numberOfClusters = 1;

        protected Results(double errorRatesMAD, double labelsMeanErrorRate) {
            this.errorRatesMAD = errorRatesMAD;
            this.labelsMeanErrorRate = labelsMeanErrorRate;
        }

        protected Results(double errorRatesMAD, double labelsMeanErrorRate, int numberOfClusters) {
            this.errorRatesMAD = errorRatesMAD;
            this.labelsMeanErrorRate = labelsMeanErrorRate;
            this.numberOfClusters = numberOfClusters;
        }

        protected double getErrorRatesMAD() {
            return errorRatesMAD;
        }

        protected double getLabelsMeanErrorRate() {
            return labelsMeanErrorRate;
        }

        protected int getNumberOfClusters() {
            return numberOfClusters;
        }
    }

    private static enum ErrorEstimationMethod {
        AR_2,
        AR_N,
        SIMPLE_GM,
        DOMAINS_DP_GM,
        DOMAINS_FAST_DP_GM,
        PAIRS_DP_GM,
        PAIRS_FAST_DP_GM,
        HDP_GM,
        DOMAINS_PER_CLASSIFIER_DP_GM,
        DOMAINS_PER_CLASSIFIER_FAST_DP_GM,
        COMPLICATED_DP_GM
    }
}
