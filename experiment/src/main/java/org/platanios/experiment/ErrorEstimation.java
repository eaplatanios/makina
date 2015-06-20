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
public class ErrorEstimation {
    public static void main(String[] args) {
        String filename = "/Users/Anthony/Development/GitHub/org.platanios/learn/src/test/resources/org/platanios/learn/classification/reflection/nell/input/";
//        filename = "/Users/Anthony/Development/GitHub/org.platanios/learn/src/test/resources/org/platanios/learn/classification/reflection/brain/input/";
        String separator = ",";
        double[] classificationThresholds = new double[] { 0.05, 0.05, 0.1, 0.05 };
//        classificationThresholds = new double[] { 0.5 };
        List<String> domainNames = new ArrayList<>();
        List<boolean[][]> functionOutputs = new ArrayList<>();
        List<boolean[]> trueLabels = new ArrayList<>();
        List<boolean[][]> evaluationFunctionOutputs = new ArrayList<>();
        for (File file : new File(filename).listFiles()) {
            if (file.isFile()) {
                DomainData data = parseLabeledDataFromCSVFile(file,
                                                              separator,
                                                              classificationThresholds,
                                                              1,
                                                              false);
                domainNames.add(data.domainName);
                functionOutputs.add(data.functionOutputs);
                trueLabels.add(data.trueLabels);
                evaluationFunctionOutputs.add(data.evaluationFunctionOutputs);
            }
        }
        ErrorEstimationMethod[] errorEstimationMethods = new ErrorEstimationMethod[] {
                ErrorEstimationMethod.AR_2,
                ErrorEstimationMethod.AR_N,
                ErrorEstimationMethod.ERROR_ESTIMATION_GM,
                ErrorEstimationMethod.COUPLED_ERROR_ESTIMATION_GM,
                ErrorEstimationMethod.HIERARCHICAL_COUPLED_ERROR_ESTIMATION_GM
        };
        Double[] alphaValues = new Double[] {
//                1e-6,
//                1e-5,
//                1e-4,
//                1e-3,
//                1e-2,
//                1e-1,
                1e0,
                1e1,
                1e2,
//                1e3,
//                1e4,
//                1e5,
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
//                1e1,
//                1e2,
                1e3,
                1e4,
                1e5,
                1e6,
                1e7
        };
        runExperiments(domainNames, errorEstimationMethods, alphaValues, gammaValues, functionOutputs, trueLabels, evaluationFunctionOutputs);
    }

    public static void runExperiments(List<String> domainNames,
                                      ErrorEstimationMethod[] errorEstimationMethods,
                                      Double[] alphaValues,
                                      Double[] gammaValues,
                                      List<boolean[][]> functionOutputs,
                                      List<boolean[]> trueLabels,
                                      List<boolean[][]> evaluationFunctionOutputs) {
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
        String trueErrorRatesString = "====================================================================================================\n";
        trueErrorRatesString += "TRUE_ERROR_RATES\n----------------------------------------------------------------------------------------------------\n";
        for (int p = 0; p < functionOutputs.size(); p++) {
            double[] realErrorRates = new double[functionOutputs.get(p)[0].length];
            for (int i = 0; i < trueLabels.get(p).length; i++)
                for (int j = 0; j < functionOutputs.get(p)[i].length; j++)
                    realErrorRates[j] += (functionOutputs.get(p)[i][j] != trueLabels.get(p)[i]) ? 1 : 0;
            for (int j = 0; j < functionOutputs.get(p)[0].length; j++)
                realErrorRates[j] /= trueLabels.get(p).length;
            trueErrorRatesString += domainNames.get(p);
            for (int j = 0; j < functionOutputs.get(p)[0].length; j++)
                trueErrorRatesString += "\t" + realErrorRates[j];
            trueErrorRatesString += "\n";
        }
        System.out.print(trueErrorRatesString);
        Arrays.asList(errorEstimationMethods).parallelStream().forEach(method -> {
            if (method != ErrorEstimationMethod.AR_2 && method != ErrorEstimationMethod.AR_N && method != ErrorEstimationMethod.ERROR_ESTIMATION_GM) {
                Arrays.asList(alphaValues).parallelStream().forEach(alpha -> {
                    if (method == ErrorEstimationMethod.HIERARCHICAL_COUPLED_ERROR_ESTIMATION_GM) {
                        Arrays.asList(gammaValues).parallelStream().forEach(gamma -> {
                            Results results = runExperiment(method, functionOutputs, trueLabels, evaluationFunctionOutputs, alpha, gamma);
                            String resultsString =
                                    "====================================================================================================\n"
                                            + method + "\t-\tγ = " + gamma
                                            + "\t-\tα = " + alpha
                                            + "\t-\tError Rates MAD Mean: " + results.getErrorRatesMAD()
                                            + "\t-\tLabels Error Rate Mean: " + results.getLabelsMeanErrorRate()
                                            + "\t-\tNumber of Clusters: " + results.getNumberOfClusters()
                                            + "\t-\tLog-likelihood: " + results.getLogLikelihood();
                            resultsString += "\n----------------------------------------------------------------------------------------------------\n";
                            for (int p = 0; p < functionOutputs.size(); p++) {
                                resultsString += domainNames.get(p);
                                for (int j = 0; j < functionOutputs.get(p)[0].length; j++)
                                    resultsString += "\t" + results.getErrorRates()[p][j];
                                resultsString += "\n";
                            }
                            System.out.print(resultsString);
                        });
                    } else {
                        Results results = runExperiment(method, functionOutputs, trueLabels, evaluationFunctionOutputs, alpha, 0);
                        String resultsString =
                                "====================================================================================================\n"
                                        + method + "\t-\tα = " + alpha
                                        + "\t-\tError Rates MAD Mean: " + results.getErrorRatesMAD()
                                        + "\t-\tLabels Error Rate Mean: " + results.getLabelsMeanErrorRate()
                                        + "\t-\tNumber of Clusters: " + results.getNumberOfClusters()
                                        + "\t-\tLog-likelihood: " + results.getLogLikelihood();
                        resultsString += "\n----------------------------------------------------------------------------------------------------\n";
                        for (int p = 0; p < functionOutputs.size(); p++) {
                            resultsString += domainNames.get(p);
                            for (int j = 0; j < functionOutputs.get(p)[0].length; j++)
                                resultsString += "\t" + results.getErrorRates()[p][j];
                            resultsString += "\n";
                        }
                        System.out.print(resultsString);
                    }
                });
            } else {
                Results results = runExperiment(method, functionOutputs, trueLabels, evaluationFunctionOutputs, 0, 0);
                String resultsString =
                        "====================================================================================================\n"
                                + method
                                + "\t-\tError Rates MAD Mean: " + results.getErrorRatesMAD()
                                + "\t-\tLabels Error Rate Mean: " + results.getLabelsMeanErrorRate();
                resultsString += "\n----------------------------------------------------------------------------------------------------\n";
                for (int p = 0; p < functionOutputs.size(); p++) {
                    resultsString += domainNames.get(p);
                    for (int j = 0; j < functionOutputs.get(p)[0].length; j++)
                        resultsString += "\t" + results.getErrorRates()[p][j];
                    resultsString += "\n";
                }
                System.out.print(resultsString);
            }
        });
        System.out.println("====================================================================================================");
    }

    public static Results runExperiment(ErrorEstimationMethod method,
                                        List<boolean[][]> functionOutputs,
                                        List<boolean[]> trueLabels,
                                        List<boolean[][]> evaluationFunctionOutputs,
                                        double alpha,
                                        double gamma) {
        double[][] errorRates = new double[functionOutputs.size()][];
        boolean[][] labels = new boolean[functionOutputs.size()][];
        int numberOfClusters = 1;
        double logLikelihood = 0;
        switch (method) {
            case AR_2:
                for (int p = 0; p < functionOutputs.size(); p++) {
                    int numberOfFunctions = functionOutputs.get(p)[0].length;
                    ErrorEstimationData errorEstimationData = new ErrorEstimationData.Builder(
                            Arrays.asList(functionOutputs.get(p)),
                            2,
                            true).build();
                    org.platanios.learn.classification.reflection.ErrorEstimation errorEstimation =
                            new org.platanios.learn.classification.reflection.ErrorEstimation.Builder(errorEstimationData)
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
                    org.platanios.learn.classification.reflection.ErrorEstimation errorEstimation =
                            new org.platanios.learn.classification.reflection.ErrorEstimation.Builder(errorEstimationData)
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
            case ERROR_ESTIMATION_GM:
                ErrorEstimationGraphicalModel eesgm = new ErrorEstimationGraphicalModel(functionOutputs, 18000, 10, 200);
                eesgm.runGibbsSampler();
                errorRates = eesgm.getErrorRatesMeans();

                double[][] labelMeansEesgm = eesgm.getLabelMeans();
                for (int p = 0; p < functionOutputs.size(); p++) {
                    labels[p] = new boolean[functionOutputs.get(p).length];
                    for (int i = 0; i < functionOutputs.get(p).length; i++) {
                        labels[p][i] = labelMeansEesgm[p][i] >= 0.5;
                    }
                }
                break;
            case COUPLED_ERROR_ESTIMATION_GM:
                CoupledErrorEstimationGraphicalModel eedfdpgm = new CoupledErrorEstimationGraphicalModel(functionOutputs, 20000, 10, alpha);
                eedfdpgm.performGibbsSampling();
                errorRates = eedfdpgm.getErrorRatesMeans();
                numberOfClusters = eedfdpgm.numberOfClusters;
                logLikelihood = eedfdpgm.logLikelihood(evaluationFunctionOutputs);

                double[][] labelMeansEedfdpgm = eedfdpgm.getLabelMeans();
                for (int p = 0; p < functionOutputs.size(); p++) {
                    labels[p] = new boolean[functionOutputs.get(p).length];
                    for (int i = 0; i < functionOutputs.get(p).length; i++) {
                        labels[p][i] = labelMeansEedfdpgm[p][i] >= 0.5;
                    }
                }
                break;
            case HIERARCHICAL_COUPLED_ERROR_ESTIMATION_GM:
                ErrorEstimationDomainsHDPNew eedfhdp = new ErrorEstimationDomainsHDPNew(functionOutputs, alpha, gamma);
                eedfhdp.run_gibbs_collapsed(1000);
                eedfhdp.run_gibbs_uncollapsed(1000, 100, 10);
                errorRates = eedfhdp.rates_to_return;
                numberOfClusters = eedfhdp.num_cluster;
                double[][] labelMeansEedfhdpmgm = eedfhdp.labels_to_return;
                int li_cnt[][] = new int[functionOutputs.size()][2];
                for (int p = 0; p < functionOutputs.size(); p++) {
                    labels[p] = new boolean[functionOutputs.get(p).length];
                    for (int i = 0; i < functionOutputs.get(p).length; i++) {
                        labels[p][i] = labelMeansEedfhdpmgm[p][i] >= 0.5;
                        int lid = labelMeansEedfhdpmgm[p][i] >= 0.5? 1:0;
                        li_cnt[p][lid]++;
                    }
                }
                logLikelihood = eedfhdp.get_log_likelihood(evaluationFunctionOutputs, alpha, gamma, 1000, li_cnt);
                break;
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
        return new Results(errorRates, errorRatesMADMean, labelsErrorRateMean, numberOfClusters, logLikelihood);
    }

    public static DomainData parseLabeledDataFromCSVFile(
            File file,
            String separator,
            double[] classificationThresholds,
            int subSampling,
            boolean evaluationData
    ) {
        String domainName = file.getName();
        BufferedReader br = null;
        String line;
        List<boolean[]> classifiersOutputsList = new ArrayList<>();
        List<Boolean> trueLabelsList = new ArrayList<>();

//        switch (domainName) {
//            case "animal.csv":
//                classificationThresholds = new double[] { 0.05, 0.05, 0.02, 0.05 };
//                break;
//            case "beverage.csv":
//                classificationThresholds = new double[] { 0.05, 0.5, 0.01, 0.03 };
//                break;
//            case "fish.csv":
//                classificationThresholds = new double[] { 0.05, 0.1, 0.03, 0.05 };
//                break;
//            case "food.csv":
//                classificationThresholds = new double[] { 0.05, 0.1, 0.02, 0.03 };
//                break;
//            case "fruit.csv":
//                classificationThresholds = new double[] { 0.05, 0.1, 0.01, 0.005 };
//                break;
//            case "muscle.csv":
//                classificationThresholds = new double[] { 0.05, 0.1, 0.01, 0.01 };
//                break;
//            case "river.csv":
//                classificationThresholds = new double[] { 0.05, 0.1, 0.01, 0.01 };
//                break;
//            default:
//                classificationThresholds = new double[] { 0.05, 0.1, 0.02, 0.05 };
//        }

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

        return new DomainData(domainName,
                              classifiersOutputsList.toArray(new boolean[classifiersOutputsList.size()][]),
                              trueLabels,
                              evaluationClassifiersOutputsList.toArray(new boolean[evaluationClassifiersOutputsList.size()][]),
                              evaluationTrueLabels);
    }

    private static class DomainData {
        private String domainName;
        private boolean[][] functionOutputs;
        private boolean[] trueLabels;
        private boolean[][] evaluationFunctionOutputs;
        private boolean[] evaluationTrueLabels;

        protected DomainData(String domainName,
                             boolean[][] functionOutputs,
                             boolean[] trueLabels,
                             boolean[][] evaluationFunctionOutputs,
                             boolean[] evaluationTrueLabels) {
            this.domainName = domainName;
            this.functionOutputs = functionOutputs;
            this.trueLabels = trueLabels;
            this.evaluationFunctionOutputs = evaluationFunctionOutputs;
            this.evaluationTrueLabels = evaluationTrueLabels;
        }

        protected String getDomainName() {
            return domainName;
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
        private double[][] errorRates;
        private double errorRatesMAD;
        private double labelsMeanErrorRate;
        private int numberOfClusters = 1;
        private double logLikelihood = 0;

        protected Results(double[][] errorRates,
                          double errorRatesMAD,
                          double labelsMeanErrorRate) {
            this.errorRates = errorRates;
            this.errorRatesMAD = errorRatesMAD;
            this.labelsMeanErrorRate = labelsMeanErrorRate;
        }

        protected Results(double[][] errorRates,
                          double errorRatesMAD,
                          double labelsMeanErrorRate,
                          int numberOfClusters,
                          double logLikelihood) {
            this.errorRates = errorRates;
            this.errorRatesMAD = errorRatesMAD;
            this.labelsMeanErrorRate = labelsMeanErrorRate;
            this.numberOfClusters = numberOfClusters;
            this.logLikelihood = logLikelihood;
        }

        protected double[][] getErrorRates() {
            return errorRates;
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

        protected double getLogLikelihood() {
            return logLikelihood;
        }
    }

    private static enum ErrorEstimationMethod {
        AR_2,
        AR_N,
        ERROR_ESTIMATION_GM,
        COUPLED_ERROR_ESTIMATION_GM,
        HIERARCHICAL_COUPLED_ERROR_ESTIMATION_GM
    }
}
