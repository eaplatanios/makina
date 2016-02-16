package org.platanios.experiment;

import com.google.common.collect.ComparisonChain;
import com.google.common.collect.Sets;
import org.platanios.learn.classification.reflection.*;
import org.platanios.learn.math.statistics.StatisticsUtilities;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentSkipListSet;
import java.util.stream.IntStream;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ErrorEstimationExperiment {
    public static void main(String[] args) {
        int numberOfExperimentRepetitions = 1;
        int loggingLevel = 2;
        String filename = args[0];
        String separator = ",";
        double[] classificationThresholds = new double[] { Double.parseDouble(args[1]) };
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
                                                              true);
                domainNames.add(data.domainName);
                functionOutputs.add(data.functionOutputs);
                trueLabels.add(data.trueLabels);
                evaluationFunctionOutputs.add(data.evaluationFunctionOutputs);
            }
        }
        CombinedResults results = new CombinedResults();
        IntStream.range(0, numberOfExperimentRepetitions).parallel().forEach(repetition -> {
            System.out.println("Running experiment repetition " + (repetition + 1) + "...");
            ErrorEstimationMethod[] errorEstimationMethods = new ErrorEstimationMethod[] {
//                    ErrorEstimationMethod.cBCC,
//                    ErrorEstimationMethod.cBCCc,
//                    ErrorEstimationMethod.AR_2,
//                    ErrorEstimationMethod.AR_N,
//                    ErrorEstimationMethod.BEE,
//                    ErrorEstimationMethod.BEE_ME,
//                    ErrorEstimationMethod.BEEc,
//                    ErrorEstimationMethod.CBEE,
//                    ErrorEstimationMethod.CBEEc,
                    ErrorEstimationMethod.HCBEE,
//                    ErrorEstimationMethod.HCBEEc
            };
            Double[] alphaValues = new Double[] {
//                    1e-11,
//                    1e-10,
//                    1e-9,
//                    1e-8,
                    1e-7,
//                    1e-6,
                    1e-5,
//                    1e-4,
                    1e-3,
//                    1e-2,
                    1e-1,
//                    1e0,
                    1e1,
//                    1e2,
                    1e3,
//                    1e4,
                    1e5,
//                    1e6,
                    1e7,
//                    1e8,
//                    1e9,
//                    1e10,
//                1e11,
//                1e12,
//                1e13,
//                    1e10,
//                    1e15,
//                    1e20,
//                    Double.MAX_VALUE
            };
            Double[] gammaValues = new Double[] {
//                    1e-11,
//                    1e-10,
//                    1e-9,
//                    1e-8,
                    1e-7,
//                    1e-6,
                    1e-5,
//                    1e-4,
                    1e-3,
//                    1e-2,
                    1e-1,
//                    1e0,
                    1e1,
//                    1e2,
                    1e3,
//                    1e4,
                    1e5,
//                    1e6,
                    1e7,
//                    1e8,
//                    1e9,
//                    1e10,
//                    1e15,
//                    1e20,
//                    Double.MAX_VALUE
            };
            runExperiments(loggingLevel, domainNames, errorEstimationMethods, alphaValues, gammaValues, functionOutputs, trueLabels, evaluationFunctionOutputs, results);
        });
        System.out.println("Combined Results:");
        Map<ErrorEstimationMethod, double[]> errorRateMADStatistics = results.getErrorRatesMADStatistics();
        Map<ErrorEstimationMethod, double[]> labelMADStatistics = results.getLabelMADStatistics();
        for (ErrorEstimationMethod method : results.getMethods()) {
            System.out.println(String.format(method.name() + ": " +
                                                     "MAD_error = %1.3e \u00B1 %1.3e, " +
                                                     "MAD_label = %1.3e \u00B1 %1.3e",
                                             errorRateMADStatistics.get(method)[0], errorRateMADStatistics.get(method)[1],
                                             labelMADStatistics.get(method)[0], labelMADStatistics.get(method)[1]));
        }
        System.out.println("Finished!");
    }

    public static void runExperiments(int loggingLevel,
                                      List<String> domainNames,
                                      ErrorEstimationMethod[] errorEstimationMethods,
                                      Double[] alphaValues,
                                      Double[] gammaValues,
                                      List<boolean[][]> functionOutputs,
                                      List<boolean[]> trueLabels,
                                      List<boolean[][]> evaluationFunctionOutputs,
                                      CombinedResults combinedResults) {
        // Compute the true error rates
        String trueErrorRatesString = "====================================================================================================\n";
        trueErrorRatesString += "TRUE_ERROR_RATES\n----------------------------------------------------------------------------------------------------\n";
        double[][] realErrorRates = new double[functionOutputs.size()][];
        for (int p = 0; p < functionOutputs.size(); p++) {
            realErrorRates[p] = new double[functionOutputs.get(p)[0].length];
            for (int i = 0; i < trueLabels.get(p).length; i++)
                for (int j = 0; j < functionOutputs.get(p)[i].length; j++)
                    realErrorRates[p][j] += (functionOutputs.get(p)[i][j] != trueLabels.get(p)[i]) ? 1 : 0;
            for (int j = 0; j < functionOutputs.get(p)[0].length; j++)
                realErrorRates[p][j] /= trueLabels.get(p).length;
            trueErrorRatesString += domainNames.get(p);
            for (int j = 0; j < functionOutputs.get(p)[0].length; j++)
                trueErrorRatesString += "\t" + realErrorRates[p][j];
            trueErrorRatesString += "\n";
        }
        if (loggingLevel > 2)
            System.out.print(trueErrorRatesString);
        // Combine the labels using a majority vote
        boolean[][] majorityVoteLabels = new boolean[functionOutputs.size()][];
        for (int p = 0; p < functionOutputs.size(); p++) {
            majorityVoteLabels[p] = new boolean[functionOutputs.get(p).length];
            for (int i = 0; i < functionOutputs.get(p).length; i++) {
                double labelsSum = 0;
                for (int j = 0; j < functionOutputs.get(p)[i].length; j++) {
                    labelsSum += (functionOutputs.get(p)[i][j] ? 1 : 0);
                }
                majorityVoteLabels[p][i] = labelsSum / functionOutputs.get(p)[i].length >= 0.5;
            }
        }
        double majorityVoteLabelMAD = 0;
        for (int p = 0; p < functionOutputs.size(); p++) {
            double labelsErrorRate = 0;
            for (int i = 0; i < trueLabels.get(p).length; i++)
                labelsErrorRate += (majorityVoteLabels[p][i] != trueLabels.get(p)[i]) ? 1 : 0;
            labelsErrorRate /= trueLabels.get(p).length;
            majorityVoteLabelMAD += labelsErrorRate;
        }
        majorityVoteLabelMAD /= functionOutputs.size();
        if (loggingLevel > 1)
            System.out.println("Majority Vote Label MAD: " + majorityVoteLabelMAD);
        double majorityVoteErrorRateMAD = 0;
        for (int p = 0; p < functionOutputs.size(); p++) {
            double[] estimatedErrorRates = new double[functionOutputs.get(p)[0].length];
            for (int i = 0; i < majorityVoteLabels[p].length; i++)
                for (int j = 0; j < functionOutputs.get(p)[i].length; j++)
                    estimatedErrorRates[j] += (functionOutputs.get(p)[i][j] != majorityVoteLabels[p][i]) ? 1 : 0;
            double errorRateMAD = 0;
            for (int j = 0; j < functionOutputs.get(p)[0].length; j++)
                errorRateMAD += Math.abs(realErrorRates[p][j] - estimatedErrorRates[j] / majorityVoteLabels[p].length);
            majorityVoteErrorRateMAD += errorRateMAD / functionOutputs.get(p)[0].length;
        }
        majorityVoteErrorRateMAD /= functionOutputs.size();
        if (loggingLevel > 1)
            System.out.println("Majority Vote Error Rate MAD: " + majorityVoteErrorRateMAD);
        combinedResults.addResult(ErrorEstimationMethod.MAJ, majorityVoteErrorRateMAD, majorityVoteLabelMAD);
        Arrays.asList(errorEstimationMethods).parallelStream().forEach(method -> {
            if (method != ErrorEstimationMethod.AR_2 && method != ErrorEstimationMethod.AR_N
                    && method != ErrorEstimationMethod.BEE && method != ErrorEstimationMethod.BEE_ME
                    && method != ErrorEstimationMethod.BEEc) {
                ConcurrentSkipListSet<ExperimentResults> differentParameterizationResults = new ConcurrentSkipListSet<>();
                Arrays.asList(alphaValues).parallelStream().forEach(alpha -> {
                    if (method == ErrorEstimationMethod.HCBEE
                            || method == ErrorEstimationMethod.HCBEEc) {
                        Arrays.asList(gammaValues).parallelStream().forEach(gamma -> {
                            ExperimentResults results = runExperiment(method, functionOutputs, trueLabels, evaluationFunctionOutputs, alpha, gamma);
                            differentParameterizationResults.add(results);
                            String resultsString =
                                    "====================================================================================================\n"
                                            + method + "\t-\tγ = " + gamma
                                            + "\t-\tα = " + alpha
                                            + "\t-\tError Rates MAD Mean: " + results.getErrorRateMAD()
                                            + "\t-\tLabels Error Rate Mean: " + results.getLabelMAD()
                                            + "\t-\tLog-likelihood: " + results.getLogLikelihood();
                            resultsString += "\n----------------------------------------------------------------------------------------------------\n";
                            if (loggingLevel > 2)
                                for (int p = 0; p < functionOutputs.size(); p++) {
                                    resultsString += domainNames.get(p);
                                    for (int j = 0; j < functionOutputs.get(p)[0].length; j++)
                                        resultsString += "\t" + results.getErrorRates()[p][j];
                                    resultsString += "\n";
                                }
                            if (loggingLevel > 1)
                                System.out.print(resultsString);
                        });
                    } else {
                        ExperimentResults results = runExperiment(method, functionOutputs, trueLabels, evaluationFunctionOutputs, alpha, 0);
                        differentParameterizationResults.add(results);
                        String resultsString =
                                "====================================================================================================\n"
                                        + method + "\t-\tα = " + alpha
                                        + "\t-\tError Rates MAD Mean: " + results.getErrorRateMAD()
                                        + "\t-\tLabels Error Rate Mean: " + results.getLabelMAD()
                                        + "\t-\tLog-likelihood: " + results.getLogLikelihood();
                        resultsString += "\n----------------------------------------------------------------------------------------------------\n";
                        if (loggingLevel > 2)
                            for (int p = 0; p < functionOutputs.size(); p++) {
                                resultsString += domainNames.get(p);
                                for (int j = 0; j < functionOutputs.get(p)[0].length; j++)
                                    resultsString += "\t" + results.getErrorRates()[p][j];
                                resultsString += "\n";
                            }
                        if (loggingLevel > 1)
                            System.out.print(resultsString);
                    }
                });
                combinedResults.addResult(method,
                                          differentParameterizationResults.last().getErrorRateMAD(),
                                          differentParameterizationResults.last().getLabelMAD());
            } else {
                ExperimentResults results = runExperiment(method, functionOutputs, trueLabels, evaluationFunctionOutputs, 0, 0);
                combinedResults.addResult(method, results.getErrorRateMAD(), results.getLabelMAD());
                String resultsString =
                        "====================================================================================================\n"
                                + method
                                + "\t-\tError Rates MAD Mean: " + results.getErrorRateMAD()
                                + "\t-\tLabels Error Rate Mean: " + results.getLabelMAD();
                resultsString += "\n----------------------------------------------------------------------------------------------------\n";
                if (loggingLevel > 2)
                    for (int p = 0; p < functionOutputs.size(); p++) {
                        resultsString += domainNames.get(p);
                        for (int j = 0; j < functionOutputs.get(p)[0].length; j++)
                            resultsString += "\t" + results.getErrorRates()[p][j];
                        resultsString += "\n";
                    }
                if (loggingLevel > 1)
                    System.out.print(resultsString);
            }
        });
        if (loggingLevel > 1)
            System.out.println("====================================================================================================");
    }

    public static ExperimentResults runExperiment(ErrorEstimationMethod method,
                                                  List<boolean[][]> functionOutputs,
                                                  List<boolean[]> trueLabels,
                                                  List<boolean[][]> evaluationFunctionOutputs,
                                                  double alpha,
                                                  double gamma) {
        double[][] errorRates = new double[functionOutputs.size()][];
        double[][] labels = new double[functionOutputs.size()][];
        int numberOfClusters = 1;
        double logLikelihood = 0;
        switch (method) {
            case cBCC:
                BayesianCombinationOfClassifiers bcc = new BayesianCombinationOfClassifiers(functionOutputs, 4000, 10, 200, alpha);
                bcc.runGibbsSampler();
                errorRates = bcc.getErrorRatesMeans();
                logLikelihood = bcc.logLikelihood(evaluationFunctionOutputs);
                labels = bcc.getLabelMeans();
//                double[][] labelMeansBcc = bcc.getLabelMeans();
//                for (int p = 0; p < functionOutputs.size(); p++) {
//                    labels[p] = new boolean[functionOutputs.get(p).length];
//                    for (int i = 0; i < functionOutputs.get(p).length; i++) {
//                        labels[p][i] = labelMeansBcc[p][i] >= 0.5;
//                    }
//                }
                break;
            case cBCCc:
                BayesianCombinationOfClassifiersConfusion bccc = new BayesianCombinationOfClassifiersConfusion(functionOutputs, 4000, 10, 200, alpha);
                bccc.runGibbsSampler();
                errorRates = bccc.getErrorRatesMeans();
                logLikelihood = bccc.logLikelihood(evaluationFunctionOutputs);
                labels = bccc.getLabelMeans();
//                double[][] labelMeansBccc = bccc.getLabelMeans();
//                for (int p = 0; p < functionOutputs.size(); p++) {
//                    labels[p] = new boolean[functionOutputs.get(p).length];
//                    for (int i = 0; i < functionOutputs.get(p).length; i++) {
//                        labels[p][i] = labelMeansBccc[p][i] >= 0.5;
//                    }
//                }
                break;
            case AR_2:
                for (int p = 0; p < functionOutputs.size(); p++) {
                    int numberOfFunctions = functionOutputs.get(p)[0].length;
                    ErrorEstimationData errorEstimationData = new ErrorEstimationData.Builder(
                            Arrays.asList(functionOutputs.get(p)),
                            2,
                            true).build();
                    ErrorEstimation errorEstimation = new ErrorEstimation.Builder(errorEstimationData)
                            .optimizationSolverType(ErrorEstimationInternalSolver.IP_OPT)
                            .build();
                    double[] allErrorRates = errorEstimation.solve().getErrorRates().array;
                    errorRates[p] = new double[numberOfFunctions];
                    System.arraycopy(allErrorRates, 0, errorRates[p], 0, numberOfFunctions);
//                    // Combine the labels using a weighted majority vote
//                    labels[p] = new boolean[functionOutputs.get(p).length];
//                    for (int i = 0; i < functionOutputs.get(p).length; i++) {
//                        double labelsSum = 0;
//                        double errorRatesSum = 0;
//                        for (int j = 0; j < functionOutputs.get(p)[i].length; j++) {
//                            labelsSum += (1 - errorRates[p][j]) * (functionOutputs.get(p)[i][j] ? 1 : 0);
//                            errorRatesSum += (1 - errorRates[p][j]);
//                        }
//                        labels[p][i] = labelsSum / errorRatesSum >= 0.5;
//                    }
                    // Combine the labels using a weighted majority vote
                    labels[p] = new double[functionOutputs.get(p).length];
                    for (int i = 0; i < functionOutputs.get(p).length; i++) {
                        double labelsSum = 0;
                        double errorRatesSum = 0;
                        for (int j = 0; j < functionOutputs.get(p)[i].length; j++) {
                            labelsSum += (1 - errorRates[p][j]) * (functionOutputs.get(p)[i][j] ? 1 : 0);
                            errorRatesSum += (1 - errorRates[p][j]);
                        }
                        labels[p][i] = labelsSum / errorRatesSum >= 0.5 ? 1 : 0;
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
//                    // Combine the labels using a weighted majority vote
//                    labels[p] = new boolean[functionOutputs.get(p).length];
//                    for (int i = 0; i < functionOutputs.get(p).length; i++) {
//                        double labelsSum = 0;
//                        double errorRatesSum = 0;
//                        for (int j = 0; j < functionOutputs.get(p)[i].length; j++) {
//                            labelsSum += (1 - errorRates[p][j]) * (functionOutputs.get(p)[i][j] ? 1 : 0);
//                            errorRatesSum += (1 - errorRates[p][j]);
//                        }
//                        labels[p][i] = labelsSum / errorRatesSum >= 0.5;
//                    }
                    // Combine the labels using a weighted majority vote
                    labels[p] = new double[functionOutputs.get(p).length];
                    for (int i = 0; i < functionOutputs.get(p).length; i++) {
                        double labelsSum = 0;
                        double errorRatesSum = 0;
                        for (int j = 0; j < functionOutputs.get(p)[i].length; j++) {
                            labelsSum += (1 - errorRates[p][j]) * (functionOutputs.get(p)[i][j] ? 1 : 0);
                            errorRatesSum += (1 - errorRates[p][j]);
                        }
                        labels[p][i] = labelsSum / errorRatesSum >= 0.5 ? 1 : 0;
                    }
                }
                break;
            case BEE:
                BayesianErrorEstimation eesgm = new BayesianErrorEstimation(functionOutputs, 4000, 10, 200);
                eesgm.runGibbsSampler();
                errorRates = eesgm.getErrorRatesMeans();
                labels = eesgm.getLabelMeans();
//                double[][] labelMeansEesgm = eesgm.getLabelMeans();
//                for (int p = 0; p < functionOutputs.size(); p++) {
//                    labels[p] = new boolean[functionOutputs.get(p).length];
//                    for (int i = 0; i < functionOutputs.get(p).length; i++) {
//                        labels[p][i] = labelMeansEesgm[p][i] >= 0.5;
//                    }
//                }
                break;
            case BEE_ME:
                BayesianErrorEstimation eesgm_me = new BayesianErrorEstimation(functionOutputs, 4000, 10, 200, true);
                eesgm_me.runGibbsSampler();
                errorRates = eesgm_me.getErrorRatesMeans();
                labels = eesgm_me.getLabelMeans();
//                double[][] labelMeansEesgm_me = eesgm_me.getLabelMeans();
//                for (int p = 0; p < functionOutputs.size(); p++) {
//                    labels[p] = new boolean[functionOutputs.get(p).length];
//                    for (int i = 0; i < functionOutputs.get(p).length; i++) {
//                        labels[p][i] = labelMeansEesgm_me[p][i] >= 0.5;
//                    }
//                }
                break;
            case BEEc:
                BayesianErrorEstimationConfusion beec = new BayesianErrorEstimationConfusion(functionOutputs, 4000, 10, 200);
                beec.runGibbsSampler();
                errorRates = beec.getErrorRatesMeans();
                labels = beec.getLabelMeans();
//                double[][] labelMeansBeec = beec.getLabelMeans();
//                for (int p = 0; p < functionOutputs.size(); p++) {
//                    labels[p] = new boolean[functionOutputs.get(p).length];
//                    for (int i = 0; i < functionOutputs.get(p).length; i++) {
//                        labels[p][i] = labelMeansBeec[p][i] >= 0.5;
//                    }
//                }
                break;
            case CBEE:
                CoupledBayesianErrorEstimation eedfdpgm = new CoupledBayesianErrorEstimation(functionOutputs, 5000, 10, alpha);
                eedfdpgm.performGibbsSampling();
                errorRates = eedfdpgm.getErrorRatesMeans();
                logLikelihood = eedfdpgm.logLikelihood(evaluationFunctionOutputs);
                labels = eedfdpgm.getLabelMeans();
//                double[][] labelMeansEedfdpgm = eedfdpgm.getLabelMeans();
//                for (int p = 0; p < functionOutputs.size(); p++) {
//                    labels[p] = new boolean[functionOutputs.get(p).length];
//                    for (int i = 0; i < functionOutputs.get(p).length; i++) {
//                        labels[p][i] = labelMeansEedfdpgm[p][i] >= 0.5;
//                    }
//                }
                break;
            case CBEEc:
                CoupledBayesianErrorEstimationConfusion cbeec = new CoupledBayesianErrorEstimationConfusion(functionOutputs, 5000, 10, alpha);
                cbeec.performGibbsSampling();
                errorRates = cbeec.getErrorRatesMeans();
                logLikelihood = cbeec.logLikelihood(evaluationFunctionOutputs);
                labels = cbeec.getLabelMeans();
//                double[][] labelMeansCbeec = cbeec.getLabelMeans();
//                for (int p = 0; p < functionOutputs.size(); p++) {
//                    labels[p] = new boolean[functionOutputs.get(p).length];
//                    for (int i = 0; i < functionOutputs.get(p).length; i++) {
//                        labels[p][i] = labelMeansCbeec[p][i] >= 0.5;
//                    }
//                }
                break;
            case HCBEE:
//                ErrorEstimationDomainsHDPNew eedfhdp = new ErrorEstimationDomainsHDPNew(functionOutputs, alpha, gamma);
//                eedfhdp.run_gibbs_collapsed(1000);
//                eedfhdp.run_gibbs_uncollapsed(1000, 100, 200);
//                errorRates = eedfhdp.rates_to_return;
//                double[][] labelMeansEedfhdpmgm = eedfhdp.labels_to_return;
//                int li_cnt[][] = new int[functionOutputs.size()][2];
//                for (int p = 0; p < functionOutputs.size(); p++) {
//                    labels[p] = new double[functionOutputs.get(p).length];
//                    for (int i = 0; i < functionOutputs.get(p).length; i++) {
//                        labels[p][i] = labelMeansEedfhdpmgm[p][i];
//                        int lid = labelMeansEedfhdpmgm[p][i] >= 0.5? 1:0;
//                        li_cnt[p][lid]++;
//                    }
//                }
//                logLikelihood = eedfhdp.get_log_likelihood(evaluationFunctionOutputs, alpha, gamma, 1000, li_cnt);
//                HierarchicalCoupledBayesianErrorEstimation hcee = new HierarchicalCoupledBayesianErrorEstimation(functionOutputs, 20000, 10, 500, alpha, gamma);
                ErrorEstimationHDPErrorRate hcee = new ErrorEstimationHDPErrorRate(functionOutputs, 20000, 10, 500, alpha, gamma);
                hcee.runGibbsSampler();
                errorRates = hcee.getErrorRatesMeans();
                logLikelihood = hcee.logLikelihood(evaluationFunctionOutputs);
                labels = hcee.getLabelMeans();
//                double[][] labelMeansHcee = hcee.getLabelMeans();
//                for (int p = 0; p < functionOutputs.size(); p++) {
//                    labels[p] = new boolean[functionOutputs.get(p).length];
//                    for (int i = 0; i < functionOutputs.get(p).length; i++) {
//                        labels[p][i] = labelMeansHcee[p][i] >= 0.5;
//                    }
//                }
                break;
            case HCBEEc:
                HierarchicalCoupledErrorEstimationConfusion hceec = new HierarchicalCoupledErrorEstimationConfusion(functionOutputs, 4000, 10, 200, alpha, gamma);
                hceec.runGibbsSampler();
                errorRates = hceec.getErrorRatesMeans();
                logLikelihood = hceec.logLikelihood(evaluationFunctionOutputs);
                labels = hceec.getLabelMeans();
//                double[][] labelMeansHceec = hceec.getLabelMeans();
//                for (int p = 0; p < functionOutputs.size(); p++) {
//                    labels[p] = new boolean[functionOutputs.get(p).length];
//                    for (int i = 0; i < functionOutputs.get(p).length; i++) {
//                        labels[p][i] = labelMeansHceec[p][i] >= 0.5;
//                    }
//                }
                break;
        }
        double[] errorRatesMSE = new double[functionOutputs.size()];
        double errorRatesMADMean = 0;
        double[] labelsMSE = new double[functionOutputs.size()];
        double labelsErrorRateMean = 0;
        for (int p = 0; p < functionOutputs.size(); p++) {
            labelsMSE[p] = 0;
            double[] realErrorRates = new double[errorRates[p].length];
            for (int i = 0; i < trueLabels.get(p).length; i++) {
                labelsMSE[p] += Math.pow(labels[p][i] - (trueLabels.get(p)[i] ? 1 : 0), 2);
//                labelsErrorRate[p] += (labels[p][i] != trueLabels.get(p)[i]) ? 1 : 0;
                for (int j = 0; j < errorRates[p].length; j++)
                    realErrorRates[j] += (functionOutputs.get(p)[i][j] != trueLabels.get(p)[i]) ? 1 : 0;
            }
            labelsMSE[p] /= trueLabels.get(p).length;
            labelsErrorRateMean += labelsMSE[p];
            errorRatesMSE[p] = 0;
            for (int j = 0; j < errorRates[p].length; j++) {
                realErrorRates[j] /= trueLabels.get(p).length;
                errorRatesMSE[p] += Math.pow(errorRates[p][j] - realErrorRates[j], 2);
            }
            errorRatesMSE[p] /= errorRates[p].length;
            errorRatesMADMean += errorRatesMSE[p];
        }
        errorRatesMADMean /= functionOutputs.size();
        labelsErrorRateMean /= functionOutputs.size();
        return new ExperimentResults(errorRates, errorRatesMADMean, labelsErrorRateMean, logLikelihood);
    }

    public static DomainData parseLabeledDataFromCSVFile(
            File file,
            String separator,
            double[] classificationThresholds,
            double subSampling,
            boolean evaluationData
    ) {
        String domainName = file.getName();
        BufferedReader br = null;
        String line;
        List<ParsedExample> parsedExamples = new ArrayList<>();
        try {
            br = new BufferedReader(new FileReader(file));
            br.readLine();
            while ((line = br.readLine()) != null) {
                String[] outputs = line.split(separator);
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
                parsedExamples.add(new ParsedExample(booleanOutputs, !outputs[0].equals("0")));
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
        int numberOfExamplesToKeep = parsedExamples.size();
        if (subSampling < 1) {
            Collections.shuffle(parsedExamples);
            numberOfExamplesToKeep = (int) Math.floor(parsedExamples.size() * subSampling);
        }
        List<boolean[]> classifiersOutputsList = new ArrayList<>();
        List<Boolean> trueLabelsList = new ArrayList<>();
        for (int exampleIndex = 0; exampleIndex < numberOfExamplesToKeep; exampleIndex++) {
            classifiersOutputsList.add(parsedExamples.get(exampleIndex).classifierOutputs);
            trueLabelsList.add(parsedExamples.get(exampleIndex).trueLabel);
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

    private static class ParsedExample {
        private final boolean[] classifierOutputs;
        private final boolean trueLabel;

        private ParsedExample(boolean[] classifierOutputs, boolean trueLabel) {
            this.classifierOutputs = classifierOutputs;
            this.trueLabel = trueLabel;
        }
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

    private static class ExperimentResults implements Comparable<ExperimentResults> {
        private double[][] errorRates;
        private double errorRateMAD;
        private double labelMAD;
        private double logLikelihood = 0;

        protected ExperimentResults(double[][] errorRates,
                                    double errorRateMAD,
                                    double labelMAD) {
            this.errorRates = errorRates;
            this.errorRateMAD = errorRateMAD;
            this.labelMAD = labelMAD;
        }

        protected ExperimentResults(double[][] errorRates,
                                    double errorRateMAD,
                                    double labelMAD,
                                    double logLikelihood) {
            this.errorRates = errorRates;
            this.errorRateMAD = errorRateMAD;
            this.labelMAD = labelMAD;
            this.logLikelihood = logLikelihood;
        }

        protected double[][] getErrorRates() {
            return errorRates;
        }

        protected double getErrorRateMAD() {
            return errorRateMAD;
        }

        protected double getLabelMAD() {
            return labelMAD;
        }

        protected double getLogLikelihood() {
            return logLikelihood;
        }

        @Override
        public int compareTo(ExperimentResults other) {
            return ComparisonChain.start()
                    .compare(logLikelihood, other.logLikelihood)
                    .compare(errorRateMAD, other.errorRateMAD)
                    .compare(labelMAD, other.labelMAD)
                    .result();
        }
    }

    private static class CombinedResults {
        private Map<ErrorEstimationMethod, List<Double>> errorRateMADs = new ConcurrentHashMap<>();
        private Map<ErrorEstimationMethod, List<Double>> labelMADs = new ConcurrentHashMap<>();

        protected CombinedResults() { }

        public Map<ErrorEstimationMethod, List<Double>> getErrorRateMADs() {
            return errorRateMADs;
        }

        public Map<ErrorEstimationMethod, List<Double>> getLabelMADs() {
            return labelMADs;
        }

        public Set<ErrorEstimationMethod> getMethods() {
            return Sets.union(errorRateMADs.keySet(), labelMADs.keySet());
        }

        public Map<ErrorEstimationMethod, double[]> getErrorRatesMADStatistics() {
            Map<ErrorEstimationMethod, double[]> errorRateMADStatistics = new HashMap<>();
            for (ErrorEstimationMethod method : errorRateMADs.keySet())
                errorRateMADStatistics.put(method, new double[]{
                        StatisticsUtilities.mean(errorRateMADs.get(method)),
                        2 * StatisticsUtilities.standardDeviation(errorRateMADs.get(method))
                });
            return errorRateMADStatistics;
        }

        public Map<ErrorEstimationMethod, double[]> getLabelMADStatistics() {
            Map<ErrorEstimationMethod, double[]> labelMADStatistics = new HashMap<>();
            for (ErrorEstimationMethod method : labelMADs.keySet())
                labelMADStatistics.put(method, new double[]{
                        StatisticsUtilities.mean(labelMADs.get(method)),
                        2 * StatisticsUtilities.standardDeviation(labelMADs.get(method))
                });
            return labelMADStatistics;
        }

        protected void addResult(ErrorEstimationMethod method, double errorRateMAD, double labelMAD) {
            if (!errorRateMADs.containsKey(method))
                errorRateMADs.put(method, new ArrayList<>());
            errorRateMADs.get(method).add(errorRateMAD);
            if (!labelMADs.containsKey(method))
                labelMADs.put(method, new ArrayList<>());
            labelMADs.get(method).add(labelMAD);
        }
    }

    private enum ErrorEstimationMethod {
        MAJ,
        cBCC,
        cBCCc,
        AR_2,
        AR_N,
        BEE,
        BEE_ME,
        BEEc,
        CBEE,
        CBEEc,
        HCBEE,
        HCBEEc
    }
}
