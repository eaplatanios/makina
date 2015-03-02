package org.platanios.learn.classification.reflection;

import org.apache.commons.math3.random.RandomDataGenerator;
import org.platanios.learn.math.matrix.MatrixUtilities;

import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import static org.apache.commons.math3.special.Beta.logBeta;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ErrorEstimationDomainsFastDPGraphicalModel {
    private final Random random = new Random();
    private final RandomDataGenerator randomDataGenerator = new RandomDataGenerator();
    private final double alpha_p = 1;
    private final double beta_p = 1;
    private final double alpha_e = 1;
    private final double beta_e = 1;

    private final double alpha;
    private final int numberOfIterations;
    private final int burnInIterations;
    private final int thinning;
    private final int numberOfSamples;
    private final int numberOfFunctions;
    private final int numberOfDomains;
    private final int[] numberOfDataSamples;
    private final int[][][] labelsSamples;
    private final int[][][] functionOutputsArray;
    private final int[][] zSamples;
    private final double[][] priorSamples;
    private final double[][][] errorRateSamples;

    private double[][] disagreements;
    private double[][] sum_1;
    private double[][] sum_2;

    private double[] priorMeans;
    private double[] priorVariances;
    private double[][] labelMeans;
    private double[][] labelVariances;
    private double[][] errorRateMeans;
    private double[][] errorRateVariances;

    public int numberOfClusters = 1;
    
    private FastDPPrior dp;

    public ErrorEstimationDomainsFastDPGraphicalModel(List<boolean[][]> functionOutputs, int numberOfIterations, int thinning, double alpha) {
        this.alpha = alpha;
        this.numberOfIterations = numberOfIterations;
        burnInIterations = numberOfIterations * 9 / 10;
        this.thinning = thinning;
        numberOfFunctions = functionOutputs.get(0)[0].length;
        numberOfDomains = functionOutputs.size();
        numberOfDataSamples = new int[numberOfDomains];
        functionOutputsArray = new int[numberOfFunctions][numberOfDomains][];
        for (int p = 0; p < numberOfDomains; p++) {
            numberOfDataSamples[p] = functionOutputs.get(p).length;
            for (int j = 0; j < numberOfFunctions; j++) {
                functionOutputsArray[j][p] = new int[numberOfDataSamples[p]];
                for (int i = 0; i < numberOfDataSamples[p]; i++)
                    functionOutputsArray[j][p][i] = functionOutputs.get(p)[i][j] ? 1 : 0;
            }
        }
        dp = new FastDPPrior(alpha, numberOfDomains);
        
        numberOfSamples = (numberOfIterations - burnInIterations) / thinning;
        priorSamples = new double[numberOfSamples][numberOfDomains];
        errorRateSamples = new double[numberOfSamples][numberOfDomains][numberOfFunctions];
        zSamples = new int[numberOfSamples][numberOfDomains];
        labelsSamples = new int[numberOfSamples][numberOfDomains][];

        disagreements = new double[numberOfFunctions][numberOfDomains];
        sum_1 = new double[numberOfFunctions][numberOfDomains];
        sum_2 = new double[numberOfFunctions][numberOfDomains];

        priorMeans = new double[numberOfDomains];
        priorVariances = new double[numberOfDomains];
        labelMeans = new double[numberOfDomains][];
        labelVariances = new double[numberOfDomains][];
        errorRateMeans = new double[numberOfDomains][numberOfFunctions];
        errorRateVariances = new double[numberOfDomains][numberOfFunctions];
        for (int p = 0; p < numberOfDomains; p++) {
            labelMeans[p] = new double[numberOfDataSamples[p]];
            labelVariances[p] = new double[numberOfDataSamples[p]];
            zSamples[0][p] = 0;
            dp.add_topic_assingment(0);
            priorSamples[0][p] = 0.5;
            labelsSamples[0][p] = new int[numberOfDataSamples[p]];
            for (int i = 0; i < numberOfDataSamples[p]; i++) {
                int sum = 0;
                for (int j = 0; j < numberOfFunctions; j++)
                    sum += functionOutputsArray[j][p][i];
//                labelsSamples[0][p][i] = sum >= (numberOfFunctions / 2) ? 1 : 0;
                labelsSamples[0][p][i] = randomDataGenerator.nextBinomial(1, 0.5);
            }
            for (int j = 0; j < numberOfFunctions; j++) {
                errorRateSamples[0][p][j] = 0.25;
                disagreements[j][p] = 0;
                for (int i = 0; i < numberOfDataSamples[p]; i++)
                    if (functionOutputsArray[j][p][i] != labelsSamples[0][p][i])
                        disagreements[j][p]++;
            }
        }
        for (int k = 0; k < numberOfDomains; k++) {
            for (int j = 0; j < numberOfFunctions; j++) {
                sum_1[j][k] = 0;
                sum_2[j][k] = 0;
                for (int p = 0; p < numberOfDomains; p++) {
                    if (zSamples[0][p] == k) {
                        sum_1[j][k] += numberOfDataSamples[p];
                        sum_2[j][k] += disagreements[j][p];
                    }
                }
            }
        }
    }

    public void performGibbsSampling() {
        for (int iterationNumber = 0; iterationNumber < burnInIterations; iterationNumber++) {
            samplePriorsAndBurn(0);
            sampleLabelsAndBurnWithCollapsedErrorRates(0);
            sampleZAndBurnWithCollapsedErrorRates(0);
//            samplePriorsAndBurn(0);
//            sampleErrorRatesAndBurn(0);
//            sampleZAndBurn(0);
//            sampleLabelsAndBurn(0);
        }
        for (int iterationNumber = 0; iterationNumber < numberOfSamples - 1; iterationNumber++) {
            for (int i = 0; i < thinning; i++) {
                samplePriorsAndBurn(iterationNumber);
                sampleErrorRatesAndBurn(iterationNumber);
                sampleZAndBurn(iterationNumber);
                sampleLabelsAndBurn(iterationNumber);
            }
            samplePriors(iterationNumber);
            sampleErrorRates(iterationNumber);
            sampleZ(iterationNumber);
            sampleLabels(iterationNumber);
        }
        Set<Integer> uniqueClusters = new HashSet<>();
        for (int p = 0; p < numberOfDomains; p++)
            uniqueClusters.add(zSamples[zSamples.length - 1][p]);
        numberOfClusters = uniqueClusters.size();
        // Aggregate values for means and variances computation
        for (int sampleNumber = 0; sampleNumber < numberOfSamples; sampleNumber++) {
            for (int p = 0; p < numberOfDomains; p++) {
//                int numberOfPhiBelowChance = 0;
//                for (int j = 0; j < numberOfFunctions; j++)
//                    if (errorRateSamples[sampleNumber][zSamples[sampleNumber][p]][j] < 0.5)
//                        numberOfPhiBelowChance++;
//                if (numberOfPhiBelowChance < numberOfFunctions / 2.0) {
//                    priorSamples[sampleNumber][p] = 1 - priorSamples[sampleNumber][p];
//                    for (int j = 0; j < numberOfFunctions; j++)
//                        errorRateSamples[sampleNumber][zSamples[sampleNumber][p]][j] = 1 - errorRateSamples[sampleNumber][zSamples[sampleNumber][p]][j];
//                }
                priorMeans[p] += priorSamples[sampleNumber][p];
                for (int j = 0; j < numberOfFunctions; j++)
                    errorRateMeans[p][j] += errorRateSamples[sampleNumber][zSamples[sampleNumber][p]][j];
                for (int i = 0; i < numberOfDataSamples[p]; i++)
                    labelMeans[p][i] += labelsSamples[sampleNumber][p][i];
            }
        }
        // Compute values for the means and the variances
        for (int p = 0; p < numberOfDomains; p++) {
            priorMeans[p] /= numberOfSamples;
            for (int j = 0; j < numberOfFunctions; j++)
                errorRateMeans[p][j] /= numberOfSamples;
            for (int i = 0; i < numberOfDataSamples[p]; i++)
                labelMeans[p][i] /= numberOfSamples;
            for (int sampleNumber = 0; sampleNumber < numberOfSamples; sampleNumber++) {
                double temp = priorSamples[sampleNumber][p] - priorMeans[p];
                priorVariances[p] += temp * temp;
                for (int j = 0; j < numberOfFunctions; j++) {
                    temp = errorRateSamples[sampleNumber][zSamples[sampleNumber][p]][j] - errorRateMeans[p][j];
                    errorRateVariances[p][j] += temp * temp;
                }
                for (int i = 0; i < numberOfDataSamples[p]; i++) {
                    temp = labelsSamples[sampleNumber][p][i] - labelMeans[p][i];
                    labelVariances[p][i] += temp * temp;
                }
            }
            priorVariances[p] /= (numberOfIterations - burnInIterations - 1);
            for (int j = 0; j < numberOfFunctions; j++)
                errorRateVariances[p][j] /= (numberOfIterations - burnInIterations - 1);
            for (int i = 0; i < numberOfDataSamples[p]; i++)
                labelVariances[p][i] /= (numberOfIterations - burnInIterations - 1);
        }
    }

    private void samplePriorsAndBurn(int iterationNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            int labelsCount = 0;
            for (int i = 0; i < numberOfDataSamples[p]; i++)
                labelsCount += labelsSamples[iterationNumber][p][i];
            priorSamples[iterationNumber][p] = randomDataGenerator.nextBeta(alpha_p + labelsCount, beta_p + numberOfDataSamples[p] - labelsCount);
        }
    }

    private void samplePriors(int iterationNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            int labelsCount = 0;
            for (int i = 0; i < numberOfDataSamples[p]; i++)
                labelsCount += labelsSamples[iterationNumber][p][i];
            priorSamples[iterationNumber + 1][p] = randomDataGenerator.nextBeta(alpha_p + labelsCount, beta_p + numberOfDataSamples[p] - labelsCount);
        }
    }

    private void sampleMAPPriors(int iterationNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            int labelsCount = 0;
            for (int i = 0; i < numberOfDataSamples[p]; i++)
                labelsCount += labelsSamples[0][p][i];
            priorSamples[iterationNumber][p] = randomDataGenerator.nextBeta(alpha_p + labelsCount, beta_p + numberOfDataSamples[p] - labelsCount);
        }
    }

    private void sampleErrorRatesAndBurn(int iterationNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            int numberOfErrorRatesBelowChance = 0;
            for (int j = 0; j < numberOfFunctions; j++) {
                int disagreementCount = 0;
                int zCount = 0;
                for (int k = 0; k < numberOfDomains; k++) {
                    if (zSamples[iterationNumber][k] == p) {
                        for (int i = 0; i < numberOfDataSamples[k]; i++)
                            if (functionOutputsArray[j][k][i] != labelsSamples[iterationNumber][k][i])
                                disagreementCount++;
                        zCount += numberOfDataSamples[k];
                    }
                }
                errorRateSamples[iterationNumber][p][j] = randomDataGenerator.nextBeta(alpha_e + disagreementCount, beta_e + zCount - disagreementCount);
                if (errorRateSamples[iterationNumber][p][j] < 0.5)
                    numberOfErrorRatesBelowChance += 1;
            }
            if (numberOfErrorRatesBelowChance < numberOfFunctions / 2.0)
                for (int j = 0; j < numberOfFunctions; j++)
                    errorRateSamples[iterationNumber][p][j] = 1 - errorRateSamples[iterationNumber][p][j];
        }
    }

    private void sampleErrorRates(int iterationNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            int numberOfErrorRatesBelowChance = 0;
            for (int j = 0; j < numberOfFunctions; j++) {
                int disagreementCount = 0;
                int zCount = 0;
                for (int k = 0; k < numberOfDomains; k++) {
                    if (zSamples[iterationNumber][k] == p) {
                        for (int i = 0; i < numberOfDataSamples[k]; i++)
                            if (functionOutputsArray[j][k][i] != labelsSamples[iterationNumber][k][i])
                                disagreementCount++;
                        zCount += numberOfDataSamples[k];
                    }
                }
                errorRateSamples[iterationNumber + 1][p][j] = randomDataGenerator.nextBeta(alpha_e + disagreementCount, beta_e + zCount - disagreementCount);
                if (errorRateSamples[iterationNumber + 1][p][j] < 0.5)
                    numberOfErrorRatesBelowChance += 1;
            }
            if (numberOfErrorRatesBelowChance < numberOfFunctions / 2.0) {
                priorSamples[iterationNumber + 1][p] = 1 - priorSamples[iterationNumber + 1][p];
                for (int j = 0; j < numberOfFunctions; j++)
                    errorRateSamples[iterationNumber + 1][p][j] = 1 - errorRateSamples[iterationNumber + 1][p][j];
            }
        }
    }

    private void sampleMAPErrorRatesAndBurn(int iterationNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
//            int numberOfErrorRatesBelowChance = 0;
            for (int j = 0; j < numberOfFunctions; j++) {
                int disagreementCount = 0;
                int zCount = 0;
                for (int k = 0; k < numberOfDomains; k++) {
                    if (zSamples[iterationNumber][k] == p) {
                        for (int i = 0; i < numberOfDataSamples[k]; i++)
                            if (functionOutputsArray[j][k][i] != labelsSamples[iterationNumber][k][i])
                                disagreementCount++;
                        zCount += numberOfDataSamples[k];
                    }
                }
                errorRateSamples[iterationNumber][p][j] = (alpha_e + disagreementCount) / (alpha_e + beta_e + zCount);
//                if (errorRateSamples[iterationNumber][p][j] < 0.5)
//                    numberOfErrorRatesBelowChance += 1;
            }
//            if (numberOfErrorRatesBelowChance < numberOfFunctions / 2.0) {
//                priorSamples[iterationNumber][p] = 1 - priorSamples[iterationNumber][p];
//                for (int j = 0; j < numberOfFunctions; j++)
//                    errorRateSamples[iterationNumber][p][j] = 1 - errorRateSamples[iterationNumber][p][j];
//            }
        }
    }

    private void sampleMAPErrorRates(int iterationNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
//            int numberOfErrorRatesBelowChance = 0;
            for (int j = 0; j < numberOfFunctions; j++) {
                int disagreementCount = 0;
                int zCount = 0;
                for (int k = 0; k < numberOfDomains; k++) {
                    if (zSamples[iterationNumber + 1][k] == p) {
                        for (int i = 0; i < numberOfDataSamples[k]; i++)
                            if (functionOutputsArray[j][k][i] != labelsSamples[iterationNumber + 1][k][i])
                                disagreementCount++;
                        zCount += numberOfDataSamples[k];
                    }
                }
                errorRateSamples[iterationNumber + 1][p][j] = (alpha_e + disagreementCount) / (alpha_e + beta_e + zCount);
//                if (errorRateSamples[iterationNumber + 1][p][j] < 0.5)
//                    numberOfErrorRatesBelowChance += 1;
            }
//            if (numberOfErrorRatesBelowChance < numberOfFunctions / 2.0) {
//                priorSamples[iterationNumber + 1][p] = 1 - priorSamples[iterationNumber + 1][p];
//                for (int j = 0; j < numberOfFunctions; j++)
//                    errorRateSamples[iterationNumber + 1][p][j] = 1 - errorRateSamples[iterationNumber + 1][p][j];
//            }
        }
    }

    private void sampleZAndBurnWithCollapsedErrorRates(int iterationNumber) {
//        for (int p = 0; p < numberOfDomains; p++) {
//            for (int j = 0; j < numberOfFunctions; j++) {
//                disagreements[j][p] = 0;
//                for (int i = 0; i < numberOfDataSamples[p]; i++)
//                    if (functionOutputsArray[j][p][i] != labelsSamples[iterationNumber][p][i])
//                        disagreements[j][p]++;
//            }
//        }
//        for (int j = 0; j < numberOfFunctions; j++) {
//            for (int k = 0; k < numberOfDomains; k++) {
//                sum_2[j][k] = 0;
//                for (int p = 0; p < numberOfDomains; p++)
//                    if (zSamples[iterationNumber][p] == k)
//                        sum_2[j][k] += disagreements[j][p];
//            }
//        }
        for (int p = 0; p < numberOfDomains; p++) {
            dp.remove_topic_assignment(zSamples[iterationNumber][p]);
            int total_cnt = dp.prob_topics();
            double z_probabilities[] = new double[total_cnt];
            for(int i = 0; i < total_cnt; i++)
                z_probabilities[i] = Math.log(dp.pdf[i].prob);
            for (int j = 0; j < numberOfFunctions; j++) {
                sum_1[j][zSamples[iterationNumber][p]] -= numberOfDataSamples[p];
                sum_2[j][zSamples[iterationNumber][p]] -= disagreements[j][p];
                for(int i = 0; i < total_cnt - 1; i++) {
                    z_probabilities[i] +=
                            logBeta(alpha_e + sum_2[j][dp.pdf[i].topic] + disagreements[j][p],
                                    beta_e + sum_1[j][dp.pdf[i].topic] - sum_2[j][dp.pdf[i].topic] + numberOfDataSamples[p] - disagreements[j][p])
                                    - logBeta(alpha_e + sum_2[j][dp.pdf[i].topic],
                                              beta_e + sum_1[j][dp.pdf[i].topic] - sum_2[j][dp.pdf[i].topic]);
                }
                z_probabilities[total_cnt - 1] +=
                        logBeta(alpha_e + disagreements[j][p],
                                beta_e + numberOfDataSamples[p] - disagreements[j][p])
                                - logBeta(alpha_e, beta_e);
            }
            for (int i = 1; i < total_cnt; i++)
                z_probabilities[i] = MatrixUtilities.computeLogSumExp(z_probabilities[i - 1], z_probabilities[i]);
            double uniform = Math.log(random.nextDouble()) + z_probabilities[total_cnt - 1];
            zSamples[iterationNumber][p] = dp.pdf[total_cnt - 1].topic;
            for(int i = 0; i < total_cnt - 1; i++)  {
                if (z_probabilities[i] > uniform) {
                    zSamples[iterationNumber][p] = dp.pdf[i].topic;
                    for (int j = 0; j < numberOfFunctions; j++) {
                        sum_1[j][dp.pdf[i].topic] += numberOfDataSamples[p];
                        sum_2[j][dp.pdf[i].topic] += disagreements[j][p];
                    }
                    dp.add_topic_assingment(dp.pdf[i].topic);
                    break;
                }
            }
            if (zSamples[iterationNumber][p] == dp.pdf[total_cnt - 1].topic) {
                for (int j = 0; j < numberOfFunctions; j++) {
                    sum_1[j][dp.pdf[total_cnt - 1].topic] += numberOfDataSamples[p];
                    sum_2[j][dp.pdf[total_cnt - 1].topic] += disagreements[j][p];
                }
                dp.add_topic_assingment(dp.pdf[total_cnt - 1].topic);
            }
        }
    }

    private void sampleZWithCollapsedErrorRates(int iterationNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            dp.remove_topic_assignment(zSamples[iterationNumber][p]);
            int total_cnt = dp.prob_topics();
            double z_probabilities[] = new double[total_cnt];
            for(int i = 0; i < total_cnt; i++)
                z_probabilities[i] = Math.log(dp.pdf[i].prob);
            for (int j = 0; j < numberOfFunctions; j++) {
                sum_1[j][zSamples[iterationNumber][p]] -= numberOfDataSamples[p];
                sum_2[j][zSamples[iterationNumber][p]] -= disagreements[j][p];
                for(int i = 0; i < total_cnt - 1; i++) {
                    z_probabilities[i] +=
                            logBeta(alpha_e + sum_2[j][dp.pdf[i].topic] + disagreements[j][p],
                                    beta_e + sum_1[j][dp.pdf[i].topic] - sum_2[j][dp.pdf[i].topic] + numberOfDataSamples[p] - disagreements[j][p])
                                    - logBeta(alpha_e + sum_2[j][dp.pdf[i].topic],
                                              beta_e + sum_1[j][dp.pdf[i].topic] - sum_2[j][dp.pdf[i].topic]);
                }
                z_probabilities[total_cnt - 1] +=
                        logBeta(alpha_e + disagreements[j][p],
                                beta_e + numberOfDataSamples[p] - disagreements[j][p])
                                - logBeta(alpha_e, beta_e);
            }
            for (int i = 1; i < total_cnt; i++)
                z_probabilities[i] = MatrixUtilities.computeLogSumExp(z_probabilities[i - 1], z_probabilities[i]);
            double uniform = Math.log(random.nextDouble()) + z_probabilities[total_cnt - 1];
            zSamples[iterationNumber + 1][p] = dp.pdf[total_cnt - 1].topic;
            for(int i = 0; i < total_cnt - 1; i++)  {
                if (z_probabilities[i] > uniform) {
                    zSamples[iterationNumber + 1][p] = dp.pdf[i].topic;
                    for (int j = 0; j < numberOfFunctions; j++) {
                        sum_1[j][dp.pdf[i].topic] += numberOfDataSamples[p];
                        sum_2[j][dp.pdf[i].topic] += disagreements[j][p];
                    }
                    dp.add_topic_assingment(dp.pdf[i].topic);
                    break;
                }
            }
            if (zSamples[iterationNumber + 1][p] == dp.pdf[total_cnt - 1].topic) {
                for (int j = 0; j < numberOfFunctions; j++) {
                    sum_1[j][dp.pdf[total_cnt - 1].topic] += numberOfDataSamples[p];
                    sum_2[j][dp.pdf[total_cnt - 1].topic] += disagreements[j][p];
                }
                dp.add_topic_assingment(dp.pdf[total_cnt - 1].topic);
            }
        }
    }

    private void sampleZAndBurn(int iterationNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            dp.remove_topic_assignment(zSamples[iterationNumber][p]);
            int total_cnt = dp.prob_topics();
            double z_probabilities[] = new double[total_cnt];
            for(int i = 0; i < total_cnt; i++)
                z_probabilities[i] = Math.log(dp.pdf[i].prob);
            for (int j = 0; j < numberOfFunctions; j++) {
                disagreements[j][p] = 0;
                for (int i = 0; i < numberOfDataSamples[p]; i++)
                    if (functionOutputsArray[j][p][i] != labelsSamples[iterationNumber][p][i])
                        disagreements[j][p]++;
                for(int i = 0; i < total_cnt - 1; i++)  {
                    int k = dp.pdf[i].topic;
                    z_probabilities[i] += disagreements[j][p] * Math.log(errorRateSamples[iterationNumber][k][j]);
                    z_probabilities[i] += (numberOfDataSamples[p] - disagreements[j][p])
                            * Math.log(1 - errorRateSamples[iterationNumber][k][j]);
                }
                z_probabilities[total_cnt - 1] +=
                        logBeta(alpha_e + disagreements[j][p],
                                beta_e + numberOfDataSamples[p] - disagreements[j][p])
                                - logBeta(alpha_e, beta_e);
            }
            for (int i = 1; i < total_cnt; i++)
                z_probabilities[i] = MatrixUtilities.computeLogSumExp(z_probabilities[i - 1], z_probabilities[i]);
            double uniform = Math.log(random.nextDouble()) + z_probabilities[total_cnt - 1];
            zSamples[iterationNumber][p] = dp.pdf[total_cnt - 1].topic;
            for(int i = 0; i < total_cnt - 1; i++)  {
                if(z_probabilities[i] > uniform){
                    zSamples[iterationNumber][p] = dp.pdf[i].topic;
                    dp.add_topic_assingment(dp.pdf[i].topic);
                    break;
                }
            }
            if (zSamples[iterationNumber][p] == dp.pdf[total_cnt - 1].topic) {
                dp.add_topic_assingment(dp.pdf[total_cnt - 1].topic);
//                int numberOfErrorRatesBelowChance = 0;
//                for (int j = 0; j < numberOfFunctions; j++) {
//                    errorRateSamples[iterationNumber][dp.pdf[total_cnt - 1].topic][j] = randomDataGenerator.nextBeta(alpha_e + disagreements[j][p], beta_e + numberOfDataSamples[p] - disagreements[j][p]);
//                    if (errorRateSamples[iterationNumber][dp.pdf[total_cnt - 1].topic][j] < 0.5)
//                        numberOfErrorRatesBelowChance += 1;
//                }
//                if (numberOfErrorRatesBelowChance < numberOfFunctions / 2.0)
//                    for (int j = 0; j < numberOfFunctions; j++)
//                        errorRateSamples[iterationNumber][dp.pdf[total_cnt - 1].topic][j] = 1 - errorRateSamples[iterationNumber][dp.pdf[total_cnt - 1].topic][j];
            }
        }
    }

    private void sampleZ(int iterationNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            dp.remove_topic_assignment(zSamples[iterationNumber][p]);
            int total_cnt = dp.prob_topics();

            double z_probabilities[] = new double[total_cnt];
            for(int i=0;i<total_cnt;i++){
                z_probabilities[i] = Math.log(dp.pdf[i].prob);
            }

            for (int j = 0; j < numberOfFunctions; j++) {
                disagreements[j][p] = 0;
                for (int i = 0; i < numberOfDataSamples[p]; i++)
                    if (functionOutputsArray[j][p][i] != labelsSamples[iterationNumber][p][i])
                        disagreements[j][p]++;
                for(int i=0;i<total_cnt - 1;i++) {
                    int k = dp.pdf[i].topic;
                    z_probabilities[i] += disagreements[j][p] * Math.log(errorRateSamples[iterationNumber + 1][k][j]);
                    z_probabilities[i] += (numberOfDataSamples[p] - disagreements[j][p]) * Math.log(1 - errorRateSamples[iterationNumber + 1][k][j]);
                }
                z_probabilities[total_cnt - 1] += logBeta(alpha_e + disagreements[j][p], beta_e + numberOfDataSamples[p] - disagreements[j][p]) - logBeta(alpha_e, beta_e);
            }
            for(int i=1;i<total_cnt;i++){
                z_probabilities[i] = MatrixUtilities.computeLogSumExp(z_probabilities[i-1],z_probabilities[i]);
            }

            double uniform = Math.log(random.nextDouble()) + z_probabilities[total_cnt-1];
            zSamples[iterationNumber + 1][p] = dp.pdf[total_cnt-1].topic;
            for(int i = 0; i < total_cnt - 1; i++) {
                if(z_probabilities[i] > uniform){
                    zSamples[iterationNumber + 1][p] = dp.pdf[i].topic;
                    dp.add_topic_assingment(dp.pdf[i].topic);
                    break;
                }
            }
            if (zSamples[iterationNumber + 1][p] == dp.pdf[total_cnt-1].topic) {
                dp.add_topic_assingment(dp.pdf[total_cnt - 1].topic);
//                int numberOfErrorRatesBelowChance = 0;
//                for (int j = 0; j < numberOfFunctions; j++) {
//                    errorRateSamples[iterationNumber + 1][dp.pdf[total_cnt - 1].topic][j] = randomDataGenerator.nextBeta(alpha_e + disagreements[j][p], beta_e + numberOfDataSamples[p] - disagreements[j][p]);
//                    if (errorRateSamples[iterationNumber + 1][dp.pdf[total_cnt - 1].topic][j] < 0.5)
//                        numberOfErrorRatesBelowChance += 1;
//                }
//                if (numberOfErrorRatesBelowChance < numberOfFunctions / 2.0)
//                    for (int j = 0; j < numberOfFunctions; j++)
//                        errorRateSamples[iterationNumber + 1][dp.pdf[total_cnt - 1].topic][j] = 1 - errorRateSamples[iterationNumber + 1][dp.pdf[total_cnt - 1].topic][j];
            }
        }
    }

    private void sampleLabelsAndBurnWithCollapsedErrorRates(int iterationNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            for (int j = 0; j < numberOfFunctions; j++)
                sum_2[j][zSamples[iterationNumber][p]] -= disagreements[j][p];
            for (int i = 0; i < numberOfDataSamples[p]; i++) {
                double a1 = alpha_e;
                double a0 = beta_e;
                double sum1 = 0;
                for (int j = 0; j < numberOfFunctions; j++) {
                    if (functionOutputsArray[j][p][i] != labelsSamples[iterationNumber][p][i]) {
                        a1 += --disagreements[j][p] + sum_2[j][zSamples[iterationNumber][p]];
                        a0 += sum_1[j][zSamples[iterationNumber][p]] - sum_2[j][zSamples[iterationNumber][p]] - disagreements[j][p];
                    }
                    if (functionOutputsArray[j][p][i] != 1)
                        sum1++;
                }
                double p1 = priorSamples[iterationNumber][p];
                double p0 = 1 - priorSamples[iterationNumber][p];
                for (int m = 0; m < sum1; m++) {
                    p1 *= a1 + m;
                    p0 *= a0 + m;
                }
                for (int m = 0; m < numberOfFunctions - sum1; m++) {
                    p1 *= a0 + m;
                    p0 *= a1 + m;
                }
                labelsSamples[iterationNumber][p][i] = randomDataGenerator.nextBinomial(1, p1 / (p0 + p1));
                for (int j = 0; j < numberOfFunctions; j++)
                    if (functionOutputsArray[j][p][i] != labelsSamples[iterationNumber][p][i])
                        disagreements[j][p]++;
            }
            for (int j = 0; j < numberOfFunctions; j++)
                sum_2[j][zSamples[iterationNumber][p]] += disagreements[j][p];
        }
    }

    private void sampleLabelsWithCollapsedErrorRates(int iterationNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            labelsSamples[iterationNumber + 1][p] = new int[numberOfDataSamples[p]];
            for (int j = 0; j < numberOfFunctions; j++)
                sum_2[j][zSamples[iterationNumber + 1][p]] -= disagreements[j][p];
            for (int i = 0; i < numberOfDataSamples[p]; i++) {
                double a1 = alpha_e;
                double a0 = beta_e;
                double sum1 = 0;
                for (int j = 0; j < numberOfFunctions; j++) {
                    if (functionOutputsArray[j][p][i] != labelsSamples[iterationNumber][p][i]) {
                        a1 += --disagreements[j][p] + sum_2[j][zSamples[iterationNumber + 1][p]];
                        a0 += sum_1[j][zSamples[iterationNumber + 1][p]] - sum_2[j][zSamples[iterationNumber + 1][p]] - disagreements[j][p];
                    }
                    if (functionOutputsArray[j][p][i] != 1)
                        sum1++;
                }
                double p1 = priorSamples[iterationNumber + 1][p];
                double p0 = 1 - priorSamples[iterationNumber + 1][p];
                for (int m = 0; m < sum1; m++) {
                    p1 *= a1 + m;
                    p0 *= a0 + m;
                }
                for (int m = 0; m < numberOfFunctions - sum1; m++) {
                    p1 *= a0 + m;
                    p0 *= a1 + m;
                }
                labelsSamples[iterationNumber + 1][p][i] = randomDataGenerator.nextBinomial(1, p1 / (p0 + p1));
                for (int j = 0; j < numberOfFunctions; j++)
                    if (functionOutputsArray[j][p][i] != labelsSamples[iterationNumber + 1][p][i])
                        disagreements[j][p]++;
            }
            for (int j = 0; j < numberOfFunctions; j++)
                sum_2[j][zSamples[iterationNumber + 1][p]] += disagreements[j][p];
        }
    }

    private void sampleLabelsAndBurn(int iterationNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            labelsSamples[iterationNumber][p] = new int[numberOfDataSamples[p]];
            for (int i = 0; i < numberOfDataSamples[p]; i++) {
                double p0 = 1 - priorSamples[iterationNumber][p];
                double p1 = priorSamples[iterationNumber][p];
                for (int j = 0; j < numberOfFunctions; j++) {
                    if (functionOutputsArray[j][p][i] == 0) {
                        p0 *= (1 - errorRateSamples[iterationNumber][zSamples[iterationNumber][p]][j]);
                        p1 *= errorRateSamples[iterationNumber][zSamples[iterationNumber][p]][j];
                    } else {
                        p0 *= errorRateSamples[iterationNumber][zSamples[iterationNumber][p]][j];
                        p1 *= (1 - errorRateSamples[iterationNumber][zSamples[iterationNumber][p]][j]);
                    }
                }
                labelsSamples[iterationNumber][p][i] = randomDataGenerator.nextBinomial(1, p1 / (p0 + p1));
            }
        }
    }

    private void sampleLabels(int iterationNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            labelsSamples[iterationNumber + 1][p] = new int[numberOfDataSamples[p]];
            for (int i = 0; i < numberOfDataSamples[p]; i++) {
                double p0 = 1 - priorSamples[iterationNumber + 1][p];
                double p1 = priorSamples[iterationNumber + 1][p];
                for (int j = 0; j < numberOfFunctions; j++) {
                    if (functionOutputsArray[j][p][i] == 0) {
                        p0 *= (1 - errorRateSamples[iterationNumber + 1][zSamples[iterationNumber + 1][p]][j]);
                        p1 *= errorRateSamples[iterationNumber + 1][zSamples[iterationNumber + 1][p]][j];
                    } else {
                        p0 *= errorRateSamples[iterationNumber + 1][zSamples[iterationNumber + 1][p]][j];
                        p1 *= (1 - errorRateSamples[iterationNumber + 1][zSamples[iterationNumber + 1][p]][j]);
                    }
                }
                labelsSamples[iterationNumber + 1][p][i] = randomDataGenerator.nextBinomial(1, p1 / (p0 + p1));
            }
        }
    }

//    private void sampleLabelsAndBurn(int iterationNumber) {
//        for (int p = 0; p < numberOfDomains; p++) {
//            labelsSamples[iterationNumber][p] = new int[numberOfDataSamples[p]];
//            for (int i = 0; i < numberOfDataSamples[p]; i++) {
//                double p0 = Math.log(1 - priorSamples[iterationNumber][p]);
//                double p1 = Math.log(priorSamples[iterationNumber][p]);
//                for (int j = 0; j < numberOfFunctions; j++) {
//                    if (functionOutputsArray[j][p][i] == 0) {
//                        p0 += Math.log(1 - errorRateSamples[iterationNumber][zSamples[iterationNumber][p]][j]);
//                        p1 += Math.log(errorRateSamples[iterationNumber][zSamples[iterationNumber][p]][j]);
//                    } else {
//                        p0 += Math.log(errorRateSamples[iterationNumber][zSamples[iterationNumber][p]][j]);
//                        p1 += Math.log(1 - errorRateSamples[iterationNumber][zSamples[iterationNumber][p]][j]);
//                    }
//                }
//                labelsSamples[iterationNumber][p][i] = randomDataGenerator.nextBinomial(1, Math.exp(p1 - MatrixUtilities.computeLogSumExp(p0, p1)));
//            }
//        }
//    }
//
//    private void sampleLabels(int iterationNumber) {
//        for (int p = 0; p < numberOfDomains; p++) {
//            labelsSamples[iterationNumber + 1][p] = new int[numberOfDataSamples[p]];
//            for (int i = 0; i < numberOfDataSamples[p]; i++) {
//                double p0 = Math.log(1 - priorSamples[iterationNumber + 1][p]);
//                double p1 = Math.log(priorSamples[iterationNumber + 1][p]);
//                for (int j = 0; j < numberOfFunctions; j++) {
//                    if (functionOutputsArray[j][p][i] == 0) {
//                        p0 += Math.log(1 - errorRateSamples[iterationNumber][zSamples[iterationNumber + 1][p]][j]);
//                        p1 += Math.log(errorRateSamples[iterationNumber][zSamples[iterationNumber + 1][p]][j]);
//                    } else {
//                        p0 += Math.log(errorRateSamples[iterationNumber][zSamples[iterationNumber + 1][p]][j]);
//                        p1 += Math.log(1 - errorRateSamples[iterationNumber][zSamples[iterationNumber + 1][p]][j]);
//                    }
//                }
//                labelsSamples[iterationNumber + 1][p][i] = randomDataGenerator.nextBinomial(1, Math.exp(p1 - MatrixUtilities.computeLogSumExp(p0, p1)));
//            }
//        }
//    }

    public double[] getPriorMeans() {
        return priorMeans;
    }

    public double[] getPriorVariances() {
        return priorVariances;
    }

    public double[][] getLabelMeans() {
        return labelMeans;
    }

    public double[][] getLabelVariances() {
        return labelVariances;
    }

    public double[][] getErrorRatesMeans() {
        return errorRateMeans;
    }

    public double[][] getErrorRatesVariances() {
        return errorRateVariances;
    }
}
