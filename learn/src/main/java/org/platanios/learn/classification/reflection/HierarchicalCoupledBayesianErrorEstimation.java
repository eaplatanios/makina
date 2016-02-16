package org.platanios.learn.classification.reflection;

import org.apache.commons.math3.random.RandomDataGenerator;

import java.lang.reflect.Array;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

import static org.apache.commons.math3.special.Beta.logBeta;

/**
 * @author Emmanouil Antonios Platanios
 */
public class HierarchicalCoupledBayesianErrorEstimation {
    private final Random random = new Random();
    private final RandomDataGenerator randomDataGenerator = new RandomDataGenerator();
    private final double labelsPriorAlpha = 1;
    private final double labelsPriorBeta = 1;
    private final double errorRatesPriorAlpha = 1;
    private final double errorRatesPriorBeta = 10;

    private final int numberOfBurnInSamples;
    private final int numberOfThinningSamples;
    private final int numberOfSamples;
    private int numberOfFunctions;
    private int numberOfDomains;
    private int[] numberOfDataSamples;
    private final int maximumNumberOfClusters;
    private int[][][] functionOutputsArray; //indexed by domain, function id, item id
    private final double[][] labelPriorsSamples;      // indexed by sample, domain
    private final double[][] labelsPriorCounts;       // indexed by domain, 0/1
    private int[][][] labelsSamples;    //indexed by sample id, domain, example
    private final double[][] errorRatesSamples;  // indexed by sample, cluster id,
    private final double[][] errorRatesCounts;     // indexed by sample, cluster id, 00 + 11/01 + 10
    private final int[][][] clusterAssignmentSamples;     //Indexed by sample, domain, function id
    private final int disagreementCounts[][];   //indexed by domain_id, classifier_id

    private final double[] labelPriorMeans;
    private final double[] labelPriorVariances;
    private final double[][] labelMeans;
    private final double[][] labelVariances;
    private final double[][] errorRateMeans;
    private final double[][] errorRateVariances;

    private FastHDPPrior hierarchicalDirichletProcess;

    public HierarchicalCoupledBayesianErrorEstimation(List<boolean[][]> functionOutputs,
                                                      int numberOfBurnInSamples,
                                                      int numberOfThinningSamples,
                                                      int numberOfSamples,
                                                      double alpha,
                                                      double gamma) {
        this.numberOfBurnInSamples = numberOfBurnInSamples;
        this.numberOfThinningSamples = numberOfThinningSamples;
        this.numberOfSamples = numberOfSamples;
        numberOfFunctions = functionOutputs.get(0)[0].length;
        numberOfDomains = functionOutputs.size();
        numberOfDataSamples = new int[numberOfDomains];
        maximumNumberOfClusters = (numberOfDomains + 1) * numberOfFunctions;
        functionOutputsArray = new int[numberOfFunctions][numberOfDomains][];
        for (int p = 0; p < numberOfDomains; p++) {
            numberOfDataSamples[p] = functionOutputs.get(p).length;
            for (int j = 0; j < numberOfFunctions; j++) {
                functionOutputsArray[j][p] = new int[numberOfDataSamples[p]];
                for (int i = 0; i < numberOfDataSamples[p]; i++)
                    functionOutputsArray[j][p][i] = functionOutputs.get(p)[i][j] ? 1 : 0;
            }
        }
        hierarchicalDirichletProcess = new FastHDPPrior(numberOfDomains, numberOfFunctions, alpha, gamma);

        disagreementCounts = new int[numberOfDomains][numberOfFunctions];
        labelsSamples = new int[numberOfSamples][numberOfDomains][];
        labelPriorsSamples = new double[numberOfSamples][numberOfDomains];
        labelsPriorCounts = new double[numberOfDomains][2];
        errorRatesSamples = new double[numberOfSamples][maximumNumberOfClusters];
        errorRatesCounts = new double[maximumNumberOfClusters][2];
        clusterAssignmentSamples = new int[numberOfSamples][numberOfDomains][numberOfFunctions];
        for (int sampleIndex = 0; sampleIndex < numberOfSamples; sampleIndex++)
            for (int p = 0; p < numberOfDomains; p++)
                labelsSamples[sampleIndex][p] = new int[numberOfDataSamples[p]];
        for (int p = 0; p < numberOfDomains; p++) {
            labelPriorsSamples[0][p] = 0.5;
            int clusterIndex = 0;
            for (int j = 0; j < numberOfFunctions; j++) {
                clusterAssignmentSamples[0][p][j] = clusterIndex;
                hierarchicalDirichletProcess.add_items_table_assignment(p, j, clusterIndex, clusterIndex);
                clusterIndex++;
            }
            for (int i = 0; i < numberOfDataSamples[p]; i++) {
                int sum = 0;
                for (int j = 0; j < numberOfFunctions; j++)
                    sum += functionOutputsArray[j][p][i];
                labelsSamples[0][p][i] = sum >= (numberOfFunctions / 2) ? 1 : 0;
                updateCountsAfterSamplingLabel(0, p, i);
            }
        }
        sampleLabelsPriors(0);
        sampleErrorRates(0);
        labelPriorMeans = new double[numberOfDomains];
        labelPriorVariances = new double[numberOfDomains];
        labelMeans = new double[numberOfDomains][];
        labelVariances = new double[numberOfDomains][];
        errorRateMeans = new double[numberOfDomains][numberOfFunctions];
        errorRateVariances = new double[numberOfDomains][numberOfFunctions];
    }

    public void runGibbsSampler() {
//        for (int sampleIndex = 0; sampleIndex < numberOfBurnInSamples; sampleIndex++) {
//            sampleLabelsPriors(0);
//            sampleClusterAssignmentsWithCollapsedErrorRates(0);
//            sampleLabelsWithCollapsedErrorRates(0);
//        }
//        for (int sampleIndex = 0; sampleIndex < 100; sampleIndex++) {
//            sampleLabelsPriors(0);
//            sampleErrorRates(0);
//            sampleClusterAssignments(0);
//            sampleLabels(0);
//        }
        for (int sampleIndex = 0; sampleIndex < numberOfBurnInSamples; sampleIndex++) {
            sampleLabelsPriors(0);
            sampleErrorRates(0);
            sampleClusterAssignments(0);
            sampleLabels(0);
        }
        for (int sampleIndex = 1; sampleIndex < numberOfSamples; sampleIndex++) {
            for (int i = 0; i < numberOfThinningSamples + 1; i++) {
                sampleLabelsPriors(sampleIndex - 1);
                sampleErrorRates(sampleIndex - 1);
                sampleClusterAssignments(sampleIndex - 1);
                sampleLabels(sampleIndex - 1);
            }
            storeSample(sampleIndex);
        }
        // Aggregate values for means and variances computation
        for (int p = 0; p < numberOfDomains; p++) {
            labelMeans[p] = new double[numberOfDataSamples[p]];
            labelVariances[p] = new double[numberOfDataSamples[p]];
        }
        for (int sampleNumber = 0; sampleNumber < numberOfSamples; sampleNumber++) {
            for (int p = 0; p < numberOfDomains; p++) {
                labelPriorMeans[p] += labelPriorsSamples[sampleNumber][p];
                for (int j = 0; j < numberOfFunctions; j++)
                    errorRateMeans[p][j] += errorRatesSamples[sampleNumber][clusterAssignmentSamples[sampleNumber][p][j]];
                for (int i = 0; i < numberOfDataSamples[p]; i++)
                    labelMeans[p][i] += labelsSamples[sampleNumber][p][i];
            }
        }
        // Compute values for the means and the variances
        for (int p = 0; p < numberOfDomains; p++) {
            labelPriorMeans[p] /= numberOfSamples;
            for (int j = 0; j < numberOfFunctions; j++)
                errorRateMeans[p][j] /= numberOfSamples;
            for (int i = 0; i < numberOfDataSamples[p]; i++)
                labelMeans[p][i] /= numberOfSamples;
            for (int sampleNumber = 0; sampleNumber < numberOfSamples; sampleNumber++) {
                double temp = labelPriorsSamples[sampleNumber][p] - labelPriorMeans[p];
                labelPriorVariances[p] += temp * temp;
                for (int j = 0; j < numberOfFunctions; j++) {
                    temp = errorRatesSamples[sampleNumber][clusterAssignmentSamples[sampleNumber][p][j]] - errorRateMeans[p][j];
                    errorRateVariances[p][j] += temp * temp;
                }
                for (int i = 0; i < numberOfDataSamples[p]; i++) {
                    temp = labelsSamples[sampleNumber][p][i] - labelMeans[p][i];
                    labelVariances[p][i] += temp * temp;
                }
            }
            labelPriorVariances[p] /= numberOfSamples;
            for (int j = 0; j < numberOfFunctions; j++)
                errorRateVariances[p][j] /= numberOfSamples;
            for (int i = 0; i < numberOfDataSamples[p]; i++)
                labelVariances[p][i] /= numberOfSamples;
        }
    }

    private void sampleLabelsPriors(int sampleNumber) {
        for (int p = 0; p < numberOfDomains; p++)
            labelPriorsSamples[sampleNumber][p] = randomDataGenerator.nextBeta(labelsPriorAlpha + labelsPriorCounts[p][1], labelsPriorBeta + labelsPriorCounts[p][0]);
    }

    private void sampleLabels(int sampleNumber){
        for (int p = 0; p < numberOfDomains; p++){
            for (int i = 0; i < numberOfDataSamples[p]; i++) {
                double p0 = 1 - labelPriorsSamples[sampleNumber][p]; // TODO: Compute this in log-space
                double p1 = labelPriorsSamples[sampleNumber][p];
                for (int j = 0; j < numberOfFunctions; j++) {
                    int clusterId = clusterAssignmentSamples[sampleNumber][p][j];
                    if (functionOutputsArray[j][p][i] == 1) {
                        p0 *= errorRatesSamples[sampleNumber][clusterId];
                        p1 *= 1 - errorRatesSamples[sampleNumber][clusterId];
                    } else {
                        p0 *= 1 - errorRatesSamples[sampleNumber][clusterId];
                        p1 *= errorRatesSamples[sampleNumber][clusterId];
                    }
                }
                int newLabel = randomDataGenerator.nextBinomial(1, p1 / (p0 + p1));
                if (labelsSamples[sampleNumber][p][i] != newLabel) {
                    updateCountsBeforeSamplingLabel(sampleNumber, p, i);
                    labelsSamples[sampleNumber][p][i] = newLabel;
                    updateCountsAfterSamplingLabel(sampleNumber, p, i);
                }
            }
        }
    }

    private void sampleLabelsWithCollapsedErrorRates(int sampleNumber) {
        for (int p = 0; p < numberOfDomains; p++){
            for (int i = 0; i < numberOfDataSamples[p]; i++) {
                double p0 = 1 - labelPriorsSamples[sampleNumber][p]; // TODO: Compute this in log-space
                double p1 = labelPriorsSamples[sampleNumber][p];
                Map<Integer, AtomicInteger> clusterCounts = new HashMap<>();
                for (int j = 0; j < numberOfFunctions; j++) {
                    if (!clusterCounts.containsKey(clusterAssignmentSamples[sampleNumber][p][j]))
                        clusterCounts.put(clusterAssignmentSamples[sampleNumber][p][j], new AtomicInteger(1));
                    clusterCounts.get(clusterAssignmentSamples[sampleNumber][p][j]).incrementAndGet();
                }
                for (int clusterID : clusterCounts.keySet()) {
                    int total = clusterCounts.get(clusterID).intValue();
                    double a1 = errorRatesPriorAlpha + errorRatesCounts[clusterID][1];
                    double a0 = errorRatesPriorBeta + errorRatesCounts[clusterID][0];
                    double sum = 0;
                    for (int j = 0; j < numberOfFunctions; j++)
                        if (functionOutputsArray[j][p][i] != 1 && clusterAssignmentSamples[sampleNumber][p][j] == clusterID)
                            sum++;
                    for (int m = 0; m < sum; m++) {
                        p0 *= a0 + m;
                        p1 *= a1 + m;
                    }
                    for (int m = 0; m < total - sum; m++) {
                        p0 *= a1 + m;
                        p1 *= a0 + m;
                    }
                }
                int newLabel = randomDataGenerator.nextBinomial(1, p1 / (p0 + p1));
                if (labelsSamples[sampleNumber][p][i] != newLabel) {
                    updateCountsBeforeSamplingLabel(sampleNumber, p, i);
                    labelsSamples[sampleNumber][p][i] = newLabel;
                    updateCountsAfterSamplingLabel(sampleNumber, p, i);
                }
            }
        }
    }

    private void updateCountsAfterSamplingLabel(int sampleNumber, int p, int i) {
        labelsPriorCounts[p][labelsSamples[sampleNumber][p][i]]++;
        int err;
        for (int j = 0; j < numberOfFunctions; j++) {
            err = (functionOutputsArray[j][p][i] != labelsSamples[sampleNumber][p][i])? 1:0;
            errorRatesCounts[clusterAssignmentSamples[sampleNumber][p][j]][err]++;
            disagreementCounts[p][j] += err;
        }
    }

    private void updateCountsBeforeSamplingLabel(int sampleNumber, int p, int i) {
        labelsPriorCounts[p][labelsSamples[sampleNumber][p][i]] --;
        int err;
        for (int j = 0; j < numberOfFunctions; j++) {
            err = (functionOutputsArray[j][p][i] != labelsSamples[sampleNumber][p][i])? 1:0;
            errorRatesCounts[clusterAssignmentSamples[sampleNumber][p][j]][err] --;
            disagreementCounts[p][j] -= err;
        }
    }

    private void sampleErrorRates(int sampleNumber) {
        for (int k = 0; k < maximumNumberOfClusters; k++)
            errorRatesSamples[sampleNumber][k] = randomDataGenerator.nextBeta(errorRatesPriorAlpha + errorRatesCounts[k][1], errorRatesPriorBeta + errorRatesCounts[k][0]);
    }

    private void sampleClusterAssignments(int sampleNumber){
        for (int p = 0; p < numberOfDomains; p++) {
            for (int j = 0; j < numberOfFunctions; j++) {
                updateCountsBeforeSamplingClusterAssignment(sampleNumber,p,j);
                int total_table_dish = hierarchicalDirichletProcess.prob_table_assignment_for_item(p, j);
                double cdf[] = new double[total_table_dish];
                for (int k = 0; k < total_table_dish; k++) {
                    cdf[k] = Math.log(hierarchicalDirichletProcess.pdf[k].prob);
                }
                int topic_id = 0;
                double max = Double.NEGATIVE_INFINITY;
                for (int k = 0; k < total_table_dish; k++) {
                    topic_id = hierarchicalDirichletProcess.pdf[k].topic;
                    cdf[k] += disagreementCounts[p][j] * Math.log(errorRatesSamples[sampleNumber][topic_id]) + (numberOfDataSamples[p] - disagreementCounts[p][j]) * Math.log(1 - errorRatesSamples[sampleNumber][topic_id]);
                    if(max < cdf[k]) max = cdf[k];
                }
                cdf[0] -= max;
                for (int k = 1; k < total_table_dish; k++) {
                    cdf[k] -= max;
                    cdf[k] = Math.log(Math.exp(cdf[k - 1]) + Math.exp(cdf[k]));
                }
                double uniform = Math.log(random.nextDouble()) + cdf[total_table_dish - 1];
                int newClusterID = total_table_dish - 1;
                clusterAssignmentSamples[sampleNumber][p][j] = hierarchicalDirichletProcess.pdf[newClusterID].topic;
                for (int k = 0; k < total_table_dish - 1; k++) {
                    if (cdf[k] > uniform) {
                        newClusterID = k;
                        clusterAssignmentSamples[sampleNumber][p][j] = hierarchicalDirichletProcess.pdf[newClusterID].topic;
                        break;
                    }
                }
                updateCountsAfterSamplingClusterAssignment(sampleNumber, p, j, hierarchicalDirichletProcess.pdf[newClusterID].topic, hierarchicalDirichletProcess.pdf[newClusterID].table);
            }
        }
    }

    private void sampleClusterAssignmentsWithCollapsedErrorRates(int sampleNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            for (int j = 0; j < numberOfFunctions; j++) {
                updateCountsBeforeSamplingClusterAssignment(sampleNumber,p,j);
                int total_table_dish = hierarchicalDirichletProcess.prob_table_assignment_for_item(p, j);
                double cdf[] = new double[total_table_dish];
                for (int k = 0; k < total_table_dish; k++) {
                    cdf[k] = Math.log(hierarchicalDirichletProcess.pdf[k].prob);
                }
                int topic_id = 0;
                double max = Double.NEGATIVE_INFINITY;
                for (int k = 0; k < total_table_dish; k++) {
                    topic_id = hierarchicalDirichletProcess.pdf[k].topic;
                    cdf[k] += logBeta(errorRatesPriorAlpha + errorRatesCounts[k][1] + disagreementCounts[p][j],
                                      errorRatesPriorBeta + errorRatesCounts[k][0] + numberOfDataSamples[p] - disagreementCounts[p][j]);
                    cdf[k] -= logBeta(errorRatesPriorAlpha + errorRatesCounts[k][1],
                                      errorRatesPriorBeta + errorRatesCounts[k][0] );
//                    cdf[k] += disagreementCounts[p][j] *
//                    		Math.log(errorRatesSamples[sampleNumber][topic_id])
//                    		+ (numberOfDataSamples[p] - disagreementCounts[p][j])
//                    		* Math.log(1 - errorRatesSamples[sampleNumber][topic_id]);
                    if(max < cdf[k]) max = cdf[k];
                }
                cdf[0] -= max;
                for (int k = 1; k < total_table_dish; k++) {
                    cdf[k] -= max;
                    cdf[k] = Math.log(Math.exp(cdf[k - 1]) + Math.exp(cdf[k]));
                }
                double uniform = Math.log(random.nextDouble()) + cdf[total_table_dish - 1];
                int newClusterID = total_table_dish - 1;
                clusterAssignmentSamples[sampleNumber][p][j] = hierarchicalDirichletProcess.pdf[newClusterID].topic;
                for (int k = 0; k < total_table_dish - 1; k++) {
                    if (cdf[k] > uniform) {
                        newClusterID = k;
                        clusterAssignmentSamples[sampleNumber][p][j] = hierarchicalDirichletProcess.pdf[newClusterID].topic;
                        break;
                    }
                }
                updateCountsAfterSamplingClusterAssignment(sampleNumber, p, j, hierarchicalDirichletProcess.pdf[newClusterID].topic, hierarchicalDirichletProcess.pdf[newClusterID].table);
            }
        }
    }

    private void updateCountsBeforeSamplingClusterAssignment(int sampleNumber, int p, int j) {
        for (int i = 0; i < numberOfDataSamples[p]; i++){
            int err = (functionOutputsArray[j][p][i] != labelsSamples[sampleNumber][p][i])? 1:0;
            errorRatesCounts[clusterAssignmentSamples[sampleNumber][p][j]][err] --;
        }
        hierarchicalDirichletProcess.remove_items_table_assignment(p, j);
    }

    private void updateCountsAfterSamplingClusterAssignment(int sampleNumber, int p, int j, int clusterID, int tableID) {
        for (int i = 0; i < numberOfDataSamples[p]; i++){
            int err = (functionOutputsArray[j][p][i] != labelsSamples[sampleNumber][p][i])? 1:0;
            errorRatesCounts[clusterAssignmentSamples[sampleNumber][p][j]][err] ++;
        }
        hierarchicalDirichletProcess.add_items_table_assignment(p, j, tableID, clusterID);
    }

    public double logLikelihood(List<boolean[][]> functionOutputs) {
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
        for (int p = 0; p < numberOfDomains; p++) {
            labelsSamples[0][p] = new int[numberOfDataSamples[p]];
            for (int i = 0; i < numberOfDataSamples[p]; i++) {
                int sum = 0;
                for (int j = 0; j < numberOfFunctions; j++)
                    sum += functionOutputsArray[j][p][i];
                labelsSamples[0][p][i] = sum >= (numberOfFunctions / 2) ? 1 : 0;
            }
        }
        double logLikelihood = 0;
        for (int sampleNumber = 0; sampleNumber < numberOfSamples; sampleNumber++) {
            sampleLabels(sampleNumber);
            // Top level HDP term
            for (int topicID : hierarchicalDirichletProcess.getTakenTopics().toArray())
                logLikelihood += Math.log(hierarchicalDirichletProcess.getNumberOfTablesForTopic(topicID)) - Math.log(hierarchicalDirichletProcess.getNumberOfTables());
            for (int p = 0; p < numberOfDomains; p++) {
                // Label prior term
                logLikelihood += (labelsPriorAlpha - 1) * Math.log(labelPriorsSamples[sampleNumber][p])
                        + (labelsPriorBeta - 1) * Math.log(1 - labelPriorsSamples[sampleNumber][p]);
                // Bottom level HDP term
                Map<Integer, AtomicInteger> clusterCounts = new HashMap<>();
                for (int j = 0; j < numberOfFunctions; j++) {
                    if (!clusterCounts.containsKey(clusterAssignmentSamples[sampleNumber][p][j]))
                        clusterCounts.put(clusterAssignmentSamples[sampleNumber][p][j], new AtomicInteger(1));
                    else
                        clusterCounts.get(clusterAssignmentSamples[sampleNumber][p][j]).incrementAndGet();
                }
                for (int j = 0; j < numberOfFunctions; j++)
                    logLikelihood += Math.log(clusterCounts.get(clusterAssignmentSamples[sampleNumber][p][j]).intValue()) - Math.log(numberOfFunctions);
                // Labels term
                for (int i = 0; i < numberOfDataSamples[p]; i++)
                    if (labelsSamples[sampleNumber][p][i] == 1)
                        logLikelihood += Math.log(labelPriorsSamples[sampleNumber][p]);
                    else
                        logLikelihood += Math.log(1 - labelPriorsSamples[sampleNumber][p]);
                // Error rates term
                for (int j = 0; j < numberOfFunctions; j++) {
                    for (int i = 0; i < numberOfDataSamples[p]; i++) {
                        int err = (functionOutputsArray[j][p][i] != labelsSamples[sampleNumber][p][i]) ? 1 : 0;
                        logLikelihood += (1 - err) * Math.log(1 - errorRatesSamples[sampleNumber][clusterAssignmentSamples[sampleNumber][p][j]]);
                        logLikelihood += err * Math.log(errorRatesSamples[sampleNumber][clusterAssignmentSamples[sampleNumber][p][j]]);
                    }
                }
                // Function outputs term
                for (int j = 0; j < numberOfFunctions; j++)
                    for (int i = 0; i < numberOfDataSamples[p]; i++) {
                        if (functionOutputsArray[j][p][i] != labelsSamples[sampleNumber][p][i])
                            logLikelihood += Math.log(errorRatesSamples[sampleNumber][clusterAssignmentSamples[sampleNumber][p][j]]);
                        else
                            logLikelihood += Math.log(1 - errorRatesSamples[sampleNumber][clusterAssignmentSamples[sampleNumber][p][j]]);
                    }
            }
        }
        logLikelihood /= numberOfSamples;
        return logLikelihood;
    }

    private void storeSample(int sampleIndex) {
        copyArray(labelPriorsSamples[sampleIndex - 1], labelPriorsSamples[sampleIndex]);
        copyArray(errorRatesSamples[sampleIndex - 1], errorRatesSamples[sampleIndex]);
        copyArray(clusterAssignmentSamples[sampleIndex - 1], clusterAssignmentSamples[sampleIndex]);
        copyArray(labelsSamples[sampleIndex - 1], labelsSamples[sampleIndex]);
    }

    private void copyArray(Object sourceArray, Object destinationArray) {
        if(sourceArray.getClass().isArray() && destinationArray.getClass().isArray()) {
            for(int i = 0; i < Array.getLength(sourceArray); i++) {
                if(Array.get(sourceArray, i) != null && Array.get(sourceArray, i).getClass().isArray())
                    copyArray(Array.get(sourceArray, i), Array.get(destinationArray, i));
                else
                    Array.set(destinationArray, i, Array.get(sourceArray, i));
            }
        }
    }

    public double[] getLabelPriorMeans() {
        return labelPriorMeans;
    }

    public double[] getLabelPriorVariances() {
        return labelPriorVariances;
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
