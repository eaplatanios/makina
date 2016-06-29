package makina.learn.classification.reflection;

import org.apache.commons.math3.random.RandomDataGenerator;

import java.lang.reflect.Array;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author Emmanouil Antonios Platanios
 */
public class HierarchicalCoupledErrorEstimationConfusion {
    private final Random random = new Random();
    private final RandomDataGenerator randomDataGenerator = new RandomDataGenerator();
    private final double labelsPriorAlpha = 1;
    private final double labelsPriorBeta = 1;
    private final double[][] confusionMatrixPrior = new double[][] { new double[] { 10, 1 }, new double[] { 1, 10 } };

    private final int numberOfBurnInSamples;
    private final int numberOfThinningSamples;
    private final int numberOfSamples;
    private final int numberOfFunctions;
    private final int numberOfDomains;
    private int[] numberOfDataSamples;
    private int[][][] labelsSamples;
    private int[][][] functionOutputsArray;
    private final double[][] labelPriorsSamples;      // indexed by sample, domain
    private final double[][] labelPriorsCounts;       // indexed by domain, 0/1
    private final double[][][][] confusionMatrixSamples;  // indexed by sample, cluster id, 0/1, 0/1
    private final double[][][] confusionMatrixCounts;     // indexed by cluster id, 0/1, 0/1
    private final int[][][] clusterAssignmentSamples;

    private double[] priorMeans;
    private double[] priorVariances;
    private double[][] labelMeans;
    private double[][] labelVariances;
    private double[][] errorRateMeans;
    private double[][] errorRateVariances;

    private FastHDPPrior hierarchicalDirichletProcess;
    private int maximumNumberOfCluster;
//    private double clusterErrorSamples[][][][];

    public HierarchicalCoupledErrorEstimationConfusion(List<boolean[][]> functionOutputs,
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
        functionOutputsArray = new int[numberOfFunctions][numberOfDomains][];
        for (int p = 0; p < numberOfDomains; p++) {
            numberOfDataSamples[p] = functionOutputs.get(p).length;
            for (int j = 0; j < numberOfFunctions; j++) {
                functionOutputsArray[j][p] = new int[numberOfDataSamples[p]];
                for (int i = 0; i < numberOfDataSamples[p]; i++)
                    functionOutputsArray[j][p][i] = functionOutputs.get(p)[i][j] ? 1 : 0;
            }
        }
        maximumNumberOfCluster = numberOfDomains*numberOfFunctions + numberOfFunctions;
        hierarchicalDirichletProcess = new FastHDPPrior(numberOfDomains, numberOfFunctions, alpha, gamma);
//        clusterErrorSamples = new double[numberOfSamples][maximumNumberOfCluster][2];


        labelsSamples = new int[numberOfSamples][numberOfDomains][];
        labelPriorsSamples = new double[numberOfSamples][numberOfDomains];
        labelPriorsCounts = new double[numberOfDomains][2];
        confusionMatrixSamples = new double[numberOfSamples][maximumNumberOfCluster][2][2];
        confusionMatrixCounts = new double[maximumNumberOfCluster][2][2];
        clusterAssignmentSamples = new int[numberOfSamples][numberOfDomains][numberOfFunctions];

        priorMeans = new double[numberOfDomains];
        priorVariances = new double[numberOfDomains];
        labelMeans = new double[numberOfDomains][];
        labelVariances = new double[numberOfDomains][];
        errorRateMeans = new double[numberOfDomains][numberOfFunctions];
        errorRateVariances = new double[numberOfDomains][numberOfFunctions];
        for (int sampleIndex = 0; sampleIndex < numberOfSamples; sampleIndex++)
            for (int p = 0; p < numberOfDomains; p++)
                labelsSamples[sampleIndex][p] = new int[numberOfDataSamples[p]];
        for (int p = 0; p < numberOfDomains; p++) {
            labelMeans[p] = new double[numberOfDataSamples[p]];
            labelVariances[p] = new double[numberOfDataSamples[p]];
//            dpPriors[p] = new DirichletProcess(alpha, numberOfFunctions);
            labelPriorsSamples[0][p] = 0.5;
            for (int j = 0; j < numberOfFunctions; j++) {
                clusterAssignmentSamples[0][p][j] = 0;
//                clusterAssignmentSamples[0][p][j] = j;
                hierarchicalDirichletProcess.add_items_table_assignment(p, j, 0, 0);
            }
            for (int i = 0; i < numberOfDataSamples[p]; i++) {
                int sum = 0;
                for (int j = 0; j < numberOfFunctions; j++)
                    sum += functionOutputsArray[j][p][i];
                labelsSamples[0][p][i] = sum >= (numberOfFunctions / 2) ? 1 : 0;
                updateCountsAfterSamplingLabel(0, p, i);
            }
        }
        samplePriors(0);
        sampleConfusionMatrix(0);
    }

    public void runGibbsSampler() {
        for (int sampleIndex = 0; sampleIndex < numberOfBurnInSamples; sampleIndex++) {
            samplePriors(0);
            sampleConfusionMatrix(0);
            sampleClusterAssignments(0);
            sampleLabels(0);
        }
        for (int sampleIndex = 1; sampleIndex < numberOfSamples; sampleIndex++) {
            for (int i = 0; i < numberOfThinningSamples + 1; i++) {
                samplePriors(sampleIndex - 1);
                sampleConfusionMatrix(sampleIndex - 1);
                sampleClusterAssignments(sampleIndex - 1);
                sampleLabels(sampleIndex - 1);
            }
            storeSample(sampleIndex);
        }
        // Aggregate values for means and variances computation
        for (int sampleNumber = 0; sampleNumber < numberOfSamples; sampleNumber++) {
            for (int p = 0; p < numberOfDomains; p++) {
                priorMeans[p] += labelPriorsSamples[sampleNumber][p];
                for (int j = 0; j < numberOfFunctions; j++) {
                    double errorRate = 0;
                    for (int i = 0; i < numberOfDataSamples[p]; i++)
                        errorRate += functionOutputsArray[j][p][i] != labelsSamples[sampleNumber][p][i] ? 1 : 0;
                    errorRateMeans[p][j] += errorRate / numberOfDataSamples[p];
                }
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
                double temp = labelPriorsSamples[sampleNumber][p] - priorMeans[p];
                priorVariances[p] += temp * temp;
                for (int j = 0; j < numberOfFunctions; j++) {
                    temp = (confusionMatrixSamples[sampleNumber][clusterAssignmentSamples[sampleNumber][p][j]][0][1] * labelPriorsSamples[sampleNumber][p]
                            + confusionMatrixSamples[sampleNumber][clusterAssignmentSamples[sampleNumber][p][j]][1][0] * (1 - labelPriorsSamples[sampleNumber][p]))
                            - errorRateMeans[p][j];
                    errorRateVariances[p][j] += temp * temp;
                }
                for (int i = 0; i < numberOfDataSamples[p]; i++) {
                    temp = labelsSamples[sampleNumber][p][i] - labelMeans[p][i];
                    labelVariances[p][i] += temp * temp;
                }
            }
            priorVariances[p] /= numberOfSamples;
            for (int j = 0; j < numberOfFunctions; j++)
                errorRateVariances[p][j] /= numberOfSamples;
            for (int i = 0; i < numberOfDataSamples[p]; i++)
                labelVariances[p][i] /= numberOfSamples;
        }
    }

    private void samplePriors(int sampleNumber) {
        for (int p = 0; p < numberOfDomains; p++)
            labelPriorsSamples[sampleNumber][p] = randomDataGenerator.nextBeta(labelsPriorAlpha + labelPriorsCounts[p][1], labelsPriorBeta + labelPriorsCounts[p][0]);
    }

    private void sampleConfusionMatrix(int sampleNumber) { // TODO:Add checking for number of error rates below chance.
        for (int k = 0; k< maximumNumberOfCluster; k++){
            confusionMatrixSamples[sampleNumber][k][0][0] = randomDataGenerator.nextBeta(confusionMatrixPrior[0][0] + confusionMatrixCounts[k][0][0],
                                                                                         confusionMatrixPrior[0][1] + confusionMatrixCounts[k][0][1]);
            confusionMatrixSamples[sampleNumber][k][1][0] = randomDataGenerator.nextBeta(confusionMatrixPrior[1][0] + confusionMatrixCounts[k][1][0],
                                                                                         confusionMatrixPrior[1][1] + confusionMatrixCounts[k][1][1]);
            confusionMatrixSamples[sampleNumber][k][0][1] = 1 - confusionMatrixSamples[sampleNumber][k][0][0];
            confusionMatrixSamples[sampleNumber][k][1][1] = 1 - confusionMatrixSamples[sampleNumber][k][1][0];
        }

    }

    private void updateCountsBeforeSamplingClusterAssignment(int sampleNumber, int p, int j) {
        for (int i = 0; i < numberOfDataSamples[p]; i++)
            confusionMatrixCounts[clusterAssignmentSamples[sampleNumber][p][j]][labelsSamples[sampleNumber][p][i]][functionOutputsArray[j][p][i]]--;
        hierarchicalDirichletProcess.remove_items_table_assignment(p, j);
    }

    private void updateCountsAfterSamplingClusterAssignment(int sampleNumber, int p, int j, int tableId) {
        for (int i = 0; i < numberOfDataSamples[p]; i++)
            confusionMatrixCounts[clusterAssignmentSamples[sampleNumber][p][j]][labelsSamples[sampleNumber][p][i]][functionOutputsArray[j][p][i]]++;
        hierarchicalDirichletProcess.add_items_table_assignment(p, j, tableId, clusterAssignmentSamples[sampleNumber][p][j]);
    }

    private void sampleClusterAssignments(int sampleNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            for (int j = 0; j < numberOfFunctions; j++) {
                int cnt_errs[][] = new int[2][2];
                for (int i = 0; i < numberOfDataSamples[p]; i++)
                    cnt_errs[labelsSamples[sampleNumber][p][i]][functionOutputsArray[j][p][i]]++;
                updateCountsBeforeSamplingClusterAssignment(sampleNumber, p, j);
                int currentNumberOfClusters = hierarchicalDirichletProcess.prob_table_assignment_for_item(p, j);
                double cdf[] = new double[currentNumberOfClusters];
                double max = Double.NEGATIVE_INFINITY;
                for (int k = 0; k < currentNumberOfClusters; k++) {
                    int clusterID = hierarchicalDirichletProcess.pdf[k].topic;
                    cdf[k] = Math.log(hierarchicalDirichletProcess.pdf[k].prob);
                    cdf[k] += cnt_errs[0][0] * Math.log(confusionMatrixSamples[sampleNumber][clusterID][0][0]);
                    cdf[k] += cnt_errs[0][1] * Math.log(confusionMatrixSamples[sampleNumber][clusterID][0][1]);
                    cdf[k] += cnt_errs[1][0] * Math.log(confusionMatrixSamples[sampleNumber][clusterID][1][0]);
                    cdf[k] += cnt_errs[1][1] * Math.log(confusionMatrixSamples[sampleNumber][clusterID][1][1]);
                    if (max < cdf[k])
                        max = cdf[k];
                }
                cdf[0] -= max;
                for (int k = 1; k < currentNumberOfClusters; k++) {
                    cdf[k] -= max;
                    cdf[k] = Math.log(Math.exp(cdf[k - 1]) + Math.exp(cdf[k]));
                }
                double uniform = Math.log(random.nextDouble()) + cdf[currentNumberOfClusters - 1];
                int newClusterID = hierarchicalDirichletProcess.pdf[currentNumberOfClusters-1].topic;
                int newTableID = hierarchicalDirichletProcess.pdf[currentNumberOfClusters-1].table;
                clusterAssignmentSamples[sampleNumber][p][j] = newClusterID;
                for (int k = 0; k < currentNumberOfClusters - 1; k++) {
                    if (cdf[k] > uniform) {
                        newClusterID = hierarchicalDirichletProcess.pdf[k].topic;
                        newTableID = hierarchicalDirichletProcess.pdf[k].table;
                        clusterAssignmentSamples[sampleNumber][p][j] = newClusterID;
                        break;
                    }
                }
                updateCountsAfterSamplingClusterAssignment(sampleNumber, p, j,newTableID);
            }
        }
    }

    private void updateCountsBeforeSamplingLabel(int sampleNumber, int p, int i) {
        labelPriorsCounts[p][labelsSamples[sampleNumber][p][i]]--;
        for (int j = 0; j < numberOfFunctions; j++)
            confusionMatrixCounts[clusterAssignmentSamples[sampleNumber][p][j]][labelsSamples[sampleNumber][p][i]][functionOutputsArray[j][p][i]]--;
    }

    private void updateCountsAfterSamplingLabel(int sampleNumber, int p, int i) {
        labelPriorsCounts[p][labelsSamples[sampleNumber][p][i]]++;
        for (int j = 0; j < numberOfFunctions; j++)
            confusionMatrixCounts[clusterAssignmentSamples[sampleNumber][p][j]][labelsSamples[sampleNumber][p][i]][functionOutputsArray[j][p][i]]++;
    }

    private void sampleLabels(int sampleNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            for (int i = 0; i < numberOfDataSamples[p]; i++) {
                double p0 = 1 - labelPriorsSamples[sampleNumber][p]; // TODO: Compute this in log-space
                double p1 = labelPriorsSamples[sampleNumber][p];
                for (int j = 0; j < numberOfFunctions; j++) {
                    p0 *= confusionMatrixSamples[sampleNumber][clusterAssignmentSamples[sampleNumber][p][j]][0][functionOutputsArray[j][p][i]];
                    p1 *= confusionMatrixSamples[sampleNumber][clusterAssignmentSamples[sampleNumber][p][j]][1][functionOutputsArray[j][p][i]];
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

    public double logLikelihood(List<boolean[][]> functionOutputs) {
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
                    clusterCounts.get(clusterAssignmentSamples[sampleNumber][p][j]).incrementAndGet();
                }
                for (int j = 0; j < numberOfFunctions; j++)
                    logLikelihood += Math.log(clusterCounts.get(clusterAssignmentSamples[sampleNumber][p][j]).intValue()) - Math.log(numberOfFunctions);
                // Labels term
                for (int i = 0; i < numberOfDataSamples[p]; i++) {
                    if (labelsSamples[sampleNumber][p][i] == 1)
                        logLikelihood += Math.log(labelPriorsSamples[sampleNumber][p]);
                    else
                        logLikelihood += Math.log(1 - labelPriorsSamples[sampleNumber][p]);
                }
                // Confusion matrix term
                for (int clusterID : clusterCounts.keySet()) {
                    logLikelihood += confusionMatrixPrior[0][0] * Math.log(confusionMatrixSamples[sampleNumber][clusterID][0][0]);
                    logLikelihood += confusionMatrixPrior[0][1] * Math.log(confusionMatrixSamples[sampleNumber][clusterID][0][1]);
                    logLikelihood += confusionMatrixPrior[1][0] * Math.log(confusionMatrixSamples[sampleNumber][clusterID][1][0]);
                    logLikelihood += confusionMatrixPrior[1][1] * Math.log(confusionMatrixSamples[sampleNumber][clusterID][1][1]);
                }
                // Function outputs term
                for (int j = 0; j < numberOfFunctions; j++)
                    for (int i = 0; i < numberOfDataSamples[p]; i++)
                        logLikelihood += Math.log(confusionMatrixSamples[sampleNumber][clusterAssignmentSamples[sampleNumber][p][j]][labelsSamples[sampleNumber][p][i]][functionOutputsArray[j][p][i]]);
            }
        }
        return logLikelihood / numberOfSamples;
    }

    private void storeSample(int sampleIndex) {
        copyArray(labelPriorsSamples[sampleIndex - 1], labelPriorsSamples[sampleIndex]);
        copyArray(confusionMatrixSamples[sampleIndex - 1], confusionMatrixSamples[sampleIndex]);
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
