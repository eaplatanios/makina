package org.platanios.learn.classification.reflection;

import gnu.trove.map.hash.TIntIntHashMap;
import org.apache.commons.math3.random.RandomDataGenerator;
import org.platanios.learn.math.matrix.MatrixUtilities;

import java.util.*;

import static org.apache.commons.math3.special.Beta.logBeta;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ErrorEstimationDomainsFastHDPMixedGraphicalModel {
    private final Random random = new Random();
    private final RandomDataGenerator randomDataGenerator = new RandomDataGenerator();
    private final double alpha_p = 1;
    private final double beta_p = 1;
    private final double alpha_e = 1;
    private final double beta_e = 100;

    private final double alpha;
    private final double gamma;
    private final int numberOfIterations;
    private final int burnInIterations;
    private final int thinning;
    private final int numberOfSamples;
    private final int numberOfFunctions;
    private final int numberOfDomains;
    private final int[] numberOfDataSamples;
    private final int[][][] labelsSamples;
    private final int[][][] functionOutputsArray;
    private final int[][][] zSamples;
    private final double[][] priorSamples;
    private final double[][] errorRateSamples;

    private double[][] disagreements;
    private double[] sum_1;
    private double[] sum_2;

    private double[] priorMeans;
    private double[] priorVariances;
    private double[][] labelMeans;
    private double[][] labelVariances;
    private double[][] errorRateMeans;
    private double[][] errorRateVariances;

    public int numberOfClusters = 1;
    
    private FastHDPPrior hdp;

    public ErrorEstimationDomainsFastHDPMixedGraphicalModel(List<boolean[][]> functionOutputs, int numberOfIterations, int thinning, double alpha, double gamma) {
        this.alpha = alpha;
        this.gamma = gamma;
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
        
        hdp = new FastHDPPrior(numberOfDomains, numberOfFunctions, alpha , gamma);
        
        numberOfSamples = (numberOfIterations - burnInIterations) / thinning;
        priorSamples = new double[numberOfSamples][numberOfDomains];
        errorRateSamples = new double[numberOfSamples][numberOfDomains * numberOfFunctions];
        zSamples = new int[numberOfSamples][numberOfDomains][numberOfFunctions];
        labelsSamples = new int[numberOfSamples][numberOfDomains][];

        disagreements = new double[numberOfFunctions][numberOfDomains];
        sum_1 = new double[numberOfDomains * numberOfFunctions];
        sum_2 = new double[numberOfDomains * numberOfFunctions];

        priorMeans = new double[numberOfDomains];
        priorVariances = new double[numberOfDomains];
        labelMeans = new double[numberOfDomains][];
        labelVariances = new double[numberOfDomains][];
        errorRateMeans = new double[numberOfDomains][numberOfFunctions];
        errorRateVariances = new double[numberOfDomains][numberOfFunctions];
        for (int p = 0; p < numberOfDomains; p++) {
            labelMeans[p] = new double[numberOfDataSamples[p]];
            labelVariances[p] = new double[numberOfDataSamples[p]];
            priorSamples[0][p] = 0.5;
            labelsSamples[0][p] = new int[numberOfDataSamples[p]];
            for (int i = 0; i < numberOfDataSamples[p]; i++)
                labelsSamples[0][p][i] = randomDataGenerator.nextBinomial(1, 0.5);
            for (int j = 0; j < numberOfFunctions; j++) {
                zSamples[0][p][j] = 0;
                hdp.add_items_table_assignment(p, j, 0, 0);
                errorRateSamples[0][p * numberOfFunctions + j] = 0.25;
                disagreements[j][p] = 0;
                for (int i = 0; i < numberOfDataSamples[p]; i++)
                    if (functionOutputsArray[j][p][i] != labelsSamples[0][p][i])
                        disagreements[j][p]++;
            }
        }
        for (int k = 0; k < numberOfDomains * numberOfFunctions; k++) {
            sum_1[k] = 0;
            sum_2[k] = 0;
            for (int j = 0; j < numberOfFunctions; j++) {
                for (int p = 0; p < numberOfDomains; p++) {
                    if (zSamples[0][p][j] == k) {
                        sum_1[k] += numberOfDataSamples[p];
                        sum_2[k] += disagreements[j][p];
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
            sampleInternalTableTopicsAndBurnWithCollapsedErrorRates(0);
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
                sampleTableTopicAndBurn(iterationNumber);
                sampleLabelsAndBurn(iterationNumber);
            }
            samplePriors(iterationNumber);
            sampleErrorRates(iterationNumber);
            sampleZ(iterationNumber);
            sampleTableTopic(iterationNumber);
            sampleLabels(iterationNumber);
        }
        Set<Integer> uniqueClusters = new HashSet<>();
        for (int p = 0; p < numberOfDomains; p++)
            for (int j = 0; j < numberOfFunctions; j++)
                uniqueClusters.add(zSamples[zSamples.length - 1][p][j]);
        numberOfClusters = uniqueClusters.size();
        // Aggregate values for means and variances computation
        for (int sampleNumber = 0; sampleNumber < numberOfSamples; sampleNumber++) {
            for (int p = 0; p < numberOfDomains; p++) {
                int numberOfPhiBelowChance = 0;
                for (int j = 0; j < numberOfFunctions; j++)
                    if (errorRateSamples[sampleNumber][zSamples[sampleNumber][p][j]] < 0.5)
                        numberOfPhiBelowChance++;
                if (numberOfPhiBelowChance < numberOfFunctions / 2.0) {
                    priorSamples[sampleNumber][p] = 1 - priorSamples[sampleNumber][p];
                    for (int j = 0; j < numberOfFunctions; j++)
                        errorRateSamples[sampleNumber][zSamples[sampleNumber][p][j]] = 1 - errorRateSamples[sampleNumber][zSamples[sampleNumber][p][j]];
                }
                priorMeans[p] += priorSamples[sampleNumber][p];
                for (int j = 0; j < numberOfFunctions; j++)
                    errorRateMeans[p][j] += errorRateSamples[sampleNumber][zSamples[sampleNumber][p][j]];
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
                    temp = errorRateSamples[sampleNumber][zSamples[sampleNumber][p][j]] - errorRateMeans[p][j];
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

    private void sampleErrorRatesAndBurn(int iterationNumber) {
        for (int k = 0; k < numberOfDomains * numberOfFunctions; k++) {
            int disagreementCount = 0;
            int zCount = 0;
            for (int p = 0; p < numberOfDomains; p++) {
                for (int j = 0; j < numberOfFunctions; j++) {
                    if (zSamples[iterationNumber][p][j] == k) {
                        for (int i = 0; i < numberOfDataSamples[p]; i++)
                            if (functionOutputsArray[j][p][i] != labelsSamples[iterationNumber][p][i])
                                disagreementCount++;
                        zCount += numberOfDataSamples[p];
                    }
                }
            }
            errorRateSamples[iterationNumber][k] = randomDataGenerator.nextBeta(alpha_e + disagreementCount, beta_e + zCount - disagreementCount);
        }
        for (int p = 0; p < numberOfDomains; p++) {
            int numberOfErrorRatesBelowChance = 0;
            for (int j = 0; j < numberOfFunctions; j++)
                if (errorRateSamples[iterationNumber][zSamples[iterationNumber][p][j]] < 0.5)
                    numberOfErrorRatesBelowChance++;
            if (numberOfErrorRatesBelowChance < numberOfFunctions / 2.0)
                for (int j = 0; j < numberOfFunctions; j++)
                    errorRateSamples[iterationNumber][zSamples[iterationNumber][p][j]] = 1 - errorRateSamples[iterationNumber][zSamples[iterationNumber][p][j]];
        }
    }

    private void sampleErrorRates(int iterationNumber) {
        for (int k = 0; k < numberOfDomains * numberOfFunctions; k++) {
            int disagreementCount = 0;
            int zCount = 0;
            for (int p = 0; p < numberOfDomains; p++) {
                for (int j = 0; j < numberOfFunctions; j++) {
                    if (zSamples[iterationNumber][p][j] == k) {
                        for (int i = 0; i < numberOfDataSamples[p]; i++)
                            if (functionOutputsArray[j][p][i] != labelsSamples[iterationNumber][p][i])
                                disagreementCount++;
                        zCount += numberOfDataSamples[p];
                    }
                }
            }
            errorRateSamples[iterationNumber + 1][k] = randomDataGenerator.nextBeta(alpha_e + disagreementCount, beta_e + zCount - disagreementCount);
        }
        for (int p = 0; p < numberOfDomains; p++) {
            int numberOfErrorRatesBelowChance = 0;
            for (int j = 0; j < numberOfFunctions; j++)
                if (errorRateSamples[iterationNumber + 1][zSamples[iterationNumber][p][j]] < 0.5)
                    numberOfErrorRatesBelowChance++;
            if (numberOfErrorRatesBelowChance < numberOfFunctions / 2.0)
                for (int j = 0; j < numberOfFunctions; j++)
                    errorRateSamples[iterationNumber + 1][zSamples[iterationNumber][p][j]] = 1 - errorRateSamples[iterationNumber + 1][zSamples[iterationNumber][p][j]];
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
//        for (int k = 0; k < numberOfDomains * numberOfFunctions; k++) {
//            sum_2[k] = 0;
//            for (int j = 0; j < numberOfFunctions; j++) {
//                for (int p = 0; p < numberOfDomains; p++) {
//                    if (zSamples[iterationNumber][p][j] == k)
//                        sum_2[k] += disagreements[j][p];
//                }
//            }
//        }
        for (int p = 0; p < numberOfDomains; p++) {
            for (int j = 0; j < numberOfFunctions; j++) {
                
                hdp.remove_items_table_assignment(p, j);
                int total_cnt = hdp.prob_table_assignment_for_item(p, j);

                double z_probabilities[] = new double[total_cnt];
                for(int i=0;i<total_cnt;i++){
                    z_probabilities[i] = Math.log(hdp.pdf[i].prob);
                }
                int previous_topic= zSamples[iterationNumber][p][j];
                sum_1[previous_topic] -= numberOfDataSamples[p];
                sum_2[previous_topic] -= disagreements[j][p];
                for(int i=0;i<total_cnt - 1;i++) {
                    int k = hdp.pdf[i].topic;
                    double alpha = alpha_e + sum_2[k] + disagreements[j][p];
                    double beta = beta_e + sum_1[k] - sum_2[k] + numberOfDataSamples[p] - disagreements[j][p];
                    z_probabilities[i] += logBeta(alpha, beta) - logBeta(alpha_e + sum_2[k], beta_e + sum_1[k] - sum_2[k]);
                }
                z_probabilities[total_cnt - 1] += logBeta(alpha_e + disagreements[j][p], beta_e + numberOfDataSamples[p] - disagreements[j][p]) - logBeta(alpha_e, beta_e);
                
                for(int i=1;i<total_cnt;i++){
                    z_probabilities[i] = MatrixUtilities.computeLogSumExp(z_probabilities[i-1],z_probabilities[i]);
                }
                
                double uniform = Math.log(random.nextDouble()) + z_probabilities[total_cnt-1];
                zSamples[iterationNumber][p][j] = hdp.pdf[total_cnt-1].topic;
                for(int i=0;i<total_cnt-1;i++){
                    if(z_probabilities[i] > uniform){
                        zSamples[iterationNumber][p][j] = hdp.pdf[i].topic;
                        sum_1[hdp.pdf[i].topic] += numberOfDataSamples[p];
                        sum_2[hdp.pdf[i].topic] += disagreements[j][p];
                        hdp.add_items_table_assignment(p, j, hdp.pdf[i].table, hdp.pdf[i].topic);
                        break;
                    }
                }
                if (zSamples[iterationNumber][p][j] == hdp.pdf[total_cnt-1].topic) {
                    sum_1[hdp.pdf[total_cnt-1].topic] += numberOfDataSamples[p];
                    sum_2[hdp.pdf[total_cnt-1].topic] += disagreements[j][p];
                    hdp.add_items_table_assignment(p, j, hdp.pdf[total_cnt-1].table, hdp.pdf[total_cnt-1].topic);
                }
            }
        }
    }

    private void sampleInternalTableTopicsAndBurnWithCollapsedErrorRates(int iterationNumber) {
//        for (int p = 0; p < numberOfDomains; p++) {
//            for (int j = 0; j < numberOfFunctions; j++) {
//                disagreements[j][p] = 0;
//                for (int i = 0; i < numberOfDataSamples[p]; i++)
//                    if (functionOutputsArray[j][p][i] != labelsSamples[iterationNumber][p][i])
//                        disagreements[j][p]++;
//            }
//        }
//        for (int k = 0; k < numberOfDomains * numberOfFunctions; k++) {
//            sum_2[k] = 0;
//            for (int j = 0; j < numberOfFunctions; j++) {
//                for (int p = 0; p < numberOfDomains; p++) {
//                    if (zSamples[iterationNumber][p][j] == k)
//                        sum_2[k] += disagreements[j][p];
//                }
//            }
//        }
        for (int p = 0; p < numberOfDomains; p++) {
            int tables_ids[] = hdp.get_tables_taken(p);
            for (int table_id:tables_ids){
                int previous_topic = hdp.get_topic_table(p, table_id);
                int itm_lc[] = hdp.remove_tables_topic_assignment(p, table_id);
                double smc_1 =0;
                double smc_2 = 0;
                for (int j:itm_lc){
                    sum_1[previous_topic] -= numberOfDataSamples[p];
                    smc_1 += numberOfDataSamples[p];
                    sum_2[previous_topic] -= disagreements[j][p];
                    smc_2 += disagreements[j][p];
                }
                int total_cnt = hdp.prob_topic_assignment_for_table(p, table_id);
                double z_probabilities[] = new double[total_cnt];
                for(int i=0;i<total_cnt;i++){
                    z_probabilities[i] = Math.log(hdp.pdf[i].prob);
                }
                for(int i=0;i<total_cnt - 1;i++) {
                    int k = hdp.pdf[i].topic;
                    double alpha = alpha_e +  sum_2[k] + smc_2;
                    double beta = beta_e + sum_1[k] -sum_2[k] + smc_1 - smc_2;
                    z_probabilities[i] += logBeta(alpha, beta) - logBeta(alpha_e + sum_2[k], beta_e + sum_1[k] - sum_2[k]);
                }
                z_probabilities[total_cnt - 1] += logBeta(alpha_e + smc_2, beta_e + smc_1-smc_2) - logBeta(alpha_e, beta_e);
                for(int i=1;i<total_cnt;i++){
                    z_probabilities[i] = MatrixUtilities.computeLogSumExp(z_probabilities[i-1],z_probabilities[i]);
                }
                double uniform = Math.log(random.nextDouble()) + z_probabilities[total_cnt-1];
                int sample_topic = hdp.pdf[total_cnt-1].topic;
                for(int i=0;i<total_cnt-1;i++){
                    if(z_probabilities[i] > uniform){
                        sample_topic = hdp.pdf[i].topic;
                        break;
                    }
                }
                hdp.add_tobles_topic_assignment(p, table_id, sample_topic);
                sum_1[sample_topic] += smc_1;
                sum_2[sample_topic] += smc_2;
                for (int j:itm_lc){
                    zSamples[iterationNumber][p][j] = sample_topic;
                }
            }
            
        }
    }

    private void sampleZAndBurn(int iterationNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            for (int j = 0; j < numberOfFunctions; j++) {
                hdp.remove_items_table_assignment(p, j);
                int total_cnt = hdp.prob_table_assignment_for_item(p, j);

                double z_probabilities[] = new double[total_cnt];
                for(int i=0;i<total_cnt;i++){
                    z_probabilities[i] = Math.log(hdp.pdf[i].prob);
                }
                disagreements[j][p] = 0;
                for (int i = 0; i < numberOfDataSamples[p]; i++)
                    if (functionOutputsArray[j][p][i] != labelsSamples[iterationNumber][p][i])
                        disagreements[j][p]++;
                for(int i=0;i<total_cnt - 1;i++) {
                    int k = hdp.pdf[i].topic;
                    z_probabilities[i] += disagreements[j][p] * Math.log(errorRateSamples[iterationNumber][k]);
                    z_probabilities[i] += (numberOfDataSamples[p] - disagreements[j][p]) * Math.log(1 - errorRateSamples[iterationNumber][k]);
                }
                z_probabilities[total_cnt - 1] += logBeta(alpha_e + disagreements[j][p], beta_e + numberOfDataSamples[p] - disagreements[j][p]) - logBeta(alpha_e, beta_e);
                for(int i=1;i<total_cnt;i++){
                    z_probabilities[i] = MatrixUtilities.computeLogSumExp(z_probabilities[i-1],z_probabilities[i]);
                }
                
                double uniform = Math.log(random.nextDouble()) + z_probabilities[total_cnt-1];
                zSamples[iterationNumber][p][j] = hdp.pdf[total_cnt-1].topic;
                for(int i=0;i<total_cnt-1;i++){
                    if(z_probabilities[i] > uniform){
                        zSamples[iterationNumber][p][j] = hdp.pdf[i].topic;
                        hdp.add_items_table_assignment(p, j, hdp.pdf[i].table, hdp.pdf[i].topic);
                        break;
                    }
                }
                if (zSamples[iterationNumber][p][j] == hdp.pdf[total_cnt-1].topic) {
                    hdp.add_items_table_assignment(p, j, hdp.pdf[total_cnt-1].table, hdp.pdf[total_cnt-1].topic);
                }
            }
        }
    }
    
    
    private void sampleTableTopicAndBurn(int iterationNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            int tables_ids[] = hdp.get_tables_taken(p);
            for (int table_id:tables_ids){
                int previous_topic = hdp.get_topic_table(p, table_id);
                int itm_lc[] = hdp.remove_tables_topic_assignment(p, table_id);
                double smc_1 =0;
                double smc_2 = 0;
                for (int j:itm_lc){
                    for(int i=0;i<numberOfDataSamples[p];i++){
                        if (functionOutputsArray[j][p][i] != labelsSamples[iterationNumber][p][i])
                            smc_2++;
                    }
                    smc_1 += numberOfDataSamples[p];
                }
                int total_cnt = hdp.prob_topic_assignment_for_table(p, table_id);
                double z_probabilities[] = new double[total_cnt];
                for(int i=0;i<total_cnt;i++){
                    z_probabilities[i] = Math.log(hdp.pdf[i].prob);
                }
                for(int i=0;i<total_cnt - 1;i++) {
                    int k = hdp.pdf[i].topic;
                    z_probabilities[i] += smc_2 * Math.log(errorRateSamples[iterationNumber][k]);
                    z_probabilities[i] += (smc_1 - smc_2) * Math.log(1 - errorRateSamples[iterationNumber][k]);
                }
                z_probabilities[total_cnt - 1] += logBeta(alpha_e + smc_2, beta_e + smc_1-smc_2) - logBeta(alpha_e, beta_e);
                for(int i=1;i<total_cnt;i++){
                    z_probabilities[i] = MatrixUtilities.computeLogSumExp(z_probabilities[i-1],z_probabilities[i]);
                }
                double uniform = Math.log(random.nextDouble()) + z_probabilities[total_cnt-1];
                int sample_topic = hdp.pdf[total_cnt-1].topic;
                for(int i=0;i<total_cnt-1;i++){
                    if(z_probabilities[i] > uniform){
                        sample_topic = hdp.pdf[i].topic;
                        break;
                    }
                }
                hdp.add_tobles_topic_assignment(p, table_id, sample_topic);
                for (int j:itm_lc){
                    zSamples[iterationNumber][p][j] = sample_topic;
                }
                
            }
        }
    }
    
    

    private void sampleZ(int iterationNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            for (int j = 0; j < numberOfFunctions; j++) {
                hdp.remove_items_table_assignment(p, j);
                int total_cnt = hdp.prob_table_assignment_for_item(p, j);

                double z_probabilities[] = new double[total_cnt];
                for(int i=0;i<total_cnt;i++){
                    z_probabilities[i] = Math.log(hdp.pdf[i].prob);
                }
                disagreements[j][p] = 0;
                for (int i = 0; i < numberOfDataSamples[p]; i++)
                    if (functionOutputsArray[j][p][i] != labelsSamples[iterationNumber][p][i])
                        disagreements[j][p]++;
                for(int i=0;i<total_cnt - 1;i++) {
                    int k = hdp.pdf[i].topic;
                    z_probabilities[i] += disagreements[j][p] * Math.log(errorRateSamples[iterationNumber + 1][k]);
                    z_probabilities[i] += (numberOfDataSamples[p] - disagreements[j][p]) * Math.log(1 - errorRateSamples[iterationNumber + 1][k]);
                }
                z_probabilities[total_cnt - 1] += logBeta(alpha_e + disagreements[j][p], beta_e + numberOfDataSamples[p] - disagreements[j][p]) - logBeta(alpha_e, beta_e);
                for(int i=1;i<total_cnt;i++){
                    z_probabilities[i] = MatrixUtilities.computeLogSumExp(z_probabilities[i-1],z_probabilities[i]);
                }
                
                double uniform = Math.log(random.nextDouble()) + z_probabilities[total_cnt-1];
                zSamples[iterationNumber + 1][p][j] = hdp.pdf[total_cnt-1].topic;
                for(int i=0;i<total_cnt-1;i++){
                    if(z_probabilities[i] > uniform){
                        zSamples[iterationNumber + 1][p][j] = hdp.pdf[i].topic;
                        hdp.add_items_table_assignment(p, j, hdp.pdf[i].table, hdp.pdf[i].topic);
                        break;
                    }
                }
                if (zSamples[iterationNumber + 1][p][j] == hdp.pdf[total_cnt-1].topic)
                    hdp.add_items_table_assignment(p, j, hdp.pdf[total_cnt-1].table, hdp.pdf[total_cnt-1].topic);
            }
        }
    }
    
    private void sampleTableTopic(int iterationNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            int tables_ids[] = hdp.get_tables_taken(p);
            for (int table_id:tables_ids){
                int itm_lc[] = hdp.remove_tables_topic_assignment(p, table_id);
                double smc_1 =0;
                double smc_2 = 0;
                for (int j:itm_lc){
                    for(int i=0;i<numberOfDataSamples[p];i++){
                        if (functionOutputsArray[j][p][i] != labelsSamples[iterationNumber][p][i])
                            smc_2++;
                    }
                    smc_1 += numberOfDataSamples[p];
                }
                int total_cnt = hdp.prob_topic_assignment_for_table(p, table_id);
                double z_probabilities[] = new double[total_cnt];
                for(int i=0;i<total_cnt;i++){
                    z_probabilities[i] = Math.log(hdp.pdf[i].prob);
                }
                for(int i=0;i<total_cnt - 1;i++) {
                    int k = hdp.pdf[i].topic;
                    z_probabilities[i] += smc_2 * Math.log(errorRateSamples[iterationNumber + 1][k]);
                    z_probabilities[i] += (smc_1 - smc_2) * Math.log(1 - errorRateSamples[iterationNumber + 1][k]);
                }
                z_probabilities[total_cnt - 1] += logBeta(alpha_e + smc_2, beta_e + smc_1-smc_2) - logBeta(alpha_e, beta_e);
                for(int i=1;i<total_cnt;i++){
                    z_probabilities[i] = MatrixUtilities.computeLogSumExp(z_probabilities[i-1],z_probabilities[i]);
                }
                double uniform = Math.log(random.nextDouble()) + z_probabilities[total_cnt-1];
                int sample_topic = hdp.pdf[total_cnt-1].topic;
                for(int i=0;i<total_cnt-1;i++){
                    if(z_probabilities[i] > uniform){
                        sample_topic = hdp.pdf[i].topic;
                        break;
                    }
                }
                hdp.add_tobles_topic_assignment(p, table_id, sample_topic);
                for (int j:itm_lc){
                    zSamples[iterationNumber + 1][p][j] = sample_topic;
                }
                
            }
        }
    }
    
    private void sampleLabelsAndBurnWithCollapsedErrorRates(int iterationNumber) {
        int topics[] = hdp.get_topics();
        TIntIntHashMap hmp = new TIntIntHashMap();
        for(int i=0;i<topics.length;i++){
            hmp.put(topics[i], i);
        }
        for (int p = 0; p < numberOfDomains; p++) {
            for(int i=0; i < numberOfDataSamples[p]; i++){
                double mistake[][] = new double[topics.length][2];
                double matches[][] = new double[topics.length][2];
                
                for(int j=0; j<numberOfFunctions; j++){
                    if(functionOutputsArray[j][p][i] == 1){
                        mistake[hmp.get(zSamples[iterationNumber][p][j])][0] +=1;
                        matches[hmp.get(zSamples[iterationNumber][p][j])][1] +=1;
                    }else{
                        mistake[hmp.get(zSamples[iterationNumber][p][j])][1] +=1;
                        matches[hmp.get(zSamples[iterationNumber][p][j])][0] +=1;
                    }
                    if(functionOutputsArray[j][p][i] != labelsSamples[iterationNumber][p][i]){
                        disagreements[j][p] --;
                        sum_2[zSamples[iterationNumber][p][j]] --;
                    }
                    sum_1[zSamples[iterationNumber][p][j]]--;
                }
                double p1 = Math.log(priorSamples[iterationNumber][p]);
                double p0 = Math.log(1 - priorSamples[iterationNumber][p]);
                for (int tp:topics){
                    for(int l=0;l<mistake[hmp.get(tp)][0];l++){
                        p0 += Math.log(alpha_e + sum_2[tp] + l);
                    }
                    for(int l=0;l<mistake[hmp.get(tp)][1];l++){
                        p1 += Math.log(alpha_e + sum_2[tp] + l);
                    }
                    for(int l=0;l<matches[hmp.get(tp)][0];l++){
                        p0 += Math.log(beta_e + sum_1[tp] - sum_2[tp] + l);
                    }
                    for(int l=0;l<matches[hmp.get(tp)][1];l++){
                        p1 += Math.log(beta_e + sum_1[tp] - sum_2[tp] + l);
                    }
                }
                double logsum = MatrixUtilities.computeLogSumExp(p1,p0);
                labelsSamples[iterationNumber][p][i] = randomDataGenerator.nextBinomial(1, Math.exp(p1 - logsum));
                for(int j=0; j<numberOfFunctions; j++){
                    if(functionOutputsArray[j][p][i] != labelsSamples[iterationNumber][p][i]){
                        disagreements[j][p] ++;
                        sum_2[zSamples[iterationNumber][p][j]] ++;
                    }
                    sum_1[zSamples[iterationNumber][p][j]]++;
                }
            }
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
                        p0 *= (1 - errorRateSamples[iterationNumber][zSamples[iterationNumber][p][j]]);
                        p1 *= errorRateSamples[iterationNumber][zSamples[iterationNumber][p][j]];
                    } else {
                        p0 *= errorRateSamples[iterationNumber][zSamples[iterationNumber][p][j]];
                        p1 *= (1 - errorRateSamples[iterationNumber][zSamples[iterationNumber][p][j]]);
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
                        p0 *= (1 - errorRateSamples[iterationNumber + 1][zSamples[iterationNumber + 1][p][j]]);
                        p1 *= errorRateSamples[iterationNumber + 1][zSamples[iterationNumber + 1][p][j]];
                    } else {
                        p0 *= errorRateSamples[iterationNumber + 1][zSamples[iterationNumber + 1][p][j]];
                        p1 *= (1 - errorRateSamples[iterationNumber + 1][zSamples[iterationNumber + 1][p][j]]);
                    }
                }
                labelsSamples[iterationNumber + 1][p][i] = randomDataGenerator.nextBinomial(1, p1 / (p0 + p1));
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
