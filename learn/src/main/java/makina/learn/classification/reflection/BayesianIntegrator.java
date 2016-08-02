package makina.learn.classification.reflection;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import org.apache.commons.math3.random.RandomDataGenerator;
import makina.learn.classification.Label;
import makina.utilities.ArrayUtilities;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public final class BayesianIntegrator extends Integrator {
    private final RandomDataGenerator randomDataGenerator = new RandomDataGenerator();
    private final BiMap<Label, Integer> labelKeysMap = HashBiMap.create();
    private final BiMap<Integer, Integer> classifierKeysMap = HashBiMap.create();

    private final List<BiMap<Integer, Integer>> instanceKeysMap;
    private final int[][] domainInstances;
    private final int[][] domainFunctions;
    private final double[][] domainValues;
    private final double labelsPriorAlpha;
    private final double labelsPriorBeta;
    private final double errorRatesPriorAlpha;
    private final double errorRatesPriorBeta;
    private final int numberOfBurnInSamples;
    private final int numberOfThinningSamples;
    private final int numberOfSamples;
    private final int numberOfFunctions;
    private final int numberOfDomains;
    private final int[] numberOfInstances;
    private final double[][] labelPriorsSamples;
    private final double[][][] errorRatesSamples;
    private final int[][][] labelsSamples;
    private final int[][] disagreements;
    private final double[] labelPriorMeans;
    private final double[][] errorRateMeans;
    private final double[][] labelMeans;

    private boolean needsInference = true;

    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>>
            extends Integrator.AbstractBuilder<T> {
        private Long seed = null;
        private double labelsPriorAlpha = 1.0;
        private double labelsPriorBeta = 1.0;
        private double errorRatesPriorAlpha = 1.0;
        private double errorRatesPriorBeta = 2.0;
        private int numberOfBurnInSamples = 4000;
        private int numberOfThinningSamples = 10;
        private int numberOfSamples = 200;

        public AbstractBuilder(Data<Data.PredictedInstance> data) {
            super(data);
        }

        private AbstractBuilder(String predictedDataFilename) {
            super(predictedDataFilename);
        }

        public T seed(Long seed) {
            this.seed = seed;
            return self();
        }

        public T labelsPriorAlpha(double labelsPriorAlpha) {
            this.labelsPriorAlpha = labelsPriorAlpha;
            return self();
        }

        public T labelsPriorBeta(double labelsPriorBeta) {
            this.labelsPriorBeta = labelsPriorBeta;
            return self();
        }

        public T errorRatesPriorAlpha(double errorRatesPriorAlpha) {
            this.errorRatesPriorAlpha = errorRatesPriorAlpha;
            return self();
        }

        public T errorRatesPriorBeta(double errorRatesPriorBeta) {
            this.labelsPriorAlpha = labelsPriorAlpha;
            return self();
        }

        public T numberOfBurnInSamples(int numberOfBurnInSamples) {
            this.numberOfBurnInSamples = numberOfBurnInSamples;
            return self();
        }

        public T numberOfThinningSamples(int numberOfThinningSamples) {
            this.numberOfThinningSamples = numberOfThinningSamples;
            return self();
        }

        public T numberOfSamples(int numberOfSamples) {
            this.numberOfSamples = numberOfSamples;
            return self();
        }

        public T options(String options) {
            if (options == null || options.length() == 0)
                return self();
            if (options.length() < 7)
                throw new IllegalArgumentException("Invalid number of arguments provided.");
            String[] optionsStringParts = options.split(":");
            if (!optionsStringParts[0].equals("-"))
                this.numberOfBurnInSamples = Integer.parseInt(optionsStringParts[0]);
            if (!optionsStringParts[1].equals("-"))
                this.numberOfThinningSamples = Integer.parseInt(optionsStringParts[1]);
            if (!optionsStringParts[2].equals("-"))
                this.numberOfSamples = Integer.parseInt(optionsStringParts[2]);
            if (!optionsStringParts[3].equals("-"))
                this.labelsPriorAlpha = Double.parseDouble(optionsStringParts[3]);
            if (!optionsStringParts[4].equals("-"))
                this.labelsPriorBeta = Double.parseDouble(optionsStringParts[4]);
            if (!optionsStringParts[5].equals("-"))
                this.errorRatesPriorAlpha = Double.parseDouble(optionsStringParts[5]);
            if (!optionsStringParts[6].equals("-"))
                this.errorRatesPriorBeta = Double.parseDouble(optionsStringParts[6]);
            return self();
        }

        public BayesianIntegrator build() {
            return new BayesianIntegrator(this);
        }
    }

    public static class Builder extends AbstractBuilder<Builder> {
        public Builder(Data<Data.PredictedInstance> data) {
            super(data);
        }

        public Builder(String predictedDataFilename) {
            super(predictedDataFilename);
        }

        @Override
        protected Builder self() {
            return this;
        }
    }

    private BayesianIntegrator(AbstractBuilder<?> builder) {
        super(builder);
        if (builder.seed != null)
            randomDataGenerator.reSeed(builder.seed);
        data.stream()
                .map(Data.PredictedInstance::label)
                .distinct()
                .forEach(label -> labelKeysMap.computeIfAbsent(label, key -> labelKeysMap.size()));
        labelsPriorAlpha = builder.labelsPriorAlpha;
        labelsPriorBeta = builder.labelsPriorBeta;
        errorRatesPriorAlpha = builder.errorRatesPriorAlpha;
        errorRatesPriorBeta = builder.errorRatesPriorBeta;
        numberOfBurnInSamples = builder.numberOfBurnInSamples;
        numberOfThinningSamples = builder.numberOfThinningSamples;
        numberOfSamples = builder.numberOfSamples;
        numberOfFunctions = (int) data.stream().map(Data.PredictedInstance::functionId).distinct().count();
        numberOfDomains = (int) data.stream().map(Data.PredictedInstance::label).distinct().count();
        instanceKeysMap = new ArrayList<>();
        numberOfInstances = new int[numberOfDomains];
        domainInstances = new int[numberOfDomains][];
        domainFunctions = new int[numberOfDomains][];
        domainValues = new double[numberOfDomains][];
        for (int p = 0; p < numberOfDomains; p++) {
            final int domain = p;
            instanceKeysMap.add(HashBiMap.create());
            int numberOfSamples = (int) data.stream().filter(instance -> instance.label().equals(labelKeysMap.inverse().get(domain))).count();
            domainInstances[p] = new int[numberOfSamples];
            domainFunctions[p] = new int[numberOfSamples];
            domainValues[p] = new double[numberOfSamples];
            numberOfInstances[p] = (int) data.stream().filter(instance -> instance.label().equals(labelKeysMap.inverse().get(domain))).map(Data.PredictedInstance::id).distinct().count();
            int[] sampleIndex = {0};
            data.stream().filter(instance -> instance.label().equals(labelKeysMap.inverse().get(domain))).forEach(instance -> {
                int i = instanceKeysMap.get(domain).computeIfAbsent(instance.id(), key -> instanceKeysMap.get(domain).size());
                int j = classifierKeysMap.computeIfAbsent(instance.functionId(), key -> classifierKeysMap.size());
                double value = instance.value() >= 0.5 ? 1.0 : 0.0;
                domainInstances[domain][sampleIndex[0]] = i;
                domainFunctions[domain][sampleIndex[0]] = j;
                domainValues[domain][sampleIndex[0]] = value;
                sampleIndex[0]++;
            });
        }
        labelPriorsSamples = new double[numberOfSamples][numberOfDomains];
        errorRatesSamples = new double[numberOfSamples][numberOfDomains][numberOfFunctions];
        labelsSamples = new int[numberOfSamples][numberOfDomains][];
        disagreements = new int[numberOfDomains][numberOfFunctions];
        for (int sampleIndex = 0; sampleIndex < numberOfSamples; sampleIndex++)
            for (int p = 0; p < numberOfDomains; p++)
                labelsSamples[sampleIndex][p] = new int[numberOfInstances[p]];
        for (int p = 0; p < numberOfDomains; p++) {
            labelPriorsSamples[0][p] = 0.5;
            int[] sum = new int[numberOfInstances[p]];
            int[] numberOfFunctions = new int[numberOfInstances[p]];
            for (int sample = 0; sample < domainInstances[p].length; sample++) {
                sum[domainInstances[p][sample]] += domainValues[p][sample];
                numberOfFunctions[domainInstances[p][sample]]++;
            }
            for (int i = 0; i < numberOfInstances[p]; i++)
                labelsSamples[0][p][i] = sum[i] >= (numberOfFunctions[i] / 2) ? 1 : 0;
            for (int j = 0; j < this.numberOfFunctions; j++) {
                errorRatesSamples[0][p][j] = 0.25;
                disagreements[p][j] = 0;
            }
            for (int sample = 0; sample < domainInstances[p].length; sample++)
                if (domainValues[p][sample] != labelsSamples[0][p][domainInstances[p][sample]])
                    disagreements[p][domainFunctions[p][sample]]++;
        }
        labelPriorMeans = new double[numberOfDomains];
        errorRateMeans = new double[numberOfDomains][numberOfFunctions];
        labelMeans = new double[numberOfDomains][];
    }

    @Override
    public ErrorRates errorRates(boolean forceComputation) {
        if (forceComputation)
            needsInference = true;
        performInference();
        return errorRates;
    }

    @Override
    public Integrator.Data<Data.PredictedInstance> integratedData(boolean forceComputation) {
        if (forceComputation)
            needsInference = true;
        performInference();
        return integratedData;
    }

    private void performInference() {
        if (!needsInference)
            return;
        for (int sampleIndex = 0; sampleIndex < numberOfBurnInSamples; sampleIndex++) {
            samplePriorsAndErrorRates(0);
            sampleLabels(0);
        }
        for (int sampleIndex = 1; sampleIndex < numberOfSamples; sampleIndex++) {
            for (int i = 0; i < numberOfThinningSamples; i++) {
                samplePriorsAndErrorRates(sampleIndex - 1);
                sampleLabels(sampleIndex - 1);
            }
            samplePriorsAndErrorRates(sampleIndex - 1);
            sampleLabels(sampleIndex - 1);
            storeSample(sampleIndex);
        }
        // Aggregate values for means and variances computation
        for (int p = 0; p < numberOfDomains; p++)
            labelMeans[p] = new double[numberOfInstances[p]];
        for (int sampleNumber = 0; sampleNumber < numberOfSamples; sampleNumber++) {
            for (int p = 0; p < numberOfDomains; p++) {
                labelPriorMeans[p] += labelPriorsSamples[sampleNumber][p];
                for (int j = 0; j < numberOfFunctions; j++)
                    errorRateMeans[p][j] += errorRatesSamples[sampleNumber][p][j];
                for (int i = 0; i < numberOfInstances[p]; i++)
                    labelMeans[p][i] += labelsSamples[sampleNumber][p][i];
            }
        }
        List<Integrator.Data.PredictedInstance> integratedDataInstances = new ArrayList<>();
        List<ErrorRates.Instance> errorRatesInstances = new ArrayList<>();
        // Compute values for the means and the variances
        for (int p = 0; p < numberOfDomains; p++) {
            labelPriorMeans[p] /= numberOfSamples;
            for (int j = 0; j < numberOfFunctions; j++) {
                errorRateMeans[p][j] /= numberOfSamples;
                errorRatesInstances.add(new ErrorRates.Instance(labelKeysMap.inverse().get(p),
                                                                classifierKeysMap.inverse().get(j),
                                                                errorRateMeans[p][j]));
            }
            for (int i = 0; i < numberOfInstances[p]; i++) {
                labelMeans[p][i] /= numberOfSamples;
                integratedDataInstances.add(new Data.PredictedInstance(instanceKeysMap.get(p).inverse().get(i),
                                                                       labelKeysMap.inverse().get(p),
                                                                       labelMeans[p][i]));
            }
        }
        integratedData = new Data<>(integratedDataInstances);
        errorRates = new ErrorRates(errorRatesInstances);
        needsInference = false;
    }

    private void samplePriorsAndErrorRates(int sampleNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            int labelsCount = 0;
            for (int i = 0; i < numberOfInstances[p]; i++)
                labelsCount += labelsSamples[sampleNumber][p][i];
            labelPriorsSamples[sampleNumber][p] = randomDataGenerator.nextBeta(
                    labelsPriorAlpha + labelsCount,
                    labelsPriorBeta + numberOfInstances[p] - labelsCount
            );
            disagreements[p] = new int[numberOfFunctions];
            int[] numberOfInstances = new int[numberOfFunctions];
            for (int sample = 0; sample < domainInstances[p].length; sample++) {
                if (labelsSamples[0][p][domainInstances[p][sample]] != domainValues[p][sample])
                    disagreements[p][domainFunctions[p][sample]]++;
                numberOfInstances[domainFunctions[p][sample]]++;
            }
            for (int j = 0; j < numberOfFunctions; j++)
                errorRatesSamples[sampleNumber][p][j] = randomDataGenerator.nextBeta(
                        errorRatesPriorAlpha + disagreements[p][j],
                        errorRatesPriorBeta + numberOfInstances[j] - disagreements[p][j]
                );
        }
    }

    private void sampleLabels(int sampleNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            double[] p0 = new double[numberOfInstances[p]];
            double[] p1 = new double[numberOfInstances[p]];
            Arrays.fill(p0, 1 - labelPriorsSamples[sampleNumber][p]);
            Arrays.fill(p1, labelPriorsSamples[sampleNumber][p]);
            for (int sample = 0; sample < domainInstances[p].length; sample++)
                if (domainValues[p][sample] == 0.0) {
                    p0[domainInstances[p][sample]] *= (1.0 - errorRatesSamples[sampleNumber][p][domainFunctions[p][sample]]);
                    p1[domainInstances[p][sample]] *= errorRatesSamples[sampleNumber][p][domainFunctions[p][sample]];
                } else {
                    p0[domainInstances[p][sample]] *= errorRatesSamples[sampleNumber][p][domainFunctions[p][sample]];
                    p1[domainInstances[p][sample]] *= (1.0 - errorRatesSamples[sampleNumber][p][domainFunctions[p][sample]]);
                }
            for (int i = 0; i < numberOfInstances[p]; i++)
                labelsSamples[sampleNumber][p][i] = randomDataGenerator.nextBinomial(1, p1[i] / (p0[i] + p1[i]));
        }
    }

    private void storeSample(int sampleIndex) {
        ArrayUtilities.copy(labelPriorsSamples[sampleIndex - 1], labelPriorsSamples[sampleIndex]);
        ArrayUtilities.copy(errorRatesSamples[sampleIndex - 1], errorRatesSamples[sampleIndex]);
        ArrayUtilities.copy(labelsSamples[sampleIndex - 1], labelsSamples[sampleIndex]);
    }
}
