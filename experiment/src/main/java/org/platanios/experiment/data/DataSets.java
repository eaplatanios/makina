package org.platanios.experiment.data;

import com.google.common.collect.Sets;
import org.apache.commons.cli.*;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.platanios.learn.classification.Label;
import org.platanios.learn.classification.constraint.Constraint;
import org.platanios.learn.classification.constraint.MutualExclusionConstraint;
import org.platanios.learn.classification.constraint.SubsumptionConstraint;
import org.platanios.math.matrix.Vector;
import org.platanios.math.matrix.Vectors;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.zip.GZIPInputStream;

/**
 * @author Emmanouil Antonios Platanios
 */
public class DataSets {
    private static final Logger logger = LogManager.getLogger("Data Sets");

    /**
     *
     * @param   workingDirectory
     * @return              A {@code Map<String, Map<String, Map<String, Double>>>} indexed by noun-phrase, category,
     *                      and classifier and containing the corresponding probabilities or {@code null} if there is no
     *                      such probability.
     */
    public static NELLData importNELLData(String workingDirectory) {
        List<NELLData.Instance> instances = new ArrayList<>();
        File[] directoryFiles = new File(workingDirectory).listFiles();
        if (directoryFiles == null)
            throw new IllegalArgumentException("The provided working directory does not contain any files.");
        Arrays.asList(directoryFiles)
                .stream()
                .filter(file -> file.getName().endsWith(".all_predictions.txt"))
                .forEach(file -> {
                    try {
                        String categoryName = file.getName().split("[.]")[0];
                        Map<Integer, String> components = new HashMap<>();
                        boolean[] firstLine = new boolean[] { true };
                        Files.newBufferedReader(Paths.get(file.getPath())).lines().forEach(line -> {
                            String[] lineParts = line.split("\t");
                            if (firstLine[0]) {
                                for (int index = 1; index < lineParts.length; index++)
                                    components.put(index, lineParts[index]);
                                firstLine[0] = false;
                            } else {
                                for (int index = 1; index < lineParts.length; index++)
                                    if (!lineParts[index].equals("-") && !lineParts[index].equals("NaN"))
                                        instances.add(new NELLData.Instance(lineParts[0],
                                                                            categoryName,
                                                                            components.get(index),
                                                                            Double.valueOf(lineParts[index])));
                            }
                        });
                    } catch (IOException e) {
                        throw new IllegalStateException("An exception occurred while processing a predictions file.");
                    }
                });
        return new NELLData(instances);
    }

    public static class NELLData implements Iterable<NELLData.Instance> {
        private final List<Instance> instances;

        public NELLData(List<Instance> instances) {
            this.instances = instances;
        }

        @Override
        public Iterator<Instance> iterator() {
            return instances.iterator();
        }

        public Stream<Instance> stream() {
            return instances.stream();
        }

        public Set<String> nounPhrases() {
            return instances.stream().map(Instance::nounPhrase).collect(Collectors.toSet());
        }

        public Set<String> categories() {
            return instances.stream().map(Instance::category).collect(Collectors.toSet());
        }

        public Set<String> components() {
            return instances.stream().map(Instance::component).collect(Collectors.toSet());
        }

        public int numberOfNounPhrases(Set<String> categories, Set<String> components) {
            return (int) instances.stream()
                    .filter(instance -> categories.contains(instance.category) && components.contains(instance.component))
                    .count();
        }

        public static class Instance {
            private final String nounPhrase;
            private final String category;
            private final String component;
            private final double probability;

            public Instance(String nounPhrase, String category, String component, double probability) {
                this.nounPhrase = nounPhrase;
                this.category = category;
                this.component = component;
                this.probability = probability;
            }

            public String nounPhrase() {
                return nounPhrase;
            }

            public String category() {
                return category;
            }

            public String component() {
                return component;
            }

            public double probability() {
                return probability;
            }
        }
    }

    public static CPLDataSet importCPLDataSet(String labeledNPsPath,
                                              String cplFeatureMapDirectory) {
        return importCPLDataSet(labeledNPsPath, cplFeatureMapDirectory, null, 0.0);
    }

    public static CPLDataSet importCPLDataSet(String labeledNPsPath,
                                              String cplFeatureMapDirectory,
                                              Set<String> categories) {
        return importCPLDataSet(labeledNPsPath, cplFeatureMapDirectory, categories, 0.0);
    }

    public static CPLDataSet importCPLDataSet(String labeledNPsPath,
                                              String cplFeatureMapDirectory,
                                              double cplThreshold) {
        return importCPLDataSet(labeledNPsPath, cplFeatureMapDirectory, null, cplThreshold);
    }

    public static CPLDataSet importCPLDataSet(String labeledNPsPath,
                                              String cplFeatureMapDirectory,
                                              Set<String> categories,
                                              double cplThreshold) {
        logger.info("Importing NELL data set...");
        Map<String, Set<String>> labeledNounPhrases = new HashMap<>();
        try {
            Files.newBufferedReader(Paths.get(labeledNPsPath)).lines().forEach(line -> {
                String[] lineParts = line.split("\t");
                if (lineParts.length == 2) {
                    Set<String> labels;
                    if (categories != null)
                        labels = Sets.intersection(new HashSet<>(Arrays.asList(lineParts[1].split(","))), categories);
                    else
                        labels = new HashSet<>(Arrays.asList(lineParts[1].split(",")));
                    if (labels.size() > 0)
                        labeledNounPhrases.put(lineParts[0], labels);
                }
            });
        } catch (IOException e) {
            throw new IllegalArgumentException("There was a problem with the provided labeled noun phrases file.");
        }
        logger.info("Importing NELL feature map...");
        Map<String, Vector> featureMap = importCPLFeatureMap(cplFeatureMapDirectory,
                                                             labeledNounPhrases.keySet(),
                                                             cplThreshold);
        Map<String, Set<String>> labels = new HashMap<>();
        Map<String, Vector> features = new HashMap<>();
        Set<String> nounPhrasesWithoutFeatures = new HashSet<>();
        for (Map.Entry<String, Set<String>> labeledNounPhraseEntry : labeledNounPhrases.entrySet()) {
            String nounPhrase = labeledNounPhraseEntry.getKey();
            if (!featureMap.containsKey(nounPhrase)) {
                nounPhrasesWithoutFeatures.add(nounPhrase);
                continue;
            }
            labels.put(nounPhrase, labeledNounPhraseEntry.getValue());
            features.put(nounPhrase, featureMap.get(nounPhrase));
        }
        logger.info("There were " + nounPhrasesWithoutFeatures.size() + " noun phrases without features.");
        return new CPLDataSet(labels, features);
    }

    public static Map<String, Vector> importCPLFeatureMap(String cplFeatureMapDirectory,
                                                          Set<String> nounPhrases) {
        return importCPLFeatureMap(cplFeatureMapDirectory, nounPhrases, 0.0);
    }

    public static Map<String, Vector> importCPLFeatureMap(String cplFeatureMapDirectory,
                                                          Set<String> nounPhrases,
                                                          double contentCountsProportionThreshold) {
        Map<String, Vector> featureMap = new HashMap<>();
        Map<String, Integer> contexts;
        try {
            Stream<String> contextsLines = new BufferedReader(new InputStreamReader(new GZIPInputStream(
                    Files.newInputStream(Paths.get(cplFeatureMapDirectory + "/cat_contexts.txt.gz"))
            ))).lines();
            Map<String, Integer> contextCounts = new HashMap<>();
            contextsLines.forEach(line -> {
                String[] lineParts = line.split("\t");
                contextCounts.put(lineParts[0], Integer.parseInt(lineParts[1]));
            });
            List<Integer> contextCountsList = new ArrayList<>();
            contextCounts.entrySet()
                    .stream()
                    .sorted(Collections.reverseOrder(Comparator.comparing(Map.Entry::getValue)))
                    .forEachOrdered(e -> contextCountsList.add(e.getValue()));
            Stream<String> npContextPairsLines = new BufferedReader(new InputStreamReader(new GZIPInputStream(
                    Files.newInputStream(Paths.get(cplFeatureMapDirectory + "/cat_pairs_np-idx.txt.gz"))
            ))).lines();
            Map<String, Map<String, Double>> preprocessedFeatureMap = new HashMap<>();
            npContextPairsLines.forEach(line -> {
                String[] lineParts = line.split("\t");
                String np = lineParts[0];
                if (nounPhrases == null || nounPhrases.contains(np)) {
                    Map<String, Double> contextValues = new HashMap<>();
                    for (int i = 1; i < lineParts.length; i++) {
                        String[] contextParts = lineParts[i].split(" -#- ");
                        if (contentCountsProportionThreshold >= 0
                                && contextCounts.containsKey(contextParts[0])
                                && contextCounts.get(contextParts[0]) > contextCountsList.get(0) * contentCountsProportionThreshold)
                            contextValues.put(contextParts[0], Double.parseDouble(contextParts[1]));
                    }
                    if (contextValues.size() > 0)
                        preprocessedFeatureMap.put(np, contextValues);
                }
            });
            contexts = importContextsMap(
                    cplFeatureMapDirectory,
                    preprocessedFeatureMap.values()
                            .stream()
                            .map(Map::keySet)
                            .flatMap(Collection::stream)
                            .collect(Collectors.toSet())
            );
            for (Map.Entry<String, Map<String, Double>> preprocessedFeatures : preprocessedFeatureMap.entrySet()) {
                Map<Integer, Double> featuresMap = new TreeMap<>();
                preprocessedFeatures.getValue().entrySet()
                        .stream()
                        .filter(preprocessedFeature -> contexts.containsKey(preprocessedFeature.getKey()))
                        .forEach(preprocessedFeature -> featuresMap.put(contexts.get(preprocessedFeature.getKey()),
                                                                        preprocessedFeature.getValue()));
                featureMap.put(preprocessedFeatures.getKey(), Vectors.sparse(contexts.size(), featuresMap));
            }
        } catch (IOException e) {
            logger.error("An exception was thrown while trying to build the CPL feature map.", e);
        }
        return featureMap;
    }

    public static Map<String, Integer> importContextsMap(String cplFeatureMapDirectory,
                                                         Set<String> uniqueContexts) throws IOException {
        Map<String, Integer> contexts = new HashMap<>();
        Stream<String> npContextPairsLines = new BufferedReader(new InputStreamReader(new GZIPInputStream(
                Files.newInputStream(Paths.get(cplFeatureMapDirectory + "/cat_contexts.txt.gz"))
        ))).lines();
        int[] contextIndex = { 0 };
        npContextPairsLines.forEachOrdered(line -> {
            String[] lineParts = line.split("\t");
            if (uniqueContexts.contains(lineParts[0]) && !contexts.containsKey(lineParts[0]))
                contexts.put(lineParts[0], contextIndex[0]++);
        });
        return contexts;
    }

    public static class CPLDataSet {
        private final Map<String, Set<String>> labels;
        private final Map<String, Vector> features;

        public CPLDataSet(Map<String, Set<String>> labels, Map<String, Vector> features) {
            this.labels = labels;
            this.features = features;
        }

        public Map<String, Set<String>> getLabels() {
            return labels;
        }

        public Map<String, Vector> getFeatures() {
            return features;
        }
    }

    public static Set<Constraint> importConstraints(String filePath) {
        Set<Constraint> constraints = new HashSet<>();
        try {
            Files.newBufferedReader(Paths.get(filePath)).lines().forEach(line -> {
                if (line.startsWith("!")) {
                    constraints.add(new MutualExclusionConstraint(Arrays.asList(line.substring(1).split(","))
                                                                          .stream()
                                                                          .map(Label::new)
                                                                          .collect(Collectors.toSet())));
                } else {
                    String[] lineParts = line.split(" -> ");
                    String[] childrenLabels = lineParts[1].split(",");
                    for (String childLabel : childrenLabels)
                        constraints.add(new SubsumptionConstraint(new Label(lineParts[0]), new Label(childLabel)));
                }
            });
        } catch (IOException e) {
            throw new IllegalArgumentException("There was a problem with the provided constraints file.");
        }
        return constraints;
    }

    public enum Type {
        NELL,
        NELL_CPL
    }

    public static void main(String[] args) {
        Options options = new Options();
        options.addOption("h", "help", false, "Prints information regarding the usage of this program.");
        options.addOption(Option.builder("t").longOpt("type").hasArg(true).desc("The type of the data set to create.").required(true).build());
        options.addOption("cpl", "cpl-feature-maps-directory", true, "The CPL feature maps directory.");
        options.addOption("cplTh", "cpl-threshold", true, "The CPL feature map threshold as a proportion of the highest-valued feature.");
        options.addOption(Option.builder("cat").longOpt("categories").hasArg(true).desc("The NELL categories to consider.").valueSeparator(',').build());
        CommandLineParser parser = new DefaultParser();
        try {
            CommandLine cmd = parser.parse(options, args);
            if (cmd.hasOption("h") || args.length == 0) {
                HelpFormatter formatter = new HelpFormatter();
                formatter.printHelp("DataSets [options] [working directory]", options);
                return;
            }
            String workingDirectory = args[args.length - 1];
            double cplThreshold = 0.0;
            if (cmd.hasOption("cpl-threshold"))
                cplThreshold = Double.parseDouble(cmd.getOptionValue("cpl-threshold"));
            Type type;
            switch (cmd.getOptionValue("t")) {
                case "NELL":
                case "nell":
                    type = Type.NELL;
                    break;
                case "NELL-CPL":
                case "nell-cpl":
                    type = Type.NELL_CPL;
                    break;
                default:
                    throw new ParseException("Unsupported data set type.");
            }
            switch (type) {
                case NELL:
                    if (!cmd.hasOption("cpl"))
                        throw new ParseException("No CPL features map directory provided.");
                    NELLData nellData = importNELLData(workingDirectory);
                    Map<String, Vector> cplFeatureMap = importCPLFeatureMap(cmd.getOptionValue("cpl"),
                                                                            nellData.nounPhrases(),
                                                                            cplThreshold);
                    File nellFeaturesFile = new File(workingDirectory + "/np_features.tsv");
                    try (BufferedWriter nellFeaturesWriter = new BufferedWriter(new FileWriter(nellFeaturesFile))) {
                        for (Map.Entry<String, Vector> featuresEntry : cplFeatureMap.entrySet()) {
                            StringJoiner features = new StringJoiner(",");
                            for (Vector.VectorElement element : featuresEntry.getValue())
                                features.add(element.index() + ":" + element.value());
                            nellFeaturesWriter.write(featuresEntry.getKey() + "\t" + features.toString() + "\n");
                        }
                    } catch (IOException e) {
                        logger.error("There was a problem while producing the output file.", e);
                    }
                    break;
                case NELL_CPL:
                    if (!cmd.hasOption("cpl"))
                        throw new ParseException("No CPL features map directory provided.");
                    Set<String> categories = null;
                    if (cmd.hasOption("cat"))
                        categories = new HashSet<>(Arrays.asList(cmd.getOptionValues("cat")));
                    CPLDataSet cplDataSet = importCPLDataSet(workingDirectory + "/np_labels_original.tsv",
                                                             cmd.getOptionValue("cpl"),
                                                             categories,
                                                             cplThreshold);
                    File labelsFile = new File(workingDirectory + "/np_labels.tsv");
                    try (BufferedWriter labelsWriter = new BufferedWriter(new FileWriter(labelsFile))) {
                        for (Map.Entry<String, Set<String>> labelsEntry : cplDataSet.getLabels().entrySet()) {
                            StringJoiner labels = new StringJoiner(",");
                            labelsEntry.getValue().forEach(labels::add);
                            labelsWriter.write(labelsEntry.getKey() + "\t" + labels.toString() + "\n");
                        }
                    } catch (IOException e) {
                        logger.error("There was a problem while producing the output file.", e);
                    }
                    File cplFeaturesFile = new File(workingDirectory + "/np_features.tsv");
                    try (BufferedWriter cplFeaturesWriter = new BufferedWriter(new FileWriter(cplFeaturesFile))) {
                        for (Map.Entry<String, Vector> featuresEntry : cplDataSet.getFeatures().entrySet()) {
                            StringJoiner features = new StringJoiner(",");
                            for (Vector.VectorElement element : featuresEntry.getValue())
                                features.add(element.index() + ":" + element.value());
                            cplFeaturesWriter.write(featuresEntry.getKey() + "\t" + features.toString() + "\n");
                        }
                    } catch (IOException e) {
                        logger.error("There was a problem while producing the output file.", e);
                    }
                    break;
            }
        } catch (ParseException e) {
            logger.error("There was a problem while parsing the command-line arguments.", e);
        }
    }
}
