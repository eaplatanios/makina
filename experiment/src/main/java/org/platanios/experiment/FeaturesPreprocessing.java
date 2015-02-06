package org.platanios.experiment;

import com.google.common.collect.Maps;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.platanios.learn.data.FeatureMapMySQL;
import org.platanios.learn.math.matrix.SparseVector;
import org.platanios.learn.math.matrix.VectorType;
import org.platanios.learn.math.matrix.Vectors;

import java.io.*;
import java.nio.charset.Charset;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.zip.GZIPInputStream;

/**
 * @author Emmanouil Antonios Platanios
 */
public class FeaturesPreprocessing {
    private static final Logger logger = LogManager.getLogger("Features Preprocessing");

    private static final String[] categories = { "animal" };
    private static final String labeledNELLNounPhrasesDirectory = "/Volumes/Macintosh HD/Users/Anthony/Development/NELL/data/server/NELL.08m.880.mttrain.csv";
    private static final String featureMapsDirectory = "/Volumes/Macintosh HD/Users/Anthony/Development/NELL/data/features";
    private static final String labeledDataDirectory = "/Volumes/Macintosh HD/Users/Anthony/Development/Data Sets/NELL/Training Data/labeled_nps.data";
    private static final String filteredLabeledDataDirectory = "/Volumes/Macintosh HD/Users/Anthony/Development/Data Sets/NELL/Training Data/filtered_labeled_nps.data";
    private static final String trainingDataVariasDirectory = "/Volumes/Macintosh HD/Users/Anthony/Development/Data Sets/NELL/varias_data/trainData";
    private static final String testingDataVariasDirectory = "/Volumes/Macintosh HD/Users/Anthony/Development/Data Sets/NELL/varias_data/testData";
    private static final String cplFeatureMapDirectory = "/Volumes/Macintosh HD/Users/Anthony/Development/Data Sets/NELL/Server/all-pairs/all-pairs-OC-2011-12-31-big2-gz";
    private static final String adjFeatureMapDirectory = "/Volumes/Macintosh HD/Users/Anthony/Development/Data Sets/NELL/Feature Cabinets/adj_shashans";

    private static Map<String, Map<String, Boolean>> labeledData;
    private static Map<String, Map<String, Boolean>> filteredLabeledData;

    private static void parseVariasCategoriesFiles() {
        if (Files.exists(Paths.get(labeledDataDirectory))) {
            labeledData = readStringStringBooleanMap(labeledDataDirectory);
            logger.info("Read the labeled data from the existing file.");
        } else {
            logger.info("Starting the collection of the labeled data...");
            labeledData = new HashMap<>();
            try {
                for (String path : new String[]{trainingDataVariasDirectory, testingDataVariasDirectory}) {
                    DirectoryStream<Path> featureTypesDirectories = Files.newDirectoryStream(Paths.get(path));
                    for (Path featureTypeDirectory : featureTypesDirectories) {
                        if (Files.isDirectory(featureTypeDirectory)) {
                            DirectoryStream<Path> categoriesFilePaths = Files.newDirectoryStream(featureTypeDirectory);
                            for (Path categoryFile : categoriesFilePaths) {
                                String category = categoryFile.getFileName().toString().replaceFirst("[.][^.]+$", "");
                                if (!labeledData.keySet().contains(category))
                                    labeledData.put(category, new HashMap<>());
                                boolean positiveExample = true;
                                Stream<String> categoryFileLines = Files.lines(categoryFile, Charset.defaultCharset());
                                for (String categoryFileLine : categoryFileLines.collect(Collectors.toList())) {
                                    switch (categoryFileLine) {
                                        case "##POSITIVE##":
                                            break;
                                        case "##ALWAYSNEGATIVE##":
                                            positiveExample = false;
                                            break;
                                        case "##NEGATIVE##":
                                            positiveExample = false;
                                            break;
                                        default:
                                            if (labeledData.get(category).getOrDefault(categoryFileLine, positiveExample) != positiveExample) {
                                                logger.info("Conflicting labels found for NP \"" + categoryFileLine + "\" for category \"" + category + "\"!");
                                                labeledData.get(category).remove(categoryFileLine);
                                            } else {
                                                labeledData.get(category).put(categoryFileLine, positiveExample);
                                            }
                                    }
                                }
                            }
                        }
                    }
                }
                writeStringStringBooleanMap(labeledData, labeledDataDirectory);
                logger.info("Finished collecting the labeled data.");
                for (String category : labeledData.keySet()) {
                    int totalNumber = labeledData.get(category).size();
                    int numberOfPositive = Maps.filterValues(labeledData.get(category), x -> x).size();
                    int numberOfNegative = Maps.filterValues(labeledData.get(category), x -> !x).size();
                    logger.info("Number of NPs for \"" + category + "\": " + totalNumber + " (" + numberOfPositive + " positive | " + numberOfNegative + " negative)");
                }
            } catch (IOException e) {
                logger.error("An exception was thrown while trying to parse the categories files.", e);
            }
        }
    }

    private static void buildFeatureMap(FeatureMapMySQL<SparseVector> featureMap) {
        featureMap.createDatabase();
        buildCPLFeatureMap(featureMap);
        buildADJFeatureMap(featureMap);
    }

    private static void buildCPLFeatureMap(FeatureMapMySQL<SparseVector> featureMap) {
        Map<String, Integer> contexts;
        try {
            if (Files.exists(Paths.get(cplFeatureMapDirectory + "/contexts.bin"))) {
                contexts = readStringIntegerMap(cplFeatureMapDirectory + "/contexts.bin");
            } else {
                contexts = new HashMap<>();
                Stream<String> npContextPairsLines = new BufferedReader(new InputStreamReader(new GZIPInputStream(
                        Files.newInputStream(Paths.get(cplFeatureMapDirectory + "/cat_pairs_np-idx.txt.gz"))
                ))).lines();
                int[] contextIndex = {0};
                npContextPairsLines.forEachOrdered(line -> {
                    String[] lineParts = line.split("\t");
                    for (int i = 1; i < lineParts.length; i++) {
                        String[] contextParts = lineParts[i].split(" -#- ");
                        if (!contexts.containsKey(contextParts[0]))
                            contexts.put(contextParts[0], contextIndex[0]++);
                    }
                });
                writeStringIntegerMap(contexts, cplFeatureMapDirectory + "/contexts.bin");
            }
            Stream<String> npContextPairsLines = new BufferedReader(new InputStreamReader(new GZIPInputStream(
                    Files.newInputStream(Paths.get(cplFeatureMapDirectory + "/cat_pairs_np-idx.txt.gz"))
            ))).lines();
            npContextPairsLines.forEachOrdered(line -> {
                String[] lineParts = line.split("\t");
                String np = lineParts[0];
                SparseVector features = (SparseVector) Vectors.build(contexts.size(), VectorType.SPARSE);
                for (int i = 1; i < lineParts.length; i++) {
                    String[] contextParts = lineParts[i].split(" -#- ");
                    features.set(contexts.get(contextParts[0]), Double.parseDouble(contextParts[1]));
                }
                lineParts = null;
                line = null;
                try {
                    featureMap.addFeatureMappings(np, features, 0);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            });
        } catch (IOException e) {
            System.out.println("An exception was thrown while trying to build the CPL feature map.");
        }
    }

    private static void buildADJFeatureMap(FeatureMapMySQL<SparseVector> featureMap) {
        Map<String, Integer> adjectives;
        try {
            if (Files.exists(Paths.get(adjFeatureMapDirectory + "/adjectives.bin"))) {
                adjectives = readStringIntegerMap(adjFeatureMapDirectory + "/adjectives.bin");
            } else {
                adjectives = new HashMap<>();
                Stream<String> npAdjectivePairsLines = new BufferedReader(new InputStreamReader(
                        Files.newInputStream(Paths.get(adjFeatureMapDirectory + "/NP1_JJ1_DependencyArcs.vocab"))
                )).lines();
                npAdjectivePairsLines.forEach(line -> {
                    String[] lineParts = line.split("\t");
                    adjectives.put(lineParts[1], Integer.parseInt(lineParts[0]));
                });
                writeStringIntegerMap(adjectives, adjFeatureMapDirectory + "/adjectives.bin");
            }
            Stream<String> npAdjectivePairsLines = new BufferedReader(new InputStreamReader(
                    Files.newInputStream(Paths.get(adjFeatureMapDirectory + "/NP1_JJ1_DependencyArcs.features"))
            )).lines();
            int[] processedNPs = new int[] { 0 };
            npAdjectivePairsLines.forEach(line -> {
                String[] lineParts = line.split("\t\t");
                String np = lineParts[0];
                lineParts = lineParts[1].split("\t");
                if (np.length() <= 100) {
                    SparseVector features = (SparseVector) Vectors.build(adjectives.size(), VectorType.SPARSE);
                    for (int i = 0; i < lineParts.length; i++) {
                        String[] contextParts = lineParts[i].split(":");
                        features.set(Integer.parseInt(contextParts[0]), Double.parseDouble(contextParts[1]));
                    }
                    lineParts = null;
                    line = null;
                    processedNPs[0]++;
                    if (processedNPs[0] % 1000 == 0)
                        logger.info("Number of NPs processed:" + processedNPs[0]);
                    try {
                        featureMap.addFeatureMappings(np, features, 2);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            });
        } catch (IOException e) {
            System.out.println("An exception was thrown while trying to build the CPL feature map.");
            e.printStackTrace();
        }
    }

    public static void filterLabeledDataByFeatureMap(FeatureMapMySQL<SparseVector> featureMap) {
        if (Files.exists(Paths.get(filteredLabeledDataDirectory))) {
            filteredLabeledData = readStringStringBooleanMap(filteredLabeledDataDirectory);
            logger.info("Read the filtered labeled data from the existing file.");
        } else {
            filteredLabeledData = new HashMap<>();
            for (String category : labeledData.keySet()) {
                filteredLabeledData.put(category, new HashMap<>());
                labeledData.get(category)
                        .keySet()
                        .stream()
                        .filter(np -> !np.equals("people") && !np.equals("People") && !np.equals("PEOPLE") && !featureMap.getFeatureVectors(np).contains(null))
                        .forEach(np -> filteredLabeledData.get(category).put(np, labeledData.get(category).get(np)));
                int totalNumber = filteredLabeledData.get(category).size();
                int numberOfPositive = Maps.filterValues(filteredLabeledData.get(category), x -> x).size();
                int numberOfNegative = Maps.filterValues(filteredLabeledData.get(category), x -> !x).size();
                logger.info("Number of NPs for \"" + category + "\": " + totalNumber + " (" + numberOfPositive + " positive | " + numberOfNegative + " negative)");
            }
            writeStringStringBooleanMap(filteredLabeledData, filteredLabeledDataDirectory);
            logger.info("Finished filtering the labeled data.");
        }
    }

    public static void writeStringStringBooleanMap(Map<String, Map<String, Boolean>> map, String filename) {
        try
        {
            FileOutputStream fos = new FileOutputStream(filename);
            ObjectOutputStream oos = new ObjectOutputStream(fos);
            oos.writeObject(map);
            oos.close();
            fos.close();
        } catch (IOException e) {
            System.out.println("An exception was thrown while trying to write the string-integer map.");
            e.printStackTrace();
        }
    }

    @SuppressWarnings("unchecked")
    public static Map<String, Map<String, Boolean>> readStringStringBooleanMap(String filename) {
        Map<String, Map<String, Boolean>> map = new HashMap<>();
        try
        {
            FileInputStream fis = new FileInputStream(filename);
            ObjectInputStream ois = new ObjectInputStream(fis);
            map = (Map<String, Map<String, Boolean>>) ois.readObject();
            ois.close();
            fis.close();
        } catch (IOException|ClassNotFoundException e) {
            System.out.println("An exception was thrown while trying to read the string-integer map.");
            e.printStackTrace();
        }
        return map;
    }

    public static void writeStringIntegerMap(Map<String, Integer> map, String filename) {
        try
        {
            FileOutputStream fos = new FileOutputStream(filename);
            ObjectOutputStream oos = new ObjectOutputStream(fos);
            oos.writeObject(map);
            oos.close();
            fos.close();
        } catch (IOException e) {
            System.out.println("An exception was thrown while trying to write the string-integer map.");
            e.printStackTrace();
        }
    }

    @SuppressWarnings("unchecked")
    public static Map<String, Integer> readStringIntegerMap(String filename) {
        Map<String, Integer> map = new HashMap<>();
        try
        {
            FileInputStream fis = new FileInputStream(filename);
            ObjectInputStream ois = new ObjectInputStream(fis);
            map = (Map<String, Integer>) ois.readObject();
            ois.close();
            fis.close();
        } catch (IOException|ClassNotFoundException e) {
            System.out.println("An exception was thrown while trying to read the string-integer map.");
            e.printStackTrace();
        }
        return map;
    }

    public static void main(String[] args) {
        FeatureMapMySQL<SparseVector> featureMap = new FeatureMapMySQL<>(
                3,
                "jdbc:mysql://localhost/",
                "root",
                null,
                "learn",
                "features"
        );
//        buildFeatureMap(featureMap);
        parseVariasCategoriesFiles();
        filterLabeledDataByFeatureMap(featureMap);
    }
}
