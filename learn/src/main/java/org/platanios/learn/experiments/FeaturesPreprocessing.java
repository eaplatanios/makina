package org.platanios.learn.experiments;

import com.google.common.collect.Maps;
import org.platanios.learn.data.FeatureMapMariaDB;
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
    private static final String[] categories = { "animal" };
    private static final String labeledNounPhrasesDirectory = "/Volumes/Macintosh HD/Users/Anthony/Development/NELL/data/server/NELL.08m.880.mttrain.csv";
    private static final String featureMapsDirectory = "/Volumes/Macintosh HD/Users/Anthony/Development/NELL/data/features";
    private static final String trainingDataDirectory = "/Volumes/Macintosh HD/Users/Anthony/Development/NELL/data/varias_data/trainData";
    private static final String cplFeatureMapDirectory = "/Volumes/Macintosh HD/Users/Anthony/Development/Data Sets/NELL/Server/all-pairs/all-pairs-OC-2011-12-31-big2-gz";

    private static Map<String, List<Map<String, Boolean>>> categoriesFiles;
    private static Map<String, Map<String, Boolean>> combinedCategoriesNounPhrases;

    private static void parseCategoriesFiles() {
        System.out.println("\nParsing categories files...");
        categoriesFiles = new TreeMap<>();
        try {
            DirectoryStream<Path> featureTypesDirectories = Files.newDirectoryStream(Paths.get(trainingDataDirectory));
            for (Path featureTypeDirectory : featureTypesDirectories) {
                if (Files.isDirectory(featureTypeDirectory)) {
                    System.out.println("\n\tParsing files for \""
                                               + featureTypeDirectory.getFileName() + "\" feature map...\n");
                    DirectoryStream<Path> categoriesFilePaths = Files.newDirectoryStream(featureTypeDirectory);
                    for (Path categoryFile : categoriesFilePaths) {
                        String category = categoryFile.getFileName().toString().replaceFirst("[.][^.]+$", "");
                        if (!categoriesFiles.keySet().contains(category))
                            categoriesFiles.put(category, new ArrayList<>());
                        Map<String, Boolean> categoryNounPhrases = new TreeMap<>();
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
                                    categoryNounPhrases.put(categoryFileLine, positiveExample);
                            }
                        }
                        categoriesFiles.get(category).add(categoryNounPhrases);
                        int totalNumber = categoryNounPhrases.size();
                        int numberOfPositive = Maps.filterValues(categoryNounPhrases, x -> x).size();
                        int numberOfNegative = Maps.filterValues(categoryNounPhrases, x -> !x).size();
                        System.out.println("\t\t" + category + ": " + totalNumber
                                                   + "("+ numberOfPositive + " positive | "
                                                   + numberOfNegative + " negative)");
                    }
                }
            }
        } catch (IOException e) {
            System.out.println("An exception was thrown while trying to parse the categories files.");
        }
    }

    private static void combineCategoriesFiles() {
        System.out.println("\nCombining categories files...\n");
        combinedCategoriesNounPhrases = new TreeMap<>();
        for (String category : categoriesFiles.keySet()) {
            Set<String> intersectionKeySet = categoriesFiles.get(category).get(0).keySet();
            for (int i = 1; i < categoriesFiles.get(category).size(); i++) {
                intersectionKeySet.retainAll(categoriesFiles.get(category).get(i).keySet());
            }
            combinedCategoriesNounPhrases.put(category, new TreeMap<>());
            for (String key : intersectionKeySet) {
                combinedCategoriesNounPhrases.get(category).put(key, categoriesFiles.get(category).get(0).get(key));
            }
            int totalNumber = combinedCategoriesNounPhrases.get(category).size();
            int numberOfPositive = Maps.filterValues(combinedCategoriesNounPhrases.get(category), x -> x).size();
            int numberOfNegative = Maps.filterValues(combinedCategoriesNounPhrases.get(category), x -> !x).size();
            System.out.println("\t" + category + ": " + totalNumber
                                       + "("+ numberOfPositive + " positive | " + numberOfNegative + " negative)");
        }
    }

    private static void buildFeatureMap() {
        FeatureMapMariaDB<SparseVector> featureMap = new FeatureMapMariaDB<>(
                3,
                "jdbc:mariadb://localhost/",
                "root",
                null,
                "learn",
                "features"
        );
        featureMap.createDatabase();
        buildCPLFeatureMap(featureMap);
    }

    private static void buildCPLFeatureMap(FeatureMapMariaDB<SparseVector> featureMap) {
        Map<String, Integer> contexts;
        try {
            if (Files.exists(Paths.get(cplFeatureMapDirectory + "/contexts.bin"))) {
                contexts = readContexts();
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
                writeContexts(contexts);
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

    public static void writeContexts(Map<String, Integer> contexts) {
        try
        {
            FileOutputStream fos = new FileOutputStream(cplFeatureMapDirectory + "/contexts.bin");
            ObjectOutputStream oos = new ObjectOutputStream(fos);
            oos.writeObject(contexts);
            oos.close();
            fos.close();
        } catch (IOException e) {
            System.out.println("An exception was thrown while trying to write the CPL contexts map.");
            e.printStackTrace();
        }
    }

    @SuppressWarnings("unchecked")
    public static Map<String, Integer> readContexts() {
        Map<String, Integer> contexts = new HashMap<>();
        try
        {
            FileInputStream fis = new FileInputStream(cplFeatureMapDirectory + "/contexts.bin");
            ObjectInputStream ois = new ObjectInputStream(fis);
            contexts = (Map<String, Integer>) ois.readObject();
            ois.close();
            fis.close();
        } catch (IOException|ClassNotFoundException e) {
            System.out.println("An exception was thrown while trying to read the CPL contexts map.");
            e.printStackTrace();
        }
        return contexts;
    }

    public static void main(String[] args) {
        buildFeatureMap();
//        parseCategoriesFiles();
//        combineCategoriesFiles();
    }
}
