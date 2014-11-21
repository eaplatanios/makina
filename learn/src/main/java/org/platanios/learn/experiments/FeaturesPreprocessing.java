package org.platanios.learn.experiments;

import com.google.common.collect.Maps;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * @author Emmanouil Antonios Platanios
 */
public class FeaturesPreprocessing {
    private static final String[] categories = { "animal" };
    private static final String labeledNounPhrasesDirectory = "/Volumes/Macintosh HD/Users/Anthony/Development/NELL/data/server/NELL.08m.880.mttrain.csv";
    private static final String featureMapsDirectory = "/Volumes/Macintosh HD/Users/Anthony/Development/NELL/data/features";
    private static final String trainingDataDirectory = "/Volumes/Macintosh HD/Users/Anthony/Development/NELL/data/varias_data/trainData";

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

    public static void main(String[] args) {
        parseCategoriesFiles();
        combineCategoriesFiles();
    }
}
