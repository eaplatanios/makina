package org.platanios.experiment.classification.reflection;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

/**
 * @author Emmanouil Antonios Platanios
 */
public class NELLDataPreprocessing {
    /**
     *
     * @param   directory
     * @return              A {@code Map<String, Map<String, Map<String, Double>>>} indexed by noun-phrase, category,
     *                      and classifier and containing the corresponding probabilities or {@code null} if there is no
     *                      such probability.
     */
    public static Data aggregatePredictions(String directory) {
        Data data = new Data();
        File[] directoryFiles = new File(directory).listFiles();
        if (directoryFiles == null)
            throw new IllegalArgumentException("The provided directory does not contain any files.");
        Arrays.asList(directoryFiles)
                .stream()
                .filter(file -> file.getName().endsWith(".all_predictions.txt"))
                .forEach(file -> {
                    try {
                        String categoryName = file.getName().split("[.]")[0];
                        data.categoryNames.add(categoryName);
                        Map<Integer, String> classifierNames = new HashMap<>();
                        Files.newBufferedReader(Paths.get(file.getPath())).lines().forEach(line -> {
                            if (line.startsWith("Input")) {
                                String[] lineParts = line.split(",");
                                for (int index = 1; index < lineParts.length; index++) {
                                    classifierNames.put(index, lineParts[index]);
                                    data.classifierNames.add(lineParts[index]);
                                }
                            } else {
                                String[] lineParts = line.split("\t");
                                if (!data.predictions.containsKey(lineParts[0]))
                                    data.predictions.put(lineParts[0], new HashMap<>());
                                if (!data.predictions.get(lineParts[0]).containsKey(categoryName))
                                    data.predictions.get(lineParts[0]).put(categoryName, new HashMap<>());
                                for (int index = 1; index < lineParts.length; index++)
                                    if (!lineParts[index].equals("-") && !lineParts[index].equals("NaN"))
                                        data.predictions.get(lineParts[0]).get(categoryName)
                                                .put(classifierNames.get(index), Double.valueOf(lineParts[index]));
                            }
                        });
                    } catch (IOException e) {
                        throw new IllegalStateException("An exception occured while processing a predictions file.");
                    }
                });
        return data;
    }

    public static int getNumberOfNounPhrases(Map<String, Map<String, Map<String, Double>>> predictions,
                                             Set<String> categoryNames,
                                             Set<String> classifierNames) {
        int numberOfNounPhrases = 0;
        for (Map.Entry<String, Map<String, Map<String, Double>>> nounPhraseEntry : predictions.entrySet()) {
            boolean missingCategoryOrClassifier = false;
            for (String categoryName : categoryNames)
                if (!nounPhraseEntry.getValue().containsKey(categoryName)
                        || !nounPhraseEntry.getValue().get(categoryName).keySet().containsAll(classifierNames)) {
                    missingCategoryOrClassifier = true;
                    break;
                }
            if (!missingCategoryOrClassifier)
                numberOfNounPhrases++;
        }
        return numberOfNounPhrases;
    }

    public static void main(String[] args) {
        if (args.length < 1)
            throw new IllegalArgumentException("A directory needs to be provided.");
        Data data = aggregatePredictions(args[0]);
        int numberOfNounPhrases = getNumberOfNounPhrases(data.predictions, data.categoryNames, data.classifierNames);
        Set<String> filteredClassifierNames = new HashSet<>();
        filteredClassifierNames.add("CPL");
        filteredClassifierNames.add("SEAL");
        filteredClassifierNames.add("CMC");
        filteredClassifierNames.add("OE");
        int numberOfFilteredNounPhrases = getNumberOfNounPhrases(data.predictions, data.categoryNames, filteredClassifierNames);
        System.out.println("Number of \"complete\" noun phrases: " + numberOfNounPhrases);
        System.out.println("Number of filtered \"complete\" noun phrases: " + numberOfFilteredNounPhrases);
    }

    public static class Data {
        private final Map<String, Map<String, Map<String, Double>>> predictions = new HashMap<>();
        private final Set<String> categoryNames = new HashSet<>();
        private final Set<String> classifierNames = new HashSet<>();

        public Data() { }

        public Map<String, Map<String, Map<String, Double>>> getPredictions() {
            return predictions;
        }

        public Set<String> getNounPhrases() {
            return predictions.keySet();
        }

        public Set<String> getCategoryNames() {
            return categoryNames;
        }

        public Set<String> getClassifierNames() {
            return classifierNames;
        }
    }
}
