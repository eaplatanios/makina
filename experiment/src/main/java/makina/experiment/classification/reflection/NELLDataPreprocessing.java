package makina.experiment.classification.reflection;

import makina.experiment.data.DataSets;

import java.util.HashSet;
import java.util.Set;

/**
 * @author Emmanouil Antonios Platanios
 */
public class NELLDataPreprocessing {
    public static void main(String[] args) {
        if (args.length < 1)
            throw new IllegalArgumentException("A directory needs to be provided.");
        DataSets.NELLData data = DataSets.importNELLData(args[0]);
        int numberOfNounPhrases = data.numberOfNounPhrases(data.categories(), data.components());
        Set<String> filteredClassifierNames = new HashSet<>();
        filteredClassifierNames.add("CPL");
        filteredClassifierNames.add("SEAL");
        filteredClassifierNames.add("CMC");
        filteredClassifierNames.add("OE");
        int numberOfFilteredNounPhrases = data.numberOfNounPhrases(data.categories(), filteredClassifierNames);
        System.out.println("Number of \"complete\" noun phrases: " + numberOfNounPhrases);
        System.out.println("Number of filtered \"complete\" noun phrases: " + numberOfFilteredNounPhrases);
    }
}
