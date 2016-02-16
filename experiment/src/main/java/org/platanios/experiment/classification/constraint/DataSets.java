package org.platanios.experiment.classification.constraint;

import org.platanios.learn.classification.Label;
import org.platanios.learn.classification.constraint.Constraint;
import org.platanios.learn.classification.constraint.MutualExclusionConstraint;
import org.platanios.learn.classification.constraint.SubsumptionConstraint;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public class DataSets {
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
}
