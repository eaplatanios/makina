package org.platanios.experiment.psl.parser;

import org.junit.Test;
import org.platanios.learn.logic.InMemoryLogicManager;
import org.platanios.learn.logic.ProbabilisticSoftLogic;
import org.platanios.experiment.psl.ProbabilisticSoftLogicReader;
import org.platanios.learn.logic.LogicManager;
import org.platanios.learn.logic.LukasiewiczLogic;
import org.platanios.learn.logic.formula.EntityType;
import org.platanios.learn.logic.formula.Variable;
import org.platanios.learn.logic.grounding.GroundPredicate;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Stream;
import java.util.zip.DataFormatException;

import static junit.framework.TestCase.fail;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ProbabilisticSoftLogicTest {
    @Test
    public void testEndToEnd() {
        String experimentName = "uci_trust";

        LogicManager logicManager = new InMemoryLogicManager(new LukasiewiczLogic());

        Set<Long> personValues = new HashSet<>();
        try {
            Stream<String> lines = Files.lines(Paths.get(ProbabilisticSoftLogicTest.class.getResource("../" + experimentName + "/knows.txt").getPath()));
            lines.forEach(line -> {
                String[] lineParts = line.split("\t");
                for (String linePart : lineParts)
                    personValues.add(Long.parseLong(linePart.trim()));
            });
            lines = Files.lines(Paths.get(ProbabilisticSoftLogicTest.class.getResource("../" + experimentName + "/train.txt").getPath()));
            lines.forEach(line -> {
                String[] lineParts = line.split("\t");
                for (int partIndex = 0; partIndex < lineParts.length - 1; partIndex++)
                    personValues.add(Long.parseLong(lineParts[partIndex].trim()));
            });
        } catch (IOException ignored) { }

        EntityType personType = logicManager.addEntityType("{person}", new HashSet<>(personValues));
        List<Variable> variables = new ArrayList<>();
        variables.add(new Variable(0, "A", personType));
        variables.add(new Variable(1, "B", personType));
        variables.add(new Variable(2, "C", personType));
        variables.add(new Variable(3, "D", personType));
        variables.add(new Variable(4, "E", personType));
        List<EntityType> argumentTypes = new ArrayList<>(2);
        argumentTypes.add(personType);
        argumentTypes.add(personType);
        logicManager.addPredicate("KNOWS", argumentTypes, false);
        logicManager.addPredicate("TRUSTS", argumentTypes, false);

        InputStream modelStream = ProbabilisticSoftLogicTest.class.getResourceAsStream("../" + experimentName + "/model.txt");
        InputStream knowsStream = ProbabilisticSoftLogicTest.class.getResourceAsStream("../" + experimentName + "/knows.txt");
        InputStream trustTrainStream = ProbabilisticSoftLogicTest.class.getResourceAsStream("../" + experimentName + "/train.txt");
        InputStream trustTestStream = ProbabilisticSoftLogicTest.class.getResourceAsStream("../" + experimentName + "/test.txt");

        List<ProbabilisticSoftLogic.LogicRule> logicRules = null;

        try (
                BufferedReader modelReader = new BufferedReader(new InputStreamReader(modelStream));
                BufferedReader knowsReader = new BufferedReader(new InputStreamReader(knowsStream));
                BufferedReader trustTrainReader = new BufferedReader(new InputStreamReader(trustTrainStream));
                BufferedReader trustTestReader = trustTestStream == null ? null : new BufferedReader(new InputStreamReader(trustTestStream))) {

            logicRules = ProbabilisticSoftLogicReader.readRules(modelReader, logicManager, variables);

            ProbabilisticSoftLogicReader.readGroundingsAndAddToManager(logicManager, "KNOWS", true, knowsReader);

            ProbabilisticSoftLogicReader.readGroundingsAndAddToManager(logicManager, "TRUSTS", false, trustTrainReader);

//                    ProbabilisticSoftLogicReader.readGroundingsAndAddToManager(testPredicateManager, logicManager, personType, "TRUSTS", false, false, trustTestReader);

        } catch (IOException | DataFormatException e) {
            fail(e.getMessage());
            System.out.println(e.getMessage());
            e.printStackTrace();
        }

//        int[] observedIndexes = new int[(int) logicManager.getNumberOfGroundPredicates()];
//        double[] observedWeights = new double[(int) logicManager.getNumberOfGroundPredicates()];
//        List<GroundPredicate> groundPredicates = logicManager.getGroundPredicates();
//        for (int index = 0; index < logicManager.getNumberOfGroundPredicates(); index++) {
//            observedIndexes[index] = (int) groundPredicates.get(index).name();
//            observedWeights[index] = (double) groundPredicates.get(index).getValue();
//        }

        ProbabilisticSoftLogic problem =
                new ProbabilisticSoftLogic.Builder(logicManager)
                        .addLogicRules(logicRules)
                        .build();
        List<GroundPredicate> result = problem.solve();

        List<GroundPredicate> filteredResults = new ArrayList<>();

        result.stream()
                .filter(groundPredicate -> groundPredicate.getValue() > 0.01)
                .forEach(filteredResults::add);

        System.out.println(result.get(0));
        System.out.println(result.get(1));

        System.out.println("Done\n");

    }
}
