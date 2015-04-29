package org.platanios.experiment.psl.parser;

import org.junit.Test;
import org.platanios.experiment.psl.CartesianProductIterator;
import org.platanios.experiment.psl.ProbabilisticSoftLogicPredicateManager;
import org.platanios.experiment.psl.ProbabilisticSoftLogicReader;
import org.platanios.learn.math.matrix.*;
import org.platanios.learn.optimization.ConsensusAlternatingDirectionsMethodOfMultipliersSolver;
import org.platanios.experiment.psl.SubProblemSolvers;
import org.platanios.learn.optimization.function.ProbabilisticSoftLogicFunction;

import java.io.*;
import java.util.*;
import java.util.zip.DataFormatException;

import static junit.framework.TestCase.fail;
import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertTrue;


/**
 * Created by Dan on 4/25/2015.
 * Test cases for parsing logic rules and populating solver
 */
public class LogicRuleParserTest {
    @Test
    public void testComplexPredicateParser() {

        ArrayList<AbstractMap.SimpleEntry<String, String>> inputOutputExpressions = new ArrayList<>(Arrays.asList(

                new AbstractMap.SimpleEntry<>(
                        " ( ( ( ( TRUSTS(A, B) & TRUSTS(B, C) ) & KNOWS(A, B) ) & KNOWS(A, C) ) & KNOWS(B, C) )",
                        "((((TRUSTS(A, B) & TRUSTS(B, C)) & KNOWS(A, B)) & KNOWS(A, C)) & KNOWS(B, C))"),

                new AbstractMap.SimpleEntry<>(
                        " ( ~( ( ~( TRUSTS(A, B) | TRUSTS(B, C) ) & KNOWS(A, B) ) | KNOWS(A, C) ) & KNOWS(B, C) )",
                        "(~((~(TRUSTS(A, B) | TRUSTS(B, C)) & KNOWS(A, B)) | KNOWS(A, C)) & KNOWS(B, C))")

        ));

        for (AbstractMap.SimpleEntry<String, String> inputOutputExpression : inputOutputExpressions) {

            String inputExpression = inputOutputExpression.getKey();
            String outputExpression = inputOutputExpression.getValue();

            try {
                PrattParserExpression expression = ComplexPredicateParser.parseRule(inputExpression);

                assertEquals(outputExpression, expression.toString());

            } catch (DataFormatException e) {
                fail(e.getMessage());
                System.out.println(e.getMessage());
                e.printStackTrace();
            }
        }
    }

    @Test
    public void testReadRules() {

        String[] lines = {
                "{constraint} ~( KNOWS(A, B) ) >> ~( TRUSTS(A, B) )",
                "{1.0909031372322524} ( ( ( ( TRUSTS(A, B) & TRUSTS(B, C) ) & KNOWS(A, B) ) & KNOWS(A, C) ) & KNOWS(B, C) ) >> TRUSTS(A, C) {squared}",
                "{1.1340018162428416} ( ( ( ( TRUSTS(A, B) & ~( TRUSTS(B, C) ) ) & KNOWS(A, B) ) & KNOWS(A, C) ) & KNOWS(B, C) ) >> ~( TRUSTS(A, C) ) {1}",
                "{1.1340018162428416} ( ( ( ~( ~TRUSTS(A, B) | ( TRUSTS(B, C) ) ) & KNOWS(A, B) ) & KNOWS(A, C) ) & KNOWS(B, C) ) >> ~( TRUSTS(A, C) )",
        };

        StringReader stringReader = new StringReader(String.join("\n", lines));
        BufferedReader reader = null;
        List<ProbabilisticSoftLogicReader.Rule> rules = null;

        try {

            reader = new BufferedReader(stringReader);
            rules = ProbabilisticSoftLogicReader.readRules(reader);

        } catch (IOException e) {
            fail(e.getMessage());
            System.out.println(e.getMessage());
            e.printStackTrace();
        } catch (DataFormatException e) {
            fail(e.getMessage());
            System.out.println(e.getMessage());
            e.printStackTrace();
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    fail(e.getMessage());
                    System.out.println(e.getMessage());
                    e.printStackTrace();
                }
            }
        }

        String expected = "{constraint} ~KNOWS(A, B) >> ~TRUSTS(A, B)";
        ProbabilisticSoftLogicReader.Rule rule = rules.get(0);

        assertEquals(expected, rule.toString());
        assertTrue("Constraint has weight", Double.isNaN(rule.Weight));
        assertTrue("Wrong number of body predicates", rule.Body.size() == 1);
        assertTrue("Wrong number of head predicates", rule.Head.size() == 1);
        assertEquals("~KNOWS(A, B)", rule.Body.get(0).toString());
        assertEquals("~TRUSTS(A, B)", rule.Head.get(0).toString());

        expected = "{1.0909031372322524} TRUSTS(A, B) & TRUSTS(B, C) & KNOWS(A, B) & KNOWS(A, C) & KNOWS(B, C) >> TRUSTS(A, C) {squared}";
        rule = rules.get(1);

        assertEquals(expected, rule.toString());
        assertTrue("Bad weight", Math.abs(1.0909031372322524 - rule.Weight) < 0.00001);
        assertEquals("Bad power", rule.Power, 2.0);
        assertTrue("Wrong number of body predicates", rule.Body.size() == 5);
        assertTrue("wrong number of head predicates", rule.Head.size() == 1);
        assertEquals("TRUSTS(A, B)", rule.Body.get(0).toString());
        assertEquals("TRUSTS(B, C)", rule.Body.get(1).toString());
        assertEquals("KNOWS(A, B)", rule.Body.get(2).toString());
        assertEquals("KNOWS(A, C)", rule.Body.get(3).toString());
        assertEquals("KNOWS(B, C)", rule.Body.get(4).toString());
        assertEquals("TRUSTS(A, C)", rule.Head.get(0).toString());

        expected = "{1.1340018162428416} TRUSTS(A, B) & ~TRUSTS(B, C) & KNOWS(A, B) & KNOWS(A, C) & KNOWS(B, C) >> ~TRUSTS(A, C) {1.0}";
        rule = rules.get(2);

        assertEquals(expected, rule.toString());
        assertTrue("Bad weight", Math.abs(1.1340018162428416 - rule.Weight) < 0.00001);
        assertEquals("Bad power", rule.Power, 1.0);
        assertTrue("Wrong number of body predicates", rule.Body.size() == 5);
        assertTrue("wrong number of head predicates", rule.Head.size() == 1);
        assertEquals("TRUSTS(A, B)", rule.Body.get(0).toString());
        assertEquals("~TRUSTS(B, C)", rule.Body.get(1).toString());
        assertEquals("KNOWS(A, B)", rule.Body.get(2).toString());
        assertEquals("KNOWS(A, C)", rule.Body.get(3).toString());
        assertEquals("KNOWS(B, C)", rule.Body.get(4).toString());
        assertEquals("~TRUSTS(A, C)", rule.Head.get(0).toString());

        expected = "{1.1340018162428416} TRUSTS(A, B) & ~TRUSTS(B, C) & KNOWS(A, B) & KNOWS(A, C) & KNOWS(B, C) >> ~TRUSTS(A, C) {1.0}";
        rule = rules.get(3);

        assertEquals(expected, rule.toString());
        assertTrue("Bad weight", Math.abs(1.1340018162428416 - rule.Weight) < 0.00001);
        assertEquals("Bad power", rule.Power, 1.0);
        assertTrue("Wrong number of body predicates", rule.Body.size() == 5);
        assertTrue("wrong number of head predicates", rule.Head.size() == 1);
        assertEquals("TRUSTS(A, B)", rule.Body.get(0).toString());
        assertEquals("~TRUSTS(B, C)", rule.Body.get(1).toString());
        assertEquals("KNOWS(A, B)", rule.Body.get(2).toString());
        assertEquals("KNOWS(A, C)", rule.Body.get(3).toString());
        assertEquals("KNOWS(B, C)", rule.Body.get(4).toString());
        assertEquals("~TRUSTS(A, C)", rule.Head.get(0).toString());

    }

    @Test
    public void testGroundings() {

        String[] bogusLines1 = {
                "1\t2\t1",
                "2\t3\t5.0",
                "1\t2\t4.7",
                "3\t4\t1",
                "2\t4\t0"
        };

        ProbabilisticSoftLogicPredicateManager predicateManager = new ProbabilisticSoftLogicPredicateManager();
        StringReader stringReader = new StringReader(String.join("\n", bogusLines1));
        BufferedReader reader = null;

        boolean gotFormatException = false;
        try {

            reader = new BufferedReader(stringReader);
            ProbabilisticSoftLogicReader.readGroundingsAndAddToManager(
                predicateManager,
                "KNOWS",
                reader);

        } catch (IOException e) {
            fail(e.getMessage());
            System.out.println(e.getMessage());
            e.printStackTrace();
        } catch (DataFormatException e) {
            gotFormatException = true;
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    fail(e.getMessage());
                    System.out.println(e.getMessage());
                    e.printStackTrace();
                }
            }
        }

        assertTrue("Didn't get format exception from bad lines", gotFormatException);

        String[] bogusLines2 = {
                "1\t2\t1",
                "2\t3",
                "1\t2\t1",
                "3\t4\t1",
                "2\t4\t0"
        };

        predicateManager = new ProbabilisticSoftLogicPredicateManager();
        stringReader = new StringReader(String.join("\n", bogusLines2));
        reader = null;
        gotFormatException = false;

        try {

            reader = new BufferedReader(stringReader);
            ProbabilisticSoftLogicReader.readGroundingsAndAddToManager(
                    predicateManager,
                    "KNOWS",
                    reader);

        } catch (IOException e) {
            fail(e.getMessage());
            System.out.println(e.getMessage());
            e.printStackTrace();
        } catch (DataFormatException e) {
            gotFormatException = true;
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    fail(e.getMessage());
                    System.out.println(e.getMessage());
                    e.printStackTrace();
                }
            }
        }

        assertTrue("Didn't get format exception from bad lines", gotFormatException);

        String[] lines = {
                "1\t2\t1",
                "2\t3\t5.0",
                "1\t2\t1",
                "3\t4\t1",
                "2\t4\t0"
        };

        String[] expected = {
                "KNOWS(1, 2)",
                "KNOWS(2, 3)",
                "KNOWS(3, 4)",
                "KNOWS(2, 4)"
        };

        boolean[] expectedFound = { false, false, false, false };
        double[] expectedWeights = { 1, 5, 1, 0 };

        String[] expectedFirstArg = { "1", "2", "3" };
        boolean[] expectedFirstArgFound = { false, false, false };
        String[] expectedSecondArg = { "2", "3", "4" };
        boolean[] expectedSecondArgFound = { false, false, false };

        predicateManager = new ProbabilisticSoftLogicPredicateManager();
        stringReader = new StringReader(String.join("\n", lines));
        reader = null;

        List<Integer> predicateIds = null;

        try {

            reader = new BufferedReader(stringReader);
            predicateIds = ProbabilisticSoftLogicReader.readGroundingsAndAddToManager(
                predicateManager,
                "KNOWS",
                reader);

        } catch (IOException e) {
            fail(e.getMessage());
            System.out.println(e.getMessage());
            e.printStackTrace();
        } catch (DataFormatException e) {
            fail(e.getMessage());
            System.out.println(e.getMessage());
            e.printStackTrace();
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    fail(e.getMessage());
                    System.out.println(e.getMessage());
                    e.printStackTrace();
                }
            }
        }

        assertTrue("Wrong number of ids returned", predicateIds.size() == expected.length);

        List<Integer> idsAgain = predicateManager.getIdsForPredicateName("KNOWS");
        assertTrue("Wrong number of ids when asked", idsAgain.size() == predicateIds.size());
        for (int idAgain : idsAgain) {
            boolean isFound = false;
            for (int id : predicateIds) {
                if (idAgain == id) {
                    isFound = true;
                    break;
                }
            }
            assertTrue("returned ids and asked ids not same", isFound);
        }

        for (int id : predicateIds) {

            boolean isFound = false;
            for (int idAgain : idsAgain) {
                if (idAgain == id) {
                    isFound = true;
                    break;
                }
            }
            assertTrue("returned ids and asked ids not same", isFound);

            ProbabilisticSoftLogicReader.Predicate predicate = predicateManager.getPredicateFromId(id);
            int idReflex = predicateManager.getIdForPredicate(predicate);
            assertEquals("Round trip failed", idReflex, id);

            for (int i = 0; i < expected.length; ++i) {
                if (predicate.toString().equals(expected[i])) {
                    assertTrue("Multiple ids match same predicate", !expectedFound[i]);
                    assertEquals("Expected weight not found", expectedWeights[i], predicateManager.getObservedWeight(id));
                    expectedFound[i] = true;
                }
            }

        }

        for (int i = 0; i < expected.length; ++i) {
            assertTrue("Not all expected predicates found", expectedFound[i]);
        }

        Iterator<String> firstArgIterator = predicateManager.getArgumentGroundings("KNOWS", 0);
        while (firstArgIterator.hasNext()) {
            String arg = firstArgIterator.next();
            for (int i = 0; i < expectedFirstArg.length; ++i) {
                if (arg.equals(expectedFirstArg[i])) {
                    assertTrue("Multiple first args match same arg", !expectedFirstArgFound[i]);
                    expectedFirstArgFound[i] = true;
                }
            }
        }

        for (int i = 0; i < expectedFirstArgFound.length; ++i) {
            assertTrue("Some expected first args not found", expectedFirstArgFound[i]);
        }

        Iterator<String> secondArgIterator = predicateManager.getArgumentGroundings("KNOWS", 1);
        while (secondArgIterator.hasNext()) {
            String arg = secondArgIterator.next();
            for (int i = 0; i < expectedSecondArg.length; ++i) {
                if (arg.equals(expectedSecondArg[i])) {
                    assertTrue("Multiple first args match same arg", !expectedSecondArgFound[i]);
                    expectedSecondArgFound[i] = true;
                }
            }
        }

        for (int i = 0; i < expectedSecondArgFound.length; ++i) {
            assertTrue("Some expected first args not found", expectedSecondArgFound[i]);
        }

    }

    @Test
    public void testCartesionProductIterator() {

        List<List<Integer>> lists =
            Arrays.asList(
                    Arrays.asList(1, 2, 3),
                    Arrays.asList(3, 4),
                    Arrays.asList(7, 8)
            );

        List<List<Integer>> expected =
            Arrays.asList(
                    Arrays.asList(1, 3, 7),
                    Arrays.asList(1, 3, 8),
                    Arrays.asList(1, 4, 7),
                    Arrays.asList(1, 4, 8),
                    Arrays.asList(2, 3, 7),
                    Arrays.asList(2, 3, 8),
                    Arrays.asList(2, 4, 7),
                    Arrays.asList(2, 4, 8),
                    Arrays.asList(3, 3, 7),
                    Arrays.asList(3, 3, 8),
                    Arrays.asList(3, 4, 7),
                    Arrays.asList(3, 4, 8)
            );

        ArrayList<List<Integer>> result = new ArrayList<>();
        CartesianProductIterator<Integer> cartesianProduct = new CartesianProductIterator<>(lists);
        for ( List<Integer> combination : cartesianProduct ) {
            result.add(combination);
        }

        Comparator<List<Integer>> comparator = new Comparator<List<Integer>>() {

            @Override
            public int compare(List<Integer> l1, List<Integer> l2) {
                if (l1.size() < l2.size()) {
                    return -1;
                } else if(l1.size() > l2.size()) {
                    return 1;
                }
                for (int i = 0; i < l1.size(); ++i) {
                    if (l1.get(i) < l2.get(i)) {
                        return -1;
                    } else if (l1.get(i) > l2.get(i)) {
                        return 1;
                    }
                }
                return 0;
            }
        };

        result.sort(comparator);
        assertEquals(expected.size(), result.size());
        for (int i = 0; i < result.size(); ++i) {
            assertEquals(expected.get(i).size(), result.get(i).size());
            for (int j = 0; j < expected.get(i).size(); ++j) {
                assertEquals(expected.get(i).get(j), result.get(i).get(j));
            }
        }

    }

    @Test
    public void testEndToEnd() {

        InputStream modelStream = LogicRuleParserTest.class.getResourceAsStream("./model.txt");
        InputStream knowsStream = LogicRuleParserTest.class.getResourceAsStream("./knows.txt");
        InputStream trustTrainStream = LogicRuleParserTest.class.getResourceAsStream("./trust_train.txt");
        InputStream trustTestStream = LogicRuleParserTest.class.getResourceAsStream("./trust_test.txt");

        BufferedReader modelReader = null;
        BufferedReader knowsReader = null;
        BufferedReader trustTrainReader = null;
        BufferedReader trustTestReader = null;

        List<ProbabilisticSoftLogicReader.Rule> rules = null;
        ProbabilisticSoftLogicPredicateManager trainPredicateManager = new ProbabilisticSoftLogicPredicateManager();
        ProbabilisticSoftLogicPredicateManager testPredicateManager = new ProbabilisticSoftLogicPredicateManager();

        try {

            InputStreamReader streamReader = new InputStreamReader(modelStream);
            modelReader = new BufferedReader(streamReader);
            rules = ProbabilisticSoftLogicReader.readRules(modelReader);

            streamReader = new InputStreamReader(knowsStream);
            knowsReader = new BufferedReader(streamReader);
            ProbabilisticSoftLogicReader.readGroundingsAndAddToManager(trainPredicateManager, "KNOWS", knowsReader);

            streamReader = new InputStreamReader(trustTrainStream);
            trustTrainReader = new BufferedReader(streamReader);
            ProbabilisticSoftLogicReader.readGroundingsAndAddToManager(trainPredicateManager, "TRUST", trustTrainReader);

            streamReader = new InputStreamReader(trustTestStream);
            trustTestReader = new BufferedReader(streamReader);
            ProbabilisticSoftLogicReader.readGroundingsAndAddToManager(testPredicateManager, "TRUST", trustTestReader);

        } catch (IOException e) {
            fail(e.getMessage());
            System.out.println(e.getMessage());
            e.printStackTrace();
        } catch (DataFormatException e) {
            fail(e.getMessage());
            System.out.println(e.getMessage());
            e.printStackTrace();
        } finally {

            if (modelReader != null) {
                try {
                    modelReader.close();
                } catch (IOException e) {
                    fail(e.getMessage());
                    System.out.println(e.getMessage());
                    e.printStackTrace();
                }
            }

            if (knowsReader != null) {
                try {
                    knowsReader.close();
                } catch (IOException e) {
                    fail(e.getMessage());
                    System.out.println(e.getMessage());
                    e.printStackTrace();
                }
            }

            if (trustTrainReader != null) {
                try {
                    trustTrainReader.close();
                } catch (IOException e) {
                    fail(e.getMessage());
                    System.out.println(e.getMessage());
                    e.printStackTrace();
                }
            }

            if (trustTestReader != null) {
                try {
                    trustTestReader.close();
                } catch (IOException e) {
                    fail(e.getMessage());
                    System.out.println(e.getMessage());
                    e.printStackTrace();
                }
            }
        }

        ProbabilisticSoftLogicFunction pslFunction = new ProbabilisticSoftLogicFunction.Builder(2)
                .addRule(new int[]{0}, new int[]{2}, new boolean[]{false}, new boolean[]{false}, new int[]{2}, new double[]{1}, 1, 1)           // C -> A  =>  1 - A
                .addRule(new int[]{1}, new int[]{0}, new boolean[]{false}, new boolean[]{false}, new int[]{}, new double[]{}, 1, 1)             // A -> B  =>  A - B
                .addRule(new int[]{1}, new int[]{3}, new boolean[]{false}, new boolean[]{false}, new int[]{3}, new double[]{0}, 1, 1)           // D -> B  =>  -B
                .build();

        ConsensusAlternatingDirectionsMethodOfMultipliersSolver consensusAlternatingDirectionsMethodOfMultipliersSolver =
                new ConsensusAlternatingDirectionsMethodOfMultipliersSolver.Builder(pslFunction,
                        Vectors.dense(new double[]{0.5, 0.5}))
                        .subProblemSolver(SubProblemSolvers::solveProbabilisticSoftLogicSubProblem)
                        .checkForObjectiveConvergence(false)
                        .checkForGradientConvergence(false)
                        .loggingLevel(5)
                        .build();

        org.platanios.learn.math.matrix.Vector result = consensusAlternatingDirectionsMethodOfMultipliersSolver.solve();
        System.out.println(result.get(0));
        System.out.println(result.get(1));
        System.out.println('\n');

        for (ProbabilisticSoftLogicReader.Rule rule : rules) {

            rule.addGroundingsToBuilder();

        }

    }

}