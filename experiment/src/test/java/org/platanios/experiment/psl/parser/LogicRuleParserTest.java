package org.platanios.experiment.psl.parser;

import com.google.common.collect.ImmutableList;
import org.junit.Test;
import org.platanios.experiment.psl.CartesianProductIterator;
import org.platanios.experiment.psl.ProbabilisticSoftLogicPredicateManager;
import org.platanios.experiment.psl.ProbabilisticSoftLogicProblem;
import org.platanios.experiment.psl.ProbabilisticSoftLogicReader;
import org.platanios.learn.optimization.ConsensusAlternatingDirectionsMethodOfMultipliersSolver;

import java.io.*;
import java.net.URISyntaxException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.zip.DataFormatException;

import static junit.framework.TestCase.*;


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

        List<ProbabilisticSoftLogicProblem.Rule> rules = null;

        try (BufferedReader reader = new BufferedReader(new StringReader(String.join("\n", lines)))) {

            rules = ProbabilisticSoftLogicReader.readRules(reader);

        } catch (IOException|DataFormatException e) {
            fail(e.getMessage());
            System.out.println(e.getMessage());
            e.printStackTrace();
        }

        String expected = "{constraint} ~KNOWS(A, B) >> ~TRUSTS(A, B)";
        ProbabilisticSoftLogicProblem.Rule rule = rules.get(0);

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

        boolean gotFormatException = false;
        try (BufferedReader reader = new BufferedReader(new StringReader(String.join("\n", bogusLines1)))) {

            ProbabilisticSoftLogicReader.readGroundingsAndAddToManager(
                predicateManager,
                "KNOWS",
                true,
                false,
                reader);

        } catch (IOException|DataFormatException e) {
            fail(e.getMessage());
            System.out.println(e.getMessage());
            e.printStackTrace();
        } catch (UnsupportedOperationException e) {
            gotFormatException = true;
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
        gotFormatException = false;

        try (BufferedReader reader = new BufferedReader(new StringReader(String.join("\n", bogusLines2)))) {

            ProbabilisticSoftLogicReader.readGroundingsAndAddToManager(
                    predicateManager,
                    "KNOWS",
                    true,
                    false,
                    reader);

        } catch (IOException e) {
            fail(e.getMessage());
            System.out.println(e.getMessage());
            e.printStackTrace();
        } catch (DataFormatException e) {
            gotFormatException = true;
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

        Set<Integer> predicateIds = null;

        try (BufferedReader reader = new BufferedReader(new StringReader(String.join("\n", lines)))) {

            ProbabilisticSoftLogicReader.readGroundingsAndAddToManager(
                predicateManager,
                "KNOWS",
                true,
                false,
                reader);
            predicateIds = predicateManager.getIdsForPredicateName("KNOWS");

        } catch (IOException|DataFormatException e) {
            fail(e.getMessage());
            System.out.println(e.getMessage());
            e.printStackTrace();
        }

        assertTrue("Wrong number of ids returned", predicateIds.size() == expected.length);

        for (int id : predicateIds) {

            ProbabilisticSoftLogicProblem.Predicate predicate = predicateManager.getPredicateFromId(id);
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

        Set<String> firstArgGroundings = predicateManager.getArgumentGroundings("KNOWS", 0);
        for (String arg : firstArgGroundings) {
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

        Set<String> secondArgGroundings = predicateManager.getArgumentGroundings("KNOWS", 1);
        for (String arg : secondArgGroundings) {
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
    public void testCartesianProductIterator() {

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
    public void testNonSymmetric() {

        ProbabilisticSoftLogicProblem.Predicate knowsAB =
                new ProbabilisticSoftLogicProblem.Predicate("KNOWS", ImmutableList.of("A", "B"), false);

        ProbabilisticSoftLogicProblem.Predicate knowsBC =
                new ProbabilisticSoftLogicProblem.Predicate(knowsAB.Name, ImmutableList.of("B", "C"), false);

        ProbabilisticSoftLogicProblem.Predicate knowsAC =
                new ProbabilisticSoftLogicProblem.Predicate(knowsAB.Name, ImmutableList.of("A", "C"), false);

        ProbabilisticSoftLogicProblem.Predicate nonSymAB =
                new ProbabilisticSoftLogicProblem.Predicate("#NONSYMMETRIC", ImmutableList.of("A", "B"), false);

        ProbabilisticSoftLogicProblem.Predicate nonSymBC =
                new ProbabilisticSoftLogicProblem.Predicate("#NONSYMMETRIC", ImmutableList.of("B", "C"), false);

        ProbabilisticSoftLogicProblem.Predicate negNonSymAB =
                new ProbabilisticSoftLogicProblem.Predicate(nonSymAB.Name, nonSymAB.Arguments, true);

        boolean isException = false;
        try {
            // "{1.5} KNOWS(A, B) & KNOWS(B, C) >> KNOWS(A, C) | #NONSYMMETRIC(A, B)"
            new ProbabilisticSoftLogicProblem.Rule(1.5, 1, ImmutableList.of(knowsAC, nonSymAB), ImmutableList.of(knowsAB, knowsBC));
        } catch (UnsupportedOperationException e) {
            isException = true;
        }

        assertTrue("No exception for nonsymmetric in head", isException);

        isException = false;
        try {
            // "{1.5} KNOWS(A, B) & KNOWS(B, C) & ~#NONSYMMETRIC(A, B) >> KNOWS(A, C)"
            new ProbabilisticSoftLogicProblem.Rule(1.5, 1, ImmutableList.of(knowsAC), ImmutableList.of(knowsAB, knowsBC, negNonSymAB));
        } catch (UnsupportedOperationException e) {
            isException = true;
        }

        assertTrue("No exception for negative nonsymmetric", isException);

        // "{1.5} KNOWS(A, B) & KNOWS(B, C) >> KNOWS(A, C)"
        ProbabilisticSoftLogicProblem.Rule rule =
                new ProbabilisticSoftLogicProblem.Rule(1.5, 1, ImmutableList.of(knowsAC), ImmutableList.of(knowsAB, knowsBC));

        // "{1.5} KNOWS(A, B) & KNOWS(B, C) & #NONSYMMETRIC(A, B) & #NONSYMMETRIC(B, C) >> KNOWS(A, C)"
        ProbabilisticSoftLogicProblem.Rule nonSymRule =
                new ProbabilisticSoftLogicProblem.Rule(1.5, 1, ImmutableList.of(knowsAC), ImmutableList.of(knowsAB, knowsBC, nonSymAB, nonSymBC));

        ProbabilisticSoftLogicProblem.Predicate knows12 =
                new ProbabilisticSoftLogicProblem.Predicate(knowsAB.Name, ImmutableList.of("1", "2"), false);
        ProbabilisticSoftLogicProblem.Predicate knows23 =
                new ProbabilisticSoftLogicProblem.Predicate(knowsAB.Name, ImmutableList.of("2", "3"), false);
        ProbabilisticSoftLogicProblem.Predicate knows34 =
                new ProbabilisticSoftLogicProblem.Predicate(knowsAB.Name, ImmutableList.of("3", "4"), false);

        ProbabilisticSoftLogicPredicateManager predicateManager = new ProbabilisticSoftLogicPredicateManager();
        ProbabilisticSoftLogicPredicateManager nonSymPredicateManager = new ProbabilisticSoftLogicPredicateManager();
        predicateManager.getOrAddPredicate(knows12, 0.5);
        nonSymPredicateManager.getOrAddPredicate(knows12, 0.5);
        predicateManager.getOrAddPredicate(knows23, 0.1);
        nonSymPredicateManager.getOrAddPredicate(knows23, 0.1);
        predicateManager.getOrAddPredicate(knows34, 0.7);
        nonSymPredicateManager.getOrAddPredicate(knows34, 0.7);

        ProbabilisticSoftLogicPredicateManager.IdWeights observedIdsAndWeights =
                predicateManager.getAllObservedWeights();

        ProbabilisticSoftLogicProblem.Builder builder =
            new ProbabilisticSoftLogicProblem.Builder(observedIdsAndWeights.Ids, observedIdsAndWeights.Weights, 100);

        ProbabilisticSoftLogicProblem.Builder nonSymBuilder =
                new ProbabilisticSoftLogicProblem.Builder(observedIdsAndWeights.Ids, observedIdsAndWeights.Weights, 100);

        ProbabilisticSoftLogicProblem.Rule.addGroundingsToBuilder(
                Arrays.asList(rule), builder, predicateManager, ProbabilisticSoftLogicProblem.GroundingMode.AllPossible);
        ProbabilisticSoftLogicProblem.Rule.addGroundingsToBuilder(
                Arrays.asList(rule), nonSymBuilder, nonSymPredicateManager, ProbabilisticSoftLogicProblem.GroundingMode.AllPossible);

        Set<String> expectedNormal = new HashSet<>(Arrays.asList(
            "KNOWS(1, 1)",
            "KNOWS(1, 2)",
            "KNOWS(1, 3)",
            "KNOWS(1, 4)",
            "KNOWS(2, 1)",
            "KNOWS(2, 2)",
            "KNOWS(2, 3)",
            "KNOWS(2, 4)",
            "KNOWS(3, 1)",
            "KNOWS(3, 2)",
            "KNOWS(3, 3)",
            "KNOWS(3, 4)",
            "KNOWS(4, 1)",
            "KNOWS(4, 2)",
            "KNOWS(4, 3)",
            "KNOWS(4, 4)"
        ));

        Set<String> expectedNonSym = new HashSet<>(Arrays.asList(
            "KNOWS(1, 2)",
            "KNOWS(1, 3)",
            "KNOWS(1, 4)",
            "KNOWS(2, 3)",
            "KNOWS(2, 4)",
            "KNOWS(3, 4)"
        ));

        Set<Integer> predicateIds = predicateManager.getIdsForPredicateName("KNOWS");
        assertTrue(predicateIds != null);
        assertTrue(predicateIds.size() == expectedNormal.size());

        Set<Integer> nonSymPredicateIds = nonSymPredicateManager.getIdsForPredicateName("KNOWS");
        assertTrue(nonSymPredicateIds != null);
        assertTrue(nonSymPredicateIds.size() == expectedNonSym.size());

        for (int predicateId : predicateIds) {
            ProbabilisticSoftLogicProblem.Predicate predicate = predicateManager.getPredicateFromId(predicateId);
            assertTrue(predicate != null);
            assertTrue(expectedNormal.contains(predicate.toString()));
        }

        for (int predicateId : nonSymPredicateIds) {
            ProbabilisticSoftLogicProblem.Predicate predicate = nonSymPredicateManager.getPredicateFromId(predicateId);
            assertTrue(predicate != null);
            assertTrue(expectedNonSym.contains(predicate.toString()));
        }

    }

    @Test
    public void testClosed() {

        ProbabilisticSoftLogicProblem.Predicate knowsAB =
                new ProbabilisticSoftLogicProblem.Predicate("KNOWS", ImmutableList.of("A", "B"), false);

        ProbabilisticSoftLogicProblem.Predicate knowsBC =
                new ProbabilisticSoftLogicProblem.Predicate(knowsAB.Name, ImmutableList.of("B", "C"), false);

        ProbabilisticSoftLogicProblem.Predicate trustsAC =
                new ProbabilisticSoftLogicProblem.Predicate("TRUSTS", ImmutableList.of("A", "C"), false);

        // "{1.5} KNOWS(A, B) & KNOWS(B, C) >> TRUSTS(A, C)"
        ProbabilisticSoftLogicProblem.Rule rule =
                new ProbabilisticSoftLogicProblem.Rule(1.5, 1, ImmutableList.of(trustsAC), ImmutableList.of(knowsAB, knowsBC));

        ProbabilisticSoftLogicProblem.Predicate knows12 =
                new ProbabilisticSoftLogicProblem.Predicate(knowsAB.Name, ImmutableList.of("1", "2"), false);
        ProbabilisticSoftLogicProblem.Predicate knows23 =
                new ProbabilisticSoftLogicProblem.Predicate(knowsAB.Name, ImmutableList.of("2", "3"), false);
        ProbabilisticSoftLogicProblem.Predicate knows34 =
                new ProbabilisticSoftLogicProblem.Predicate(knowsAB.Name, ImmutableList.of("3", "4"), false);

        ProbabilisticSoftLogicPredicateManager predicateManager = new ProbabilisticSoftLogicPredicateManager();
        ProbabilisticSoftLogicPredicateManager closedPredicateManager = new ProbabilisticSoftLogicPredicateManager();
        predicateManager.getOrAddPredicate(knows12, 0.5);
        closedPredicateManager.getOrAddPredicate(knows12, 0.5);
        predicateManager.getOrAddPredicate(knows23, 0.1);
        closedPredicateManager.getOrAddPredicate(knows23, 0.1);
        predicateManager.getOrAddPredicate(knows34, 0.7);
        closedPredicateManager.getOrAddPredicate(knows34, 0.7);

        closedPredicateManager.closePredicate("KNOWS");

        ProbabilisticSoftLogicPredicateManager.IdWeights observedIdsAndWeights =
                predicateManager.getAllObservedWeights();

        ProbabilisticSoftLogicProblem.Builder builder =
                new ProbabilisticSoftLogicProblem.Builder(observedIdsAndWeights.Ids, observedIdsAndWeights.Weights, 100);

        ProbabilisticSoftLogicProblem.Builder closedBuilder =
                new ProbabilisticSoftLogicProblem.Builder(observedIdsAndWeights.Ids, observedIdsAndWeights.Weights, 100);

        ProbabilisticSoftLogicProblem.Rule.addGroundingsToBuilder(
                Arrays.asList(rule), builder, predicateManager, ProbabilisticSoftLogicProblem.GroundingMode.AllPossible);
        ProbabilisticSoftLogicProblem.Rule.addGroundingsToBuilder(
                Arrays.asList(rule), closedBuilder, closedPredicateManager, ProbabilisticSoftLogicProblem.GroundingMode.AllPossible);

        Set<String> expectedNormalKnows = new HashSet<>(Arrays.asList(
                "KNOWS(1, 1)",
                "KNOWS(1, 2)",
                "KNOWS(1, 3)",
                "KNOWS(1, 4)",
                "KNOWS(2, 1)",
                "KNOWS(2, 2)",
                "KNOWS(2, 3)",
                "KNOWS(2, 4)",
                "KNOWS(3, 1)",
                "KNOWS(3, 2)",
                "KNOWS(3, 3)",
                "KNOWS(3, 4)",
                "KNOWS(4, 1)",
                "KNOWS(4, 2)",
                "KNOWS(4, 3)",
                "KNOWS(4, 4)"
        ));

        Set<String> expectedNormalTrusts = new HashSet<>(Arrays.asList(
                "TRUSTS(1, 1)",
                "TRUSTS(1, 2)",
                "TRUSTS(1, 3)",
                "TRUSTS(1, 4)",
                "TRUSTS(2, 1)",
                "TRUSTS(2, 2)",
                "TRUSTS(2, 3)",
                "TRUSTS(2, 4)",
                "TRUSTS(3, 1)",
                "TRUSTS(3, 2)",
                "TRUSTS(3, 3)",
                "TRUSTS(3, 4)",
                "TRUSTS(4, 1)",
                "TRUSTS(4, 2)",
                "TRUSTS(4, 3)",
                "TRUSTS(4, 4)"
        ));

        Set<String> expectedClosedKnows = new HashSet<>(Arrays.asList(
                "KNOWS(1, 2)",
                "KNOWS(2, 3)",
                "KNOWS(3, 4)"
        ));

        Set<String> expectedClosedTrusts = new HashSet<>(Arrays.asList(
                "TRUSTS(1, 3)",
                "TRUSTS(2, 4)"
        ));

        Set<Integer> predicateIdsKnows = predicateManager.getIdsForPredicateName("KNOWS");
        assertTrue(predicateIdsKnows != null);
        assertTrue(predicateIdsKnows.size() == expectedNormalKnows.size());

        for (int predicateId : predicateIdsKnows) {
            ProbabilisticSoftLogicProblem.Predicate predicate = predicateManager.getPredicateFromId(predicateId);
            assertTrue(predicate != null);
            assertTrue(expectedNormalKnows.contains(predicate.toString()));
        }

        Set<Integer> predicateIdsTrusts = predicateManager.getIdsForPredicateName("TRUSTS");
        assertTrue(predicateIdsTrusts != null);
        assertTrue(predicateIdsTrusts.size() == expectedNormalKnows.size());

        for (int predicateId : predicateIdsTrusts) {
            ProbabilisticSoftLogicProblem.Predicate predicate = predicateManager.getPredicateFromId(predicateId);
            assertTrue(predicate != null);
            assertTrue(expectedNormalTrusts.contains(predicate.toString()));
        }

        Set<Integer> closedPredicateIdsKnows = closedPredicateManager.getIdsForPredicateName("KNOWS");
        assertTrue(closedPredicateIdsKnows != null);
        assertTrue(closedPredicateIdsKnows.size() == expectedClosedKnows.size());

        for (int predicateId : closedPredicateIdsKnows) {
            ProbabilisticSoftLogicProblem.Predicate predicate = closedPredicateManager.getPredicateFromId(predicateId);
            assertTrue(predicate != null);
            assertTrue(expectedClosedKnows.contains(predicate.toString()));
        }

        Set<Integer> closedPredicateIdsTrusts = closedPredicateManager.getIdsForPredicateName("TRUSTS");
        assertTrue(closedPredicateIdsTrusts != null);
        assertTrue(closedPredicateIdsTrusts.size() == expectedClosedTrusts.size());

        for (int predicateId : closedPredicateIdsTrusts) {
            ProbabilisticSoftLogicProblem.Predicate predicate = closedPredicateManager.getPredicateFromId(predicateId);
            assertTrue(predicate != null);
            assertTrue(expectedClosedTrusts.contains(predicate.toString()));
        }
    }

    @Test
    public void testPredicateExpansion() {

        ProbabilisticSoftLogicProblem.Predicate knowsAB =
                new ProbabilisticSoftLogicProblem.Predicate("KNOWS", ImmutableList.of("A", "B"), false);

        ProbabilisticSoftLogicProblem.Predicate knowsBC =
                new ProbabilisticSoftLogicProblem.Predicate(knowsAB.Name, ImmutableList.of("B", "C"), false);

        ProbabilisticSoftLogicProblem.Predicate knowsAC =
                new ProbabilisticSoftLogicProblem.Predicate("KNOWS", ImmutableList.of("A", "C"), false);

        // "{1.5} KNOWS(A, B) & KNOWS(B, C) >> KNOWS(A, C)"
        ProbabilisticSoftLogicProblem.Rule rule =
                new ProbabilisticSoftLogicProblem.Rule(1.5, 1, ImmutableList.of(knowsAC), ImmutableList.of(knowsAB, knowsBC));

        ProbabilisticSoftLogicProblem.Predicate knows12 =
                new ProbabilisticSoftLogicProblem.Predicate(knowsAB.Name, ImmutableList.of("1", "2"), false);
        ProbabilisticSoftLogicProblem.Predicate knows23 =
                new ProbabilisticSoftLogicProblem.Predicate(knowsAB.Name, ImmutableList.of("2", "3"), false);
        ProbabilisticSoftLogicProblem.Predicate knows34 =
                new ProbabilisticSoftLogicProblem.Predicate(knowsAB.Name, ImmutableList.of("3", "4"), false);
        ProbabilisticSoftLogicProblem.Predicate knows42 =
                new ProbabilisticSoftLogicProblem.Predicate(knowsAB.Name, ImmutableList.of("4", "2"), false);

        ProbabilisticSoftLogicPredicateManager predicateManager = new ProbabilisticSoftLogicPredicateManager();
        predicateManager.getOrAddPredicate(knows12, 0.5);
        predicateManager.getOrAddPredicate(knows23, 0.1);
        predicateManager.getOrAddPredicate(knows34, 0.7);
        predicateManager.getOrAddPredicate(knows42, 0.9);

        ProbabilisticSoftLogicPredicateManager.IdWeights observedIdsAndWeights =
                predicateManager.getAllObservedWeights();

        ProbabilisticSoftLogicProblem.Builder builder =
                new ProbabilisticSoftLogicProblem.Builder(observedIdsAndWeights.Ids, observedIdsAndWeights.Weights, 100);

        ProbabilisticSoftLogicProblem.Rule.addGroundingsToBuilder(Arrays.asList(rule), builder, predicateManager, ProbabilisticSoftLogicProblem.GroundingMode.ByExtension);

        // from initial observations we have:
        // K(1, 2) & K(2, 3) >> K(1, 3) --> body does not hold
        // K(2, 3) & K(3, 4) >> K(2, 4) --> body does not hold
        // K(3, 4) & K(4, 2) >> K(3, 2) --> body holds --> new predicate K(3, 2)               // term 1
        // then in round 2 we have new rule:
        // K(2, 3) & K(3, 2) >> K(2, 2) --> body holds (worst case) --> new predicate K(2, 2)  // term 2
        // K(3, 2) & K(2, 3) >> K(3, 3) --> body holds (worst case) --> new predicate K(3, 3)  // term 3
        // then in round 3 we have:
        // K(2, 2) & K(2, 2) >> K(2, 2)                                                        // term 4
        // K(2, 2) & K(2, 3) >> K(2, 3)                                                        // eliminated by worst case
        // K(1, 2) & K(2, 2) >> K(1, 2)                                                        // eliminated by worst case
        // K(3, 2) & K(2, 2) >> K(3, 2)                                                        // term 5
        // K(4, 2) & K(2, 2) >> K(4, 2)                                                        // eliminated by worst case
        // K(3, 3) & K(3, 2) >> K(3, 2)                                                        // term 6
        // K(3, 3) & K(3, 3) >> K(3, 3)                                                        // term 7
        // K(3, 3) & K(3, 4) >> K(3, 4)                                                        // eliminated by worst case
        // K(2, 3) & K(3, 3) >> K(2, 3)                                                        // eliminated by worst case

        Set<String> expected = new HashSet<>(Arrays.asList(
                "KNOWS(1, 2)",
                "KNOWS(2, 3)",
                "KNOWS(3, 4)",
                "KNOWS(4, 2)",
                "KNOWS(3, 2)",
                "KNOWS(2, 2)",
                "KNOWS(3, 3)"
        ));

        Set<Integer> predicateIds = predicateManager.getIdsForPredicateName("KNOWS");
        assertTrue(predicateIds != null);
        assertTrue(predicateIds.size() == expected.size());

        for (int predicateId : predicateIds) {
            ProbabilisticSoftLogicProblem.Predicate predicate = predicateManager.getPredicateFromId(predicateId);
            assertTrue(predicate != null);
            assertTrue(expected.contains(predicate.toString()));
        }

        assertTrue(builder.getNumberOfTerms() == 7);
    }

    @Test
    public void testSerialization() {

        ProbabilisticSoftLogicProblem.Predicate knowsAB =
                new ProbabilisticSoftLogicProblem.Predicate("KNOWS", ImmutableList.of("A", "B"), false);

        ProbabilisticSoftLogicProblem.Predicate knowsBC =
                new ProbabilisticSoftLogicProblem.Predicate(knowsAB.Name, ImmutableList.of("B", "C"), false);

        ProbabilisticSoftLogicProblem.Predicate trustsAC =
                new ProbabilisticSoftLogicProblem.Predicate("TRUSTS", ImmutableList.of("A", "C"), false);

        // "{1.5} KNOWS(A, B) & KNOWS(B, C) >> TRUSTS(A, C)"
        ProbabilisticSoftLogicProblem.Rule rule =
                new ProbabilisticSoftLogicProblem.Rule(1.5, 1, ImmutableList.of(trustsAC), ImmutableList.of(knowsAB, knowsBC));

        ProbabilisticSoftLogicProblem.Predicate knows12 =
                new ProbabilisticSoftLogicProblem.Predicate(knowsAB.Name, ImmutableList.of("1", "2"), false);
        ProbabilisticSoftLogicProblem.Predicate knows23 =
                new ProbabilisticSoftLogicProblem.Predicate(knowsAB.Name, ImmutableList.of("2", "3"), false);
        ProbabilisticSoftLogicProblem.Predicate knows34 =
                new ProbabilisticSoftLogicProblem.Predicate(knowsAB.Name, ImmutableList.of("3", "4"), false);

        ProbabilisticSoftLogicPredicateManager predicateManager = new ProbabilisticSoftLogicPredicateManager();

        predicateManager.getOrAddPredicate(knows12, 0.5);
        predicateManager.getOrAddPredicate(knows23, 0.1);
        predicateManager.getOrAddPredicate(knows34, 0.7);

        ProbabilisticSoftLogicPredicateManager.IdWeights observedIdsAndWeights =
                predicateManager.getAllObservedWeights();

        ProbabilisticSoftLogicProblem.Builder builder =
                new ProbabilisticSoftLogicProblem.Builder(observedIdsAndWeights.Ids, observedIdsAndWeights.Weights, 100);

        ProbabilisticSoftLogicProblem.Rule.addGroundingsToBuilder(Arrays.asList(rule), builder, predicateManager, ProbabilisticSoftLogicProblem.GroundingMode.AllPossible);

        ProbabilisticSoftLogicProblem problem = builder.build();

        ProbabilisticSoftLogicProblem deserializedProblem = null;

        try {
            ByteArrayOutputStream byteStream = new ByteArrayOutputStream(2048);
            ProbabilisticSoftLogicProblem.ProblemSerializer.write(byteStream, Arrays.asList(rule), predicateManager, ProbabilisticSoftLogicProblem.GroundingMode.AllPossible);
            byteStream.close();
            byte[] bytes = byteStream.toByteArray();
            ByteArrayInputStream inputStream = new ByteArrayInputStream(bytes);
            Map.Entry<ProbabilisticSoftLogicPredicateManager, ProbabilisticSoftLogicProblem.Builder> deserialized =
                    ProbabilisticSoftLogicProblem.ProblemSerializer.read(inputStream);
            deserializedProblem = deserialized.getValue().build();
        } catch (IOException|ClassNotFoundException e) {
            fail(e.getMessage());
            System.out.println(e.getMessage());
            e.printStackTrace();
        }

        assertTrue("Serialization round-trip not equal to initial object", problem.equals(deserializedProblem));

    }

    @Test
    public void testEndToEnd() {

        String experimentName = "epinions_200";

        for (ProbabilisticSoftLogicProblem.GroundingMode groundingMode : ProbabilisticSoftLogicProblem.GroundingMode.values()) {

            // for testing one mode
            if (groundingMode != ProbabilisticSoftLogicProblem.GroundingMode.AsRead) {
                continue;
            }

            String outputStreamName = null;
            try {
                Path knowsPath = Paths.get(this.getClass().getResource("../" + experimentName + "/knows.txt").toURI());
                outputStreamName = Paths.get(knowsPath.getParent().toString(), "problem_serialized.bin").toString();
            } catch (URISyntaxException e) {
                fail(e.getMessage());
                System.out.println(e.getMessage());
                e.printStackTrace();
                return;
            }


            File outputStreamFile = new File(outputStreamName);
            ProbabilisticSoftLogicProblem.Builder problemBuilder = null;
            ProbabilisticSoftLogicPredicateManager trainPredicateManager = null;
            if (outputStreamFile.exists()) {

                try (FileInputStream inputStream = new FileInputStream(outputStreamFile)) {

                    Map.Entry<ProbabilisticSoftLogicPredicateManager, ProbabilisticSoftLogicProblem.Builder> deserialized =
                            ProbabilisticSoftLogicProblem.ProblemSerializer.read(inputStream);

                    trainPredicateManager = deserialized.getKey();
                    problemBuilder = deserialized.getValue();

                } catch (IOException|ClassNotFoundException e) {
                    fail(e.getMessage());
                    System.out.println(e.getMessage());
                    e.printStackTrace();
                }

            } else {

                InputStream groundingStream = null;
                if (groundingMode == ProbabilisticSoftLogicProblem.GroundingMode.AsRead) {
                    groundingStream = LogicRuleParserTest.class.getResourceAsStream("../" + experimentName + "/trust_groundings.txt");
                }

                InputStream modelStream = LogicRuleParserTest.class.getResourceAsStream("../" + experimentName + "/model.txt");
                InputStream knowsStream = LogicRuleParserTest.class.getResourceAsStream("../" + experimentName + "/knows.txt");
                InputStream trustTrainStream = LogicRuleParserTest.class.getResourceAsStream("../" + experimentName + "/train.txt");
                InputStream trustTestStream = LogicRuleParserTest.class.getResourceAsStream("../" + experimentName + "/test.txt");

                List<ProbabilisticSoftLogicProblem.Rule> rules = null;
                trainPredicateManager = new ProbabilisticSoftLogicPredicateManager();
                ProbabilisticSoftLogicPredicateManager testPredicateManager = new ProbabilisticSoftLogicPredicateManager();

                try (
                        BufferedReader groundingReader = groundingStream == null ? null : new BufferedReader(new InputStreamReader(groundingStream));
                        BufferedReader modelReader = new BufferedReader(new InputStreamReader(modelStream));
                        BufferedReader knowsReader = new BufferedReader(new InputStreamReader(knowsStream));
                        BufferedReader trustTrainReader = new BufferedReader(new InputStreamReader(trustTrainStream));
                        BufferedReader trustTestReader = trustTestStream == null ? null : new BufferedReader(new InputStreamReader(trustTestStream))) {

                    rules = ProbabilisticSoftLogicReader.readRules(modelReader);

                    ProbabilisticSoftLogicReader.readGroundingsAndAddToManager(
                            trainPredicateManager, "KNOWS", groundingMode == ProbabilisticSoftLogicProblem.GroundingMode.AllPossible, false, knowsReader);

                    ProbabilisticSoftLogicReader.readGroundingsAndAddToManager(
                            trainPredicateManager, "TRUSTS", false, false, trustTrainReader);

                    if (groundingMode == ProbabilisticSoftLogicProblem.GroundingMode.AsRead) {
                        ProbabilisticSoftLogicReader.readGroundingsAndAddToManager(
                                trainPredicateManager, "TRUSTS", false, true, groundingReader);
                    }

                    ProbabilisticSoftLogicReader.readGroundingsAndAddToManager(testPredicateManager, "TRUSTS", false, false, trustTestReader);

                } catch (IOException | DataFormatException e) {
                    fail(e.getMessage());
                    System.out.println(e.getMessage());
                    e.printStackTrace();
                }

                try (FileOutputStream outputStream = new FileOutputStream(outputStreamFile)) {

                    problemBuilder =
                            ProbabilisticSoftLogicProblem.ProblemSerializer.write(
                                    outputStream,
                                    rules,
                                    trainPredicateManager,
                                    groundingMode);

                } catch (IOException e) {
                    fail(e.getMessage());
                    System.out.println(e.getMessage());
                    e.printStackTrace();
                }

            }

            ProbabilisticSoftLogicProblem problem = problemBuilder
                    .subProblemSelectionMethod(ConsensusAlternatingDirectionsMethodOfMultipliersSolver.SubProblemSelectionMethod.UNIFORM_SAMPLING)
                    .numberOfSubProblemSamples(8)
                    .build();

            Map<Integer, Double> result = problem.solve();
            Map<String, Double> filteredResults = new HashMap<>();

            // to make compiler happy that this is effectively final
            ProbabilisticSoftLogicPredicateManager temp = trainPredicateManager;

            result.keySet().stream()
                    .filter(key -> result.get(key) > Math.sqrt(Double.MIN_VALUE))
                    .forEach(key -> filteredResults.put(temp.getPredicateFromId(key).toString(), result.get(key)));


            System.out.println(result.get(0));
            System.out.println(result.get(1));

            System.out.println("Done\n");

        }

    }


    // DBC: Added test to do pre-grounded rules, from PSL (based on testEndToEnd)
    @Test
    public void testPregroundRules() {

        String experimentName = "epinions_30";

        for (ProbabilisticSoftLogicProblem.GroundingMode groundingMode : ProbabilisticSoftLogicProblem.GroundingMode.values()) {

            // for testing one mode
            if (groundingMode != ProbabilisticSoftLogicProblem.GroundingMode.AsRead) {
                continue;
            }

            String outputStreamName = null;
            try {
                Path knowsPath = Paths.get(this.getClass().getResource("../" + experimentName + "/knows.txt").toURI());
                outputStreamName = Paths.get(knowsPath.getParent().toString(), "problem_serialized.bin").toString();
            } catch (URISyntaxException e) {
                fail(e.getMessage());
                System.out.println(e.getMessage());
                e.printStackTrace();
                return;
            }


            // DBC: Not really doing the serialization thing yet, but I'll leave it here for now
            File outputStreamFile = new File(outputStreamName);
            ProbabilisticSoftLogicProblem.Builder problemBuilder = null;
            ProbabilisticSoftLogicPredicateManager trainPredicateManager = null;

            // Don't look for serialized file
            if (false) {
            //if (outputStreamFile.exists()) {

                try (FileInputStream inputStream = new FileInputStream(outputStreamFile)) {

                    Map.Entry<ProbabilisticSoftLogicPredicateManager, ProbabilisticSoftLogicProblem.Builder> deserialized =
                            ProbabilisticSoftLogicProblem.ProblemSerializer.read(inputStream);

                    trainPredicateManager = deserialized.getKey();
                    problemBuilder = deserialized.getValue();

                } catch (IOException|ClassNotFoundException e) {
                    fail(e.getMessage());
                    System.out.println(e.getMessage());
                    e.printStackTrace();
                }

            } else {

                InputStream groundingStream = null;
                if (groundingMode == ProbabilisticSoftLogicProblem.GroundingMode.AsRead) {
                    groundingStream = LogicRuleParserTest.class.getResourceAsStream("../" + experimentName + "/trust_groundings.txt");
                }

                InputStream modelStream = LogicRuleParserTest.class.getResourceAsStream("../" + experimentName + "/model.txt");
                InputStream knowsStream = LogicRuleParserTest.class.getResourceAsStream("../" + experimentName + "/knows.txt");
                InputStream trustTrainStream = LogicRuleParserTest.class.getResourceAsStream("../" + experimentName + "/train.txt");
                InputStream trustTestStream = LogicRuleParserTest.class.getResourceAsStream("../" + experimentName + "/test.txt");
                // DBC: The rules file contains all grounded rules
                InputStream groundRuleStream = LogicRuleParserTest.class.getResourceAsStream("../" + experimentName + "/rules.txt");

                List<ProbabilisticSoftLogicProblem.Rule> rules = null;
                List<ProbabilisticSoftLogicProblem.Rule> groundRules = null;
                trainPredicateManager = new ProbabilisticSoftLogicPredicateManager();
                ProbabilisticSoftLogicPredicateManager testPredicateManager = new ProbabilisticSoftLogicPredicateManager();

                try (
                        BufferedReader groundingReader = groundingStream == null ? null : new BufferedReader(new InputStreamReader(groundingStream));
                        BufferedReader modelReader = new BufferedReader(new InputStreamReader(modelStream));
                        BufferedReader knowsReader = new BufferedReader(new InputStreamReader(knowsStream));
                        BufferedReader trustTrainReader = new BufferedReader(new InputStreamReader(trustTrainStream));
                        BufferedReader groundRuleReader = new BufferedReader(new InputStreamReader(groundRuleStream));
                        BufferedReader trustTestReader = trustTestStream == null ? null : new BufferedReader(new InputStreamReader(trustTestStream))) {

                    rules = ProbabilisticSoftLogicReader.readRules(modelReader);

                    ProbabilisticSoftLogicReader.readGroundingsAndAddToManager(
                            trainPredicateManager, "KNOWS", groundingMode == ProbabilisticSoftLogicProblem.GroundingMode.AllPossible, false, knowsReader);

                    ProbabilisticSoftLogicReader.readGroundingsAndAddToManager(
                            trainPredicateManager, "TRUSTS", false, false, trustTrainReader);

                    if (groundingMode == ProbabilisticSoftLogicProblem.GroundingMode.AsRead) {
                        ProbabilisticSoftLogicReader.readGroundingsAndAddToManager(
                                trainPredicateManager, "TRUSTS", false, true, groundingReader);
                    }

                    ProbabilisticSoftLogicReader.readGroundingsAndAddToManager(testPredicateManager, "TRUSTS", false, false, trustTestReader);

                    // DBC: Use the model parser to parse the pre-grounded rules
                    groundRules = ProbabilisticSoftLogicReader.readRules(groundRuleReader);

                    // DBC: Creata a problem builder using the pre-grounded rules
                    problemBuilder = ProbabilisticSoftLogicProblem.PregroundRuleHandler.createBuilder(
                            rules,
                            trainPredicateManager,
                            groundRules);


                } catch (IOException | DataFormatException e) {
                    fail(e.getMessage());
                    System.out.println(e.getMessage());
                    e.printStackTrace();
                }


                /*
                try (FileOutputStream outputStream = new FileOutputStream(outputStreamFile)) {

                    problemBuilder =
                            ProbabilisticSoftLogicProblem.ProblemSerializer.write(
                                    outputStream,
                                    rules,
                                    trainPredicateManager,
                                    groundingMode);

                } catch (IOException e) {
                    fail(e.getMessage());
                    System.out.println(e.getMessage());
                    e.printStackTrace();
                }
                */

            }


            ProbabilisticSoftLogicProblem problem = problemBuilder
                    //.subProblemSelectionMethod(ConsensusAlternatingDirectionsMethodOfMultipliersSolver.SubProblemSelectionMethod.UNIFORM_SAMPLING)
                    //.numberOfSubProblemSamples(100)
                    .build();


            Map<Integer, Double> result = problem.solve();
            Map<String, Double> filteredResults = new HashMap<>();

            // to make compiler happy that this is effectively final
            ProbabilisticSoftLogicPredicateManager temp = trainPredicateManager;

            result.keySet().stream()
                    .filter(key -> result.get(key) > Math.sqrt(Double.MIN_VALUE))
                    .forEach(key -> filteredResults.put(temp.getPredicateFromId(key).toString(), result.get(key)));



            Iterator it = filteredResults.entrySet().iterator();
            while (it.hasNext()) {
                Map.Entry pair = (Map.Entry)it.next();
                System.out.println(pair.getKey() + " = " + pair.getValue());
                it.remove(); // avoids a ConcurrentModificationException
            }


            System.out.println("Done\n");

        }

    }

}