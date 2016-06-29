//package org.platanios.learn.logic.formula;
//
//import org.junit.Assert;
//import org.junit.Test;
//
//import java.util.ArrayList;
//import java.util.List;
//
///**
// * @author Emmanouil Antonios Platanios
// */
//public class ToStringTest {
//    @Test
//    public void testAtomToString() {
//        List<VariableType<Integer>> variableTypes = new ArrayList<>();
//        variableTypes.add(new VariableType<Integer>(0, Integer.class).setName("{person}"));
//        variableTypes.add(new VariableType<Integer>(0, Integer.class).setName("{person}"));
//
//        Predicate<Integer> knowsPredicate = new Predicate<>(0, variableTypes).setName("knows");
//
//        Variable<Integer> personA = new Variable<>(0, 0, variableTypes.get(0)).setName("A");
//        Variable<Integer> personB = new Variable<>(1, 0, variableTypes.get(0)).setName("B");
//
//        List<Variable<Integer>> knowsABArguments = new ArrayList<>(2);
//        knowsABArguments.add(personA);
//        knowsABArguments.add(personB);
//
//        Formula<Integer> knowsABAtom = new Atom<>(knowsPredicate, knowsABArguments);
//
//        Assert.assertEquals("knows(A, B)", knowsABAtom.toString());
//    }
//
//    @Test
//    public void testNegationToString() {
//        List<VariableType<Integer>> variableTypes = new ArrayList<>();
//        variableTypes.add(new VariableType<Integer>(0, Integer.class).setName("{person}"));
//        variableTypes.add(new VariableType<Integer>(0, Integer.class).setName("{person}"));
//
//        Predicate<Integer> knowsPredicate = new Predicate<>(0, variableTypes).setName("knows");
//
//        Variable<Integer> personA = new Variable<>(0, 0, variableTypes.get(0)).setName("A");
//        Variable<Integer> personB = new Variable<>(1, 0, variableTypes.get(0)).setName("B");
//
//        List<Variable<Integer>> knowsABArguments = new ArrayList<>(2);
//        knowsABArguments.add(personA);
//        knowsABArguments.add(personB);
//
//        Formula<Integer> knowsABNegation = new Negation<>(new Atom<>(knowsPredicate, knowsABArguments));
//
//        Assert.assertEquals("!knows(A, B)", knowsABNegation.toString());
//    }
//
//    @Test
//    public void testConjunctionToString() {
//        List<VariableType<Integer>> variableTypes = new ArrayList<>();
//        variableTypes.add(new VariableType<Integer>(0, Integer.class).setName("{person}"));
//        variableTypes.add(new VariableType<Integer>(0, Integer.class).setName("{person}"));
//
//        Predicate<Integer> knowsPredicate = new Predicate<>(0, variableTypes).setName("knows");
//        Predicate<Integer> trustsPredicate = new Predicate<>(0, variableTypes).setName("trusts");
//
//        Variable<Integer> personA = new Variable<>(0, 0, variableTypes.get(0)).setName("A");
//        Variable<Integer> personB = new Variable<>(1, 0, variableTypes.get(0)).setName("B");
//        Variable<Integer> personC = new Variable<>(2, 0, variableTypes.get(0)).setName("C");
//
//        List<Variable<Integer>> knowsABArguments = new ArrayList<>(2);
//        knowsABArguments.add(personA);
//        knowsABArguments.add(personB);
//
//        List<Variable<Integer>> trustsBCArguments = new ArrayList<>(2);
//        trustsBCArguments.add(personB);
//        trustsBCArguments.add(personC);
//
//        List<Formula<Integer>> conjunctionComponents = new ArrayList<>();
//        conjunctionComponents.add(new Atom<>(knowsPredicate, knowsABArguments));
//        conjunctionComponents.add(new Negation<>(new Atom<>(trustsPredicate, trustsBCArguments)));
//
//        Formula<Integer> conjunction = new Conjunction<>(conjunctionComponents);
//
//        Assert.assertEquals("knows(A, B) & !trusts(B, C)", conjunction.toString());
//    }
//}
