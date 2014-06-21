package org.platanios.learn.combination;

import com.ziena.knitro.KnitroJava;
import org.platanios.math.combinatorics.CombinatoricsUtilities;

import java.util.*;

/**
 * @author Emmanouil Antonios Platanios
 */
public class KNitroOptimizationProblem {
    int n, m;
    final int numberOfFunctions;
    final int maximumOrder;
    final ErrorRatesVector errorRates;
    final AgreementRatesVector agreementRates;

    Map<ArrayList<Integer>, Integer> agreementRatesIndexToKeyMapping;
    Map<ArrayList<Integer>, Integer> errorRatesIndexToKeyMapping;
    Map<ArrayList<Integer>, Integer> hessianIndexToKeyMapping;
    List<Integer[]> constraintsIndexes;
    KnitroJava solver;
    double[] daX;
    double[] daLambda;
    double[] daObj;
    double[] daC;
    double[] daObjGrad;
    double[] daJac;
    double[] daHess;

    public KNitroOptimizationProblem(int numberOfFunctions, int maximumOrder, ErrorRatesVector errorRates, AgreementRatesVector agreementRates) {
        this.numberOfFunctions = numberOfFunctions;
        this.maximumOrder = maximumOrder;
        this.errorRates = errorRates;
        this.agreementRates = agreementRates;

        agreementRatesIndexToKeyMapping = agreementRates.getIndexToKeyMapping();
        errorRatesIndexToKeyMapping = errorRates.getIndexToKeyMapping();

        n = errorRates.getLength();
        int  objGoal = KnitroJava.KTR_OBJGOAL_MINIMIZE;
        int  objType = KnitroJava.KTR_OBJTYPE_GENERAL;

        double[] bndsLo = new double[n];
        double[] bndsUp = new double[n];

        for (int i = 0; i < n; i++) {
            bndsLo[i] = 0.0;
            bndsUp[i] = 0.5;
        }

        m = agreementRatesIndexToKeyMapping.size();

        for (int k = 2; k <= maximumOrder; k++) {
            m += CombinatoricsUtilities.binomialCoefficient(numberOfFunctions, k) * k;
        }

        int[] cType = new int[m];

        for (int k = 0; k < m; k++) {
            cType[k] = KnitroJava.KTR_CONTYPE_LINEAR;
        }

        double[] cBndsLo = new double[m];
        double[] cBndsUp = new double[m];

        for (int i = 0; i < agreementRatesIndexToKeyMapping.size(); i++) {
            cBndsLo[i] = agreementRates.agreementRates[i] - 1;
            cBndsUp[i] = agreementRates.agreementRates[i] - 1;
        }

        for (int i = agreementRatesIndexToKeyMapping.size(); i < m; i++) {
            cBndsLo[i] = -KnitroJava.KTR_INFBOUND;
            cBndsUp[i] = 0;
        }

        int nnzJ = 0;
        int constraintIndex = 0;
        constraintsIndexes = new ArrayList<Integer[]>();

        for (Map.Entry<ArrayList<Integer>, Integer> entry : agreementRatesIndexToKeyMapping.entrySet()) {
            int k = entry.getKey().size();
            constraintsIndexes.add(new Integer[] { constraintIndex, errorRatesIndexToKeyMapping.get(entry.getKey()), 2 } );
            nnzJ += 1;
            for (int l = 1; l < k; l++) {
                List<ArrayList<Integer>> inner_indexes = CombinatoricsUtilities.getCombinations(k, l);
                for (ArrayList<Integer> inner_index : inner_indexes) {
                    ArrayList<Integer> temp_index = new ArrayList<Integer>();
                    for (int i : inner_index) {
                        temp_index.add(entry.getKey().get(i));
                    }
                    constraintsIndexes.add(new Integer[] { constraintIndex, errorRatesIndexToKeyMapping.get(temp_index), (int) Math.pow(-1, l) });
                }
                nnzJ += inner_indexes.size();
            }
            constraintIndex++;
        }

        for (Map.Entry<ArrayList<Integer>, Integer> entry : errorRatesIndexToKeyMapping.entrySet()) {
            int k = entry.getKey().size();
            if (entry.getKey().size() > 1) {
                List<ArrayList<Integer>> inner_indexes = CombinatoricsUtilities.getCombinations(k, k - 1);
                for (ArrayList<Integer> inner_index : inner_indexes) {
                    ArrayList<Integer> temp_index = new ArrayList<Integer>();
                    for (int i : inner_index) {
                        temp_index.add(entry.getKey().get(i));
                    }
                    constraintsIndexes.add(new Integer[] { constraintIndex, errorRatesIndexToKeyMapping.get(entry.getKey()), 1 } );
                    constraintsIndexes.add(new Integer[] { constraintIndex++, errorRatesIndexToKeyMapping.get(temp_index), -1 });
                }
                nnzJ += 2 * inner_indexes.size();
            }
        }

        int[] jacIxConstr = new int[nnzJ];
        int[] jacIxVar = new int[nnzJ];
        constraintIndex = 0;

        for (Integer[] constraintsIndex : constraintsIndexes) {
            jacIxConstr[constraintIndex] = constraintsIndex[0];
            jacIxVar[constraintIndex++] = constraintsIndex[1];
        }

        int nnzH = n * (n + 1) / 2;
        int[] hessRow = new int[n * (n + 1) / 2];
        int[] hessCol = new int[n * (n + 1) / 2];
        int hessianEntryIndex = 0;
        hessianIndexToKeyMapping = new LinkedHashMap<ArrayList<Integer>, Integer>();

        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                hessRow[hessianEntryIndex] = i;
                hessCol[hessianEntryIndex] = j;
                hessianIndexToKeyMapping.put(new ArrayList<Integer>(Arrays.asList(new Integer[] { i, j })), hessianEntryIndex++);
            }
        }

        double[]  daXInit = errorRates.errorRates;

        try
        {
            solver = new KnitroJava();
        }
        catch (java.lang.Exception  e)
        {
            System.err.println (e);
            return;
        }

        if (!solver.setIntParamByName("algorithm", 3))
        {
            System.err.println ("Error setting parameter 'algorithm'");
            return;
        }
        if (!solver.setIntParamByName("blasoption", 1))
        {
            System.err.println ("Error setting parameter 'blasoption'");
            return;
        }
        if (!solver.setDoubleParamByName("feastol", 1.0E-20))
        {
            System.err.println ("Error setting parameter 'feastol'");
            return;
        }
        if (!solver.setIntParamByName("gradopt", 1))
        {
            System.err.println ("Error setting parameter 'gradopt'");
            return;
        }
        if (!solver.setIntParamByName("hessian_no_f", 1))
        {
            System.err.println ("Error setting parameter 'hessian_no_f'");
            return;
        }
        if (!solver.setIntParamByName("hessopt", 1))
        {
            System.err.println ("Error setting parameter 'hessopt'");
            return;
        }
        if (!solver.setIntParamByName("honorbnds", 1))
        {
            System.err.println ("Error setting parameter 'honorbnds'");
            return;
        }
        if (!solver.setIntParamByName("linsolver", 0))
        {
            System.err.println ("Error setting parameter 'linsolver'");
            return;
        }
        if (!solver.setIntParamByName("maxit", 100000))
        {
            System.err.println ("Error setting parameter 'maxit'");
            return;
        }
        if (!solver.setDoubleParamByName("opttol", 1E-20))
        {
            System.err.println ("Error setting parameter 'opttol'");
            return;
        }
        if (!solver.setIntParamByName("outlev", 6))
        {
            System.err.println ("Error setting parameter 'outlev'");
            return;
        }
        if (!solver.setIntParamByName("par_numthreads", 1))
        {
            System.err.println ("Error setting parameter 'par_numthreads'");
            return;
        }
        if (!solver.setIntParamByName("scale", 1))
        {
            System.err.println ("Error setting parameter 'scale'");
            return;
        }
        if (!solver.setIntParamByName("soc", 0))
        {
            System.err.println ("Error setting parameter 'soc'");
            return;
        }
        if (!solver.setDoubleParamByName("xtol", 1.0E-20))
        {
            System.err.println ("Error setting parameter 'xtol'");
            return;
        }

        if (!solver.initProblem(n, objGoal, objType, bndsLo, bndsUp,
                m, cType, cBndsLo, cBndsUp,
                nnzJ, jacIxVar, jacIxConstr,
                nnzH, hessRow, hessCol,
                daXInit, null))
        {
            System.err.println ("Error initializing the problem, "
                    + "KNITRO status = "
                    + solver.getKnitroStatusCode());
            return;
        }

        daX       = new double[n];
        daLambda  = new double[m + n];
        daObj     = new double[1];
        daC       = new double[m];
        daObjGrad = new double[n];
        daJac     = new double[nnzJ];
        daHess    = new double[nnzH];
    }

    /** Compute the function and constraint values at x.
     *
     *  For more information about the arguments, refer to the KNITRO
     *  manual, especially the section on the Callable Library.
     */
    public double evaluateFC (double[]  daX,
                               double[]  daC)
    {
        double dObj = 0;
        for (Map.Entry<ArrayList<Integer>, Integer> entry : errorRatesIndexToKeyMapping.entrySet()) {
            if (entry.getKey().size() > 1) {
                double term = 1;
                ArrayList<Integer> indexes = entry.getKey();
                for (int index : indexes) {
                    term *= daX[index];
                }
                dObj += Math.pow(daX[entry.getValue()] - term, 2);
            }
        }

        int constraintIndex = 0;

        for (Map.Entry<ArrayList<Integer>, Integer> entry : agreementRatesIndexToKeyMapping.entrySet()) {
            int k = entry.getKey().size();
            daC[constraintIndex] = 2 * daX[errorRatesIndexToKeyMapping.get(entry.getKey())];
            for (int l = 1; l < k; l++) {
                List<ArrayList<Integer>> inner_indexes = CombinatoricsUtilities.getCombinations(k, l);
                for (ArrayList<Integer> inner_index : inner_indexes) {
                    ArrayList<Integer> temp_index = new ArrayList<Integer>();
                    for (int i : inner_index) {
                        temp_index.add(entry.getKey().get(i));
                    }
                    daC[constraintIndex] += Math.pow(-1, l) * daX[errorRatesIndexToKeyMapping.get(temp_index)];
                }
            }
            constraintIndex++;
        }

        for (Map.Entry<ArrayList<Integer>, Integer> entry : errorRatesIndexToKeyMapping.entrySet()) {
            int k = entry.getKey().size();
            if (k > 1) {
                List<ArrayList<Integer>> inner_indexes = CombinatoricsUtilities.getCombinations(k, k - 1);
                for (ArrayList<Integer> inner_index : inner_indexes) {
                    ArrayList<Integer> temp_index = new ArrayList<Integer>();
                    for (int i : inner_index) {
                        temp_index.add(entry.getKey().get(i));
                    }
                    daC[constraintIndex++] = daX[errorRatesIndexToKeyMapping.get(entry.getKey())] - daX[errorRatesIndexToKeyMapping.get(temp_index)];
                }
            }
        }

        return dObj;
    }

    //----------------------------------------------------------------
    //   METHOD evaluateGA
    //----------------------------------------------------------------
    /** Compute the function and constraint first deriviatives at x.
     *
     *  For more information about the arguments, refer to the KNITRO
     *  manual, especially the section on the Callable Library.
     */
    public void  evaluateGA (double[]  daX,
                             double[]  daObjGrad,
                             double[]  daJac)
    {
        for (int i = 0; i < daObjGrad.length; i++) {
            daObjGrad[i] = 0;
        }

        for (Map.Entry<ArrayList<Integer>, Integer> entry : errorRatesIndexToKeyMapping.entrySet()) {
            if (entry.getKey().size() > 1) {
                double temp_product = 1;
                ArrayList<Integer> indexes = entry.getKey();
                for (int index : indexes) {
                    temp_product *= daX[index];
                }
                double term = daX[entry.getValue()] - temp_product;
                daObjGrad[entry.getValue()] += 2 * term;
                for (int i : indexes) {
                    daObjGrad[i] -= 2 * term * temp_product / daX[i];
                }
            }
        }

        int i = 0;

        for (Integer[] constraintsIndex : constraintsIndexes) {
            daJac[i++] = constraintsIndex[2];
        }
    }


    //----------------------------------------------------------------
    //   METHOD evaluateH
    //----------------------------------------------------------------
    /** Compute the Hessian of the Lagrangian at x and lambda.
     *
     *  For more information about the arguments, refer to the KNITRO
     *  manual, especially the section on the Callable Library.
     */
    public void  evaluateH (double[]  daX,
                            double[]  daLambda,
                            double    dSigma,
                            double[]  daHess)
    {
        for (Map.Entry<ArrayList<Integer>, Integer> entry : errorRatesIndexToKeyMapping.entrySet()) {
            if (entry.getKey().size() > 1) {
                double temp_product = 1;
                ArrayList<Integer> indexes = entry.getKey();
                for (int index : indexes) {
                    temp_product *= daX[index];
                }
                int jointTermIndex = entry.getValue();
                daHess[hessianIndexToKeyMapping.get(new ArrayList<Integer>(Arrays.asList(new Integer[] { jointTermIndex, jointTermIndex })))] += 2;
                for (int i : indexes) {
                    if (jointTermIndex <= i) {
                        daHess[hessianIndexToKeyMapping.get(new ArrayList<Integer>(Arrays.asList(new Integer[] { jointTermIndex, i })))] -= 2 * temp_product / daX[i];
                    }

                    if (i <= jointTermIndex) {
                        daHess[hessianIndexToKeyMapping.get(new ArrayList<Integer>(Arrays.asList(new Integer[] { i, jointTermIndex })))] -= 2 * temp_product / daX[i];
                    }

                    daHess[hessianIndexToKeyMapping.get(new ArrayList<Integer>(Arrays.asList(new Integer[] { i, i })))] += 2 * daX[jointTermIndex] * temp_product / Math.pow(daX[i], 2);

                    for (int j : indexes) {
                        if (i <= j) {
                            daHess[hessianIndexToKeyMapping.get(new ArrayList<Integer>(Arrays.asList(new Integer[] { i, j })))] -= (2 * daX[jointTermIndex] * temp_product + 2 * Math.pow(temp_product, 2))
                                    / (daX[i] * daX[j]);
                        }
                    }
                }
            }
        }
    }

    public double[] solve() {
        int  nKnStatus;
        int  nEvalStatus = 0;

        do
        {
            nKnStatus = solver.solve(nEvalStatus, daObj, daC,
                    daObjGrad, daJac, daHess);
            if (nKnStatus == KnitroJava.KTR_RC_EVALFC)
            {
                //---- KNITRO WANTS daObj AND daC EVALUATED AT THE POINT x.
                daX = solver.getCurrentX();
                daObj[0] = this.evaluateFC(daX, daC);
            }
            else if (nKnStatus == KnitroJava.KTR_RC_EVALGA)
            {
                //---- KNITRO WANTS daObjGrad AND daJac EVALUATED AT THE POINT x.
                daX = solver.getCurrentX();
                this.evaluateGA(daX, daObjGrad, daJac);
            }
            else if (nKnStatus == KnitroJava.KTR_RC_EVALH)
            {
                //---- KNITRO WANTS daHess EVALUATED AT THE POINT x.
                daX = solver.getCurrentX();
                daLambda = solver.getCurrentLambda();
                this.evaluateH(daX, daLambda, 1.0, daHess);
            }
            else if (nKnStatus == KnitroJava.KTR_RC_EVALH_NO_F)
            {
                //---- KNITRO WANTS daHess EVALUATED AT THE POINT x
                //---- WITHOUT OBJECTIVE COMPONENT.
                daX = solver.getCurrentX();
                daLambda = solver.getCurrentLambda();
                this.evaluateH(daX, daLambda, 0.0, daHess);
            }

            //---- ASSUME THAT PROBLEM EVALUATION IS ALWAYS SUCCESSFUL.
            //---- IF A FUNCTION OR ITS DERIVATIVE COULD NOT BE EVALUATED
            //---- AT THE GIVEN (x, lambda), THEN SET nEvalStatus = 1 BEFORE
            //---- CALLING solve AGAIN.
            nEvalStatus = 0;
        }
        while (nKnStatus > 0);

        //---- DISPLAY THE RESULTS.
        System.out.print ("KNITRO finished, status " + nKnStatus + ": ");
        switch (nKnStatus)
        {
            case KnitroJava.KTR_RC_OPTIMAL:
                System.out.println ("converged to optimality.");
                break;
            case KnitroJava.KTR_RC_ITER_LIMIT:
                System.out.println ("reached the maximum number of allowed iterations.");
                break;
            case KnitroJava.KTR_RC_NEAR_OPT:
            case KnitroJava.KTR_RC_FEAS_XTOL:
            case KnitroJava.KTR_RC_FEAS_FTOL:
            case KnitroJava.KTR_RC_FEAS_NO_IMPROVE:
                System.out.println ("could not improve upon the current iterate.");
                break;
            case KnitroJava.KTR_RC_TIME_LIMIT:
                System.out.println ("reached the maximum CPU time allowed.");
                break;
            default:
                System.out.println ("failed.");
        }

        //---- EXAMPLES OF OBTAINING SOLUTION INFORMATION.
        System.out.println ("  optimal value = " + daObj[0]);
        daX = solver.getCurrentX();
        daLambda = solver.getCurrentLambda();
        System.out.println ("  solution feasibility violation    = "
                + solver.getAbsFeasError());
        System.out.println ("           KKT optimality violation = "
                + solver.getAbsOptError());
        System.out.println ("  number of function evaluations    = "
                + solver.getNumberFCEvals());

        //---- BE CERTAIN THE NATIVE OBJECT INSTANCE IS DESTROYED.
        solver.destroyInstance();

        return daX;
    }
}
