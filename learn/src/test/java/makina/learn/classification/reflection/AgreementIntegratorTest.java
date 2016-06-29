package makina.learn.classification.reflection;

import org.junit.Assert;
import org.junit.Test;
import makina.learn.classification.Label;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class AgreementIntegratorTest {
    @Test
    public void testAgreementRatesPowerSetVector() {
        InputStream dataInputStream = AgreementIntegratorTest.class.getResourceAsStream("./nell/animal.csv");
        double[] classificationThresholds = new double[] { 0.05, 0.05, 0.1, 0.05 };
        InputData data = parseLabeledDataFromCSVFile(dataInputStream, ",", classificationThresholds);
        AgreementIntegrator errorEstimation =
                new AgreementIntegrator.Builder(data.predictedData, data.observedData)
                        .highestOrder(4)
                        .onlyEvenCardinalitySubsetsAgreements(false)
                        .build();
        double[] actualResult = errorEstimation.agreementRates().get(new Label("label")).array();
        double[] expectedResult = new double[]{
                0.646698499975884,
                0.698548208170549,
                0.575362947957363,
                0.693290888921044,
                0.583803598128587,
                0.607678580041480,
                0.519268798533739,
                0.402932523030917,
                0.440794868084696,
                0.442386533545555,
                0.337481309988907
        };
        Assert.assertArrayEquals(expectedResult, actualResult, 1E-15);
    }

    @Test
    public void testErrorRatesEstimationVector() {
        InputStream dataInputStream = AgreementIntegratorTest.class.getResourceAsStream("./nell/animal.csv");
        double[] classificationThresholds = new double[] { 0.05, 0.05, 0.1, 0.05 };
        InputData data = parseLabeledDataFromCSVFile(dataInputStream, ",", classificationThresholds);
        AgreementIntegrator errorEstimation =
                new AgreementIntegrator.Builder(data.predictedData, data.observedData)
                        .highestOrder(4)
                        .onlyEvenCardinalitySubsetsAgreements(true)
                        .build();
        errorEstimation.errorRates();
        double[] obtainedErrorRates = errorEstimation.errorRatesVectors().get(new Label("label")).array;
        double[] sampleErrorRates = errorEstimation.sampleErrorRates().get(new Label("label")).array;
        double mad = 0.0;
        double mad_ind = 0.0;
        for (int i = 0; i < obtainedErrorRates.length; i++) {
            mad += Math.abs(sampleErrorRates[i] - obtainedErrorRates[i]);
            if (i < 4)
                mad_ind += Math.abs(sampleErrorRates[i] - obtainedErrorRates[i]);
        }
        mad /= sampleErrorRates.length;
        mad_ind /= 4;
        Assert.assertEquals(0.02369546827159496, mad, 1E-10);
        Assert.assertEquals(0.04165557759034837, mad_ind, 1E-10);
    }

//    @Test
//    public void testErrorEstimationSimpleGraphicalModel() {
//        InputStream dataInputStream = AgreementErrorEstimationTest.class.getResourceAsStream("./nell/animal.csv");
//        String separator = ",";
//        int highestOrder = 4;
//        double[] classificationThresholds = new double[] { 0.05, 0.05, 0.1, 0.05 };
//
//        AgreementErrorEstimation.SufficientData data = parseLabeledDataFromCSVFile(dataInputStream,
//                                                                                   separator,
//                                                                                   classificationThresholds,
//                                                                                   highestOrder,
//                                                                                   true);
//        dataInputStream = AgreementErrorEstimationTest.class.getResourceAsStream("./nell/animal.csv");
//        List<boolean[][]> functionOutputs = parseLabeledDataFromCSVFileForSimpleGM(dataInputStream,
//                                                                                   separator,
//                                                                                   classificationThresholds);
//        BayesianErrorEstimation eegm = new BayesianErrorEstimation(functionOutputs, 90, 1, 10);
//        eegm.runGibbsSampler();
//
//        double[] obtainedErrorRates = eegm.getErrorRatesMeans()[0];
//        double[] sampleErrorRates = data.getSampleErrorRates().array;
//
//        double mad_ind = 0.0;
//        for (int i = 0; i < data.getNumberOfFunctions(); i++)
//            mad_ind += Math.abs(sampleErrorRates[i] - obtainedErrorRates[i]);
//        mad_ind /= data.getNumberOfFunctions();
//
//        Assert.assertEquals(0.03873716293662867, mad_ind, 1E-10);
//    }

//    @Test
//    public void testErrorRatesEstimationVectorWith3Classifiers() {
//        InputStream dataInputStream = ErrorEstimationTest.class.getResourceAsStream("./nell/input/animal.csv");
//        String separator = ",";
//        int highestOrder = 3;
//        double[] classificationThresholds = new double[] { 0.05, 0.05, 0.1 };
//        ErrorEstimationData data = parseLabeledDataFromCSVFileWith3Classifiers(dataInputStream,
//                                                                               separator,
//                                                                               classificationThresholds,
//                                                                               highestOrder,
//                                                                               true);
//        ErrorEstimation ere = new ErrorEstimation.Builder(data)
//                .optimizationSolverType(ErrorEstimationInternalSolver.IP_OPT)
//                .build();
//        data = ere.solve();
//
//        double[] obtainedErrorRates = data.getErrorRates().array;
//        double[] sampleErrorRates = data.getSampleErrorRates().array;
//
//        double mad = 0.0;
//        double mad_ind = 0.0;
//
//        for (int i = 0; i < obtainedErrorRates.length; i++) {
//            mad += Math.abs(sampleErrorRates[i] - obtainedErrorRates[i]);
//            if (i < data.getNumberOfFunctions()) {
//                mad_ind += Math.abs(sampleErrorRates[i] - obtainedErrorRates[i]);
//            }
//        }
//
//        mad /= sampleErrorRates.length;
//        mad_ind /= data.getNumberOfFunctions();
//
//        Assert.assertEquals(0.022264846242268578, mad, 1E-10);
//        Assert.assertEquals(0.03873716293662867, mad_ind, 1E-10);
//    }

    private static InputData parseLabeledDataFromCSVFile(InputStream dataInputStream, String separator) {
        return parseLabeledDataFromCSVFile(dataInputStream, separator, null);
    }

    private static InputData parseLabeledDataFromCSVFile(InputStream dataInputStream,
                                                        String separator,
                                                        double classificationThreshold) {
        return parseLabeledDataFromCSVFile(dataInputStream, separator, new double[] { classificationThreshold });
    }

    private static InputData parseLabeledDataFromCSVFile(InputStream dataInputStream,
                                                        String separator,
                                                        double[] classificationThresholds) {
        List<Integrator.Data.PredictedInstance> predictedData = new ArrayList<>();
        List<Integrator.Data.ObservedInstance> observedData = new ArrayList<>();
        BufferedReader br = null;
        String line;
        try {
            br = new BufferedReader(new InputStreamReader(dataInputStream));
            br.readLine();
            Label label = new Label("label");
            int lineNumber = 0;
            while ((line = br.readLine()) != null) {
                String[] outputs = line.split(separator);
                observedData.add(new Integrator.Data.ObservedInstance(lineNumber, label, !outputs[0].equals("0")));
                for (int i = 1; i < outputs.length; i++) {
                    double value;
                    if (classificationThresholds == null)
                        value = Double.parseDouble(outputs[i]) >= 0.5 ? 1.0 : 0.0;
                    else if (classificationThresholds.length == 1)
                        value = Double.parseDouble(outputs[i]) >= classificationThresholds[0] ? 1.0 : 0.0;
                    else
                        value = Double.parseDouble(outputs[i]) >= classificationThresholds[i - 1] ? 1.0 : 0.0;
                    predictedData.add(new Integrator.Data.PredictedInstance(lineNumber, label, i - 1, value));
                }
                lineNumber++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (br != null)
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
        }
        return new InputData(new Integrator.Data<>(predictedData), new Integrator.Data<>(observedData));
    }
//
//    public static List<boolean[][]> parseLabeledDataFromCSVFileForSimpleGM(InputStream dataInputStream,
//                                                                           String separator,
//                                                                           double[] classificationThresholds) {
//        BufferedReader br = null;
//        String line;
//        String[] classifiersNames = null;
//        List<boolean[]> classifiersOutputsList = new ArrayList<>();
//        List<Boolean> trueLabelsList = new ArrayList<Boolean>();
//
//        try {
//            br = new BufferedReader(new InputStreamReader(dataInputStream));
//            line = br.readLine();
//            classifiersNames = line.split(separator);
//            while ((line = br.readLine()) != null) {
//                String[] outputs = line.split(separator);
//                trueLabelsList.add(!outputs[0].equals("0"));
//                boolean[] booleanOutputs = new boolean[outputs.length - 1];
//                for (int i = 1; i < outputs.length; i++) {
//                    if (classificationThresholds == null) {
//                        booleanOutputs[i - 1] = Double.parseDouble(outputs[i]) >= 0.5;
//                    } else if (classificationThresholds.length == 1) {
//                        booleanOutputs[i - 1] = Double.parseDouble(outputs[i]) >= classificationThresholds[0];
//                    } else {
//                        booleanOutputs[i - 1] = Double.parseDouble(outputs[i]) >= classificationThresholds[i - 1];
//                    }
//                }
//                classifiersOutputsList.add(booleanOutputs);
//            }
//        } catch (IOException e) {
//            e.printStackTrace();
//        } finally {
//            if (br != null) {
//                try {
//                    br.close();
//                } catch (IOException e) {
//                    e.printStackTrace();
//                }
//            }
//        }
//
//        List<boolean[][]> functionOutputs = new ArrayList<>();
//        functionOutputs.add(classifiersOutputsList.toArray(new boolean[classifiersOutputsList.size()][]));
//
//        return functionOutputs;
//    }
//
//    public static AgreementErrorEstimation.SufficientData parseLabeledDataFromCSVFileWith3Classifiers(InputStream dataInputStream,
//                                                                                                      String separator,
//                                                                                                      double[] classificationThresholds,
//                                                                                                      int highestOrder,
//                                                                                                      boolean onlyEvenCardinalitySubsetsAgreements) {
//        int numberOfFunctions = 0;
//
//        BufferedReader br = null;
//        String line;
//        String[] classifiersNames = null;
//        List<boolean[]> classifiersOutputsList = new ArrayList<>();
//        List<Boolean> trueLabelsList = new ArrayList<Boolean>();
//
//        try {
//            br = new BufferedReader(new InputStreamReader(dataInputStream));
//            line = br.readLine();
//            classifiersNames = line.split(separator);
//            numberOfFunctions = classifiersNames.length - 1;
//            while ((line = br.readLine()) != null) {
//                String[] outputs = line.split(separator);
//                trueLabelsList.add(!outputs[0].equals("0"));
//                boolean[] booleanOutputs = new boolean[3];
//                for (int i = 1; i < 3; i++) {
//                    if (classificationThresholds == null) {
//                        booleanOutputs[i - 1] = Double.parseDouble(outputs[i]) >= 0.5;
//                    } else if (classificationThresholds.length == 1) {
//                        booleanOutputs[i - 1] = Double.parseDouble(outputs[i]) >= classificationThresholds[0];
//                    } else {
//                        booleanOutputs[i - 1] = Double.parseDouble(outputs[i]) >= classificationThresholds[i - 1];
//                    }
//                }
//                classifiersOutputsList.add(booleanOutputs);
//            }
//        } catch (IOException e) {
//            e.printStackTrace();
//        } finally {
//            if (br != null) {
//                try {
//                    br.close();
//                } catch (IOException e) {
//                    e.printStackTrace();
//                }
//            }
//        }
//
//        if (highestOrder == -1) {
//            highestOrder = numberOfFunctions;
//        }
//
//        return new AgreementErrorEstimation.SufficientData.Builder(classifiersOutputsList,
//                                                                   trueLabelsList,
//                                                                   highestOrder,
//                                                                   onlyEvenCardinalitySubsetsAgreements)
//                .functionNames(classifiersNames)
//                .build();
//    }

    private static class InputData {
        private final Integrator.Data<Integrator.Data.PredictedInstance> predictedData;
        private final Integrator.Data<Integrator.Data.ObservedInstance> observedData;

        private InputData(Integrator.Data<Integrator.Data.PredictedInstance> predictedData,
                          Integrator.Data<Integrator.Data.ObservedInstance> observedData) {
            this.predictedData = predictedData;
            this.observedData = observedData;
        }
    }
}
