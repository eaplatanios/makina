package org.platanios.learn.evaluation;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.AreaChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.stage.Stage;
import javafx.util.converter.NumberStringConverter;
import org.platanios.learn.data.DataSet;
import org.platanios.learn.data.PredictedDataInstance;
import org.platanios.learn.math.MathUtilities;
import org.platanios.learn.math.matrix.Vector;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.function.Function;

/**
 * @author Emmanouil Antonios Platanios
 */
public class PrecisionRecall<T extends Vector, S> {
    private final double epsilon = MathUtilities.computeMachineEpsilonDouble();
    private final List<Curve> curves = new ArrayList<>();
    private final List<Double> areaUnderCurves = new ArrayList<>();

    public PrecisionRecall() { }

    // TODO: It is maybe better to have a getList() method in the data set classes.
    public void addCurve(String name,
                         DataSet<PredictedDataInstance<T, S>> predictions,
                         Function<PredictedDataInstance<T, S>, Boolean> groundTruth) {
        List<PredictedDataInstance<T, S>> predictionsList = new ArrayList<>();
        predictions.forEach(predictionsList::add);
        addCurve(name, predictionsList, groundTruth);
    }

    public void addCurve(String name,
                         List<PredictedDataInstance<T, S>> predictions,
                         Function<PredictedDataInstance<T, S>, Boolean> groundTruth) {
        List<CurvePoint> points = new ArrayList<>();
        Collections.sort(predictions, Comparator.comparing(PredictedDataInstance::probability));
        int truePositivesNumber = 0;
        int trueNegativesNumber = 0;
        int falsePositivesNumber = 0;
        int falseNegativesNumber = 0;
        for (PredictedDataInstance<T, S> prediction : predictions)
            if (groundTruth.apply(prediction))
                falseNegativesNumber++;
            else
                trueNegativesNumber++;
        points.add(
                new CurvePoint((truePositivesNumber + epsilon) / (truePositivesNumber + falsePositivesNumber + epsilon),
                               (truePositivesNumber + epsilon) / (truePositivesNumber + falseNegativesNumber + epsilon))
        );
        double areaUnderCurve = 0;
        int previousThresholdPredictionIndex = 0;
        for (double thresholdIndex = 0; thresholdIndex <= 1000; thresholdIndex++) {
            double threshold = 1 - thresholdIndex / 1000;
            PredictedDataInstance<T, S> prediction = predictions.get(previousThresholdPredictionIndex);
            while (prediction.probability() >= threshold) {
                if (groundTruth.apply(prediction)) {
                    falseNegativesNumber--;
                    truePositivesNumber++;
                } else {
                    trueNegativesNumber--;
                    falsePositivesNumber++;
                }
                points.add(
                        new CurvePoint((truePositivesNumber + epsilon) / (truePositivesNumber + falsePositivesNumber + epsilon),
                                       (truePositivesNumber + epsilon) / (truePositivesNumber + falseNegativesNumber + epsilon))
                );
                int k = points.size() - 1;
                areaUnderCurve += 0.5
                        * (points.get(k).recall - points.get(k - 1).recall)
                        * (points.get(k).precision + points.get(k - 1).precision);
                if (++previousThresholdPredictionIndex < predictions.size())
                    prediction = predictions.get(previousThresholdPredictionIndex);
                else
                    break;
            }
            if (previousThresholdPredictionIndex == predictions.size())
                break;
        }
        areaUnderCurve += 0.5
                * (1 - points.get(points.size() - 2).recall)
                * (1 + points.get(points.size() - 2).precision);
        curves.add(new Curve(name, points));
        areaUnderCurves.add(areaUnderCurve);
    }

    public List<Curve> getCurves() {
        return curves;
    }

    public void plotCurves() {
        Plot.addCurves(curves, areaUnderCurves);
        Application.launch(Plot.class);
    }

    public static class CurvePoint {
        private final double precision;
        private final double recall;

        public CurvePoint(double precision, double recall) {
            this.precision = precision;
            this.recall = recall;
        }

        public double getPrecision() {
            return precision;
        }

        public double getRecall() {
            return recall;
        }
    }

    public static class Curve {
        private final String name;
        private final List<CurvePoint> points;

        public Curve(String name, List<CurvePoint> points) {
            this.name = name;
            this.points = points;
        }

        public String getName() {
            return name;
        }

        public List<CurvePoint> getPoints() {
            return points;
        }
    }

    public static class Plot extends Application {
        private static final List<Curve> curves = new ArrayList<>();
        private static final List<Double> areaUnderCurves = new ArrayList<>();

        private final NumberAxis xAxis = new NumberAxis("Recall", 0, 1, 0.2);
        private final NumberAxis yAxis = new NumberAxis("Precision", 0, 1, 0.2);
        private final AreaChart<Number, Number> areaChart = new AreaChart<>(xAxis, yAxis);

        public Plot() { }

        private static void addCurves(List<Curve> curves, List<Double> areaUnderCurves) {
            Plot.curves.addAll(curves);
            Plot.areaUnderCurves.addAll(areaUnderCurves);
        }

        @Override
        public void start(Stage stage) {
            stage.setTitle("Precision-Recall Curve");
            areaChart.setTitle("Precision-Recall Curve");
            areaChart.setCreateSymbols(false);
            xAxis.setTickLabelFormatter(new NumberStringConverter("0.0"));
            yAxis.setTickLabelFormatter(new NumberStringConverter("0.0"));
            for (int curveIndex = 0; curveIndex < curves.size(); curveIndex++) {
                XYChart.Series<Number, Number> curveSeries = new XYChart.Series<>();
                curveSeries.setName(curves.get(curveIndex).name
                                            + " - AUC: " + String.format("%.4f", areaUnderCurves.get(curveIndex)));
                for (CurvePoint point : curves.get(curveIndex).points)
                    curveSeries.getData().add(new XYChart.Data<>(point.recall, point.precision));
                areaChart.getData().add(curveSeries);
            }
            Scene scene = new Scene(areaChart, 350, 400);
            scene.getStylesheets().add("org.platanios.learn.evaluation/ReceiverOperatingCharacteristic.Plot.css");
            stage.setScene(scene);
            stage.show();
        }
    }
}
