package org.platanios.learn.evaluation;

import javafx.application.Application;
import javafx.embed.swing.SwingFXUtils;
import javafx.scene.Scene;
import javafx.scene.chart.AreaChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.image.WritableImage;
import javafx.stage.Stage;
import org.platanios.learn.data.DataSet;
import org.platanios.learn.data.PredictedDataInstance;
import org.platanios.learn.math.matrix.Vector;

import javax.imageio.ImageIO;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.function.Function;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ReceiverOperatingCharacteristic<T extends Vector, S> {
    private final List<Curve> curves = new ArrayList<>();

    public ReceiverOperatingCharacteristic() { }

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
                new CurvePoint((double) truePositivesNumber / (truePositivesNumber + falseNegativesNumber + 1),
                               (double) falsePositivesNumber / (falsePositivesNumber + trueNegativesNumber + 1))
        );
        for (PredictedDataInstance<T, S> prediction : predictions) {
            if (groundTruth.apply(prediction)) {
                truePositivesNumber++;
                falseNegativesNumber--;
            } else {
                falsePositivesNumber++;
                trueNegativesNumber--;
            }
            points.add(
                    new CurvePoint((double) truePositivesNumber / (truePositivesNumber + falseNegativesNumber + 1),
                                   (double) falsePositivesNumber / (falsePositivesNumber + trueNegativesNumber + 1))
            );
        }
        points.add(new CurvePoint(1, 1));
        curves.add(new Curve(name, points));
    }

    public List<Curve> getCurves() {
        return curves;
    }

    public void plotCurves() {
        Plot.addCurves(curves);
        Application.launch(Plot.class);
    }

    public static class CurvePoint {
        private final double truePositiveRate;
        private final double falsePositiveRate;

        public CurvePoint(double truePositiveRate, double falsePositiveRate) {
            this.truePositiveRate = truePositiveRate;
            this.falsePositiveRate = falsePositiveRate;
        }

        public double getTruePositiveRate() {
            return truePositiveRate;
        }

        public double getFalsePositiveRate() {
            return falsePositiveRate;
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
        private final NumberAxis xAxis = new NumberAxis("False Positive Rate", 0, 1, 0.2);
        private final NumberAxis yAxis = new NumberAxis("True Positive Rate", 0, 1, 0.2);
        private final AreaChart<Number, Number> areaChart = new AreaChart<>(xAxis, yAxis);

        public Plot() { }

        private static void addCurves(List<Curve> curves) {
            Plot.curves.addAll(curves);
        }

        @Override
        public void start(Stage stage) {
            stage.setTitle("Receiver Operating Characteristic Curve");
            areaChart.setTitle("Receiver Operating Characteristic Curve");
            areaChart.setCreateSymbols(false);
            for (Curve curve : curves) {
                XYChart.Series<Number, Number> curveSeries = new XYChart.Series<>();
                curveSeries.setName(curve.name);
                for (CurvePoint point : curve.points)
                    curveSeries.getData().add(new XYChart.Data<>(point.falsePositiveRate, point.truePositiveRate));
                areaChart.getData().add(curveSeries);
            }
            Scene scene = new Scene(areaChart, 500, 500);
            scene.getStylesheets().add("org.platanios.learn.evaluation/ReceiverOperatingCharacteristic.Plot.css");
            stage.setScene(scene);
            stage.show();
        }
    }
}
