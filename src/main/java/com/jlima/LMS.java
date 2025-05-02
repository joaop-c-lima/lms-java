package com.jlima;

import org.ejml.simple.SimpleMatrix;
import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.BitmapEncoder.BitmapFormat;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.style.markers.None;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Implements the Least Mean Squares (LMS) algorithm for supervised learning,
 * using mini-batch gradient descent on data stored in a CSV file.
 * <p>
 * This implementation supports lazy loading of batches from disk to handle large datasets.
 */
public class LMS {

    // Configuration parameters
    private final String datasetPath;
    private final int testSetSize;
    private final String target;
    private final int maxEpochs;
    private final int batchSize;
    private final double stepSize;
    private final boolean recordMse;

    // Internal state
    private List<String> labels;
    private int targetColumn;
    private int dimension;

    private SimpleMatrix weights;
    private List<Sample> testSet;
    private List<Double> mseOverIterations;
    private List<Long> trainingOffsets;
    private List<Sample> batch;

    /**
     * Constructs an LMS model with the provided hyperparameters.
     *
     * @param datasetPath    Path to the CSV dataset.
     * @param testSetSize    Number of samples to use for the test set.
     * @param target         Name of the target column.
     * @param initialWeights Optional initial weights (including bias).
     * @param maxEpochs      Maximum number of training epochs.
     * @param batchSize      Batch size for each training iteration.
     * @param stepSize       Learning rate (step size).
     * @param recordMse      Whether to record MSE on the test set over iterations.
     */
    public LMS(String datasetPath, int testSetSize, String target, double[] initialWeights,
               int maxEpochs, int batchSize, double stepSize, boolean recordMse) {
        if (!datasetPath.endsWith(".csv")) {
            throw new IllegalArgumentException("Only CSV files are supported.");
        }

        this.datasetPath = datasetPath;
        this.testSetSize = validateTestSetSize(testSetSize);
        this.target = target;
        this.maxEpochs = validateEpochs(maxEpochs);
        this.batchSize = batchSize;
        this.stepSize = stepSize;
        this.recordMse = recordMse;
        if (recordMse) this.mseOverIterations = new ArrayList<>();

        try {
            loadTrainingOffsets();
            prepareInitialWeights(initialWeights);
            loadTestSet();
        } catch (IOException e) {
            throw new RuntimeException("Failed to initialize LMS: " + e.getMessage(), e);
        }
    }

    /**
     * Trains the LMS model using mini-batch gradient descent.
     *
     * @throws IOException if reading the CSV fails.
     */
    public void train() throws IOException {
        int lastLine = 0;
        if (recordMse) mseOverIterations.add(mse());

        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            lastLine = prepareBatch(lastLine);
            SimpleMatrix gradient = new SimpleMatrix(dimension + 1, 1);
            for (Sample sample : batch) {
                gradient = gradient.plus(sample.x().scale(error(sample)));
            }
            weights = weights.plus(gradient.scale(stepSize / batchSize));
            if (recordMse) mseOverIterations.add(mse());
        }
    }

    /**
     * Extracts a sample from a CSV line.
     *
     * @param line Line from the CSV file.
     * @return A Sample containing input vector and target value.
     */
    public Sample extractSample(String line) {
        String[] tokens = line.trim().split(",");
        if (tokens.length != labels.size()) {
            throw new IllegalArgumentException("Expected " + labels.size() +
                    " columns but got " + tokens.length);
        }

        double[] x = new double[dimension + 1];
        x[0] = 1.0; // Bias term
        double target = 0.0;
        int xi = 1;

        for (int i = 0; i < tokens.length; i++) {
            double value;
            try {
                value = Double.parseDouble(tokens[i].trim());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("Invalid numeric value: " + tokens[i]);
            }

            if (i == targetColumn) {
                target = value;
            } else {
                x[xi++] = value;
            }
        }

        return new Sample(new SimpleMatrix(x), target);
    }

    /**
     * Calculates prediction error for a sample.
     *
     * @param sample The input sample.
     * @return The prediction error (target - prediction).
     */
    private double error(Sample sample) {
        return sample.target() - weights.dot(sample.x());
    }

    /**
     * Computes the mean squared error over the test set.
     *
     * @return MSE value.
     */
    private double mse() {
        return testSet.stream().mapToDouble(s -> Math.pow(error(s), 2)).average().orElse(0.0);
    }

    private void loadTestSet() throws IOException {
        if (testSetSize > trainingOffsets.size())
            throw new IllegalArgumentException("Test set size must be smaller than the number of samples.");
        Collections.shuffle(trainingOffsets);
        try (RandomAccessFile reader = new RandomAccessFile(datasetPath, "r")) {
            List<Long> testOffsets = new ArrayList<>(trainingOffsets.subList(0, testSetSize));
            trainingOffsets = new ArrayList<>(trainingOffsets.subList(testSetSize, trainingOffsets.size()));
            testSet = new ArrayList<>();
            for (Long offset : testOffsets) {
                reader.seek(offset);
                String line = reader.readLine();
                testSet.add(extractSample(line));
            }
        }
    }

    private int prepareBatch(int lastLine) throws IOException {
        try (RandomAccessFile reader = new RandomAccessFile(datasetPath, "r")) {
            batch = new ArrayList<>();
            if (lastLine + batchSize > trainingOffsets.size()) {
                Collections.shuffle(trainingOffsets);
                lastLine = 0;
            }
            for (int i = 0; i < batchSize; i++) {
                reader.seek(trainingOffsets.get(lastLine));
                String line = reader.readLine();
                batch.add(extractSample(line));
                lastLine++;
            }
        }
        return lastLine;
    }

    private void loadTrainingOffsets() throws IOException {
        try (RandomAccessFile reader = new RandomAccessFile(datasetPath, "r")) {
            String header = reader.readLine();
            this.labels = Arrays.asList(header.trim().split(","));
            labels.replaceAll(String::trim);
            this.targetColumn = labels.indexOf(target);
            if (this.targetColumn == -1) {
                throw new IllegalArgumentException("Target column not found: " + target);
            }
            this.dimension = labels.size() - 1;

            trainingOffsets = new ArrayList<>();
            long pointer;
            while ((pointer = reader.getFilePointer()) >= 0 && (reader.readLine()) != null) {
                trainingOffsets.add(pointer);
            }
        }
    }

    private void prepareInitialWeights(double[] initialWeights) {
        if (initialWeights == null) {
            weights = new SimpleMatrix(new double[dimension + 1]);
        } else if (initialWeights.length == dimension + 1) {
            weights = new SimpleMatrix(initialWeights);
        } else {
            throw new IllegalArgumentException("Initial weights must match dimension + bias.");
        }
    }

    private int validateEpochs(int epochs) {
        if (epochs < 1) throw new IllegalArgumentException("Number of max epochs must be at least 1.");
        return epochs;
    }

    private int validateTestSetSize(int testSetSize) {
        if (testSetSize < 1) throw new IllegalArgumentException("Test set size must be bigger than 0.");
        return testSetSize;
    }

    /**
     * Exports the learning curve (MSE over epochs) to a PNG image using XChart.
     *
     * @param outputPath Path to save the PNG file.
     * @throws IOException If the file cannot be written.
     */
    public void exportLearningCurve(String outputPath) throws IOException {
        if (!recordMse || mseOverIterations == null || mseOverIterations.isEmpty()) {
            throw new IllegalStateException("MSE was not recorded. Enable recordMse to track learning curve.");
        }

        List<Double> xData = IntStream.range(0, mseOverIterations.size())
                .mapToDouble(i -> i).boxed().toList();

        XYChart chart = new XYChartBuilder()
                .width(800)
                .height(600)
                .title("Learning Curve")
                .xAxisTitle("Epoch")
                .yAxisTitle("Mean Squared Error")
                .build();

        chart.getStyler().setLegendVisible(false);
        chart.getStyler().setMarkerSize(4);
        //chart.getStyler().setDefaultSeriesRenderStyle(XYChart.XYSeriesRenderStyle.Line);

        chart.addSeries("MSE", xData, mseOverIterations).setMarker(new None());

        BitmapEncoder.saveBitmap(chart, outputPath, BitmapFormat.PNG);
    }

    public void printLearningCurveCSV() {
        System.out.println("Iteration,Mean Squared Error"); // Print CSV header
        for (int i = 0; i < mseOverIterations.size(); i++) {
            System.out.println((i) + "," + mseOverIterations.get(i));
        }
    }

    /**
     * @return Path to the dataset CSV file.
     */
    public String getDatasetPath() {
        return datasetPath;
    }

    /**
     * @return Most recent training batch.
     */
    public List<Sample> getBatch() {
        return batch;
    }

    /**
     * @return List of file offsets used for training.
     */
    public List<Long> getTrainingOffsets() {
        return trainingOffsets;
    }

    /**
     * @return List of MSE values recorded during training.
     */
    public List<Double> getMseOverIterations() {
        return mseOverIterations;
    }

    /**
     * @return List of samples in the test set.
     */
    public List<Sample> getTestSet() {
        return testSet;
    }

    /**
     * @return Current weights vector.
     */
    public SimpleMatrix getWeights() {
        return weights;
    }

    /**
     * @return Number of features (excluding bias).
     */
    public int getDimension() {
        return dimension;
    }

    /**
     * @return Index of the target column.
     */
    public int getTargetColumn() {
        return targetColumn;
    }

    /**
     * @return Whether MSE is being recorded.
     */
    public boolean isRecordMse() {
        return recordMse;
    }

    /**
     * @return List of column labels from the CSV header.
     */
    public List<String> getLabels() {
        return labels;
    }

    /**
     * @return Learning rate used during training.
     */
    public double getStepSize() {
        return stepSize;
    }

    /**
     * @return Size of the training batch.
     */
    public int getBatchSize() {
        return batchSize;
    }

    /**
     * @return Number of samples in the test set.
     */
    public int getTestSetSize() {
        return testSetSize;
    }

    /**
     * @return Name of the target column.
     */
    public String getTarget() {
        return target;
    }

    /**
     * @return Maximum number of training epochs.
     */
    public int getMaxEpochs() {
        return maxEpochs;
    }
}
