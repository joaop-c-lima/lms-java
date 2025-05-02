package com.jlima;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Objects;

import static org.junit.jupiter.api.Assertions.assertEquals;

class LMSTest {

    private Path getResourcePath(String fileName) {
        try {
            return Paths.get(Objects.requireNonNull(getClass().getClassLoader().getResource(fileName)).toURI());
        } catch (URISyntaxException | NullPointerException e) {
            throw new RuntimeException("Resource not found: " + fileName, e);
        }
    }

    @Test
    void test2DFile() {
        Path dataset = getResourcePath("2d-12samples.csv");
        double[] initialWeights = new double[]{1.0,1.0};

        LMS lms = new LMS(dataset.toString(), 5, "d", initialWeights, 100, 1, 0.0001, false);

        assertEquals(List.of("x", "d"), lms.getLabels());
        assertEquals(1, lms.getDimension());
        assertEquals(1, lms.getTargetColumn());
        assertEquals(5, lms.getTestSet().size());
    }

    @Test
    void test3DFile() {
        Path dataset = getResourcePath("3d-15samples.csv");

        LMS lms = new LMS(dataset.toString(), 10, "d", null, 100, 1, 0.0001, true);

        assertEquals(List.of("x1", "x2", "d"), lms.getLabels());
        assertEquals(2, lms.getDimension());
        assertEquals(2, lms.getTargetColumn());
        assertEquals(10, lms.getTestSet().size());
    }

    @Test
    void testTraining() throws IOException {
        Path dataset = getResourcePath("2d-300samples.csv");

        LMS lms = new LMS(dataset.toString(), 25, "y", null, 1000, 25, 0.000001, true);
        lms.train();

    }

}
