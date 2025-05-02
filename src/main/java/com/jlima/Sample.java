package com.jlima;

import org.ejml.simple.SimpleMatrix;

/**
 * Represents a single data sample used in the LMS algorithm.
 * <p>
 * Each sample contains:
 * <ul>
 *   <li>{@code x}: the input feature vector, including a bias term at index 0</li>
 *   <li>{@code target}: the expected output (label) for supervised learning</li>
 * </ul>
 *
 * @param x      Feature vector as a {@link SimpleMatrix}, including bias.
 * @param target Target value (label) for the input.
 */
public record Sample(SimpleMatrix x, double target) {
}
