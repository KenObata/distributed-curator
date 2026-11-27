package fasttext.cpp_mirror

/**
 * Pure mirror of fastText's `SoftmaxLoss::computeOutput` (src/loss.cc). Given raw logits, returns the softmax
 * probability distribution — nothing else. No label selection, no Option wrapping: that's PR-7a-specific behavior and
 * belongs in FastTextScorer, not here.
 *
 * CONTRACT: same as FastTextDictionary — this file should be boringly faithful to loss.cc so a parity bug narrows to
 * "check this file" without wading through our own scoring decisions.
 */
object FastTextSoftmaxLoss {

  /**
   * SoftmaxLoss::computeOutput (loss.cc):
   *
   * real max = output[0], z = 0.0; for (i = 0; i < osz_; i++) max = std::max(output[i], max); for (i = 0; i < osz_;
   * i++) { output[i] = std::exp(output[i] - max); z += output[i]; } for (i = 0; i < osz_; i++) output[i] /= z;
   *
   * Max-shift is for numerical stability (avoids exp() overflow on large logits) and does not change the mathematical
   * result. Accumulation and the exp() result are float32 (real = float); JVM's Math.exp is double->double, narrowed
   * back to float to match C++'s std::exp(float) as closely as the JVM allows — this is the one knowingly inexact step,
   * covered by the 1e-6 parity gate.
   *
   * Mutates and returns `logits` in place (matching the C++ in-place `output` array), to avoid an extra allocation per
   * score() call at 2.53B-document scale.
   */
  def softmax(logits: Array[Float]): Array[Float] = {
    val n    = logits.length
    var maxV = Float.NegativeInfinity
    var k    = 0
    while (k < n) {
      if (logits(k) > maxV) maxV = logits(k)
      k += 1
    }
    var z = 0.0f
    k = 0
    while (k < n) {
      logits(k) = math.exp((logits(k) - maxV).toDouble).toFloat
      z += logits(k)
      k += 1
    }
    k = 0
    while (k < n) {
      logits(k) = logits(k) / z
      k += 1
    }
    logits
  }

}
