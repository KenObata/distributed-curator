package fasttext
/*
Goal of this file:
 - The inference core, with every parity-critical line annotated with the C++ it mirrors.
Hash function:
 - FNV-1a (Fowler–Noll–Vo, variant "a") is a simple, fast, non-cryptographic hash function.
 - This is to mirror fastText's C++ implementation
 */

import java.nio.charset.StandardCharsets

import FastTextModelLoader.Loaded
import fasttext.cpp_mirror.{FastTextDictionary, FastTextModel, FastTextSoftmaxLoss}

/**
 * PR-7a's own logic: wires the C++ mirrors together and picks one label's probability out of the result. Nothing in
 * this file should need to cite a specific C++ source line — that's the point of having pulled everything that does
 * into its own file:
 *
 * FastTextDictionary -> dictionary.cc (tokenize, hash, bigram) FastTextModel -> model.cc (hidden layer, output logits)
 * FastTextSoftmaxLoss -> loss.cc (softmax)
 *
 * Artifact loading (manifest, mmap, vocab) is FastTextModelLoader — deliberately a different name from FastTextModel,
 * which mirrors fastText's C++ `Model` class specifically.
 *
 * This class's only jobs: build those collaborators from the loaded artifacts, run a document through them in order,
 * and select the probability the caller actually wants.
 *
 * Concurrency contract for PR-7b: this class is immutable; all mutable scratch (hidden vector, id list, logits) is
 * method-local per score() call. Share one instance per executor; call score() from any thread.
 */
final class FastTextScorer(model: Loaded, positiveLabel: String) {

  private val config = model.config

  private val dictionary = new FastTextDictionary(
    model.wordIds,
    config.labelPrefix,
    config.wordNgrams,
    config.nWords,
    config.bucket
  )

  private val positiveIndex: Int = {
    val idx = model.labels.indexOf(positiveLabel)
    require(idx >= 0, s"label $positiveLabel not in labels.txt: ${model.labels.mkString(", ")}")
    idx
  }

  /**
   * Probability of `positiveLabel` for one document, or None when the document produces no input features (fastText
   * returns no prediction for an empty line — in practice this never happens here since EOS alone still yields one id,
   * but the branch is kept for defensiveness / future non-EOS-appending callers).
   *
   * With n_labels = 2 (verified at load in FastTextModelLoader), the complementary label's probability is exactly 1 -
   * result, by construction of softmax over exactly two logits — not an approximation.
   *
   * PARITY CONTRACT (verify against the PR-5 oracle in the fixture test): fastText's Python predict() processes ONE
   * line — it rejects embedded '\n' and appends '\n' itself, which becomes the EOS token. This scorer therefore treats
   * all 7 separator bytes (including \n) as plain word boundaries and appends exactly one EOS at the end. That matches
   * an oracle that replaces newlines with spaces before predict(). If PR-5 preprocessed differently, mirror it here.
   */
  def score(text: String): Option[Double] = {
    val ids = dictionary.collectInputIds(text.getBytes(StandardCharsets.UTF_8))
    if (ids.isEmpty) return None

    val logits = FastTextModel.forward(
      model.inputMatrix,
      model.outputMatrix,
      ids,
      config.dim,
      config.nLabels
    )
    val probs = FastTextSoftmaxLoss.softmax(logits)
    Some(probs(positiveIndex).toDouble)
  }

}

object FastTextScorer {

  /**
   * Convenience: load artifacts and build a scorer for __label__hq. reason why only need __label__hq is because if we
   * now __label__hq, we can derive __label__cc as complement
   */
  def forQualityModel(artifactDir: String): FastTextScorer =
    new FastTextScorer(FastTextModelLoader.load(artifactDir), "__label__hq")

}
