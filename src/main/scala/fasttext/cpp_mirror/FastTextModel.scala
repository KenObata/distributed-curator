package fasttext.cpp_mirror

import org.eclipse.collections.impl.list.mutable.primitive.IntArrayList

import fasttext.RowMajorFloatMatrix

/**
 * Pure mirror of fastText's `Model` class (src/model.cc) — the part that runs before the loss function. Two
 * responsibilities, matching the two things model.cc does before handing off to Loss:
 *
 * Model::computeHidden -> computeHidden() (average input rows) Model::computeOutputSoftmax (the Wo * hidden loop,
 * before it delegates to loss_->computeOutput) -> forward() (dot-product logits)
 *
 * Softmax itself is NOT here — that's loss.cc, mirrored separately in FastTextSoftmaxLoss. Label selection / Option
 * wrapping is NOT here either — that's PR-7a's own logic, in FastTextScorer.
 *
 * CONTRACT: same as FastTextDictionary and FastTextSoftmaxLoss — every line here should trace to model.cc. Stateless,
 * no mutable fields; safe to share across threads.
 */
object FastTextModel {

  /**
   * Runs a document's input ids through Model's forward pass and returns raw logits (pre-softmax). `ids` are the row
   * indices from FastTextDictionary.collectInputIds.
   */
  def forward(
    inputMatrix: RowMajorFloatMatrix,
    outputMatrix: RowMajorFloatMatrix,
    ids: IntArrayList,
    dim: Int,
    nLabels: Int
  ): Array[Float] = {
    val hidden = computeHidden(inputMatrix, ids, dim)

    // Model::computeOutputSoftmax (model.cc), the part before the
    // loss: output[i] = wo_->dotRow(hidden, i) for each label i.
    val logits = new Array[Float](nLabels)
    var k      = 0
    while (k < nLabels) {
      logits(k) = outputMatrix.dotRow(k, hidden)
      k += 1
    }
    logits
  }

  /**
   * Model::computeHidden (model.cc):
   *
   * hidden.zero(); for (id : input) hidden.addRow(*wi_, id); hidden.mul(1.0 / input.size());
   *
   * real = float throughout; the 1.0/size quotient is computed in double then narrowed, so this narrows the same way.
   */
  private[fasttext] def computeHidden(
    inputMatrix: RowMajorFloatMatrix,
    ids: IntArrayList,
    dim: Int
  ): Array[Float] = {
    val hidden = new Array[Float](dim)
    var i      = 0
    while (i < ids.size) {
      inputMatrix.addRowInto(ids.get(i), hidden)
      i += 1
    }
    val inv = (1.0 / ids.size).toFloat
    var d   = 0
    while (d < dim) {
      hidden(d) *= inv
      d += 1
    }
    hidden
  }

}
