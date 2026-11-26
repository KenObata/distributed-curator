package fasttext

import java.nio.channels.FileChannel

import java.nio.file.{Files, Path, StandardOpenOption}
import java.nio.{ByteBuffer, ByteOrder, FloatBuffer}

/**
 * Row-major float32 matrix backed by memory-mapped file segments. This file contains mostly matrix operation for each
 * doc. Only mmap() is called once by the loader. Why mmap one-off function lives in this per doc loop process file?
 * because mmap's segment layout (rowsPerSegment) is the invariant that addRowInto/dotRow's index arithmetic depends on.
 * They change together.
 *
 * Why segments: a single MappedByteBuffer is capped at Integer.MAX_VALUE bytes (~2 GiB). oh-eli5's input matrix is
 * (3,777,339 + 2,000,000) rows x 100 cols x 4 B = 2,310,935,600 B — over the cap. We map fixed-size row-aligned
 * segments instead.
 *
 * Reads are absolute-indexed (no buffer position mutation) → thread-safe for concurrent readers.
 */
final class RowMajorFloatMatrix private (
  val rows: Int,
  val cols: Int,
  rowsPerSegment: Int,
  segments: Array[FloatBuffer]
) {

  /**
   * hidden(i) += this(row, i) for all i. Sequential adds in index order to match C++ Vector::addRow accumulation order
   * (float32).
   */
  def addRowInto(row: Int, acc: Array[Float]): Unit = {
    val seg  = segments(row / rowsPerSegment)
    val base = (row % rowsPerSegment) * cols
    var i    = 0
    while (i < cols) {
      acc(i) += seg.get(base + i)
      i += 1
    }
  }

  /** Float32 dot product of matrix row with vec, accumulated in float to match C++ DenseMatrix::dotRow (real = float). */
  def dotRow(row: Int, vec: Array[Float]): Float = {
    val seg  = segments(row / rowsPerSegment)
    val base = (row % rowsPerSegment) * cols
    var d    = 0.0f
    var i    = 0
    while (i < cols) {
      d += seg.get(base + i) * vec(i)
      i += 1
    }
    d
  }

}

object RowMajorFloatMatrix {
  // 1 GiB worth of rows per segment, row-aligned. Any value with
  // rowsPerSegment * cols * 4 <= Integer.MAX_VALUE is correct; 1 GiB
  // keeps segment count small (3 for oh-eli5's input matrix).
  private val TargetSegmentBytes: Long = 1L << 30

  def mmap(path: Path, rows: Int, cols: Int): RowMajorFloatMatrix = {
    val rowBytes      = cols.toLong * 4L
    val expectedBytes = rows.toLong * rowBytes
    val actualBytes   = Files.size(path)
    require(
      actualBytes == expectedBytes,
      s"$path holds $actualBytes bytes, manifest declares $expectedBytes"
    )

    val rowsPerSegment = math.max(1L, TargetSegmentBytes / rowBytes).toInt
    val nSegments      = ((rows.toLong + rowsPerSegment - 1) / rowsPerSegment).toInt
    val segments       = new Array[FloatBuffer](nSegments)

    val channel = FileChannel.open(path, StandardOpenOption.READ)
    try {
      var s = 0
      while (s < nSegments) {
        val firstRow = s.toLong * rowsPerSegment
        val segRows  = math.min(rowsPerSegment.toLong, rows.toLong - firstRow)
        val byteBuf: ByteBuffer = channel.map(
          FileChannel.MapMode.READ_ONLY,
          firstRow * rowBytes,
          segRows * rowBytes
        )
        // Artifacts are written little-endian by the converter
        // (manifest byte_order = "little"); asFloatBuffer inherits
        // the order set here.
        segments(s) = byteBuf.order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        s += 1
      }
    } finally
      // Mappings stay valid after the channel is closed.
      channel.close()
    new RowMajorFloatMatrix(rows, cols, rowsPerSegment, segments)
  }

}
