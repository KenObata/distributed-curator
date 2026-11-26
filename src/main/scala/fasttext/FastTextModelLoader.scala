package fasttext
// NOTE: adjust package to match the repo's existing Scala package layout.

/*
FastTextModelLoader.scala — artifact loading.
- The non-obvious piece: oh-eli5's input matrix is 2.31 GB, which exceeds Java's 2 GiB single-MappedByteBuffer cap,
so the matrix is mapped as row-aligned 1 GiB segments with index arithmetic on top.
- Reads are absolute-indexed (get(base + i), no buffer position mutation),
  which is what makes one instance shareable across threads. the property 7b will rely on.

 */

import java.nio.channels.FileChannel
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths, StandardOpenOption}
import java.nio.{ByteBuffer, ByteOrder, FloatBuffer}
import java.util.{Arrays => JArrays}

import com.fasterxml.jackson.databind.{JsonNode, ObjectMapper}
import org.eclipse.collections.impl.map.mutable.primitive.ObjectIntHashMap

/**
 * Loads the converted fastText artifacts produced by PR-6's `fasttext_convert.py`. Never parses Facebook's binary
 * format.
 *
 * PR-7a scope: quality classifier (oh-eli5) only — plain softmax, supervised, minn = maxn = 0. Anything else hard-fails
 * at load time, mirroring the converter's own export-and-assert stance.
 *
 * Thread-safety: everything here is immutable after construction. Matrix reads use absolute FloatBuffer.get(index) (no
 * position state), so a single instance is safe to share across threads — that is the property PR-7b's per-executor
 * sharing relies on.
 */
object FastTextModelLoader {

  /**
   * Byte-exact map key. Vocab lookup must match C++ std::string equality on raw bytes; decoding to java.lang.String
   * would collapse distinct invalid-UTF-8 byte sequences onto U+FFFD and diverge from the oracle.
   */
  final class ByteKey(val bytes: Array[Byte]) {
    override val hashCode: Int = JArrays.hashCode(bytes)

    override def equals(other: Any): Boolean = other match {
      case that: ByteKey => JArrays.equals(bytes, that.bytes)
      case _             => false
    }

  }

  final case class Config(
    dim: Int,
    wordNgrams: Int,
    bucket: Int,
    minn: Int,
    maxn: Int,
    loss: String,
    model: String,
    labelPrefix: String,
    nWords: Int,
    nLabels: Int
  )

  /**
   * Row-major float32 matrix backed by memory-mapped file segments.
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
     * hidden(i) += this(row, i) for all i. Sequential adds in index order to match C++ Vector::addRow accumulation
     * order (float32).
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

  final class Loaded(
    val config: Config,
    val inputMatrix: RowMajorFloatMatrix,
    val outputMatrix: RowMajorFloatMatrix,
    val wordIds: ObjectIntHashMap[ByteKey],
    val labels: Array[String]
  )

  private val mapper = new ObjectMapper()

  /** Load a converted-artifact directory (manifest.json + *.f32 + vocab/labels). */
  def load(artifactDir: String): Loaded = {
    val dir      = Paths.get(artifactDir)
    val manifest = mapper.readTree(Files.readAllBytes(dir.resolve("manifest.json")))

    val config = readConfig(manifest.get("config"))
    validateSupported(config)

    val inputSpec  = manifest.get("matrices").get("input")
    val outputSpec = manifest.get("matrices").get("output")
    validateMatrixSpec(inputSpec, "input")
    validateMatrixSpec(outputSpec, "output")

    val inputMatrix = RowMajorFloatMatrix.mmap(
      dir.resolve(inputSpec.get("file").asText()),
      inputSpec.get("rows").asInt(),
      inputSpec.get("cols").asInt()
    )
    val outputMatrix = RowMajorFloatMatrix.mmap(
      dir.resolve(outputSpec.get("file").asText()),
      outputSpec.get("rows").asInt(),
      outputSpec.get("cols").asInt()
    )
    require(
      inputMatrix.rows == config.nWords + config.bucket,
      s"input matrix rows ${inputMatrix.rows} != n_words + bucket " +
        s"${config.nWords + config.bucket}"
    )
    require(outputMatrix.rows == config.nLabels, "output rows != n_labels")
    require(inputMatrix.cols == config.dim && outputMatrix.cols == config.dim, "matrix cols != dim")

    val wordIds = loadVocab(
      dir.resolve(manifest.get("vocab").get("file").asText()),
      config.nWords
    )
    val labels = loadLines(
      dir.resolve(manifest.get("labels").get("file").asText()),
      config.nLabels
    ).map(new String(_, StandardCharsets.UTF_8))

    new Loaded(config, inputMatrix, outputMatrix, wordIds, labels)
  }

  private def readConfig(node: JsonNode): Config = Config(
    dim = node.get("dim").asInt(),
    wordNgrams = node.get("wordNgrams").asInt(),
    bucket = node.get("bucket").asInt(),
    minn = node.get("minn").asInt(),
    maxn = node.get("maxn").asInt(),
    loss = node.get("loss").asText(),
    model = node.get("model").asText(),
    labelPrefix = node.get("label").asText(),
    nWords = node.get("n_words").asInt(),
    nLabels = node.get("n_labels").asInt()
  )

  private def validateSupported(c: Config): Unit = {
    require(c.loss == "softmax", s"loss=${c.loss} unsupported; PR-7a implements plain softmax only")
    require(c.model == "supervised", s"model=${c.model} unsupported")
    require(
      c.minn == 0 && c.maxn == 0,
      s"minn=${c.minn}/maxn=${c.maxn}: character subwords are out of " +
        "PR-7a scope (oh-eli5 has none)"
    )
    require(c.wordNgrams >= 1, "wordNgrams must be >= 1")
  }

  private def validateMatrixSpec(spec: JsonNode, name: String): Unit = {
    require(spec.get("dtype").asText() == "float32", s"$name dtype != float32")
    require(spec.get("byte_order").asText() == "little", s"$name byte_order != little")
    require(spec.get("layout").asText() == "row_major", s"$name layout != row_major")
  }

  /**
   * vocab.txt line i = input-matrix row i. Byte-exact keys; line split on 0x0A only, matching the converter's write
   * format.
   */
  private def loadVocab(path: Path, expectedCount: Int): ObjectIntHashMap[ByteKey] = {
    val lines = loadLines(path, expectedCount)
    val map   = new ObjectIntHashMap[ByteKey](expectedCount)
    var i     = 0
    while (i < lines.length) {
      map.put(new ByteKey(lines(i)), i)
      i += 1
    }
    map
  }

  private def loadLines(path: Path, expectedCount: Int): Array[Array[Byte]] = {
    val all   = Files.readAllBytes(path)
    val out   = new Array[Array[Byte]](expectedCount)
    var count = 0
    var start = 0
    var i     = 0
    while (i < all.length) {
      if (all(i) == '\n'.toByte) {
        require(count < expectedCount, s"$path has more lines than declared")
        out(count) = JArrays.copyOfRange(all, start, i)
        count += 1
        start = i + 1
      }
      i += 1
    }
    require(start == all.length, s"$path does not end with a newline (converter always writes one)")
    require(count == expectedCount, s"$path holds $count lines, manifest declares $expectedCount")
    out
  }

}
