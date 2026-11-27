/*
This is a util file for FastTextScorer.scala
 and this is a mirror from fasttext's C++ mirror.cpp
 so the logic is copied form them. But written in scala.
 */
package fasttext.cpp_mirror

import java.nio.charset.StandardCharsets

import org.eclipse.collections.impl.list.mutable.primitive.IntArrayList
import org.eclipse.collections.impl.map.mutable.primitive.ObjectIntHashMap

import fasttext.FastTextModelLoader.ByteKey

/**
 * Pure mirror of fastText's `Dictionary` class (src/dictionary.cc).
 *
 * CONTRACT: every method here should trace to a specific C++ function and match it bit-for-bit. Nothing PR-7a-specific
 * (label selection, softmax, Option[Double] wrapping) belongs in this file — that lives in FastTextScorer.scala, which
 * mirrors Model/Loss instead. Keeping this file "boringly faithful" is the point: if a parity bug shows up, this is the
 * first and only file that needs re-checking against dictionary.cc.
 *
 * Immutable, no mutable fields — safe to share across threads (see FastTextModelLoader.Loaded / PR-7b's per-executor
 * sharing contract).
 */
final class FastTextDictionary(
  word2Id: ObjectIntHashMap[ByteKey],
  labelPrefix: String,
  wordNgrams: Int,
  nWords: Int,
  bucket: Int
) {
  private val labelPrefixBytes = labelPrefix.getBytes(StandardCharsets.UTF_8)
  private val eosBytes         = "</s>".getBytes(StandardCharsets.UTF_8)

  /**
   * Dictionary::getLine equivalent for prediction, followed by Dictionary::addWordNgrams. Returns the full list of
   * input_matrix row indices for one document: vocab-word rows, hashed-OOV/bigram rows, in fastText's own order.
   *
   * for each token: in vocab -> push its row id; push its hash label-prefixed OOV -> skip entirely (no id, no hash)
   * other OOV -> push its hash ONLY (minn=maxn=0, no subword fallback — see addSubwords) then addWordNgrams over the
   * collected hashes.
   *
   * EOS ("</s>") is a real vocab entry: it gets an id and a hash like any word, so the final bigram is (last_word,
   * </s>).
   *
   * Args:
   *   - word2Id: dictionary from vocab.txt {"not": 1024}
   *   - wordNgrams: defined in manifest.json ex) 2
   *   - nWords: how many words are in the dictionary, which is the same as the number of lines in vocab.txt
   *   - bucket: how many rows do we allocate in the fasttext matrix for bigram, defined in manifest.json
   *
   * Return: IntArrayList
   *   - ArrayList because The size isn't known upfront. It has to grow as you scan the document
   *   - IntArrayList instead of ArrayBuffer[Int] because Every ArrayBuffer.add() allocates a box
   */
  def collectInputIds(bytes: Array[Byte]): IntArrayList = {
    val ids        = new IntArrayList()
    val wordHashes = new IntArrayList() // int32 hashes, as C++ vector<int32_t>

    var start = 0
    var i     = 0
    while (i <= bytes.length) {
      val atSeparator = i == bytes.length || isSeparator(bytes(i))
      if (atSeparator) {
        if (i > start) addToken(bytes, start, i, ids, wordHashes)
        start = i + 1
      }
      i += 1
    }
    addToken(eosBytes, 0, eosBytes.length, ids, wordHashes)

    addWordNgrams(ids, wordHashes, wordNgrams)
    ids
  }

  /**
   * Dictionary::readWord separators — these 7 raw byte values and NOTHING else. Deliberately not Character.isWhitespace
   * / (?U)\s: fastText's C++ never treats multi-byte Unicode whitespace (e.g. U+3000) as a separator, and the trained
   * weights reflect that.
   */
  @inline private[fasttext] def isSeparator(b: Byte): Boolean = b == ' ' || b == '\n' || b == '\r' || b == '\t' ||
    b == 0x0b || b == 0x0c || b == 0x00

  private def addToken(
    bytes: Array[Byte],
    start: Int,
    end: Int,
    ids: IntArrayList,
    wordHashes: IntArrayList
  ): Unit = {
    val token = java.util.Arrays.copyOfRange(bytes, start, end)
    val id    = word2Id.getIfAbsent(new ByteKey(token), -1)
    if (id >= 0) {
      ids.add(id)
      wordHashes.add(fnv1aHash(token))
    } else if (!startsWithLabelPrefix(token)) {
      wordHashes.add(fnv1aHash(token))
    }
    // label-prefixed OOV token: dropped, matching getLine's label branch.
  }

  private[fasttext] def startsWithLabelPrefix(token: Array[Byte]): Boolean = {
    if (token.length < labelPrefixBytes.length) return false
    var i = 0
    while (i < labelPrefixBytes.length) {
      if (token(i) != labelPrefixBytes(i)) return false
      i += 1
    }
    true
  }

  /**
   * Dictionary::hash (dictionary.cc):
   *
   * uint32_t h = 2166136261; for (...) { h = h ^ uint32_t(int8_t(str[i])); h = h * 16777619; }
   *
   * The int8_t cast SIGN-EXTENDS bytes >= 0x80 before the XOR (GitHub issue #553: models were trained with signed-char
   * semantics). Java bytes are already signed, so plain `h ^ b` promotes with sign extension and matches exactly. Int
   * overflow on the multiply wraps mod 2^32, same as uint32_t.
   *
   * Verified against reference values computed from the C++ formula: "not" -> 699505802, "good" -> -94359080 (as
   * int32), "</s>" -> -677604519, "日本語" -> 308035559.
   */
  private[fasttext] def fnv1aHash(token: Array[Byte]): Int = {
    var h = FastTextDictionary.Fnv1aSeed
    var i = 0
    while (i < token.length) {
      h = h ^ token(i) // byte -> int promotion sign-extends, as C++ does
      h = h * FastTextDictionary.Fnv1aPrime
      i += 1
    }
    h
  }

  /**
   * Dictionary::addWordNgrams (dictionary.cc):
   *
   * for i: uint64_t h = hashes[i]; // int32 -> uint64 SIGN-EXTENDS for j in (i+1, i+n): h = h*116049371 + hashes[j]; //
   * sign-extends too pushHash(line, h % bucket) // unsigned modulo pushHash appends nwords_ + id (unpruned model).
   *
   * Scala Int -> Long conversion sign-extends, matching the C++ int32_t -> uint64_t bit pattern. The modulo must be
   * UNSIGNED: Long.remainderUnsigned, never %, because the running product routinely has the top bit set.
   */
  private[fasttext] def addWordNgrams(
    ids: IntArrayList,
    hashes: IntArrayList,
    n: Int
  ): Unit = {
    if (n <= 1) return
    val bucketL = bucket.toLong
    var i       = 0
    while (i < hashes.size) {
      var h: Long = hashes.get(i).toLong // sign-extends
      var j       = i + 1
      while (j < hashes.size && j < i + n) {
        h = h * 116049371L + hashes.get(j).toLong // sign-extends
        ids.add(nWords + java.lang.Long.remainderUnsigned(h, bucketL).toInt)
        j += 1
      }
      i += 1
    }
  }

}

object FastTextDictionary {
  private[fasttext] val Fnv1aSeed: Int  = 0x811c9dc5 // 2166136261 as int32
  private[fasttext] val Fnv1aPrime: Int = 16777619
}
