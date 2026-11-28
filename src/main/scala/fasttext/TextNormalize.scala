package fasttext

/*
Goal is to do input normalization (= convert to one line),
 applied BEFORE fastText scoring.
Why? because fasttext can only take a line, not a paragraph.

Not a C++ mirror (fastText itself does no normalization),
so it lives outside cpp_mirror/.
It reproduces, character for character, the Python preprocessing used to build the PR-5
oracle fixtures (originally DCLM's classify_fasttext_hq_prob):

    " ".join(text.strip().splitlines())

Character sets below were enumerated from CPython 3.12 (str.isspace / str.splitlines)
over the whole BMP. All of them are BMP chars, so iterating Java UTF-16 `Char`s is exact
(surrogate halves never match).
 */
object TextNormalizer {

  /**
   * Equivalent of Python `" ".join(text.strip().splitlines())`.
   *
   * ex)
   *   - "a\nb" -> "a b"
   *   - "a\r\nb" -> "a b" (\r\n is ONE line break)
   *   - "a\n\nb" -> "a b" (empty line in the middle -> two spaces, as Python)
   *   - "a\u2028b" -> "a b" (Unicode line separator)
   *   - " a b \n" -> "a b"
   *   - "a\u3000b" -> "a\u3000b" (U+3000 is whitespace, NOT a line break: kept in the middle)
   *   - " \n " or "" -> ""
   */
  def normalize(text: String): String = {
    // str.strip(): drop leading/trailing Python whitespace
    var start: Int = 0
    var end: Int   = text.length
    while (start < end && isPyWhitespace(text.charAt(start))) start += 1
    while (end > start && isPyWhitespace(text.charAt(end - 1))) end -= 1

    // in JVM, string is immutable, so every modification
    // creates a new String obj, StringBuilder modifies without obj create.
    val stringBuilder = new java.lang.StringBuilder(end - start)
    var i: Int        = start
    while (i < end) {
      val char: Char = text.charAt(i)
      if (isPyLineBreak(char)) {
        // splitlines() + " ".join(): every line break becomes one space.
        // After strip there is no leading/trailing break (all breaks are whitespace),
        // so splitlines' "no empty last line" rule never comes into play.
        if (char == '\r' && i + 1 < end && text.charAt(i + 1) == '\n') i += 1
        stringBuilder.append(' ')
      } else {
        stringBuilder.append(char)
      }
      i += 1
    }
    stringBuilder.toString
  }

  /** Python str.isspace() — 29 BMP code points (CPython 3.12). */
  private[fasttext] def isPyWhitespace(c: Char): Boolean = (c >= '\u0009' && c <= '\u000d') || // \t \n \v \f \r
    (c >= '\u001c' && c <= '\u001f') || // file/group/record/unit separators
    c == ' ' || c == '\u0085' || c == '\u00a0' || c == '\u1680' ||
    (c >= '\u2000' && c <= '\u200a') ||
    c == '\u2028' || c == '\u2029' || c == '\u202f' || c == '\u205f' || c == '\u3000'

  /* Python str.splitlines() boundaries
   *  10 code points (\r\n handled as one in normalize).
   */
  private[fasttext] def isPyLineBreak(c: Char): Boolean = c == '\n' || c == '\u000b' || c == '\u000c' || c == '\r' ||
    c == '\u001c' || c == '\u001d' || c == '\u001e' ||
    c == '\u0085' || c == '\u2028' || c == '\u2029'

}
