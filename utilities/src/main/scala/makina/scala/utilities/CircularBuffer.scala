package makina.scala.utilities

/**
  * @author Emmanouil Antonios Platanios
  */
object CircularBuffer {
  def apply[T](size: Int)(implicit m: Manifest[T]) = new CircularBuffer[T](size)(m)
}

class CircularBuffer[T](size: Int)(implicit m: Manifest[T]) extends Seq[T] {
  val buffer = new Array[T](size)
  var bufferIndex: Int = 0

  override def apply(index: Int): T = buffer(Math.floorMod(bufferIndex + index, size))
  override def length = size
  override def iterator = new CircularBufferIterator[T](buffer, bufferIndex)

  def add(e: T) = {
    buffer(bufferIndex) = e
    bufferIndex = (bufferIndex + 1) % size
  }
}

class CircularBufferIterator[T](buffer: Array[T], start: Int) extends Iterator[T] {
  var index = 0
  override def hasNext = index < buffer.length
  override def next() = {
    val i = index
    index = index + 1
    buffer(i)
  }
}
