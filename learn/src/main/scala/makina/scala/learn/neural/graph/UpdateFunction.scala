package makina.scala.learn.neural.graph

import breeze.linalg.{DenseMatrix, DenseVector}
import makina.scala.learn.graph.Vertex
import makina.scala.learn.neural.graph.DeepGraph.VertexContent
import makina.scala.learn.neural.network.Network

import scala.collection.mutable
import scala.util.Random

/**
  * @author Emmanouil Antonios Platanios
  */
object UpdateFunction {
  def apply[E >: Null <: AnyRef](name: String, featuresSize: Int, vertices: Traversable[Vertex[VertexContent, E]]) =
    name match {
      case "sigmoid" => SigmoidUpdateFunction[E](featuresSize, vertices)
      case "gru" => GRUUpdateFunction[E](featuresSize, vertices)
      case _ => throw new IllegalArgumentException("The provided update function name is not supported.")
    }
}

case class SigmoidUpdateFunction[E >: Null <: AnyRef](featuresSize: Int,
                                                      vertices: Traversable[Vertex[VertexContent, E]])
  extends DeepGraph.UpdateFunction[E] {
  val random = Random
  val networksBuilder = mutable.HashMap[Int, Network]()
  vertices.foreach(vertex => {
    val networkBuilder = Network.Builder()
    val phiLayer          = networkBuilder.addInputLayer(featuresSize, "phi")
    val phiInLayer        = networkBuilder.addInputLayer(featuresSize, "phi_in")
    val phiOutLayer       = networkBuilder.addInputLayer(featuresSize, "phi_out")
    val phiHiddenLayer    = networkBuilder.addFullyConnectedLayer(phiLayer,
                                                                  featuresSize,
                                                                  weightsVariableName = "W_phi",
                                                                  biasVariableName = "b")
    val phiInHiddenLayer  = networkBuilder.addFullyConnectedLayer(phiInLayer,
                                                                  featuresSize,
                                                                  useBias = false,
                                                                  weightsVariableName = "W_phi_in")
    val phiOutHiddenLayer = networkBuilder.addFullyConnectedLayer(phiOutLayer,
                                                                  featuresSize,
                                                                  useBias = false,
                                                                  weightsVariableName = "W_phi_out")
    val additionLayer     = networkBuilder.addAdditionLayer(Array(phiHiddenLayer, phiInHiddenLayer, phiOutHiddenLayer))
    networkBuilder.addSigmoidLayer(additionLayer, isOutput = true)
    networksBuilder += vertex.content.id -> networkBuilder.build()
  })
  val networks: Map[Int, Network] = networksBuilder.toMap

  override def parametersSize: Int = 3 * featuresSize * featuresSize + featuresSize

  override def initialParameters: DenseVector[Double] = {
    val initialParameters = DenseVector.zeros[Double](parametersSize)
    var index = 0
    for (layerIndex <- 0 until 3; i <- 0 until featuresSize; j <- 0 until featuresSize) {
      initialParameters(index) = (random.nextDouble() - 0.5) * 2.0 / Math.sqrt(featuresSize)
      index += 1
    }
    initialParameters
  }

  private def setupNetwork(parameters: DenseVector[Double],
                           networkVertex: Vertex[VertexContent, E],
                           contentVertex: Vertex[VertexContent, E],
                           step: Int): Network = {
    val network = networks.get(networkVertex.content.id).get
    network.set("phi", contentVertex.content.allFeatures(step))
    network.set("phi_in", contentVertex.content.incomingFeaturesSum(step))
    network.set("phi_out", contentVertex.content.outgoingFeaturesSum(step))
    network.set("W_phi", parameters(0 until (featuresSize * featuresSize)))
    network.set("W_phi_in", parameters((featuresSize * featuresSize) until (2 * featuresSize * featuresSize)))
    network.set("W_phi_out", parameters((2 * featuresSize * featuresSize) until (3 * featuresSize * featuresSize)))
    network.set("b", parameters((3 * featuresSize * featuresSize) until parametersSize))
    network
  }

  override def value(parameters: DenseVector[Double],
                     vertex: Vertex[VertexContent, E],
                     step: Int): DenseVector[Double] = setupNetwork(parameters, vertex, vertex, step).value

  override def gradient(parameters: DenseVector[Double],
                        vertex: Vertex[VertexContent, E],
                        step: Int): DenseMatrix[Double] = {
    val networkGradients =
      setupNetwork(parameters, vertex, vertex, step).gradient(Array("W_phi", "W_phi_in", "W_phi_out", "b"))
    val gradient = DenseMatrix.zeros[Double](featuresSize, parametersSize)
    gradient(0 until featuresSize, 0 until (featuresSize * featuresSize)) := networkGradients(0)
    gradient(0 until featuresSize,
             (featuresSize * featuresSize) until (2 * featuresSize * featuresSize)) := networkGradients(1)
    gradient(0 until featuresSize,
             (2 * featuresSize * featuresSize) until (3 * featuresSize * featuresSize)) := networkGradients(2)
    gradient(0 until featuresSize, (3 * featuresSize * featuresSize) until parametersSize) := networkGradients(3)
    gradient
  }

  override def featuresGradient(parameters: DenseVector[Double],
                                vertex: Vertex[VertexContent, E],
                                differentiatingVertex: Vertex[VertexContent, E],
                                step: Int): DenseMatrix[Double] = {
    if (vertex.content.id == differentiatingVertex.content.id)
      setupNetwork(parameters, differentiatingVertex, vertex, step).gradient("phi")
    else if (vertex.incomingEdges.exists(_.source.content.id == differentiatingVertex.content.id))
      setupNetwork(parameters, differentiatingVertex, vertex, step).gradient("phi_in")
    else if (vertex.outgoingEdges.exists(_.destination.content.id == differentiatingVertex.content.id))
      setupNetwork(parameters, differentiatingVertex, vertex, step).gradient("phi_out")
    else
      DenseMatrix.zeros[Double](featuresSize, featuresSize)
  }
}

case class GRUUpdateFunction[E >: Null <: AnyRef](featuresSize: Int,
                                                  vertices: Traversable[Vertex[VertexContent, E]])
  extends DeepGraph.UpdateFunction[E] {
  val random = Random
  val networksBuilder = mutable.HashMap[Int, Network]()
  vertices.foreach(vertex => {
    val networkBuilder = Network.Builder()
    val phiLayer                = networkBuilder.addInputLayer(featuresSize, "phi")
    val phiInLayer              = networkBuilder.addInputLayer(featuresSize, "phi_in")
    val phiOutLayer             = networkBuilder.addInputLayer(featuresSize, "phi_out")
    val rhoPhiHiddenLayer       = networkBuilder.addFullyConnectedLayer(phiLayer,
                                                                        featuresSize,
                                                                        weightsVariableName = "rho_W_phi",
                                                                        biasVariableName = "rho_b")
    val rhoPhiInHiddenLayer     = networkBuilder.addFullyConnectedLayer(phiInLayer,
                                                                        featuresSize,
                                                                        useBias = false,
                                                                        weightsVariableName = "rho_W_phi_in")
    val rhoPhiOutHiddenLayer    = networkBuilder.addFullyConnectedLayer(phiOutLayer,
                                                                        featuresSize,
                                                                        useBias = false,
                                                                        weightsVariableName = "rho_W_phi_out")
    val rhoAdditionHiddenLayer  = networkBuilder.addAdditionLayer(Array(rhoPhiHiddenLayer, rhoPhiInHiddenLayer, rhoPhiOutHiddenLayer))
    val rhoSigmoidHiddenLayer   = networkBuilder.addSigmoidLayer(rhoAdditionHiddenLayer)
    val zPhiHiddenLayer         = networkBuilder.addFullyConnectedLayer(phiLayer,
                                                                        featuresSize,
                                                                        weightsVariableName = "z_W_phi",
                                                                        biasVariableName = "z_b")
    val zPhiInHiddenLayer       = networkBuilder.addFullyConnectedLayer(phiInLayer,
                                                                        featuresSize,
                                                                        useBias = false,
                                                                        weightsVariableName = "z_W_phi_in")
    val zPhiOutHiddenLayer      = networkBuilder.addFullyConnectedLayer(phiOutLayer,
                                                                        featuresSize,
                                                                        useBias = false,
                                                                        weightsVariableName = "z_W_phi_out")
    val zAdditionHiddenLayer    = networkBuilder.addAdditionLayer(Array(zPhiHiddenLayer, zPhiInHiddenLayer, zPhiOutHiddenLayer))
    val zSigmoidHiddenLayer     = networkBuilder.addSigmoidLayer(zAdditionHiddenLayer)
    val hTildePhiMulHiddenLayer = networkBuilder.addElementwiseMultiplicationLayer(Array(rhoSigmoidHiddenLayer, phiLayer))
    val hTildeRhoPhiHiddenLayer = networkBuilder.addFullyConnectedLayer(hTildePhiMulHiddenLayer,
                                                                        featuresSize,
                                                                        weightsVariableName = "h_tilde_W_phi",
                                                                        biasVariableName = "h_tilde_b")
    val hTildePhiInHiddenLayer  = networkBuilder.addFullyConnectedLayer(phiInLayer,
                                                                        featuresSize,
                                                                        useBias = false,
                                                                        weightsVariableName = "h_tilde_W_phi_in")
    val hTildePhiOutHiddenLayer = networkBuilder.addFullyConnectedLayer(phiOutLayer,
                                                                        featuresSize,
                                                                        useBias = false,
                                                                        weightsVariableName = "h_tilde_W_phi_out")
    val hTildeAddHiddenLayer    = networkBuilder.addAdditionLayer(Array(hTildeRhoPhiHiddenLayer, hTildePhiInHiddenLayer, hTildePhiOutHiddenLayer))
    val hTildeTanhHiddenLayer   = networkBuilder.addTanhLayer(hTildeAddHiddenLayer)
    val hZPhiMulHiddenLayer     = networkBuilder.addElementwiseMultiplicationLayer(Array(zSigmoidHiddenLayer, phiLayer))
    val constantOneLayerId      = networkBuilder.addConstantLayer(DenseVector.ones[Double](featuresSize))
    val zSubtractionHiddenLayer = networkBuilder.addSubtractionLayer(constantOneLayerId, zSigmoidHiddenLayer)
    val hZHTildeMulHiddenLayer  = networkBuilder.addElementwiseMultiplicationLayer(Array(zSubtractionHiddenLayer, hTildeTanhHiddenLayer))
    networkBuilder.addAdditionLayer(Array(hZPhiMulHiddenLayer, hZHTildeMulHiddenLayer), isOutput = true)
    networksBuilder += vertex.content.id -> networkBuilder.build()
  })
  val networks: Map[Int, Network] = networksBuilder.toMap

  override def parametersSize: Int = 9 * featuresSize * featuresSize + 3 * featuresSize

  override def initialParameters: DenseVector[Double] = {
    val initialParameters = DenseVector.zeros[Double](parametersSize)
    var index = 0
    for (layerIndex <- 0 until 3; i <- 0 until featuresSize; j <- 0 until featuresSize) {
      initialParameters(index) = (random.nextDouble() - 0.5) * 2.0 / Math.sqrt(featuresSize)
      index += 1
    }
    initialParameters
  }

  private def setupNetwork(parameters: DenseVector[Double],
                           networkVertex: Vertex[VertexContent, E],
                           contentVertex: Vertex[VertexContent, E],
                           step: Int): Network = {
    val network = networks.get(networkVertex.content.id).get
    network.set("phi", contentVertex.content.allFeatures(step))
    network.set("phi_in", contentVertex.content.incomingFeaturesSum(step))
    network.set("phi_out", contentVertex.content.outgoingFeaturesSum(step))
    network.set("rho_W_phi", parameters(0 until (featuresSize * featuresSize)))
    network.set("rho_W_phi_in", parameters((featuresSize * featuresSize) until (2 * featuresSize * featuresSize)))
    network.set("rho_W_phi_out", parameters((2 * featuresSize * featuresSize) until (3 * featuresSize * featuresSize)))
    network.set("z_W_phi", parameters((3 * featuresSize * featuresSize) until (4 * featuresSize * featuresSize)))
    network.set("z_W_phi_in", parameters((4 * featuresSize * featuresSize) until (5 * featuresSize * featuresSize)))
    network.set("z_W_phi_out", parameters((5 * featuresSize * featuresSize) until (6 * featuresSize * featuresSize)))
    network.set("h_tilde_W_phi", parameters((6 * featuresSize * featuresSize) until (7 * featuresSize * featuresSize)))
    network.set("h_tilde_W_phi_in", parameters((7 * featuresSize * featuresSize) until (8 * featuresSize * featuresSize)))
    network.set("h_tilde_W_phi_out", parameters((8 * featuresSize * featuresSize) until (9 * featuresSize * featuresSize)))
    network.set("rho_b", parameters((9 * featuresSize * featuresSize) until (9 * featuresSize * featuresSize + featuresSize)))
    network.set("z_b", parameters((9 * featuresSize * featuresSize + featuresSize) until (9 * featuresSize * featuresSize + 2 * featuresSize)))
    network.set("h_tilde_b", parameters((9 * featuresSize * featuresSize + 2 * featuresSize) until (9 * featuresSize * featuresSize + 3 * featuresSize)))
    network
  }

  override def value(parameters: DenseVector[Double],
                     vertex: Vertex[VertexContent, E],
                     step: Int): DenseVector[Double] = setupNetwork(parameters, vertex, vertex, step).value

  override def gradient(parameters: DenseVector[Double],
                        vertex: Vertex[VertexContent, E],
                        step: Int): DenseMatrix[Double] = {
    val networkGradients =
      setupNetwork(parameters, vertex, vertex, step).gradient(Array("rho_W_phi", "rho_W_phi_in", "rho_W_phi_out",
                                                                    "z_W_phi", "z_W_phi_in", "z_W_phi_out",
                                                                    "h_tilde_W_phi", "h_tilde_W_phi_in", "h_tilde_W_phi_out",
                                                                    "rho_b", "z_b", "h_tilde_b"))
    val gradient = DenseMatrix.zeros[Double](featuresSize, parametersSize)
    gradient(0 until featuresSize, 0 until (featuresSize * featuresSize)) := networkGradients(0)
    gradient(0 until featuresSize, (featuresSize * featuresSize) until (2 * featuresSize * featuresSize)) := networkGradients(1)
    gradient(0 until featuresSize, (2 * featuresSize * featuresSize) until (3 * featuresSize * featuresSize)) := networkGradients(2)
    gradient(0 until featuresSize, (3 * featuresSize * featuresSize) until (4 * featuresSize * featuresSize)) := networkGradients(3)
    gradient(0 until featuresSize, (4 * featuresSize * featuresSize) until (5 * featuresSize * featuresSize)) := networkGradients(4)
    gradient(0 until featuresSize, (5 * featuresSize * featuresSize) until (6 * featuresSize * featuresSize)) := networkGradients(5)
    gradient(0 until featuresSize, (6 * featuresSize * featuresSize) until (7 * featuresSize * featuresSize)) := networkGradients(6)
    gradient(0 until featuresSize, (7 * featuresSize * featuresSize) until (8 * featuresSize * featuresSize)) := networkGradients(7)
    gradient(0 until featuresSize, (8 * featuresSize * featuresSize) until (9 * featuresSize * featuresSize)) := networkGradients(8)
    gradient(0 until featuresSize, (9 * featuresSize * featuresSize) until (9 * featuresSize * featuresSize + featuresSize)) := networkGradients(9)
    gradient(0 until featuresSize, (9 * featuresSize * featuresSize + featuresSize) until (9 * featuresSize * featuresSize + 2 * featuresSize)) := networkGradients(10)
    gradient(0 until featuresSize, (9 * featuresSize * featuresSize + 2 * featuresSize) until (9 * featuresSize * featuresSize + 3 * featuresSize)) := networkGradients(11)
    gradient
  }

  override def featuresGradient(parameters: DenseVector[Double],
                                vertex: Vertex[VertexContent, E],
                                differentiatingVertex: Vertex[VertexContent, E],
                                step: Int): DenseMatrix[Double] = {
    if (vertex.content.id == differentiatingVertex.content.id)
      setupNetwork(parameters, differentiatingVertex, vertex, step).gradient("phi")
    else if (vertex.incomingEdges.exists(_.source.content.id == differentiatingVertex.content.id))
      setupNetwork(parameters, differentiatingVertex, vertex, step).gradient("phi_in")
    else if (vertex.outgoingEdges.exists(_.destination.content.id == differentiatingVertex.content.id))
      setupNetwork(parameters, differentiatingVertex, vertex, step).gradient("phi_out")
    else
      DenseMatrix.zeros[Double](featuresSize, featuresSize)
  }
}
