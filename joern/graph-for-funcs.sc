import scala.jdk.CollectionConverters._
import io.shiftleft.semanticcpg.language._
import io.shiftleft.codepropertygraph.generated.nodes
import overflowdb.traversal._
import java.io.PrintWriter
import io.shiftleft.codepropertygraph.Cpg
import upickle.default._
import upickle.default.{ReadWriter => RW, macroRW}

case class NodeInfo(
    id: String,
    label: String,
    line: Int,
    column: Int,
    code: String
)

case class HeteroFunctionGraph(
    name: String,
    file: String,
    id: String,
    nodes: Map[String, List[NodeInfo]],
    edges: Map[String, List[List[String]]]
)

object NodeInfo {
  implicit val rw: RW[NodeInfo] = macroRW
}

object HeteroFunctionGraph {
  implicit val rw: RW[HeteroFunctionGraph] = macroRW
}

@main def main(cpgFile: String, outputFile: String): Unit = {
  val cpg = Cpg.withStorage(cpgFile)
  
  try {
    val vipNodeLabels = Set(
      "METHOD", 
      "CALL", 
      "IDENTIFIER", 
      "LITERAL",
      "CONTROL_STRUCTURE"
    )

    val returnNodeLabels = Set(
      "METHOD_RETURN",
      "RETURN",
    )

    val functionGraphs = cpg.method.l
      .filterNot(_.name == "<global>")
      .filterNot(_.location.filename == "<empty>") 
      .map { method =>

        val allNodesInMethod = method.ast.l.distinct
        
        // TRÍCH XUẤT VÀ NHÓM CÁC NODE 
        val groupedNodes = allNodesInMethod
          .map { node =>
            val lineNum = try { node.property("LINE_NUMBER").asInstanceOf[Int] } catch { case _: Exception => 0 }
            val colNum = try { node.property("COLUMN_NUMBER").asInstanceOf[Int] } catch { case _: Exception => 0 }
            val codeStr = try { node.property("CODE").asInstanceOf[String] } catch { case _: Exception => "" }
            val label = node.label
            
            val finalLabel = 
              if (vipNodeLabels.contains(label)) label
              else if (returnNodeLabels.contains(label)) "RETURN"
              else "OTHER_NODE"

            (finalLabel, NodeInfo(node.id.toString, node.label, lineNum, colNum, codeStr))
          }
          // Sử dụng groupBy để nhóm các NodeInfo theo finalLabel đã được chuẩn hóa
          .groupBy(_._1)
          .view.mapValues(_.map(_._2)).toMap

        val astEdges = method.ast.flatMap(n => n.outE("AST").map(e => List(e.outNode.id.toString, e.inNode.id.toString))).toList

        val cfgNodesList = method.ast.isCfgNode.l
        val cfgEdges = cfgNodesList.flatMap(n => n.outE("CFG").map(e => List(e.outNode.id.toString, e.inNode.id.toString))).toList

        val cdgEdges = cfgNodesList.flatMap(n => n.controlledBy.map(controller => List(controller.id.toString, n.id.toString))).toList

        val ddgEdges = cfgNodesList.flatMap(n => n.outE("REACHING_DEF").map(e => List(e.outNode.id.toString, e.inNode.id.toString))).toList
        
        val allEdges = Map(
            "AST" -> astEdges,
            "CFG" -> cfgEdges,
            "CDG" -> cdgEdges,
            "DDG" -> ddgEdges
        ).filter { case (_, edges) => edges.nonEmpty } 


        HeteroFunctionGraph(
          name = method.fullName,
          file = method.location.filename,
          id = method.id.toString,
          nodes = groupedNodes,
          edges = allEdges
        )
      }

    val jsonString = upickle.default.write(functionGraphs, indent = 2)
    new PrintWriter(outputFile) { write(jsonString); close() }
    println(s"Output successfully written to ${outputFile}")
    
  } finally {
    cpg.close()
  }
}