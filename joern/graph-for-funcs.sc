import scala.jdk.CollectionConverters._
import io.shiftleft.semanticcpg.language._
import io.shiftleft.codepropertygraph.generated.nodes
import overflowdb.traversal._
import upickle.default._
import upickle.default.{ReadWriter => RW, macroRW}
import java.io.PrintWriter
import io.shiftleft.codepropertygraph.Cpg

case class NodeInfo(
    id: String,
    label: String,
    line: Int,
    column: Int,
    code: String
)

case class FunctionGraph(
    name: String, 
    file: String, 
    id: String,
    nodes: List[NodeInfo],
    ast_edges: List[List[Long]],
    cfg_edges: List[List[Long]],
    cdg_edges: List[List[Long]],
    ddg_edges: List[List[Long]]
)

object NodeInfo {
  implicit val rw: RW[NodeInfo] = macroRW
}

object FunctionGraph {
  implicit val rw: RW[FunctionGraph] = macroRW
}

@main def main(cpgFile: String, outputFile: String): Unit = {
  val cpg = Cpg.withStorage(cpgFile)
  
  try {
    val functionGraphs = cpg.method.l
      .filter(_.name != "<global>")
      .filter(_.location.filename != "<empty>") 
      .map { method =>

        val methodName = method.fullName
        val methodFile = method.location.filename
        val methodId = method.id.toString

        val allNodes = method.ast.l.distinct
        
        val nodes = allNodes.map { node =>
          val lineNum = try { node.property("LINE_NUMBER").asInstanceOf[Int] } catch { case _: Exception => 0 }
          val colNum = try { node.property("COLUMN_NUMBER").asInstanceOf[Int] } catch { case _: Exception => 0 }
          val codeStr = try { node.property("CODE").asInstanceOf[String] } catch { case _: Exception => "" }
          
          NodeInfo(
            id = node.id.toString,
            label = node.label,
            line = lineNum,
            column = colNum,
            code = codeStr
          )
        }

        val astEdges = method.ast.flatMap(n => n.outE("AST").map(e => List(e.outNode.id, e.inNode.id))).l

        val cfgNodesList = method.ast.isCfgNode.l
        val cfgEdges = cfgNodesList.flatMap(n => n.outE("CFG").map(e => List(e.outNode.id, e.inNode.id)))

        val cdgEdges = cfgNodesList.flatMap(n => n.controlledBy.map(controller => List(controller.id, n.id)))

        val ddgEdges = cfgNodesList.flatMap(n => n.outE("REACHING_DEF").map(e => List(e.outNode.id, e.inNode.id)))
        
        FunctionGraph(
          name = methodName,
          file = methodFile,
          id = methodId,
          nodes = nodes,
          ast_edges = astEdges,
          cfg_edges = cfgEdges,
          cdg_edges = cdgEdges,
          ddg_edges = ddgEdges
        )
      }

    val jsonString = upickle.default.write(functionGraphs, indent = 2)
    new PrintWriter(outputFile) { write(jsonString); close() }
    println(s"Output successfully written to ${outputFile}")
    
  } finally {
    cpg.close()
  }
}