package limmen.github.com.feature_engineering_spark.featuregroup

import org.apache.log4j.{ Level, LogManager, Logger }
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.graphx._
import org.apache.spark.sql.Row
import io.hops.util.Hops
import scala.collection.JavaConversions._
import collection.JavaConverters._
import org.apache.spark.sql.Dataset

/**
 * Contains logic for computing the trx_graph_summary_features featuregroup
 */
object TrxGraphSummaryFeatures {

  case class TrxNode(cust_id: Int)

  case class GraphSummaryFeature(
    cust_id: Int,
    pagerank: Float,
    triangle_count: Int)

  /**
   * Computes the featuregroup
   *
   * @param input path to the input dataset to read (csv)
   * @param output name of the output featuregroup table
   * @param version version of the featuregroup
   * @param partitions number of spark partitions to parallelize the compute on
   * @param logger spark logger
   * @param create boolean flag whether to create new featuregroup or insert into an existing ones
   * @param jobId the id of the hopsworks job
   * @param edgesLookup optional pre-computed map with edges
   */
  def computeFeatures(spark: SparkSession, input: String, featuregroupName: String,
    version: Int, partitions: Int, log: Logger, create: Boolean, jobId: Int, edgesLookup: Dataset[TrxEdge] = null): Unit = {
    log.info(s"Running computeFeatures for featuregroup: ${featuregroupName}")
    import spark.implicits._
    val featurestore = Hops.getProjectFeaturestore
    if (edgesLookup == null) {
    }
    var edgesDs: Dataset[TrxEdge] = null
    if (edgesLookup == null) {
      val edges = Hops.getFeaturegroup(spark, "trx_graph_edge_list", featurestore, 1)
      edgesDs = edges.as[TrxEdge]
    } else {
      edgesDs = edgesLookup
    }
    val inNodesDs = edgesDs.map((edge: TrxEdge) => TrxNode(edge.cust_id_1))
    val outNodesDs = edgesDs.map((edge: TrxEdge) => TrxNode(edge.cust_id_2))
    val propertyEdges = edgesDs.map((edge: TrxEdge) => Edge(edge.cust_id_1, edge.cust_id_2, edge))
    val nodesDs = inNodesDs.union(outNodesDs).distinct.map((node: TrxNode) => (node.cust_id.toLong, node))
    val graph = Graph(nodesDs.rdd, propertyEdges.rdd)
    val alpha = 0.01
    val numIter = 100
    val pageRanks = graph.pageRank(numIter, resetProb = alpha).vertices.toDF
      .withColumnRenamed("_1", "cust_id")
      .withColumnRenamed("_2", "pagerank")
    val triangleCounts = graph.triangleCount().vertices.toDF
      .withColumnRenamed("_1", "cust_id")
      .withColumnRenamed("_2", "triangle_count")
    val joined = pageRanks.join(triangleCounts, "cust_id")
    val features = joined.rdd.map((row: Row) => {
      val cust_id = row.getAs[Long]("cust_id").toInt
      val pagerank = row.getAs[Double]("pagerank").toFloat
      val triangleCount = row.getAs[Int]("triangle_count")
      GraphSummaryFeature(cust_id, pagerank, triangleCount)
    }).toDS
    log.info(s"Features: \n: ${features.show(5)}")
    val descriptiveStats = true
    val featureCorr = true
    val featureHistograms = true
    val clusterAnalysis = true
    val statColumns = List[String]().asJava
    val numBins = 20
    val corrMethod = "pearson"
    val numClusters = 5
    val description = "Contain aggregate graph features of a customers transactions"
    val primaryKey = "cust_id"
    val dependencies = List[String](input).asJava
    if (create) {
      log.info(s"Creating featuregroup $featuregroupName version $version in featurestore $featurestore")
      Hops.createFeaturegroup(spark, features.toDF, featuregroupName, featurestore, version, description,
        jobId, dependencies, primaryKey, descriptiveStats, featureCorr, featureHistograms, clusterAnalysis,
        statColumns, numBins, corrMethod, numClusters)
      log.info(s"Creation of featuregroup $featuregroupName complete")
    } else {
      log.info(s"Inserting into featuregroup $featuregroupName version $version in featurestore $featurestore")
      Hops.insertIntoFeaturegroup(spark, features.toDF, featuregroupName, featurestore, version, "overwrite",
        descriptiveStats, featureCorr, featureHistograms, clusterAnalysis, statColumns, numBins, corrMethod,
        numClusters)
      log.info(s"Insertion into featuregroup $featuregroupName complete")
    }
  }
}
