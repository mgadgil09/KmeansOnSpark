/* Kmeans.scala */
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature.IDF


object Kmeans {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Kmeans")
    val sc = new SparkContext(conf)
    val vocab = sc.textFile("vocab.nips.txt")
    val mappedVocab = vocab.zipWithIndex().map{case(line,i) => (i+1).toString + ","+s"$line"}
    val docWord = sc.textFile("docword.nips.txt")
    val newDoc = docWord.zipWithIndex().filter{_._2>2}.map{_._1}
    val vocabCount = docWord.zipWithIndex().filter{_._2 == 1}.map{_._1.toInt}
    val splittedNewDoc = newDoc.map(line => line.split(" "))
    val firstJoin = splittedNewDoc.map(line => (line(1),line(0).toString+":"+line(2).toString))
    val data = mappedVocab.map{line => (line.split(",")(0),line.split(",")(1))}
    val docWithWords = firstJoin.join(data).map{case(k1,(k2,k3)) => (k2.split(":")(0),k3,k2.split(":")(1))}
    val mapDoc = docWithWords.map{case(v1,v2,v3) => ((v1.toString + "," + v2.toString),v3)}
    val mappedDoc = mapDoc.flatMap{case(k,v) => for(i <- 1 to v.toInt)yield (k)}.map{line => ((line.split(",")(0)),(line.split(",")(1)))}
    val fullDoc = mappedDoc.reduceByKey((a,b) => a.toString + " " + b.toString).sortByKey().map{case(k,v) => v}
    val documents = fullDoc.map(_.split(" ").toSeq) 
    val hashingTF = new HashingTF(vocabCount.collect()(0)+500)
    val tf = hashingTF.transform(documents)
    tf.cache()
    val idf = new IDF().fit(tf)
    val tfidf = idf.transform(tf)
    val k:Int = 2
    val x:Int = 100
    val t:Float = 0.1.toFloat
    var converge:Float = 1.0.toFloat
    var centroids = tfidf.takeSample(false,k)
    //var oldCentroids = centroids.zipWithIndex.map{case(k,v) => (v,k)}
    var iterations = 0
    var bcastedclusters = sc.broadcast(centroids)
    def convert(vec: org.apache.spark.mllib.linalg.SparseVector) : Vector = vec
    def valCase(x: Option[Long]) : Double = x match {
      case Some(s) => s.toDouble
      case None => 0.0
   } 
    while(iterations < x && converge > t) {
	
	val indexedtfidf = tfidf.zipWithIndex.map{case(k,v) => (v,k)}
	val var1 = indexedtfidf.reduceByKey((a,b)=>b).map{case(k,v)=> v}.map{vec => {
		val x = for (a<- bcastedclusters.value)yield{(Vectors.sqdist(a,vec))}
		val y:Int = x.indexOf(x.min)
  		(y,vec)
  			}
  		}
 	val countVectors = sc.broadcast(var1.countByKey())
   	val avg = var1.reduceByKey((a,b) => {
     		val v1 = Vectors.dense(a.toArray)
      		val v2 = Vectors.dense(b.toArray)
      		val bv1 = new  breeze.linalg.DenseVector(v1.toArray)
      		val bv2 = new breeze.linalg.DenseVector(v2.toArray)
      		Vectors.dense((bv1 + bv2).toArray)})
        //got new centroids
        val newCentroids = avg.map{vector => {
       		 val avgCount = valCase(countVectors.value.get(vector._1))
    	       	 val breezeVec = new breeze.linalg.DenseVector(vector._2.toArray)
        	 (vector._1.toInt,Vectors.dense((breezeVec:/avgCount).toArray).toSparse)
        		}
      		 }
	//now converge
	val oldCentroids = centroids.toArray
	//oldCentroids.foreach(println)
	
	val newCentroids1 = newCentroids.map(line => (convert(line._2))).toArray
	//newCentroids1.foreach(println)
	//val unioned = oldCentroids.union(newCentroids1)
	var i:Int =0
	converge = 0.toFloat
	i=0
	println("before for loop")
	println("this is i:"+i)
	for(i <- 0 until oldCentroids.length){
	println("inside for loop"+ i)
	   converge += Vectors.sqdist(oldCentroids(i),newCentroids1(i)).toFloat
	   
	   println("this is euclidean value:"+ Vectors.sqdist(oldCentroids(i),newCentroids1(i)).toFloat)
	   println("this is converge value:"+converge)
	 }
	 println("New Converge Value:"+converge)
	 
	 centroids = newCentroids1
	 bcastedclusters = sc.broadcast(centroids)
	 iterations+=1
    }
    
   println("Took "+ iterations+" iterations to converge")
	 println(converge+" centroid residual left")
	 centroids.foreach(println)
}
}

