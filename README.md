## Synopsis  
Clustering on Spark. Cluster documents containing information on NIPS abstracts, NY Times news articles, and PubMed abstracts using a special case of expectation-maximization namely K-Means.

##Dataset Details  
The data consists of text documents from the UCI Machine Learning Repository
Datasets available at:http://ugammd.blob.core.windows.net/bagofwords/

##Algorithm in Detail  
Algorithm takes three parameters: 
• k, an integer number of clusters 
• x, an integer number of maximum iterations (set to 100) 
• t , a float convergence tolerance (set to 1.0) 
Convert the documents to TF-IDF format and choose k initial centroids
randomly from the dataset. 
old_centroids = tfidf_vectors.takeSample(k) 
iterations = 0 
converge = 1.0 
while iterations < x and converge > t: 
  e-step: assign each point to a cluster, based on smallest Euclidean distance 
  m-step: update centroids by averaging points in each cluster
  converge = sum( (old_centroids - new_centroids) ** 2 ) 
  old_centroids = new_centroids 
  increment iterations 
## run instructions  
To run the program, place src folder and simple.sbt inside the spark folder and the input files as well and run the following commands on terminal:
```
 ~/spark$ sbt package
 ~/spark$ bin/spark-submit   --class "Kmeans"   --master local[*]   target/scala-2.10/kmeans_2.10-1.0.jar
 ```
