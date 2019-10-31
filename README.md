# Clustering: A neural network approach
In [this paper](https://www.sciencedirect.com/science/article/pii/S089360800900207X), a comprehensive overview of 
competitive learning based clustering methods is given. Importance is attached to a number of competitive learning based clustering 
neural networks such as the self-organizing map (SOM), the learning vector quantization (LVQ), the neural gas, and the ART model.


# About this project
In this project, two of the most famous NN-based clustering methods, SOM and LVQ, will be implemented. Their performance and efficacy,
will then be investigated through analysing different criteria such as
[Calinski-Harabasz index](https://www.oreilly.com/library/view/machine-learning-algorithms/9781785889622/8dba1062-2dbe-43ce-a9b0-9ea49203ea9a.xhtml), 
[Davies-Bouldin index](https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index), and **silhouette values** as well as **silhouette plot**. 
The programming language for this project is chosen to be MATLAB and the dataset used is artificially created by sampling 
Gaussian distribution. Moreover, a linear dimensionality reduction based on 
[PCA (Principal Component Analysis)](https://en.wikipedia.org/wiki/Principal_component_analysis) is performed as a pre-processing step.        
> ***NOTE:* Self-organizing maps learn to cluster data based on similarity, topology, with a preference (but no guarantee) of assigning  the same number of instances to each class; competitive layers learn to classify input vectors into a given 
> number of classes, according to similarity between vectors, (AGAIN!) with a preference for equal numbers of vectors per 
> class. (excerpted from MATLAB help)**      
> **Hence, if you are sure that your dataset is imbalanced and your expected clusters are not supposed to be of similar population, then I
> recommend that you try to change the loss function of the neural network (which I feel would be arduous) or implement other
> clustering methods such as [hierarchical clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering) and
> [k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering).**



## Files' description:
* **sofm.m** >> in this MATLAB script, a SOFM neural network is trained to cluster the data into 3 classes.

* **lvq.m** >> in this MATLAB script, a LVQ-based neural network is trained to cluster the data into 3 classes.

* **Performance_Evaluation.m** >> in this MATLAB script, two clustering algorithms, SOFM and LVQ, are compared together through well-known quantitative criteria such as ***Calinski-Harabasz index*** and ***Davies-Bouldin index***, as well as ***silhouette values*** and ***silhouette plot***. 
