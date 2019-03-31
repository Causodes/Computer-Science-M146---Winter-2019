"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Famous Faces
"""

# python libraries
import collections

# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt

# libraries specific to project
import util
import time
from util import *
from cluster import *

######################################################################
# helper functions
######################################################################

def build_face_image_points(X, y) :
    """
    Translate images to (labeled) points.
    
    Parameters
    --------------------
        X     -- numpy array of shape (n,d), features (each row is one image)
        y     -- numpy array of shape (n,), targets
    
    Returns
    --------------------
        point -- list of Points, dataset (one point for each image)
    """
    
    n,d = X.shape
    
    images = collections.defaultdict(list) # key = class, val = list of images with this class
    for i in xrange(n) :
        images[y[i]].append(X[i,:])
    
    points = []
    for face in images :
        count = 0
        for im in images[face] :
            points.append(Point(str(face) + '_' + str(count), face, im))
            count += 1

    return points


def plot_clusters(clusters, title, average) :
    """
    Plot clusters along with average points of each cluster.

    Parameters
    --------------------
        clusters -- ClusterSet, clusters to plot
        title    -- string, plot title
        average  -- method of ClusterSet
                    determines how to calculate average of points in cluster
                    allowable: ClusterSet.centroids, ClusterSet.medoids
    """
    
    plt.figure()
    np.random.seed(20)
    label = 0
    colors = {}
    centroids = average(clusters)
    for c in centroids :
        coord = c.attrs
        plt.plot(coord[0],coord[1], 'ok', markersize=12)
    for cluster in clusters.members :
        label += 1
        colors[label] = np.random.rand(3,)
        for point in cluster.points :
            coord = point.attrs
            plt.plot(coord[0], coord[1], 'o', color=colors[label])
    plt.title(title)
    plt.show()


def generate_points_2d(N, seed=1234) :
    """
    Generate toy dataset of 3 clusters each with N points.
    
    Parameters
    --------------------
        N      -- int, number of points to generate per cluster
        seed   -- random seed
    
    Returns
    --------------------
        points -- list of Points, dataset
    """
    np.random.seed(seed)
    
    mu = [[0,0.5], [1,1], [2,0.5]]
    sigma = [[0.1,0.1], [0.25,0.25], [0.15,0.15]]
    
    label = 0
    points = []
    for m,s in zip(mu, sigma) :
        label += 1
        for i in xrange(N) :
            x = util.random_sample_2d(m, s)
            points.append(Point(str(label)+'_'+str(i), label, x))
    
    return points


######################################################################
# k-means and k-medoids
######################################################################

def random_init(points, k) :
    """
    Randomly select k unique elements from points to be initial cluster centers.
    
    Parameters
    --------------------
        points         -- list of Points, dataset
        k              -- int, number of clusters
    
    Returns
    --------------------
        initial_points -- list of k Points, initial cluster centers
    """
    ### ========== TODO : START ========== ###
    # part 2c: implement (hint: use np.random.choice)
    """indexes = np.random.choice(len(points), k)
    initial_points = []
    for i in indexes:
        initial_points.append(points[i])
    return np.array(initial_points)"""
    return np.random.choice(points, k, replace=False)
    ### ========== TODO : END ========== ###


def cheat_init(points) :
    """
    Initialize clusters by cheating!
    
    Details
    - Let k be number of unique labels in dataset.
    - Group points into k clusters based on label (i.e. class) information.
    - Return medoid of each cluster as initial centers.
    
    Parameters
    --------------------
        points         -- list of Points, dataset
    
    Returns
    --------------------
        initial_points -- list of k Points, initial cluster centers
    """
    ### ========== TODO : START ========== ###
    # part 2f: implement
    initial_points = []
    clusset = ClusterSet()
    labels = []
    for point in points:
        labels.append(point.label)
    labels = np.unique(np.array(labels))
    for lab in labels:
        clusset.add(Cluster([point for point in points if point.label == lab]))
    initial_points = clusset.medoids()
    return initial_points
    ### ========== TODO : END ========== ###


def kMeans(points, k, init='random', plot=False) :
    """
    Cluster points into k clusters using variations of k-means algorithm.
    
    Parameters
    --------------------
        points  -- list of Points, dataset
        k       -- int, number of clusters
        average -- method of ClusterSet
                   determines how to calculate average of points in cluster
                   allowable: ClusterSet.centroids, ClusterSet.medoids
        init    -- string, method of initialization
                   allowable: 
                       'cheat'  -- use cheat_init to initialize clusters
                       'random' -- use random_init to initialize clusters
        plot    -- bool, True to plot clusters with corresponding averages
                         for each iteration of algorithm
    
    Returns
    --------------------
        k_clusters -- ClusterSet, k clusters
    """
    
    ### ========== TODO : START ========== ###
    # part 2c: implement
    # Hints:
    #   (1) On each iteration, keep track of the new cluster assignments
    #       in a separate data structure. Then use these assignments to create
    #       a new ClusterSet object and update the centroids.
    #   (2) Repeat until the clustering no longer changes.
    #   (3) To plot, use plot_clusters(...).
    #k_clusters = ClusterSet()
    return kAverages(points=points, k=k, average="centroids", init=init, plot=plot)
    ### ========== TODO : END ========== ###

def kAverages(points, k, average, init='random', plot=False) :
    initpoints = None
    if init == "random":
        initpoints = random_init(points, k)
    elif init == "cheat":
        initpoints = cheat_init(points)
    else:
        return None
    
    prevclusterset = None
    currentclusterset = None
    centerpoints = initpoints
    iters = 1
    while True:
        if prevclusterset != None and currentclusterset.equivalent(prevclusterset):
            return currentclusterset
        prevclusterset = currentclusterset
        currentclusterset = ClusterSet()
        for cluster in assignclusters(centerpoints, points):
            currentclusterset.add(cluster)
        if plot:
            if average == "centroids":
                plot_clusters(currentclusterset, "Iteration #" + str(iters) + " with " + str(init) + " inital points", ClusterSet.centroids)
            elif average == "medoids":
                plot_clusters(currentclusterset, "Iteration #" + str(iters) + " with " + str(init) + " inital points", ClusterSet.medoids)
        if average == "centroids":
            centerpoints = currentclusterset.centroids()
        elif average == "medoids":
            centerpoints = currentclusterset.medoids()
        iters += 1
    return currentclusterset   


def assignclusters(centerpoints, points):
    closestcenters = []
    for point in points:
        mindist = None
        mincenter = None
        for center in centerpoints:
            dist = point.distance(center)
            if mindist == None or dist < mindist:
                mindist = dist
                mincenter = center
        closestcenters.append(mincenter)
    
    groupedpoints = []

    for center in centerpoints:
        pointset = []
        for i, assignedcenter in enumerate(closestcenters):
            if assignedcenter == center:
                pointset.append(points[i])
        groupedpoints.append(pointset)
    return [Cluster(pointset) for pointset in groupedpoints]


def kMedoids(points, k, init='random', plot=False) :
    """
    Cluster points in k clusters using k-medoids clustering.
    See kMeans(...).
    """
    ### ========== TODO : START ========== ###
    # part 2e: implement
    return kAverages(points=points, k=k, average="medoids", init=init, plot=plot)
    ### ========== TODO : END ========== ###


######################################################################
# main
######################################################################

def main() :
    ### ========== TODO : START ========== ###
    # part 1: explore LFW data set
    X, y = get_lfw_data()
    n,d = X.shape
    for im in X:
        show_image(im=im)
    meanim = []
    for i in range(0, d):
        temparr = []
        for im in X:
            temparr.append(im[i])
        tempnp = np.array(temparr)
        meanim.append(np.mean(tempnp))
    numpim = np.array(meanim)
    show_image(im=numpim)

    #part b
    U, mu = PCA(X)
    #plot_gallery([vec_to_image(U[:,i]) for i in xrange(12)])

    #part c
    test_l = [1, 10, 50, 100, 500, 1288]
    for l in test_l:
        """
        Z   -- numpy matrix of shape (n,l), n l-dimensional features
            each row is a sample, each column is one dimension of the sample
        Ul  -- numpy matrix of shape (d,l), l d-dimensional eigenvectors
            each column is a unit eigenvector; columns are sorted by eigenvalue
                (Ul is a subset of U, specifically the d-dimensional eigenvectors
                 of U corresponding to largest l eigenvalues)"""
        Z, Ul = apply_PCA_from_Eig(X=X, U=U, l=l, mu=mu)
        X_rec = reconstruct_from_PCA(Z=Z, U=Ul, mu=mu)
        #print("current l = %d" % l)
        #plot_gallery(X_rec[[:12]])

    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part 2d-2f: cluster toy dataset
    # np.random.seed(1234)
    # points = generate_points_2d(N=20)
    # kMedoids(points=points, k=3, init="cheat", plot=True)
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###    
    """X1, y1 = util.limit_pics(X, y, [4, 6, 13, 16], 40)
    points = build_face_image_points(X1, y1)
    # part 3a: cluster faces
    np.random.seed(1234)
    kmeansset = []
    kmedoidsset = []
    
    kmeanstimes = []
    kmedoidstimes = []

    #start1 = timeit.default_timer()
    for i in range(0, 10):
        start1 = time.time()
        clusset1 = kMeans(points=points, k=4, init="random", plot=False)
        kmeanstimes.append(time.time() - start1)
        score = clusset1.score()
        print "kMeans scores"
        print(score)
        kmeansset.append(score)
        
        start2 = time.time()
        clusset2= kMedoids(points=points, k=4, init="random", plot=False)
        kmedoidstimes.append(time.time() - start2)
        print "kMedoids scores"
        score = clusset2.score()
        print(score)
        kmedoidsset.append(score)
    # stop1 = timeit.default_timer()
    print "kMeans average"
    print(np.mean(np.array(kmeansset)))
    # print "kMeans time"
    # print str(stop1 - start1)


    
    #start2 = timeit.default_timer()
    for i in range(0, 10):
        clusset= kMedoids(points=points, k=4, init="random", plot=False)
        score = clusset.score()
        print(score)
        kmedoidsset.append(score)
    # stop2 = timeit.default_timer()
    print "kMedoid average"
    print(np.mean(np.array(kmedoidsset)))
    # print "kMedois time"
    # print str(stop2 - start2)

    print "kMeans times"
    print(np.mean(np.array(kmeanstimes)))

    print "kMedoids times"
    print(np.mean(np.array(kmedoidstimes)))"""
        
    # part 3b: explore effect of lower-dimensional representations on clustering performance
    np.random.seed(1234)
    X2, y2 = util.limit_pics(X, y, [4, 13], 40)
    #points = build_face_image_points(X2, y2)

    U, mu = PCA(X)
    
    lvals = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41]
    kmnscor = []
    kmdscor = []

    for l in lvals:
        Z, Ul = apply_PCA_from_Eig(X=X2, U=U, l=l, mu=mu)
        points = build_face_image_points(Z, y2)
        clusset1 = kMeans(points=points, k=2, init="cheat", plot=False)
        kmnscor.append(clusset1.score())

        clusset2 = kMedoids(points=points, k=2, init="cheat", plot=False)
        kmdscor.append(clusset2.score())
    
    # plt.xlabel("Number of components (l)")
    # plt.ylabel("Clustering score")

    # plt.plot(lvals, kmnscor, 'rs-', label="k-means")
    # plt.plot(lvals, kmdscor, 'b^-', label="k-medoids")
    #plt.show()

    # part 3c: determine ``most discriminative'' and ``least discriminative'' pairs of images
#    np.random.seed(1234)
#    currmini = 0
#    currminj = 0
#    currmaxi = 0
#    currmaxj = 0
#    currmin = 1
#    currmax = 0
#    for i in range(0, 19):
#        for j in range(i + 1, 19):
#            tempX, tempy = util.limit_pics(X, y, [i, j], 40)
#            points = build_face_image_points(tempX, tempy)
#            #print "i: " + str(i) + " j: " + str(j)
#            clusset = kMedoids(points=points, k=2, init="cheat", plot=False)
#            #print clusset.score()
#            currscore = clusset.score()
#            if currscore < currmin:
#                currmin = currscore
#                currmini = i
#                currminj = j
#            if currscore > currmax:
#                currmax = currscore
#                currmaxi = i
#                currmaxj = j
#    print "very difficult"
#    print currmin
#    print "i " + str(currmini) + " j " + str(currminj)
#
#    tX, tY = util.limit_pics(X, y, [currmini, currminj], 40)
#    points = build_face_image_points(tX, tY)
#    labels1 = []
#    for point in points:
#        labels1.append(point.label)
#    labels1 = np.unique(np.array(labels1))
#    plot_representative_images(tX, tY, labels1)
#
#
#    print "very well"
#    print currmax
#    print "i " + str(currmaxi) + " j " + str(currmaxj)
#    tX, tY = util.limit_pics(X, y, [currmaxi, currmaxj], 40)
#    points = build_face_image_points(tX, tY)
#    labels2 = []
#    for point in points:
#        labels2.append(point.label)
#    labels2 = np.unique(np.array(labels2))
#    plot_representative_images(tX, tY, labels2)

    
    ### ========== TODO : END ========== ###


if __name__ == "__main__" :
    main()
