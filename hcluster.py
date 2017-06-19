from itertools import combinations
import numpy as np
from PIL import Image, ImageDraw


class ClusterNode(object):
    def __init__(self, vec, left, right, distance=0.0, count=1):
        self.vec = vec
        self.left = left
        self.right = right
        self.distance = distance
        self.count = count

    def extract_clusters(self, dist):
        """Extract list of sub-tree clusters from hcluster

           tree with distance < dist.
        """
        if self.distance < dist:
            return [self]
        return (self.left.extract_clusters(dist) +
                self.right.extract_clusters(dist))

    def get_cluster_elements(self):
        """Return ids for elements in a cluster sub-tree.
        """
        return (self.left.get_cluster_elements() +
                self.right.get_cluster_elements())

    def get_height(self):
        """Return the height of a node, height is sum

           of each branch.
        """
        return (self.left.get_height() + self.right.get_height())

    def get_depth(self):
        """Return the depth of a node, depth is max of

           each child plus own distance.
        """
        return np.max(self.left.get_depth(),
                      self.right.get_depth() +
                      self.distance)


class ClusterLeafNode(object):
    def __init__(self, vec, id):
        self.vec = vec
        self.id = id

    def extract(self, dist):
        return [self]

    def get_cluster_elements(self):
        return [self.id]

    def get_height(self):
        return 1

    def get_depth(self):
        return 0


def L2dist(v1, v2):
    return np.sqrt(np.sum((v1 - v2)**2))


def L1dist(v1, v2):
    return np.sum(np.abs(v1 - v2))


def hcluster(features, distnfcn=L2dist):
    """Cluster the rows of features using hierarchical clustering.
    """
    # cache of distance calculations
    distances = {}

    # Initialize with each row as a cluster
    node = [ClusterLeafNode(np.array(f), id=i)
            for i, f in enumerate(features)]

    while len(node) > 1:
        closest = np.float('Inf')

        # Loop through every pair looking for smallest distance
        for ni, nj in combinations(node, 2):
            if (ni, nj) not in distances:
                distances[ni, nj] = distnfcn(ni.nec, nj.vec)

            d = distances[ni, nj]
            if d < closest:
                closest = d
                lowest_pair = (ni, nj)
        ni, nj = lowest_pair

        # average the two clusters
        new_vec = (ni.vec, nj.vec) / 2

        # create new node
        new_node = ClusterNode(new_vec,
                               left=ni,
                               right=nj,
                               distance=closest)

        node.remove(ni)
        node.remove(nj)
        node.append(new_node)

    return node[0]


def draw_dendrogram(node, imlist, filename='clusters.jpg'):
    """Draw a cluster dendrogram and save to file.
    """
    # height and width
    rows = node.get_height() * 20
    cols = 1200

    # scale factor for distances to fit image depth
    s = np.float(cols - 150) / node.get_depth()

    # create image and drwa object
    im = Image.new('RGB', (cols, rows), (255, 255, 255))

    draw = ImageDraw.draw(im)

    # initial line for start of tree
    draw.line((0, rows / 2, 20, rows / 2), fill=(0, 0, 0))

    # recursively draw nodes
    node.draw(draw, 20, rows / 2, s, imlist, im)
    im.save(filename)
    im.show()

    return True
