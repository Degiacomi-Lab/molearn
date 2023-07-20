import heapq
import numpy as np

"""
.. module:: path
   :synopsis: Tools for linking waypoints with paths in latent space
"""


class PriorityQueue:
    '''
    Queue for shortest path algorithms.
    
    :meta private:
    '''

    def __init__(self):
        self.elements = []

    def empty(self):
        '''
        clear priority queue
        '''
        return len(self.elements) == 0

    def put(self, item, priority):
        '''
        add element in priority queue.
        
        :param item: item to add in queue
        :param priority: item's priority
        '''
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        '''
        :return: pop top priority element from queue
        '''
        return heapq.heappop(self.elements)[1]

    
def _heuristic(pt1, pt2, graph=None, euclidean=True):
    '''
    :param pt1: 2D coordinate of starting point
    :param pt2: 2D coordinate of end point
    :param euclidean: if True, evaluate value of graph at regularly spaced points on a straight line between pt1 and pt2
    :param graph: only used if euclidean=False, graph for euclidean penalty evaluation
    :return: penalty associated with the distance between points
    '''

    if not euclidean:
        pts = oversample(np.array([pt1, pt2]), 1000).round().astype(int)
        pts2 = np.vstack({tuple(e) for e in pts})
        h = 0
        for p in pts2:
            h += graph[p[0], p[1]]

    else:
        h = np.sum(np.dot(pt2-pt1, pt2-pt1))
    
    return h
    
    
def _neighbors(idx, gridshape, flattened=True):
    '''
    :param idx: index of point in a grid. Can be either a flattened index or a 2D coordinate.
    :param gridshape: tuple defining grid shape
    :param flattened: if False, return 2D coordinates, flattened index otherwise (default) 
    :return: coordinates of gridpoints adjacent to a given point in a grid
    '''

    try:
        if type(idx) != int:
            idx = np.unravel_index(idx, gridshape)
        elif len(idx) != 2:
            raise Exception("Expecting 2D coordinates")
    except Exception:
        raise Exception("idx should be either integer or an iterable")

    # generate neighbour list
    n = []
    for x in range(idx[0]-1, idx[0]+2, 1):
        for y in range(idx[1]-1, idx[1]+2, 1):

            if x==idx[0] and y==idx[1]:
                continue

            # apply boundary conditions
            if x>=gridshape[0] or y>=gridshape[1]:
                continue

            if x<0 or y<0:
                continue

            if flattened:
                n.append(np.ravel_multi_index(np.array([x, y]), gridshape))
            else:
                n.append([x, y])

    return np.array(n)

    
def _cost(pt, graph):
    '''
    :return: scalar value, reporting on the cost of moving onto a grid cell
    '''
    
    # separate function for clarity, and in case in the future we want to alter this
    return graph[pt]
    
    
def _astar(start_2d, goal_2d, in_graph, euclidean=True):
    '''
    A* algorithm, find path connecting two points in a landscape.
    
    :param start: starting point
    :param goal: end point
    :param in_graph: 2D landscape
    :return: connectivity dictionary, total path cost (same type as graph)
    '''
    
    graph = in_graph.copy()
    graph -= np.min(graph) 
    graphshape = graph.shape

    start = np.ravel_multi_index(start_2d, graphshape)
    goal = np.ravel_multi_index(goal_2d, graphshape)
    
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = start
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()

        if current == goal:
            break

        for thenext in _neighbors(current, graphshape, True):

            thenext_2d = np.unravel_index(thenext, graphshape)            
            new_cost = cost_so_far[current] + _cost(thenext_2d, graph)

            if (thenext not in cost_so_far) or (new_cost < cost_so_far[thenext]):
                cost_so_far[thenext] = new_cost
                                
                h = _heuristic(goal_2d, thenext_2d, graph=graph, euclidean=euclidean)
                priority = new_cost + h
                frontier.put(thenext, priority)
                came_from[thenext] = current

    return came_from, cost_so_far


def get_path(idx_start, idx_end, landscape, xvals, yvals, smooth=3):
    '''
    Find shortest path between two points on a weighted grid
    
    :param int idx_start: index on a 2D grid, as start point for a path
    :param int idx_end: index on a 2D grid, as end point for a path
    :param numpy.array landscape: 2D grid
    :param numpy.array xvals: x-axis values, to yield actual coordinates
    :param numpy.array yvals: y-axis values, to yield actual coordinates
    :param int smooth: size of kernel for running average (must be >=1, default 3)
    :return: array of 2D coordinates each with an associated value on lanscape
    '''

    if type(smooth) != int or smooth<1:
        raise Exception("Smooth parameter should be an integer number >=1")

    # get raw A* data
    mypath, mycost = _astar(idx_start, idx_end, landscape) 

    # extract path and cost
    cnt = 0
    coords = []
    score = []
    idx_flat = np.ravel_multi_index(idx_end, landscape.shape)
    
    # safeguard for (unlikely) unfinished paths
    while cnt<1000:

        if idx_flat == mypath[idx_flat]:
            break

        idx_flat = mypath[idx_flat]
        crd = np.unravel_index(idx_flat, landscape.shape)
        coords.append([xvals[crd[0]], yvals[crd[1]]])
        score.append(landscape[crd[0], crd[1]])
        
        cnt += 1

    if smooth == 1:
        return np.array(coords)[::-1], np.array(score)[::-1]
    
    else:
    
        traj = np.array(coords)[::-1]    
        x_ave = np.convolve(traj[:, 0], np.ones(smooth), 'valid') / smooth
        y_ave = np.convolve(traj[:, 1], np.ones(smooth), 'valid') / smooth
        traj_smooth = np.array([x_ave, y_ave]).T
        
        traj_smooth = np.concatenate((np.array([traj[0]]), traj_smooth, np.array([traj[-1]])))
        return traj_smooth, np.array(score)[::-1]


def _get_point_index(crd, xvals, yvals):
    '''
    Extract index (of 2D surface) closest to a given real value coordinate
    
    :param numpy.array/list crd: coordinate
    :param numpy.array xvals: x-axis of surface
    :param numpy.array yvals: y-axis of surface
    :return: 1D array with x,y coordinates
    '''

    my_x = np.argmin(np.abs(xvals - crd[0]))
    my_y = np.argmin(np.abs(yvals - crd[1]))
    return np.array([my_x, my_y])


def get_path_aggregate(crd, landscape, xvals, yvals, input_is_index=False):
    '''
    Create a chain of shortest paths via give waypoints
    
    :param numpy.array crd: waypoints coordinates (Nx2 array)
    :param numpy.array landscape: 2D grid
    :param numpy.array xvals: x-axis values, to yield actual coordinates
    :param numpy.array yvals: y-axis values, to yield actual coordinates
    :param bool input_is_index: if False, assume crd contains actual coordinates, graph indexing otherwise
    :return: array of 2D coordinates each with an associated value on lanscape
    '''
    
    if len(crd)<2:
        return crd
    
    crd2 = []
    for i in range(1, len(crd)):

        if not input_is_index:
            idx_start = _get_point_index(crd[i-1], xvals, yvals)
            idx_end = _get_point_index(crd[i], xvals, yvals)      
        else:
            idx_start = crd[i-1]
            idx_end = crd[i]
        
        crdtmp = get_path(idx_start, idx_end, landscape, xvals, yvals)[0]
        crd2.extend(list(crdtmp))
        
    crd = np.array(crd2)
    return crd


def oversample(crd, pts=10):
    '''
    Add extra equally spaced points between a list of points.
    
    :param numpy.array crd: Nx2 numpy array with latent space coordinates
    :param int pts: number of extra points to add in each interval
    :return: Mx2 numpy array, with M>=N.
    ''' 
    
    pts += 1
    steps = np.linspace(1./pts, 1, pts)
    pts = [crd[0]]
    for i in range(1, crd.shape[0]):
        for j in steps:
            newpt = crd[i-1] + (crd[i]-crd[i-1])*j
            pts.append(newpt)

    return np.array(pts)
