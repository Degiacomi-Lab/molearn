# Copyright (c) 2021 Venkata K. Ramaswamy, Samuel C. Musson, Chris G. Willcocks, Matteo T. Degiacomi
#
# Molearn is free software ;
# you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation ;
# either version 2 of the License, or (at your option) any later version.
# molearn is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY ;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with molearn ;
# if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
#
# Author: Matteo Degiacomi


import heapq
import numpy as np

class PriorityQueue(object):
    '''
    Queue for shortest path algorithms.
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
        add element in priority queue
        :param item: item to add in queue
        :param priority: item's priority
        '''
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        '''
        pop top priority element from queue
        '''
        return heapq.heappop(self.elements)[1]

    
def _heuristic(pt1, pt2):
    '''
    return a penalty associated with the distance between points.
    '''
    # TODO: would be nice to alter the heuristic with an estimation of local
    # space warping sum of the local warping for each pixel represented by
    # drift score get pixels on line-of-sight using the Bresenham algorithm
    return np.sum(np.dot(pt2-pt1, pt2-pt1))
    
    
def _neighbors(idx, graphshape, flattened=True):

    n = []
    p = np.unravel_index(idx, graphshape)

    # generate neighbour list in a way Sam does not like
    for x in range(p[0]-1, p[0]+2, 1):
        for y in range(p[1]-1, p[1]+2, 1):

            if x==p[0] and y==p[1]:
                continue

            # apply boundary conditions
            if x>=graphshape[0] or y>=graphshape[1]:
                continue

            if x<0 or y<0:
                continue

            if flattened:
                n.append(np.ravel_multi_index(np.array([x, y]), graphshape))
            else:
                n.append([x, y])

    return np.array(n)

    
def _cost(pt, graph):
    return graph[pt]
    
    
def _astar(start_2d, goal_2d, graph):
    '''
    A* algorithm, find path connecting two points in the graph.

    :param start : starting point
    :param goal : end point
    :param graph : 2D landscape
    :returns connectivity dictionary, total path cost (same type as graph)
    '''
    
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

            #current_2d = np.unravel_index(current, graphshape)
            thenext_2d = np.unravel_index(thenext, graphshape)
            
            new_cost = cost_so_far[current] + _cost(thenext_2d, graph)

            if thenext not in cost_so_far or new_cost < cost_so_far[thenext]:
                cost_so_far[thenext] = new_cost
                               
                priority = new_cost + _heuristic(goal_2d, thenext_2d)
                frontier.put(thenext, priority)
                came_from[thenext] = current

    return came_from, cost_so_far


def get_path(idx_start, idx_end, landscape, xvals, yvals):
    '''
    : param idx_start : index on a 2D grid, as start point for a path
    : param idx_end : index on a 2D grid, as end point for a path
    : param landscape : 2D grid
    : param xvals : x-axis values, to yield actual coordinates
    : param yvals : y-axis values, to yield actual coordinates
    : returns array of 2D coordinates each with an associated value on lanscape
    '''

    # get raw A* data
    mypath, mycost = _astar(idx_start, idx_end, landscape) 

    # extract path and cost
    cnt = 0
    coords = []
    score = []
    idx_flat = np.ravel_multi_index(idx_end, landscape.shape)
    while cnt<1000: #safeguad for (unlikely) unfinished paths

        if idx_flat == mypath[idx_flat]:
            break

        idx_flat = mypath[idx_flat]
        crd = np.unravel_index(idx_flat, landscape.shape)
        coords.append([xvals[crd[0]], yvals[crd[1]]])
        score.append(landscape[crd[0], crd[1]])
        
        cnt += 1

    return np.array(coords), np.array(score)


def oversample(crd, pts=10):
    '''
    add extra equally spaced points between a list of points ("pts" per interval)
    ''' 
    pts += 1
    steps = np.linspace(1./pts, 1, pts)
    pts = [crd[0,0]]
    for i in range(1, len(crd[0])):
        for j in steps:
            newpt = crd[0, i-1] + (crd[0, i]-crd[0, i-1])*j
            pts.append(newpt)

    return np.array([pts])