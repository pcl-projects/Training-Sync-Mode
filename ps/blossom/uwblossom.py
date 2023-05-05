import logging

logger = logging.getLogger("blossom")


from blossom.graph import Blossom, UGraph
from blossom.tree import Tree


def find_perfect_matching(graph, match=None):
    """
    find perfect matching with max cardinality on unweighted undirected graph

    :param graph: UGraph object, not compatible with weighted UGraph
    :param match: UGraph object, default None at the beginning
    :returns: UGraph object, self.edge attribute give the mathcing edge tuple list
    """
    if match is None:
        match = UGraph([])
    path = find_aug_path(graph, match)
    if not path:
        return match
    else:
        new_edges = match.edge
        logger.info("find augment path as %s" % path)
        for i in range(len(path) - 1):
            if i % 2 == 0:
                new_edges.append((path[i].name, path[i + 1].name))
            else:
                if path[i].name < path[i + 1].name:
                    new_edges.remove((path[i].name, path[i + 1].name))
                else:
                    new_edges.remove((path[i + 1].name, path[i].name))
        match = UGraph(new_edges)
        return find_perfect_matching(graph, match)


def find_aug_path(graph, match):
    for e in match.edge:
        if e not in graph.edge:
            raise Exception("illegal match")
    forest = []
    tocheck = []
    exposed_vertices = graph.vertex - match.vertex
    nodes = {}
    for v in graph.vertex:
        nodes[v] = Tree(name=v)
    for v in exposed_vertices:
        forest.append(nodes[v])
        tocheck.append(nodes[v])
    logger.debug("initialized forest with exposed vertices %s" % forest)
    mark = {}
    for e in graph.edge:
        if e in match.edge:
            mark[e] = True
        else:
            mark[e] = False
    logger.debug("initialized mark dict %s" % mark)

    while len(tocheck) != 0:
        v = tocheck[0]
        logger.debug("check on %s node now..." % v)
        for w in graph.adj[v.name]:  # w only is an element in unweighted case: so this is not compatible with weighted case
            if v.name > w:
                e = (w, v.name)
            else:
                e = (v.name, w)

            if mark[e] is False:
                if nodes[w].root().name in exposed_vertices:  # w is already in the forest
                    if nodes[w].dist_to_root() % 2 == 0:
                        if v.root() != nodes[w].root():
                            path = return_aug_path(forest, v, nodes[w])
                        else:
                            path = blossom_recursion(graph, match, forest, v, nodes[w])
                        return path
                else:
                    add_to_forest(match, forest, v, nodes[w], tocheck, nodes)
                mark[e] = True

        tocheck.pop(0)
    return []


def add_to_forest(match, forest, v, w, tocheck, nodes):
    x = match.adj[w.name][0]
    v.add_child(w)
    w.add_child(nodes[x])
    tocheck.append(nodes[x])


def return_aug_path(forest, v, w):
    p1 = v.path_to_root()
    p1.reverse()
    p2 = w.path_to_root()
    return p1 + p2


def blossom_recursion(graph, match, forest, v, w):
    b = Blossom(v.path_to_node(w))
    graphr = reduce_graph(graph, b)
    matchr = reduce_graph(match, b)
    path = find_aug_path(graphr, matchr)
    logger.debug("the reduced path is %s" % path)
    if b.base.name in [p.name for p in path]:
        path = lift_blossom(path, graph, b, match)
        return path
    else:
        return path


def reduce_graph(graph, blossom):
    new_edges = []
    blossom_ele = [m.name for m in blossom.loop]
    for e in graph.edge:
        ee = list(e)
        if (e[0] in blossom_ele) and (e[1] not in blossom_ele):
            ee[0] = blossom.base.name
            if tuple(ee) not in new_edges:
                new_edges.append(tuple(ee))
        elif (e[1] in blossom_ele) and (e[0] not in blossom_ele):
            ee[1] = blossom.base.name
            if tuple(ee) not in new_edges:
                new_edges.append(tuple(ee))
        elif (e[1] in blossom_ele) and (e[0] in blossom_ele):
            pass
        else:
            if e not in new_edges:
                new_edges.append(e)
    logger.debug("new edges list in the reduced graph or match: %s" % new_edges)
    graphr = UGraph(new_edges)
    return graphr


def path_recover(graph, blossom, outer):
    candidate_path = []
    for v in blossom.loop:
        if v.name < outer.name:
            e = (v.name, outer.name)
        else:
            e = (outer.name, v.name)
        logger.debug("try recover path between %s,%s" % (e[0], e[1]))
        if e in graph.edge:
            temppath = blossom.path_to_base(v)
            candidate_path.append((len(temppath), temppath))
    pathb = max(candidate_path, key=lambda x: x[0])[1]
    return pathb


def lift_blossom(path, graph, blossom, match):
    logger.debug("lift the blossom %s with base %s now..." % (blossom.loop, blossom.base))
    if blossom.base == path[0]:
        logger.debug("the blossom is at the head of the reduced path")
        pathb = path_recover(graph, blossom, path[1])
        pathb.reverse()
        return pathb + path[1:]
    elif blossom.base == path[-1]:
        logger.debug("the blossom is at the tail of the reduced path")
        pathb = path_recover(graph, blossom, path[-2])
        return path[:-1] + pathb
    else:
        bid = path.index(blossom.base)
        logger.debug("the blossom is at the middle of the reduced path: %s" % bid)
        if path[bid - 1].name < path[bid].name:
            eb = (path[bid - 1].name, path[bid].name)
        else:
            eb = (path[bid].name, path[bid - 1].name)
        if eb in match.edge:
            pathb = path_recover(graph, blossom, path[bid + 1])
            pathb.reverse()
            path[bid : bid + 1] = pathb
        else:
            pathb = path_recover(graph, blossom, path[bid - 1])
            path[bid : bid + 1] = pathb
        return path
