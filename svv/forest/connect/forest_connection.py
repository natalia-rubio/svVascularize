from svv.forest.connect.tree_connection import TreeConnection


class ForestConnection:
    def __init__(self, forest, **kwargs):
        self.forest = forest
        self.tree_connections = []
        self.networks = []

    def solve(self, *args, num_vessels=20, attempts=5, **kwargs):
        self.tree_connections = []
        for i in range(self.forest.n_networks):
            tree_connections = TreeConnection(self.forest, i, **kwargs)
            if len(self.tree_connections) > 0:
                other_vessels = []
                for j in range(len(self.tree_connections)):
                    other_vessels.extend(self.tree_connections[j].vessels)
                tree_connections.other_vessels = other_vessels
            tree_connections.solve(*args, num_vessels=num_vessels, attempts=attempts)
            self.tree_connections.append(tree_connections)

    def export_solid(self, cap_resolution=10, extrude_roots=False):
        network_solids = []
        network_lines = []
        network_tubes = []
        for i in range(len(self.tree_connections)):
            network_solid, network_line, network_tube = self.tree_connections[i].export_solid(cap_resolution=cap_resolution,
                                                                                         extrude_roots=extrude_roots)
            network_solids.extend(network_solid)
            network_lines.extend(network_line)
            network_tubes.extend(network_tube)
        return network_solids, network_lines, network_tubes

    def show(self, **kwargs):
        pass