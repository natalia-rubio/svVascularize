import pyvista


def show(forest, plot_domain=False, return_plotter=False, **kwargs):
    """
    This function is used to visualize a synthetic vascular tree.
    """
    colors = kwargs.get('colors', ['red', 'blue', 'green', 'yellow', 'purple',
                                   'orange', 'cyan', 'magenta', 'white', 'black'])
    plotter = pyvista.Plotter(**kwargs)
    count = 0
    for network in forest.networks:
        for tree in network:
            for i in range(tree.data.shape[0]):
                center = (tree.data[i, 0:3] + tree.data[i, 3:6]) / 2
                direction = tree.data.get('w_basis', i)
                radius = tree.data.get('radius', i)
                length = tree.data.get('length', i)
                vessel = pyvista.Cylinder(center=center, direction=direction, radius=radius, height=length)
                plotter.add_mesh(vessel, color=colors[count % len(colors)])
            count += 1
    if plot_domain:
        plotter.add_mesh(forest.domain.boundary, color='grey', opacity=0.25)
    if return_plotter:
        return plotter
    else:
        plotter.show()
