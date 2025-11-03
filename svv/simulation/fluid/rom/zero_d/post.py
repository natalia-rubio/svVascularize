make_results = """import pyvista as pv
import numpy as np
import os
from tqdm import tqdm
import csv
from collections import defaultdict, OrderedDict


def read_csv_into_nested_dict(file_path):
    # Initialize a nested dictionary structure
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    time_steps = defaultdict(lambda: defaultdict(dict))
    time_counter = defaultdict(int)
    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            name = row['name']
            time = float(row['time'])
            flow_in = float(row['flow_in'])
            flow_out = float(row['flow_out'])
            pressure_in = float(row['pressure_in'])
            pressure_out = float(row['pressure_out'])
            parts = name.split('_')
            branch = int(parts[0].replace('branch', ''))
            segment = int(parts[1].replace('seg', ''))
            # Check if the time has already been encountered for this branch and segment
            if time not in time_steps[branch][segment]:
                time_steps[branch][segment][time] = time_counter[(branch, segment)]
                time_counter[(branch, segment)] += 1
            time_index = time_steps[branch][segment][time]
            data['time'][branch][segment][time_index] = time
            data['flow_in'][branch][segment][time_index] = flow_in
            data['flow_out'][branch][segment][time_index] = flow_out
            data['pressure_in'][branch][segment][time_index] = pressure_in
            data['pressure_out'][branch][segment][time_index] = pressure_out
    return data


geom_data = np.genfromtxt("geom.csv",delimiter=",")
data = read_csv_into_nested_dict("output.csv")

total = None
timepoints = []
min_pressure = np.inf
max_pressure = -np.inf
min_flow = np.inf
max_flow = -np.inf
min_wss = np.inf
max_wss = -np.inf
for idx in tqdm(range(len(data['time'][0][0])),desc="Building Timeseries ",position=0):
    #time_merge = None
    time = data['time'][0][0][idx]
    tmp_vessels = []
    for jdx in tqdm(range(len(data['flow_in'])),desc="Building Vessel Data",position=1,leave=False):
        vessel = list(data['flow_in'].keys())[jdx]
        start = geom_data[vessel,0:3]
        end   = geom_data[vessel,3:6]
        direction = (geom_data[vessel,3:6] - geom_data[vessel,0:3])/np.linalg.norm(geom_data[vessel,3:6] - geom_data[vessel,0:3])
        length    = geom_data[vessel,6]
        radius    = geom_data[vessel,7]
        number_segments = len(data['flow_in'][vessel]) #- 1 #assume only 1 segment right now
        number_points   = len(data['flow_in'][vessel])
        vdx = vessel
        for kdx in range(number_segments):
            center = (1/2)*direction*length + start
            vessel = pv.Cylinder(center=center,direction=direction,height=length,radius=radius)
            vessel = vessel.elevation(low_point=end,high_point=start,scalar_range=[data['pressure_out'][vdx][kdx][idx]/1333.33, data['pressure_in'][vdx][kdx][idx]/1333.33])
            if data['pressure_in'][vdx][kdx][idx]/1333.33 > max_pressure:
                max_pressure = data['pressure_in'][vdx][kdx][idx]/1333.33
            if data['pressure_out'][vdx][kdx][idx]/1333.33 < min_pressure:
                min_pressure = data['pressure_out'][vdx][kdx][idx]/1333.33
            vessel.rename_array('Elevation','Pressure [mmHg]',preference='point')
            vessel.cell_data['Flow [mL/s]'] = data['flow_in'][vdx][kdx][idx]
            re = (1.06*2*radius*((data['flow_in'][vdx][kdx][idx]/(np.pi*radius**2))/2))/0.04
            fd = 64/re
            wss = ((data['flow_in'][vdx][kdx][idx]/(np.pi*radius**2))/2)*fd*1.06
            vessel.cell_data['WSS [dyne/cm^2]']  = wss
            if max_flow < data['flow_in'][vdx][kdx][idx]:
                max_flow = data['flow_in'][vdx][kdx][idx]
            if min_flow > data['flow_in'][vdx][kdx][idx]:
                min_flow = data['flow_in'][vdx][kdx][idx]
            if max_wss < wss:
                max_wss = wss
            if min_wss > wss:
                min_wss = wss
            tmp_vessels.append(vessel)
        #if time_merge is None:
        #    time_merge = vessel
        #else:
        #    time_merge = time_merge.merge(vessel)
    time_merge = tmp_vessels[0].merge(tmp_vessels[1:])
    time_merge.field_data['time'] = time
    timepoints.append(time_merge)
    #if total is None:
    #    total = time_merge
    #else:
    #    total = total.merge(time_merge)
    if not os.path.isdir("timeseries"):
        os.mkdir("timeseries")

if not os.path.isdir("timeseries_for_pressure_gif"):
    os.mkdir("timeseries_for_pressure_gif")
if not os.path.isdir("timeseries_for_flow_gif"):
    os.mkdir("timeseries_for_flow_gif")
if not os.path.isdir("timeseries_for_wss_gif"):
    os.mkdir("timeseries_for_wss_gif")
total = timepoints[0].merge(timepoints[1:])
for i in tqdm(range(len(timepoints)),desc="Saving Timeseries",position=1):
    p = pv.Plotter(off_screen=True)
    p.add_mesh(timepoints[i],scalars='Pressure [mmHg]',clim=[round(min_pressure,4),round(max_pressure,4)],cmap="coolwarm")
    p.show(auto_close=True,screenshot=os.getcwd()+os.sep+"timeseries_for_pressure_gif"+os.sep+"time_point_{}.png".format(i))
    p = pv.Plotter(off_screen=True)
    p.add_mesh(timepoints[i],scalars='Flow [mL/s]',clim=[round(min_flow,4),round(max_flow,4)],cmap="GnBu")
    p.show(auto_close=True,screenshot=os.getcwd()+os.sep+"timeseries_for_flow_gif"+os.sep+"time_point_{}.png".format(i))
    p = pv.Plotter(off_screen=True)
    p.add_mesh(timepoints[i],scalars='WSS [dyne/cm^2]',clim=[round(min_wss,2),round(max_wss,2)],cmap="coolwarm")
    p.show(auto_close=True,screenshot=os.getcwd()+os.sep+"timeseries_for_wss_gif"+os.sep+"time_point_{}.png".format(i))
    timepoints[i].save(os.getcwd()+os.sep+"timeseries"+os.sep+"time_point_{}.vtp".format(i))

total.save(os.getcwd()+os.sep+"timeseries"+os.sep+"total.vtp")

"""

view_plots="""import numpy as np
import matplotlib.pyplot as plt
data = np.load("solver_0d_branch_results.npy",allow_pickle=True).item()
vessels = []
fig_flow = plt.figure()
ax_flow = fig_flow.add_subplot()
fig_pressure = plt.figure()
ax_pressure = fig_pressure.add_subplot()
fig_flow_outlets = plt.figure()
ax_flow_outlets = fig_flow_outlets.add_subplot()
fig_pressure_outlets = plt.figure()
ax_pressure_outlets = fig_pressure_outlets.add_subplot()
for vessel in data["flow"]:
    vessels.append(vessel)
for vessel in vessels:
    ax_flow.plot(data["time"],data["flow"][vessel][0],label="vessel_"+str(vessel))
    ax_pressure.plot(data["time"],data["pressure"][vessel][0]/1333.22,label="vessel_"+str(vessel))
import json
info = json.load(open("solver_0d.in"))
all_inlets = []
all_outlets = []
for i in info["junctions"]:
    for j in i["inlet_vessels"]:
        all_inlets.append(j)
    for j in i["outlet_vessels"]:
        all_outlets.append(j)
all_inlets = set(all_inlets)
all_outlets = set(all_outlets)
true_outlets = list(all_outlets.difference(all_inlets))
true_inlets = list(all_inlets.difference(all_outlets))
for vessel in true_outlets:
    ax_flow_outlets.plot(data["time"],data["flow"][vessel][-1],label="vessel_"+str(vessel))
    ax_pressure_outlets.plot(data["time"],data["pressure"][vessel][-1]/1333.22,label="vessel_"+str(vessel))
ax_flow.set_xlabel("Time (sec)")
ax_flow.set_ylabel("Flow (mL/s)")
ax_pressure.set_xlabel("Time (sec)")
ax_pressure.set_ylabel("Pressure (mmHg)")
ax_flow_outlets.set_xlabel("Time (sec)")
ax_pressure_outlets.set_xlabel("Time (sec)")
ax_flow_outlets.set_ylabel("Flow (mL/s)")
ax_flow_outlets.set_title("Outlets Only")
ax_pressure_outlets.set_ylabel("Pressure (mmHg)")
ax_pressure_outlets.set_title("Outlets Only")
#ax_pressure_outlets.set_ylim([-10*np.finfo(float).eps,10*np.finfo(float).eps])
plt.show()

"""

post_data = """import pyvista as pv
import numpy as np
import os
from tqdm import tqdm
import csv
from collections import defaultdict, OrderedDict


def read_csv_into_nested_dict(file_path):
    # Initialize a nested dictionary structure
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    time_steps = defaultdict(lambda: defaultdict(dict))
    time_counter = defaultdict(int)
    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            name = row['name']
            time = float(row['time'])
            flow_in = float(row['flow_in'])
            flow_out = float(row['flow_out'])
            pressure_in = float(row['pressure_in'])
            pressure_out = float(row['pressure_out'])
            parts = name.split('_')
            branch = int(parts[0].replace('branch', ''))
            segment = int(parts[1].replace('seg', ''))
            # Check if the time has already been encountered for this branch and segment
            if time not in time_steps[branch][segment]:
                time_steps[branch][segment][time] = time_counter[(branch, segment)]
                time_counter[(branch, segment)] += 1
            time_index = time_steps[branch][segment][time]
            data['time'][branch][segment][time_index] = time
            data['flow_in'][branch][segment][time_index] = flow_in
            data['flow_out'][branch][segment][time_index] = flow_out
            data['pressure_in'][branch][segment][time_index] = pressure_in
            data['pressure_out'][branch][segment][time_index] = pressure_out
    return data


geom_data = np.genfromtxt("geom.csv",delimiter=",")
data = read_csv_into_nested_dict("output.csv")

# Storing 5 data results: time, pressure_in, pressure_out, flow_in, flow_out
results = np.zeros((geom_data.shape[0], geom_data.shape[1] + 5, len(data['time'][0][0])))
for idx in tqdm(range(len(data['time'][0][0])),desc="Building Timeseries ",position=0):
    time = data['time'][0][0][idx]
    # Assign data from the csv file to the vessels in the results array for a given time point
    for jdx in range(results.shape[0]):
        results[jdx, 0:3, idx] = geom_data[jdx, 0:3]
        results[jdx, 3:6, idx] = geom_data[jdx, 3:6]
        results[jdx, 6, idx] = geom_data[jdx, 6]
        results[jdx, 7, idx] = geom_data[jdx, 7]
        results[jdx, 8, idx] = data['time'][jdx][0][idx]
        results[jdx, 9, idx] = data['pressure_in'][jdx][0][idx]
        results[jdx, 10, idx] = data['pressure_out'][jdx][0][idx]
        results[jdx, 11, idx] = data['flow_in'][jdx][0][idx]
        results[jdx, 12, idx] = data['flow_out'][jdx][0][idx]

# Save the results array to a numpy file
np.save("results.npy", results)
"""