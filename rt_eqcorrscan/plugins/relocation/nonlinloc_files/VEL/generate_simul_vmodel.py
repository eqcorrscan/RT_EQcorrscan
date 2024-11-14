"""
Script to generate SIMUL2K formatted grid-files from Donna's NZ3D model 2.1

Calum J. Chamberlain

19/09/2019
"""

import numpy as np

input_file = "NZWmod2.1_3Dvelocity_tbl/vlnzw2p1dnxyzltln.tbl.txt"

def read_donna(filename: str) -> dict:
    with open(filename, "r") as f:
        lines = f.read().splitlines()
    head = lines[0].split()
    origin = (float(head[4]), float(head[5]))
    rotation = float(head[7])
    nzco = int(head[9])
    central_meridian = float(head[11])

    columns = lines[1].split()
    n_obs = 0
    for line in lines[2:]:
        if len(line.split()) == len(columns):
            n_obs += 1
    out = {column: np.empty(n_obs) for column in columns}

    i = 0
    for line in lines[2:]:
        values = line.split()
        if len(values) == len(columns):
            for column, value in zip(columns, values):
                out[column][i] = float(value)
            i += 1
    out.update({"origin": origin, "rotation": rotation, "nzco": nzco,
                "central_meridian": central_meridian})
    return out


def donna_to_simul(donna: dict, outfile_prefix: str = "donna_nz3d_2.1"):
    x_values = sorted(list(set(donna["x(km)"])))
    y_values = sorted(list(set(donna["y(km)"])))
    z_values = sorted(list(set(donna["Depth(km_BSL)"])))
    head = (" 1.0 {0} {1} {2} 2       Velocity model made using "
            "Calum's Python Script".format(
                len(x_values), len(y_values), len(z_values)))
    x_val_line = ' '.join(["{0:4.1f}".format(val) for val in x_values])
    y_val_line = ' '.join(["{0:4.1f}".format(val) for val in y_values])
    z_val_line = ' '.join(["{0:4.1f}".format(val) for val in z_values])

    head = "\n".join([head, x_val_line, y_val_line, z_val_line, "  0  0  0"])
    head += "\n\n"

    """
    Format is:

    x0y0z0 x1y0z0 ... x-1y0z0
    x0y1z0 x1y1z0 ... x-1y1z0
       .
       .
       .
    x0y-1z0 ...
    x0y0z1

    etc
    """

    # This is pretty fugly, sorry.
    vp_grid = np.empty((len(x_values), len(y_values), len(z_values)))
    vs_grid = np.empty_like(vp_grid)
    for i, x_val in enumerate(x_values):
        for j, y_val in enumerate(y_values):
            for k, z_val in enumerate(z_values):
                x_slice = np.where(donna["x(km)"] == x_val)[0]
                y_slice = np.where(donna["y(km)"] == y_val)[0]
                z_slice = np.where(donna["Depth(km_BSL)"] == z_val)[0]
                ind = set(x_slice).intersection(set(y_slice)).intersection(set(z_slice))
                assert len(ind) == 1
                ind = ind.pop()
                vp_grid[i, j, k] = donna["Vp"][ind]
                vs_grid[i, j, k] = donna["Vs"][ind]
    
    # Generate the formatted thingy
    vp_lines, vs_lines = ([], [])
    for i in range(len(z_values)):
        vp_block, vs_block = ([], [])
        for j in range(len(y_values)):
            vp_line, vs_line = ([], [])
            for k in range(len(x_values)):
                vp_line.append("{0:4.2f}".format(vp_grid[k, j, i]))
                vs_line.append("{0:4.2f}".format(vs_grid[k, j, i]))
            vp_block.append(" ".join(vp_line))
            vs_block.append(" ".join(vs_line))
        vp_lines.append("\n".join(vp_block))
        vs_lines.append("\n".join(vs_block))
    vp_lines = "\n".join(vp_lines)
    vs_lines = "\n".join(vs_lines)

    # Write the files
    with open("{0}_P.inp".format(outfile_prefix), "w") as f:
        f.write(head)
        f.write(vp_lines)

    with open("{0}_S.inp".format(outfile_prefix), "w") as f:
        f.write(head)
        f.write(vs_lines)