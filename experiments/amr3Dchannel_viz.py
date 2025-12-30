"""
AMR-Wind 3D Channel Flow: Visualizations for paper
"""

#%% Load modules
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

# Add parent directory to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.channel_data_source import ChannelDataSource

#%% %============================= Main Script ================================%
if __name__ == "__main__":
    # Load the data source
    ds = ChannelDataSource(
        hfname="../../Data/nrel/channel_5200_data_0_10000.h5",
        subsample=[1, 1, 1],
        y_slice=96,
        no_pressure=True,
        which_velocity="u"
    )

    # Data parameters
    n_snapshots = 1000

    tidx = 500
    # Load the data
    Xu = ds.get_matrix(snapshot_range=slice(0, n_snapshots))
    xu = Xu[:, tidx]

    # Load the data source
    ds = ChannelDataSource(
        hfname="../../Data/nrel/channel_5200_data_0_10000.h5",
        subsample=[1, 1, 1],
        y_slice=96,
        no_pressure=True,
        which_velocity="w"
    )

    # Load the data
    Xw = ds.get_matrix(snapshot_range=slice(0, n_snapshots))
    xw = Xw[:, tidx]

#%% Plot the spectral decay
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": [
             'Ubuntu', # 'DejaVu Sans', 'Helvetica', 'Arial', 'Liberation Sans',
        ],
        "font.monospace": "Ubuntu Mono",
        "axes.labelweight": "bold",
    })

    Xu_svd = np.linalg.svd(Xu, compute_uv=False)
    Xw_svd = np.linalg.svd(Xw, compute_uv=False)

    fig, ax = plt.subplots(figsize=(24,5))
    ax.semilogy(np.arange(1, n_snapshots+1), Xu_svd, 
                'o-', label='Streamwise Velocity', markersize=10)
    ax.semilogy(np.arange(1, n_snapshots+1), Xw_svd, 
                's--', label='Wall-Normal Velocity', markersize=10)
    ax.set_xlabel(r'Mode Number, $i$', fontsize=33)
    ax.set_ylabel(r'Singular Value, $\sigma_i$', fontsize=33)
    # ax.set_title('Spectral Decay of Velocity Data Matrices', fontsize=28)
    ax.tick_params(labelsize=30)
    leg = ax.legend(fontsize=27, loc='center right', frameon=False, 
                    handlelength=4, bbox_to_anchor=(1.0, 0.68))
    for line in leg.get_lines():
        line.set_linewidth(10)
        line.set_markersize(15)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5)
    
    # Create inset axis
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    x1, x2, y1, y2 = -20, 120, 2, 7e3
    axins = ax.inset_axes(
        [1.1, 0.1, 0.7, 0.8], xlim=(x1, x2), ylim=(y1, y2),
    )
    axins.semilogy(np.arange(1, 101), Xu_svd[:100], 'o-', markersize=10)
    axins.semilogy(np.arange(1, 101), Xw_svd[:100], 's--', markersize=10)
    axins.tick_params(labelsize=25)
    axins.grid(True, which='both', linestyle='--', linewidth=0.3, alpha=0.7)
    
    # Add zoom indicator box
    # axins.set_xlim(1, 100)
    rect = (x1, y1, x2 - x1, y2 - y1)
    inset_indicator = ax.indicate_inset(rect, edgecolor="black", linewidth=2)
    from matplotlib.patches import ConnectionPatch
    cp1 = ConnectionPatch(xyA=(x2, y1), xyB=(0, 0), axesA=ax, axesB=axins,
                        coordsA="data", coordsB="axes fraction", lw=1.2, ls=":")
    cp2 = ConnectionPatch(xyA=(x2, y2), xyB=(0, 1), axesA=ax, axesB=axins,
                        coordsA="data", coordsB="axes fraction", lw=1.2, ls=":")
    ax.add_patch(cp1)
    ax.add_patch(cp2)
    
    plt.tight_layout()
    plt.savefig('figures/amr3Dchannel/spectral_decay.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

#%% Load the 3D data
    ds = ChannelDataSource(
        hfname="../../Data/nrel/channel_5200_data_0_10000.h5",
        subsample=[1, 1, 1],
        no_pressure=True,
        which_velocity="u"
    )
    nx, ny, nz = ds.shape[2:]
    x3d_u = ds.get_matrix(snapshot_range=slice(tidx, tidx+1)).reshape(nx,ny,nz)
    u_max, u_min = np.max(x3d_u), np.min(x3d_u)

#%% Plot 3D channel flow u-velocity with isosurfaces

    # Create a PyVista grid
    grid = pv.StructuredGrid(*np.meshgrid(ds.x, ds.y, ds.z, indexing='ij'))
    grid["streamwise velocity"] = x3d_u.flatten(order="F")

    # Calculate mean velocity for fluctuations (optional, for better visualization)
    u_mean = np.mean(x3d_u)
    u_fluctuations = x3d_u - u_mean
    grid["u_fluctuations"] = u_fluctuations.flatten(order="F")

    # Define plotter
    p = pv.Plotter(off_screen=True)

    # Alternative Method 2: Using multiple isosurfaces with colormap
    contours = grid.contour(np.linspace(u_min, u_max, 10), 
                            scalars="streamwise velocity")
    p.add_mesh(contours, cmap="coolwarm", opacity=1.0, show_scalar_bar=False)
    sbar = p.add_scalar_bar(
        title='streamwise velocity',
        title_font_size=70,
        label_font_size=50,
    )
    # Access the underlying VTK object for fine-tuning
    sbar.GetTitleTextProperty().SetLineOffset(-60)  # Adjust this value for spacing

    # Add axes and set background
    p.add_axes(
        box=False,  # Remove the box around axes
        x_color='red',
        y_color='green', 
        z_color='blue',
        xlabel='X',
        ylabel='Y',
        zlabel='Z',
        line_width=5,  # Make the axes lines thicker
        cone_radius=1,  # Make the cone tips bigger
        shaft_length=1,  # Adjust shaft length
        tip_length=0.2,  # Adjust tip length
        ambient=0.5,  # Lighting parameter
        label_size=(1.0, 0.25),  # Make the labels bigger
    )
    p.set_background("white")

    # Camera position - adjust for best view
    p.camera_position = [
        (15.0, 17, 7.0),  # Camera position
        (5.6, 1.5, 0.0),  # Focal point
        (0.0, 0.0, 1.0)   # View up direction
    ]
    p.add_text("Streamwise Velocity Isosurface in Channel Flow", 
           position=[180, 1350],
           font_size=45,
           color='black',
           font='arial')
    # Add additional lights
    p.add_light(pv.Light(position=(5, 5, 5), intensity=0.1))
    p.add_light(pv.Light(position=(-5, 5, 5), intensity=0.1))
    # Save screenshot
    p.show(screenshot='figures/amr3Dchannel/u_velocity_3d_isosurfaces.png', 
           window_size=[2400,1600])

#%% Plot the slice
    # Reshape the 2D slice data
    nx, ny, nz = ds.shape[2:]  # Should be (nx, nz) for a y-slice
    xu_2d = xu.reshape(nx, nz)
    nx_3d, ny_3d, nz_3d = nx, ny, nz

    # Create plotter
    p = pv.Plotter(off_screen=True)

    # 1. Create the y-slice with velocity data
    y_position = ds.y[96]  # Get actual y coordinate at index 96
    xx_y, zz_y = np.meshgrid(ds.x, ds.z, indexing='ij')
    yy_y = np.ones_like(xx_y) * y_position
    y_slice = pv.StructuredGrid(xx_y, yy_y, zz_y)
    y_slice["streamwise velocity"] = xu_2d.flatten(order="F")

    # Add the y-slice with colormap
    u_max = np.max(xu_2d)
    actor = p.add_mesh(
        y_slice, cmap="coolwarm", clim=[u_min, u_max], opacity=1, 
        show_scalar_bar=False,
        ambient=0.5,  # Increase ambient light (0-1)
        diffuse=0.8,  # Increase diffuse reflection
        specular=0.3
    )
    sbar = p.add_scalar_bar(
        title='streamwise velocity',
        title_font_size=70,
        label_font_size=50,
    )
    # Access the underlying VTK object for fine-tuning
    sbar.GetTitleTextProperty().SetLineOffset(-60)  # Adjust this value for spacing

    # 2. Create empty x-slice (vertical plane perpendicular to x)
    x_position = ds.x[nx_3d//2]  # Middle of x domain
    yy_x, zz_x = np.meshgrid(ds.y, ds.z, indexing='ij')
    xx_x = np.ones_like(yy_x) * x_position
    x_slice = pv.StructuredGrid(xx_x, yy_x, zz_x)

    # Add x-slice as a semi-transparent plane without data
    p.add_mesh(x_slice, color="grey", opacity=0.1, show_edges=False)

    # 3. Create empty z-slice (horizontal plane perpendicular to z)
    z_position = ds.z[nz_3d//2]  # Middle of z domain
    xx_z, yy_z = np.meshgrid(ds.x, ds.y, indexing='ij')
    zz_z = np.ones_like(xx_z) * z_position
    z_slice = pv.StructuredGrid(xx_z, yy_z, zz_z)

    # Add z-slice as a semi-transparent plane without data
    p.add_mesh(z_slice, color="grey", opacity=0.1, show_edges=False)

    # Optional: Add edges to make the planes more visible
    p.add_mesh(x_slice.extract_all_edges(), color="white", line_width=1, opacity=0.0)
    p.add_mesh(z_slice.extract_all_edges(), color="white", line_width=1, opacity=0.0)
    p.add_mesh(y_slice.extract_all_edges(), color="black", line_width=1, opacity=0.05)

    # Add axes and set background
    p.add_axes(
        box=False,  # Remove the box around axes
        x_color='red',
        y_color='green', 
        z_color='blue',
        xlabel='X',
        ylabel='Y',
        zlabel='Z',
        line_width=5,  # Make the axes lines thicker
        cone_radius=1,  # Make the cone tips bigger
        shaft_length=1,  # Adjust shaft length
        tip_length=0.2,  # Adjust tip length
        ambient=0.5,  # Lighting parameter
        label_size=(1.0, 0.25),  # Make the labels bigger
    )
    p.set_background("white")

    # Camera position
    p.camera_position = [
        (15.0, 17, 7.0),  # Camera position
        (5.6, 1.5, 0.0),  # Focal point
        (0.0, 0.0, 1.0)   # View up direction
    ]
    p.add_text("Spanwise Slice of the Streamwise Velocity Snapshot", 
           position=[100, 1350],
           font_size=45,
           color='black',
           font='arial')
    # Add additional lights
    p.add_light(pv.Light(position=(5, 5, 5), intensity=0.1))
    p.add_light(pv.Light(position=(-5, 5, 5), intensity=0.1))
    # Save screenshot
    p.show(screenshot='figures/amr3Dchannel/u_velocity_yslice_orthogonal.png', 
           window_size=[2400,1600])

# #%% Plot 3D channel flow u-velocity
#     import pyvista as pv

#     # Create a PyVista grid
#     u_max = np.max(x3d_u)
#     grid = pv.StructuredGrid(*np.meshgrid(ds.x, ds.y, ds.z, indexing='ij'))
#     grid["u_velocity"] = x3d_u.flatten(order="F")
#     # Define plotter
#     p = pv.Plotter(off_screen=True)
#     p.add_mesh(grid.slice_orthogonal(), cmap="coolwarm", 
#                  clim=[0, u_max], opacity=0.75)
#     p.add_axes()
#     p.set_background("white")
#     # Camera position
#     p.camera_position = [
#         (15.0, 17, 7.0),  # Camera position
#         (5.6, 1.5, 0.0),  # Focal point
#         (0.0, 0.0, 1.0)   # View up direction
#     ]
#     # Save screenshot
#     p.show(screenshot='figures/amr3Dchannel/u_velocity_3d.png', window_size=[1200,800])

#%% Load the 3D data
    ds = ChannelDataSource(
        hfname="../../Data/nrel/channel_5200_data_0_10000.h5",
        subsample=[1, 1, 1],
        no_pressure=True,
        which_velocity="w"
    )
    nx, ny, nz = ds.shape[2:]
    x3d_w = ds.get_matrix(snapshot_range=slice(tidx, tidx+1)).reshape(nx,ny,nz)
    w_max, w_min = np.max(x3d_w), np.min(x3d_w)

#%% Plot 3D channel flow w-velocity with isosurfaces

    # Create a PyVista grid
    grid = pv.StructuredGrid(*np.meshgrid(ds.x, ds.y, ds.z, indexing='ij'))
    grid["wall-normal velocity"] = x3d_w.flatten(order="F")

    # Calculate mean velocity for fluctuations (optional, for better visualization)
    w_mean = np.mean(x3d_w)
    w_fluctuations = x3d_w - u_mean
    grid["w_fluctuations"] = w_fluctuations.flatten(order="F")

    # Define plotter
    p = pv.Plotter(off_screen=True)

    # Alternative Method 2: Using multiple isosurfaces with colormap
    contours = grid.contour(np.linspace(w_min, w_max, 10), 
                            scalars="wall-normal velocity")
    p.add_mesh(
        contours, cmap="coolwarm", opacity=1.0, show_scalar_bar=False,
    )
    sbar = p.add_scalar_bar(
        title='wall-normal velocity',
        title_font_size=70,
        label_font_size=50,
    )
    # Access the underlying VTK object for fine-tuning
    sbar.GetTitleTextProperty().SetLineOffset(-60)  # Adjust this value for spacing

    # Add axes and set background
    p.add_axes(
        box=False,  # Remove the box around axes
        x_color='red',
        y_color='green', 
        z_color='blue',
        xlabel='X',
        ylabel='Y',
        zlabel='Z',
        line_width=5,  # Make the axes lines thicker
        cone_radius=1,  # Make the cone tips bigger
        shaft_length=1,  # Adjust shaft length
        tip_length=0.2,  # Adjust tip length
        ambient=0.5,  # Lighting parameter
        label_size=(1.0, 0.25),  # Make the labels bigger
    )
    p.set_background("white")

    # Camera position - adjust for best view
    p.camera_position = [
        (15.0, 17, 7.0),  # Camera position
        (5.6, 1.5, 0.0),  # Focal point
        (0.0, 0.0, 1.0)   # View up direction
    ]
    p.add_text("Wall-Normal Velocity Isosurface in Channel Flow", 
           position=[180, 1350],
           font_size=45,
           color='black',
           font='arial')
    # Add additional lights
    p.add_light(pv.Light(position=(5, 5, 5), intensity=0.1))
    p.add_light(pv.Light(position=(-5, 5, 5), intensity=0.1))
    # Save screenshot
    p.show(screenshot='figures/amr3Dchannel/w_velocity_3d_isosurfaces.png', 
           window_size=[2400,1600])

#%% Plot the slice
    # Reshape the 2D slice data
    nx, ny, nz = ds.shape[2:]  # Should be (nx, nz) for a y-slice
    xw_2d = xw.reshape(nx, nz)
    nx_3d, ny_3d, nz_3d = nx, ny, nz

    # Create plotter
    p = pv.Plotter(off_screen=True)

    # 1. Create the y-slice with velocity data
    y_position = ds.y[96]  # Get actual y coordinate at index 96
    xx_y, zz_y = np.meshgrid(ds.x, ds.z, indexing='ij')
    yy_y = np.ones_like(xx_y) * y_position
    y_slice = pv.StructuredGrid(xx_y, yy_y, zz_y)
    y_slice["wall-normal velocity"] = xw_2d.flatten(order="F")

    # Add the y-slice with colormap
    w_max = np.max(xw_2d)
    actor = p.add_mesh(
        y_slice, cmap="coolwarm", clim=[w_min, w_max], opacity=1, 
        show_scalar_bar=False,
    )
    sbar = p.add_scalar_bar(
        title='wall-normal velocity',
        title_font_size=70,
        label_font_size=50,
    )
    # Access the underlying VTK object for fine-tuning
    sbar.GetTitleTextProperty().SetLineOffset(-60)  # Adjust this value for spacing

    # 2. Create empty x-slice (vertical plane perpendicular to x)
    x_position = ds.x[nx_3d//2]  # Middle of x domain
    yy_x, zz_x = np.meshgrid(ds.y, ds.z, indexing='ij')
    xx_x = np.ones_like(yy_x) * x_position
    x_slice = pv.StructuredGrid(xx_x, yy_x, zz_x)

    # Add x-slice as a semi-transparent plane without data
    p.add_mesh(x_slice, color="grey", opacity=0.1, show_edges=False)

    # 3. Create empty z-slice (horizontal plane perpendicular to z)
    z_position = ds.z[nz_3d//2]  # Middle of z domain
    xx_z, yy_z = np.meshgrid(ds.x, ds.y, indexing='ij')
    zz_z = np.ones_like(xx_z) * z_position
    z_slice = pv.StructuredGrid(xx_z, yy_z, zz_z)

    # Add z-slice as a semi-transparent plane without data
    p.add_mesh(z_slice, color="grey", opacity=0.1, show_edges=False)

    # Optional: Add edges to make the planes more visible
    p.add_mesh(x_slice.extract_all_edges(), color="white", line_width=1, opacity=0.0)
    p.add_mesh(z_slice.extract_all_edges(), color="white", line_width=1, opacity=0.0)
    p.add_mesh(y_slice.extract_all_edges(), color="black", line_width=1, opacity=0.05)

    # Add axes and set background
    p.add_axes(
        box=False,  # Remove the box around axes
        x_color='red',
        y_color='green', 
        z_color='blue',
        xlabel='X',
        ylabel='Y',
        zlabel='Z',
        line_width=5,  # Make the axes lines thicker
        cone_radius=1,  # Make the cone tips bigger
        shaft_length=1,  # Adjust shaft length
        tip_length=0.2,  # Adjust tip length
        ambient=0.5,  # Lighting parameter
        label_size=(1.0, 0.25),  # Make the labels bigger
    )
    p.set_background("white")

    # Camera position
    p.camera_position = [
        (15.0, 17, 7.0),  # Camera position
        (5.6, 1.5, 0.0),  # Focal point
        (0.0, 0.0, 1.0)   # View up direction
    ]
    p.add_text("Spanwise Slice of the Wall-Normal Velocity Snapshot", 
           position=[70, 1350],
           font_size=45,
           color='black',
           font='arial')
    # Add additional lights
    p.add_light(pv.Light(position=(5, 5, 5), intensity=0.3))
    p.add_light(pv.Light(position=(-5, 5, 5), intensity=0.3))
    # Save screenshot
    p.show(screenshot='figures/amr3Dchannel/w_velocity_yslice_orthogonal.png', 
           window_size=[2400,1600])

# %%
