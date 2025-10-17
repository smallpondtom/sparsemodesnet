import h5py
import numpy as np

class ChannelDataSource:
    def __init__(
        self,
        hfname="../../Data/nrel/channel_5200_data_0_10000.h5",
        subsample=None,
        no_pressure=False,
        y_slice=None,
        z_slice=None,
        which_velocity="uvw"
    ):
        self.hfname = hfname
        self.subsample = subsample if subsample is not None else [1, 1, 1]
        self.no_pressure = no_pressure
        self.y_slice = y_slice  # Store y_slice
        self.z_slice = z_slice  # Store z_slice

        # Select specific velocity components indvidually or all
        if which_velocity == "u":
            self.field_indices = [0]
        elif which_velocity == "v":
            self.field_indices = [1]
        elif which_velocity == "w":
            self.field_indices = [2]
        else:
            self.field_indices = [0, 1, 2]  # Default to all velocity components
        
        if not no_pressure:
            self.field_indices.append(3)  # Include pressure (index 3)
        
        with h5py.File(self.hfname, "r") as f:
            dset = f["data"]
            assert isinstance(dset, h5py.Dataset)
            self.shape = dset.shape
            self.fields = f["fields"][()][self.field_indices]
            # Apply subsampling to coordinate arrays
            self.x = f["x"][()][::self.subsample[0]]
            if y_slice is None:
                self.y = f["y"][()][::self.subsample[1]]
            else:
                self.y = f["y"][()][y_slice]
            if z_slice is None:
                self.z = f["z"][()][::self.subsample[2]]
            else:
                self.z = f["z"][()][z_slice]
            self.t = f["times"][()]
        self.dim_order = ["times", "fields", "x", "y", "z"]
        self.n_snapshots = self.shape[0]

    def __len__(self):
        return self.n_snapshots

    
    def __getitem__(self, key):
        with h5py.File(self.hfname, "r") as f:
            dset = f["data"]
            assert isinstance(dset, h5py.Dataset)

            # Determine y-dimension slice
            if self.y_slice is not None:
                y_dim_slice = self.y_slice
            else:
                y_dim_slice = slice(None, None, self.subsample[1])
            
            # Determine z-dimension slice
            if self.z_slice is not None:
                z_dim_slice = self.z_slice
            else:
                z_dim_slice = slice(None, None, self.subsample[2])
            
            # Build the full indexing tuple including subsampling and field selection
            if isinstance(key, int):
                # Single snapshot: 
                # (key, field_indices, ::subsample[0], ::subsample[1], z_slice or ::subsample[2])
                full_key = (key, self.field_indices, 
                           slice(None, None, self.subsample[0]),
                           y_dim_slice,
                           z_dim_slice)
            elif isinstance(key, slice):
                # Multiple snapshots: 
                # (key, field_indices, ::subsample[0], ::subsample[1], z_slice or ::subsample[2])
                full_key = (key, self.field_indices,
                           slice(None, None, self.subsample[0]),
                           y_dim_slice,
                           z_dim_slice)
            else:
                # Handle other key types (tuple, list, etc.)
                if isinstance(key, tuple):
                    # Extend the key tuple with field selection and subsampling
                    spatial_slices = (slice(None, None, self.subsample[0]),
                                      y_dim_slice,
                                      z_dim_slice)
                    full_key = (key[0], self.field_indices) + spatial_slices
                else:
                    full_key = (key, self.field_indices,
                               slice(None, None, self.subsample[0]),
                               y_dim_slice,
                               z_dim_slice)
            
            data = np.array(dset[full_key])
        return data

    
    def get_matrix(self, snapshot_range=None):
        """
        Reshape the data into a matrix where each row contains all spatial points
        for each field variable (u, v, w, p), and columns are snapshots.
        
        Parameters:
        -----------
        snapshot_range : slice, int, or None
            Range of snapshots to load. If None, loads all snapshots.
            
        Returns:
        --------
        numpy.ndarray
            Matrix of shape (n_spatial_points * n_fields, n_snapshots)
            where rows are organized as:
            [u_all_points, v_all_points, w_all_points, p_all_points] (if pressure included)
            or [u_all_points, v_all_points, w_all_points] (if no_pressure=True)
        """
        if snapshot_range is None:
            data = self[:]  # Load all snapshots
        else:
            data = self[snapshot_range]
        
        # Handle case where z_slice is an integer (selects single z-plane)
        # data shape: (n_snapshots, n_fields, nx, ny, nz) or (n_snapshots, n_fields, nx, ny) if z_slice is int
        if data.ndim == 4:
            # z_slice was an integer, so we have (n_snapshots, n_fields, nx, ny)
            n_snapshots, n_fields, nx, ny = data.shape
            n_spatial_points = nx * ny
            # Reshape to (n_snapshots, n_fields, n_spatial_points)
            data_reshaped = data.reshape(n_snapshots, n_fields, n_spatial_points)
        else:
            # Normal case with 5 dimensions
            n_snapshots, n_fields, nx, ny, nz = data.shape
            n_spatial_points = nx * ny * nz
            # Reshape to (n_snapshots, n_fields, n_spatial_points)
            data_reshaped = data.reshape(n_snapshots, n_fields, n_spatial_points)
        
        # Transpose and reshape to get the desired matrix format
        # (n_fields * n_spatial_points, n_snapshots)
        matrix = data_reshaped.transpose(1, 2, 0).reshape(
            n_fields * n_spatial_points, n_snapshots
        )
        
        return matrix


    # def __getitem__(self, key):
    #     with h5py.File(self.hfname, "r") as f:
    #         dset = f["data"]
    #         assert isinstance(dset, h5py.Dataset)
            
    #         # Build the full indexing tuple including subsampling and field selection
    #         if isinstance(key, int):
    #             # Single snapshot: 
    #             # (key, field_indices, ::subsample[0], ::subsample[1], ::subsample[2])
    #             full_key = (key, self.field_indices, 
    #                        slice(None, None, self.subsample[0]),
    #                        slice(None, None, self.subsample[1]), 
    #                        slice(None, None, self.subsample[2]))
    #         elif isinstance(key, slice):
    #             # Multiple snapshots: 
    #             # (key, field_indices, ::subsample[0], ::subsample[1], ::subsample[2])
    #             full_key = (key, self.field_indices,
    #                        slice(None, None, self.subsample[0]),
    #                        slice(None, None, self.subsample[1]),
    #                        slice(None, None, self.subsample[2]))
    #         else:
    #             # Handle other key types (tuple, list, etc.)
    #             if isinstance(key, tuple):
    #                 # Extend the key tuple with field selection and subsampling
    #                 spatial_slices = (slice(None, None, self.subsample[0]),
    #                                 slice(None, None, self.subsample[1]),
    #                                 slice(None, None, self.subsample[2]))
    #                 full_key = (key[0], self.field_indices) + spatial_slices
    #             else:
    #                 full_key = (key, self.field_indices,
    #                            slice(None, None, self.subsample[0]),
    #                            slice(None, None, self.subsample[1]),
    #                            slice(None, None, self.subsample[2]))
            
    #         data = np.array(dset[full_key])
    #     return data

    
    # def get_matrix(self, snapshot_range=None):
    #     """
    #     Reshape the data into a matrix where each row contains all spatial points
    #     for each field variable (u, v, w, p), and columns are snapshots.
        
    #     Parameters:
    #     -----------
    #     snapshot_range : slice, int, or None
    #         Range of snapshots to load. If None, loads all snapshots.
            
    #     Returns:
    #     --------
    #     numpy.ndarray
    #         Matrix of shape (n_spatial_points * n_fields, n_snapshots)
    #         where rows are organized as:
    #         [u_all_points, v_all_points, w_all_points, p_all_points] (if pressure included)
    #         or [u_all_points, v_all_points, w_all_points] (if no_pressure=True)
    #     """
    #     if snapshot_range is None:
    #         data = self[:]  # Load all snapshots
    #     else:
    #         data = self[snapshot_range]
        
    #     # data shape: (n_snapshots, n_fields, nx, ny, nz)
    #     n_snapshots, n_fields, nx, ny, nz = data.shape
    #     n_spatial_points = nx * ny * nz
        
    #     # Reshape to (n_snapshots, n_fields, n_spatial_points)
    #     data_reshaped = data.reshape(n_snapshots, n_fields, n_spatial_points)
        
    #     # Transpose and reshape to get the desired matrix format
    #     # (n_fields * n_spatial_points, n_snapshots)
    #     matrix = data_reshaped.transpose(1, 2, 0).reshape(
    #         n_fields * n_spatial_points, n_snapshots
    #     )
        
    #     return matrix

# import h5py
# import numpy as np

# class ChannelDataSource:
#     def __init__(
#         self,
#         hfname="../../Data/nrel/channel_5200_data_0_10000.h5",
#         subsample=None,
#         no_pressure=False
#     ):
#         self.hfname = hfname
#         self.subsample = subsample if subsample is not None else [1, 1, 1]
#         if no_pressure:
#             field_indices = [0, 1, 2]  # Exclude pressure (index 3)
#         else:
#             field_indices = [0, 1, 2, 3]
        
#         with h5py.File(self.hfname, "r") as f:
#             dset = f["data"]
#             assert isinstance(dset, h5py.Dataset)
#             shape = dset.shape
#             self.fields = f["fields"][()][field_indices]
#             # Apply subsampling to coordinate arrays
#             self.x = f["x"][()][::self.subsample[0]]
#             self.y = f["y"][()][::self.subsample[1]]
#             self.z = f["z"][()][::self.subsample[2]]
#         self.dim_order = ["times", "fields", "x", "y", "z"]
#         self.n_snapshots = shape[0]

#     def __len__(self):
#         return self.n_snapshots

#     def __getitem__(self, key):
#         with h5py.File(self.hfname, "r") as f:
#             dset = f["data"]
#             assert isinstance(dset, h5py.Dataset)
            
#             # Build the full indexing tuple including subsampling
#             if isinstance(key, int):
#                 # Single snapshot: (key, :, ::subsample[0], ::subsample[1], ::subsample[2])
#                 full_key = (key, slice(None), 
#                            slice(None, None, self.subsample[0]),
#                            slice(None, None, self.subsample[1]), 
#                            slice(None, None, self.subsample[2]))
#             elif isinstance(key, slice):
#                 # Multiple snapshots: (key, :, ::subsample[0], ::subsample[1], ::subsample[2])
#                 full_key = (key, slice(None),
#                            slice(None, None, self.subsample[0]),
#                            slice(None, None, self.subsample[1]),
#                            slice(None, None, self.subsample[2]))
#             else:
#                 # Handle other key types (tuple, list, etc.)
#                 if isinstance(key, tuple):
#                     # Extend the key tuple with subsampling for spatial dimensions
#                     spatial_slices = (slice(None, None, self.subsample[0]),
#                                     slice(None, None, self.subsample[1]),
#                                     slice(None, None, self.subsample[2]))
#                     full_key = key + spatial_slices[len(key)-1:]
#                 else:
#                     full_key = (key, slice(None),
#                                slice(None, None, self.subsample[0]),
#                                slice(None, None, self.subsample[1]),
#                                slice(None, None, self.subsample[2]))
            
#             data = np.array(dset[full_key])
#         return data