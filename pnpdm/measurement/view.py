import h5py


# Open the .mat file in read mode
file_path = 'Sim_d=512_x=1-512_y=1-512_NBFkeep=500_illum=choose.mat'
with h5py.File(file_path, 'r') as f:
    # List all keys
    keys = list(f.keys())
    print("Variables in the file:", keys)
    
    # Now access a specific variable
    variable_name = keys[0]  # Replace with the desired key
    variable_data = f[variable_name][:]
    print(f"Data for {variable_name}: {variable_data}")