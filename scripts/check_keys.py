import scipy.io as sio

file_path = r"D:\Python Projects\haptics\data\LMT_FinalDatabase\CBM_FinalDatabase\C1\S1\M1\C1_S1_M1_P1_X1_I1.mat"

print(f"Inspecting: {file_path}\n" + "-"*40)

try:
    mat = sio.loadmat(file_path)
    
    # Access the 1x1 struct
    recording = mat['finalMaterialRecording'][0, 0]
    
    print("SUCCESS: Opened 'finalMaterialRecording' struct.")
    print("\nFields found inside:")
    
    # In scipy, MATLAB structs become numpy structured arrays. 
    # The field names are stored in the array's dtype.names.
    for field_name in recording.dtype.names:
        data = recording[field_name]
        print(f" - {field_name}: shape {data.shape}")

except Exception as e:
    print(f"Failed to read struct: {e}")