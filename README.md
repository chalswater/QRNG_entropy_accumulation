
#            Improving semi-device-independent randomness                     #
#               certification by entropy accumulation                         #

INTRUCTIONS TO USE THE CODE: The whole code consists of 8 python files. Here are presented in a suggested order of compilation.

  1. "read_and_write.py" reads and writes saved data on the Outputs folder
  2. "functions.py": contains all the functions required to compute all the
     rest of python files.
   
  - DATA GENERATING FILES:
    
      3. "read_experimental_data.py": reads the data from the experiment
          stored in the folder "extracted" using the function "SDI_QRNG.hdf5".
          Generates the single datapoints from our experiment plus the
          scalability of  our method with different subsets of data.

      4. "color_maps.py": computes the Shannon and min-entropies of the
         colormaps in the paper.

      5. "slices_of_datapoints.py": computes the Shannon and min-entropies
         of the slices of colormaps corresponding to the coherent state
         amplitudes used in the experiment.

  - PLOTTING FILES:

      6. "plot_color_map.py": generates a plot with the color maps.
        
      7. "plot_slices_datapoints.py": generates a plot with the slices
         of datapoints.

      8. "plot_scalability.py": generates a plot with the scalability of our
          method using different subsets of datapoints.

All data from the paper is already saved in the Outputs folder, and can be read comiling the "read_and_write.py" file.

PS: Extracted data is too heavy for GitHub. All plots can be reproduced with files in Outputs folder.
To generate new data, extracted data from experiment can be available upon request.




