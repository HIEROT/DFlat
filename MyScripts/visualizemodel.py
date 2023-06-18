import dflat.neural_optical_layer as df_neural
import dflat.plot_utilities as df_plt
import dflat.datasets_metasurface_cells as df_library
import matplotlib.pyplot as plt
import numpy as np
if __name__ == "__main__":
    library = df_library.loadLibrary('Nanocylinders_U650nm_H800nm')
    # Phase and transmission data has shape [wl_m, radius]
    # radius, wl_m = library.param1, library.param2

    mlp_model = df_neural.MLP_Layer("MLP_Nanocylinders_Dense128_U650_H800")
    wavelength_m = np.linspace(1565e-9, 1530e-9, 100)
    radius = np.linspace(80e-9, 325e-9, 100)
    norm_param = mlp_model.shape_to_param(radius[np.newaxis, :, np.newaxis]) # Make it match the input shape of [D, nY, Nx]
    transmission, phase = mlp_model(norm_param, wavelength_m)

    fig2 = plt.figure(figsize=(25, 10))
    axisList = df_plt.addAxis(fig2, 1, 2)
    tt = axisList[0].imshow(transmission[:, 0, :, 0], extent=(min(radius), max(radius), max(wavelength_m), min(wavelength_m)), vmin=0, vmax=1)
    phi = axisList[1].imshow(phase[:,0,:,0], extent=(min(radius), max(radius), max(wavelength_m), min(wavelength_m)), cmap="hsv")
    df_plt.formatPlots(fig2, axisList[0], tt, "len r (nm)", "wavelength (nm)", "transmission", addcolorbar=True)
    df_plt.formatPlots(fig2, axisList[1], phi, "len r (nm)", "wavelength (nm)", "phase", addcolorbar=True)


    library.plotLibrary()

    param_dimensionality = mlp_model.param_dimensionality






