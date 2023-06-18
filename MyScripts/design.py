import collections

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

import dflat.optimization_helpers as df_optimizer
import dflat.fourier_layer as df_fourier
import dflat.neural_optical_layer as df_neural
import dflat.data_structure as df_struct
import dflat.plot_utilities as plt_util
import dflat.tools as df_tools

from dflat.physical_optical_layer.core.ms_parameterization import generate_cell_perm
from myutils import *


class loss_fn():
    def __init__(self, shape, unit, mode_radius):
        self.target_mode = tf.convert_to_tensor(gaussian(shape, unit, mode_radius), dtype=tf.complex128)

    def __call__(self, pipeline_output):
        # (len(sim_wavelengths), batch_size, sensor_pixel_number["y"], sensor_pixel_number["x"])
        temp = 0.
        for i in range(pipeline_output.shape[0]):
            for j in range(pipeline_output.shape[1]):
                temp += cross_integral(pipeline_output[i, j], self.target_mode)
        return -temp


class pipeline_Metalens_MLP(df_optimizer.Pipeline_Object):
    def __init__(self, rcwa_parameters, propagation_parameters, savepath, saveAtEpochs=None):
        super(pipeline_Metalens_MLP, self).__init__(savepath, saveAtEpochs)

        # Add inputs to class attributes
        self.propagation_parameters = propagation_parameters
        self.rcwa_parameters = rcwa_parameters
        # define computational layers
        mlp_model = "MLP_Nanocylinders_Dense128_U650_H800"
        self.cell_parameterization = "cylindrical_nanoposts"
        self.mlp_latent_layer = df_neural.MLP_Latent_Layer(mlp_model)

        self.f_layer = df_fourier.Propagate_Planes_Layer(propagation_parameters)

        # Define initial starting condition for the metasurface latent tensor
        self.latent_tensor_variable = [*pipeline_Metalens_MLP.createRow(self.propagation_parameters["grid_shape"], self.propagation_parameters["num_rows_per_MLP_forward"], 150e-9)]
        self.customLoad()  # Load the last saved state of the optimization, if it exists

        return

    def __call__(self):
        wavelength_set_m = self.propagation_parameters["wavelength_set_m"]
        # trans = []
        # phase = []
        num_slices = len(self.latent_tensor_variable)
        num_wavelengths = len(wavelength_set_m)
        def lambdaCond(_idx, _trans, _phase):
            return tf.less(_idx, num_slices)
        def lambdaBody(_idx, _trans, _phase):
            trans, phase = self.mlp_latent_layer(self.latent_tensor_variable[_idx], wavelength_set_m)
            _trans = tf.concat([_trans, trans], 3)
            _phase = tf.concat([_phase, phase], 3)
            _idx += 1
            return [_idx, _trans, _phase]
        idx = tf.constant(0)
        hold_trans = tf.zeros([num_wavelengths, 1, self.propagation_parameters["grid_shape"][1], 0], dtype=tf.float32)
        hold_phase = tf.zeros([num_wavelengths, 1, self.propagation_parameters["grid_shape"][1], 0], dtype=tf.float32)
        loopData = tf.while_loop(
            lambdaCond,
            lambdaBody,
            loop_vars=[idx, hold_trans, hold_phase],
            shape_invariants=[
                idx.get_shape(),
                tf.TensorShape([num_wavelengths, 1, self.propagation_parameters["grid_shape"][1], None]),
                tf.TensorShape([num_wavelengths, 1, self.propagation_parameters["grid_shape"][1], None]),
            ],
            swap_memory=True
        )
        # for latenti in self.latent_tensor_variable:
        #     tmp1, tmp2 = self.mlp_latent_layer(latenti, wavelength_set_m)
        #     trans.append(tmp1)
        #     phase.append(tmp2)
        # trans = tf.concat(trans, 3)
        # phase = tf.concat(phase, 3)
        f_amp, f_phase = self.f_layer([loopData[1], loopData[2]])
        # Save the last lens and psf for plotting later
        self.last_lens = [loopData[1], loopData[2]]
        self.last_amp = f_amp
        self.last_pha = f_phase
        return arg_phase_to_complex(f_amp, f_phase)

    @staticmethod
    def createRow(grid_shape, num_rows_per_MLP_forward, init_value):
        num_row = grid_shape[2]
        index1 = 0
        index2 = min(num_rows_per_MLP_forward, num_row)
        while index2 <= num_row:
            yield tf.Variable(tf.zeros([grid_shape[0], grid_shape[1], index2 - index1], dtype=tf.float32) + init_value, trainable=True)
            if index2 == num_row: break
            index1 = index2
            index2 += num_rows_per_MLP_forward
            index2 = min(index2, num_row)

    def visualizeTrainingCheckpoint(self, saveto: str = None):
        '''
        This function is called during training to visualize the current state of the optimization
        Args:
            saveto: conventionally the number of iterations, but can be any string, used to generate a unique filename
        Returns:

        '''
        # This overrides the baseclass visualization call function, called during checkpoints
        savefigpath = self.savepath + "/trainingOutput/"

        # Get parameters for plotting
        # Helper call that returns simple definition of cartesian axis on lens and output space (um)
        xl, yl = plt_util.get_lens_pixel_coordinates(self.propagation_parameters)
        xd, yd = plt_util.get_detector_pixel_coordinates(self.propagation_parameters)
        xl, yl = xl * 1e6, yl * 1e6
        xd, yd = xd * 1e6, yd * 1e6
        Lx = self.rcwa_parameters["Lx"]
        Ly = self.rcwa_parameters["Ly"]

        sim_wavelengths = self.propagation_parameters["wavelength_set_m"]
        num_wl = len(sim_wavelengths)

        ### Display the learned phase and transmission profile on first row
        # and wavelength dependent PSFs on the second
        trans = self.last_lens[0]
        phase = self.last_lens[1]

        fig = plt.figure(figsize=(25, 10))
        ax = plt_util.addAxis(fig, 2, num_wl)
        for i in range(num_wl):
            # ax[i].plot(xl, phase[i, 0, 0, :], "k-")
            # ax[i].plot(xl, phase[i, 1, 0, :], "b-")
            # # ax[i].plot(xl, trans[i, 0, 0, :], "k*")
            # # ax[i].plot(xl, trans[i, 1, 0, :], "b*")
            ax[i].imshow(self.last_pha[i, 0, :, :], extent=(min(xd), max(xd), min(yd), max(yd)))
            plt_util.formatPlots(
                fig,
                ax[i],
                None,
                xlabel="det x (um)",
                ylabel="det y (um)",
                title=f"field phase on fiber end, wavelength {sim_wavelengths[i] * 1e9:3.0f} nm",
                setAspect="equal",
                fontsize_text=12,
                fontsize_title=12,
                fontsize_ticks=12,
            )

            ax[i + num_wl].imshow(self.last_amp[i, 0, :, :], extent=(min(xd), max(xd), min(yd), max(yd)))
            plt_util.formatPlots(
                fig,
                ax[i + num_wl],
                None,
                xlabel="det x (um)",
                ylabel="det y (um)",
                title=f"field amp on fiber end, wavelength {sim_wavelengths[i] * 1e9:3.0f} nm",
                setAspect="equal",
                fontsize_text=12,
                fontsize_title=12,
                fontsize_ticks=12,
            )
        plt.savefig(savefigpath + f"field_{saveto}.png", dpi=300)
        fig = plt.figure(figsize=(25, 10))
        ax = plt_util.addAxis(fig, 2, num_wl)
        for i in range(num_wl):
            ax[i].imshow(self.last_lens[1][i, 0, :, :], extent=(min(xl), max(xl), min(yl), max(yl)))
            plt_util.formatPlots(
                fig,
                ax[i],
                None,
                xlabel="det x (um)",
                ylabel="det y (um)",
                title=f"lens phase, wavelength {sim_wavelengths[i] * 1e9:3.0f} nm",
                setAspect="equal",
                fontsize_text=12,
                fontsize_title=12,
                fontsize_ticks=12,
            )
            ax[i + num_wl].imshow(self.last_lens[0][i, 0, :, :], extent=(min(xl), max(xl), min(yl), max(yl)))
            plt_util.formatPlots(
                fig,
                ax[i + num_wl],
                None,
                xlabel="det x (um)",
                ylabel="det y (um)",
                title=f"lens trans, wavelength {sim_wavelengths[i] * 1e9:3.0f} nm",
                setAspect="equal",
                fontsize_text=12,
                fontsize_title=12,
                fontsize_ticks=12,
            )
        plt.savefig(savefigpath + f"lens_{saveto}.png", dpi=300)
        return

    def calculate_efficiency(self, loss_fn, unit, saveto: str = None):
        sim_wavelengths = self.propagation_parameters["wavelength_set_m"]
        num_wl = len(sim_wavelengths)
        efficiency = np.zeros(num_wl)
        out = self.__call__()
        for i in range(num_wl):
            efficiency[i] = cross_integral(out[i, 0], loss_fn.target_mode) ** 2 * unit ** 2\
                            / (np.pi * self.propagation_parameters["radius_m"]**2)
            # here we have already normalized the target mode to 1, but the unit there is ignored, and the unit in the
            # cross_integral is ignored as well, so we need to multiply by the unit squared
            # because the incident field amplitude is 1, the area of the aperture is equal to its enclosed power
        fig = plt.figure(figsize=(5, 5))
        ax = plt_util.addAxis(fig, 1, 1)
        ax[0].plot(sim_wavelengths * 1e6, efficiency * 100, "k-")
        plt_util.formatPlots(
            fig,
            ax[0],
            None,
            xlabel="wavelength (um)",
            ylabel="coupling efficiency (%)",
            title=f"coupling efficiency",
            fontsize_text=12,
            fontsize_title=12,
            fontsize_ticks=12,
        )
        savefigpath = self.savepath + "/trainingOutput/"
        plt.savefig(savefigpath + f"efficiency_{saveto}.png", dpi=300)
    def exportMetalens(self, saveto: str = None):
        pass
        ### Display some of the learned metacells
        # We want to assemble the cell's dielectric profile, so we can plot it
        # latent_tensor_state = self.latent_tensor_variable
        # norm_shape_param = df_tools.latent_to_param(latent_tensor_state)
        # ER, _ = generate_cell_perm(norm_shape_param, self.rcwa_parameters, self.cell_parameterization, feature_layer=0)
        # disp_num = 5
        # cell_idx = np.linspace(0, ER.shape[1] - 1, disp_num).astype(int)

def optimize_metalens_mlp(radial_symmetry, num_epochs=30, try_gpu=True):
    # Define save path
    savepath = "output/multi_wavelength_mlp_metalens_design_650_800/"
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # Define Fourier parameters
    wavelength_list = np.linspace(1535e-9, 1565e-9, 5)
    propagation_parameters = df_struct.prop_params(
        {
            "wavelength_set_m": wavelength_list,
            "ms_samplesM": {"x": 1540, "y": 1540},
            "ms_dx_m": {"x": 650e-9, "y": 650e-9},
            "radius_m": 1e-3 / 2.,
            "sensor_distance_m": 1e-3,
            "initial_sensor_dx_m": {"x": 1e-6, "y": 1e-6},
            "sensor_pixel_size_m": {"x": 1e-6, "y": 1e-6},
            "sensor_pixel_number": {"x": 256, "y": 256},
            "radial_symmetry": radial_symmetry,
            "diffractionEngine": "fresnel_fourier",
            ### Optional keys
            "automatic_upsample": False,
            # If true, it will try to automatically determine good upsample factor for calculations
            # "manual_upsample_factor": 1,  # Otherwise you can manually dictate upsample factor
            "num_rows_per_MLP_forward": 100
            # limit memory usage by breaking up the forward pass into chunks
        })

    '''
    ms_samplesM: number of samples in the input field along x and y
    ms_dx_m: Cartesian grid discretization size along x and y for the input field
    radius_m: The radius of a circular aperture to be placed before the field in subsequent calculations. If set to "None", 
    then no aperture will be considered.
    sensor_distance_m: distance from the input plane to the output plane to propagate
    initial_sensor_dx_m: Grid size to explicitly compute the output field at (you can think of this as the field just before
     the detector)
    sensor_pixel_size_m: You can consider this as the actual sensor pixels which must have a pitch/discretization equal to 
    or larger than the initial_sensor_dx_m. The intensity on the pixels is the sum of the intensity of field points computed
     within the pixel and the phase is the average of such sub-field points.
    sensor_pixel_number: Specifies the number of grid points along x and y.
    '''

    df_struct.print_full_settings(propagation_parameters)

    gridshape = propagation_parameters["grid_shape"]

    # Define RCWA parameters
    fourier_modes = 5
    rcwa_parameters = df_struct.rcwa_params(
        {
            "wavelength_set_m": wavelength_list,
            "thetas": [0.0 for i in wavelength_list],
            "phis": [0.0 for i in wavelength_list],
            "pte": [1.0 for i in wavelength_list],
            "ptm": [1.0 for i in wavelength_list],
            "pixelsX": gridshape[2],
            "pixelsY": gridshape[1],
            "PQ": [fourier_modes, fourier_modes],
            "Lx": 650e-9,
            "Ly": 650e-9,
            "L": [800e-9],
            "Lay_mat": ["Vacuum"],
            "material_dielectric": 3.42 + 0.0j,
            "er1": "Vacuum",
            "er2": "Vacuum",
            "Nx": 200,
            "Ny": 200,
        }
    )

    # Call the pipeline
    saveAtEpoch = 10
    pipeline = pipeline_Metalens_MLP(rcwa_parameters, propagation_parameters, savepath, saveAtEpochs=saveAtEpoch)
    # pipeline.customLoad()  # restore previous training checkpoint if it exists

    # Define custom Loss function (Should always have pipeline_output as the function input if you use the helper)
    # Otherwise you can easily write your own train loop for more control
    sensor_pixel_number = propagation_parameters["sensor_pixel_number"]
    learning_rate = 2e-2
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    lf = loss_fn((256, 256), 1e-6, 5e-6)
    # pipeline.calculate_efficiency(lf, 1e-6, str(len(pipeline.loss_vector) if len(pipeline.loss_vector) else 0))
    # pipeline.visualizeTrainingCheckpoint(str(len(pipeline.loss_vector) if len(pipeline.loss_vector) else 0))
    df_optimizer.run_pipeline_optimization(pipeline, optimizer, num_epochs=num_epochs, loss_fn=tf.function(lf),
                                           allow_gpu=try_gpu)
    return


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    optimize_metalens_mlp(radial_symmetry=False, num_epochs=150, try_gpu=False)
