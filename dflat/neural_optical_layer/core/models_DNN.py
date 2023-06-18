import numpy as np
import tensorflow as tf
from .util_neural import leakyrelu100, gaussian_activation
from .arch_Parent_class import MLP_Nanofins_U350_H600, MLP_Nanocylinders_U180_H600, MLP_Nanocylinders_U540_H750
from .arch_Core_class import GFF_Projection_layer, GFF_Projection_layer, MLP_Object
from dflat.datasets_metasurface_cells import libraryClass as library

mlp_model_names = [
    "MLP_Nanocylinders_Dense256_U180_H600",
    "MLP_Nanocylinders_Dense128_U540_H750",
    "MLP_Nanocylinders_Dense128_U650_H800",
    # "MLP_Nanocylinders_Dense128_U180_H600",
    # "MLP_Nanocylinders_Dense64_U180_H600",
    "MLP_Nanofins_Dense1024_U350_H600",
    "MLP_Nanofins_Dense512_U350_H600",
    "MLP_Nanofins_Dense256_U350_H600",
]

class MLP_Nanocylinders_Dense128_U540_H750(MLP_Nanocylinders_U540_H750):
    def __init__(self, dtype=tf.float64):
        super(MLP_Nanocylinders_Dense128_U540_H750, self).__init__(dtype)

        self.set_model_name("MLP_Nanocylinders_Dense128_U540_H750")
        self.set_modelSavePath("trained_MLP_models/MLP_Nanocylinders_Dense128_U540_H750/")

        # Define a new architecture
        self._arch = [
            tf.keras.layers.Dense(
                128,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(
                128,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(3, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]

class MLP_Nanocylinders_Dense128_U650_H800(MLP_Object):
    def __init__(self, dtype=tf.float32):
        super().__init__()

        self.set_model_name("MLP_Nanocylinders_Dense128_U650_H800")
        self.set_modelSavePath("trained_MLP_models/MLP_Nanocylinders_Dense128_U650_H800/")
        self.set_preprocessDataBounds([[80e-9, 325e-9], [1530e-9, 1565e-9]], ["radius_m", "wavelength_m"])
        self.set_model_dtype(dtype)
        self.set_input_shape((2,))
        self.set_output_pol_state(1)
        # Define a new architecture
        self._arch = [
            tf.keras.layers.Dense(
                64,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(
                64,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(3, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]

    def returnLibraryAsTrainingData(self):
        # FDTD generated data loaded from library class file
        useLibrary = library.Nanocylinders_U650nm_H800nm()
        params = useLibrary.params
        phase = useLibrary.phase
        transmission = useLibrary.transmission

        # Normalize inputs (always done based on self model normalize function)
        normalizedParams = self.normalizeInput(params)
        trainx = np.stack([param.flatten() for param in normalizedParams], -1)
        trainy = np.stack(
            [
                np.cos(phase[:, :]).flatten(),  # cos of phase x polarized light
                np.sin(phase[:, :]).flatten(),  # sin of phase x polarized light
                transmission[:, :].flatten(),  # x transmission
            ],
            -1,
        )

        return trainx, trainy

    def get_trainingParam(self):
        useLibrary = library.Nanocylinders_U650nm_H800nm()
        return useLibrary.params

    def convert_output_complex(self, y_model, reshapeToSize=None):
        phasex = tf.math.atan2(y_model[:, 1], y_model[:, 0])
        transx = y_model[:, 2]

        # allow an option to reshape to a grid size (excluding data stack width)
        if reshapeToSize is not None:
            phasex = tf.reshape(phasex, reshapeToSize)
            transx = tf.reshape(transx, reshapeToSize)

        return transx, phasex


## USABLE MLP MODELS
class MLP_Nanocylinders_Dense256_U180_H600(MLP_Nanocylinders_U180_H600):
    def __init__(self, dtype=tf.float64):
        super(MLP_Nanocylinders_Dense256_U180_H600, self).__init__(dtype)

        self.set_model_name("MLP_Nanocylinders_Dense256_U180_H600")
        self.set_modelSavePath("trained_MLP_models/MLP_Nanocylinders_Dense256_U180_H600/")

        # Define a new architecture
        self._arch = [
            tf.keras.layers.Dense(
                256,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(
                256,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(3, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]


class MLP_Nanocylinders_Dense128_U180_H600(MLP_Nanocylinders_U180_H600):
    def __init__(self, dtype=tf.float64):
        super(MLP_Nanocylinders_Dense128_U180_H600, self).__init__(dtype)

        self.set_model_name("MLP_Nanocylinders_Dense128_U180_H600")
        self.set_modelSavePath("trained_MLP_models/MLP_Nanocylinders_Dense128_U180_H600/")

        # Define a new architecture
        self._arch = [
            tf.keras.layers.Dense(
                128,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(
                128,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(3, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]


class MLP_Nanocylinders_Dense64_U180_H600(MLP_Nanocylinders_U180_H600):
    def __init__(self, dtype=tf.float64):
        super(MLP_Nanocylinders_Dense64_U180_H600, self).__init__(dtype)

        self.set_model_name("MLP_Nanocylinders_Dense64_U180_H600")
        self.set_modelSavePath("trained_MLP_models/MLP_Nanocylinders_Dense64_U180_H600/")

        # Define a new architecture
        self._arch = [
            tf.keras.layers.Dense(
                64,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(
                64,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(3, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]


##
class MLP_Nanofins_Dense1024_U350_H600(MLP_Nanofins_U350_H600):
    def __init__(self, dtype=tf.float64):
        super(MLP_Nanofins_Dense1024_U350_H600, self).__init__(dtype)
        self.set_model_name("MLP_Nanofins_Dense1024_U350_H600")
        self.set_modelSavePath("trained_MLP_models/MLP_Nanofins_Dense1024_U350_H600/")

        # Define a new architecture
        self._arch = [
            tf.keras.layers.Dense(
                1024,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(
                1024,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(6, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]


class MLP_Nanofins_Dense512_U350_H600(MLP_Nanofins_U350_H600):
    def __init__(self, dtype=tf.float64):
        super(MLP_Nanofins_Dense512_U350_H600, self).__init__(dtype)

        self.set_model_name("MLP_Nanofins_Dense512_U350_H600")
        self.set_modelSavePath("trained_MLP_models/MLP_Nanofins_Dense512_U350_H600/")

        # Define a new architecture
        self._arch = [
            tf.keras.layers.Dense(
                512,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(
                512,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(6, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]


class MLP_Nanofins_Dense256_U350_H600(MLP_Nanofins_U350_H600):
    def __init__(self, dtype=tf.float64):
        super(MLP_Nanofins_Dense256_U350_H600, self).__init__(dtype)

        self.set_model_name("MLP_Nanofins_Dense256_U350_H600")
        self.set_modelSavePath("trained_MLP_models/MLP_Nanofins_Dense256_U350_H600/")

        # Define a new architecture
        self._arch = [
            tf.keras.layers.Dense(
                256,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(
                256,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(6, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]


class MLP_Nanofins_Dense64_U350_H600(MLP_Nanofins_U350_H600):
    def __init__(self, dtype=tf.float64):
        super(MLP_Nanofins_Dense64_U350_H600, self).__init__(dtype)

        self.set_model_name("MLP_Nanofins_Dense64_U350_H600")
        self.set_modelSavePath("trained_MLP_models/MLP_Nanofins_Dense64_U350_H600/")

        # Define a new architecture
        self._arch = [
            tf.keras.layers.Dense(
                64,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(
                64,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(6, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]


### TESTING
class MLP_Nanofins_GFFDense256_256s1p0_U350_H600(MLP_Nanofins_U350_H600):
    def __init__(self, emb_dim=256, gauss_scale=1.0, dtype=tf.float64):
        super().__init__(dtype)

        model_name = "MLP_Nanofins_GFFDense256_256s1p0_U350_H600"
        self.set_model_name(model_name)
        self.set_modelSavePath("trained_MLP_models/" + model_name)

        # Define a new architecture
        self._arch = [
            GFF_Projection_layer(emb_dim, gauss_scale),
            tf.keras.layers.Dense(
                256,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(
                256,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(6, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]


class MLP_Nanofins_GFFDense256_256s0p5_U350_H600(MLP_Nanofins_U350_H600):
    def __init__(self, emb_dim=256, gauss_scale=0.5, dtype=tf.float64):
        super().__init__(dtype)

        model_name = "MLP_Nanofins_GFFDense256_256s0p5_U350_H600"
        self.set_model_name(model_name)
        self.set_modelSavePath("trained_MLP_models/" + model_name)

        # Define a new architecture
        self._arch = [
            GFF_Projection_layer(emb_dim, gauss_scale),
            tf.keras.layers.Dense(
                256,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(
                256,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(6, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]


class MLP_Nanofins_GFF2Dense_256_U350_H600(MLP_Nanofins_U350_H600):
    def __init__(self, emb_dim=256, gauss_scale=10.0, dtype=tf.float64):
        super().__init__(dtype)

        d_str = str(int(emb_dim))
        gscale_str = str(gauss_scale).replace(".", "p")
        model_name = "MLP_Nanofins_GFF" + d_str + "_" + gscale_str + "Dense512x2_U350_H600"
        self.set_model_name(model_name)
        self.set_modelSavePath("trained_MLP_models/" + model_name)

        # Define a new architecture
        self._arch = [
            GFF_Projection_layer(emb_dim, gauss_scale),
            tf.keras.layers.Dense(
                256,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(
                256,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(6, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]


# class MLP_Nanofins_2GDense1024_U350_H600(MLP_Nanofins_U350_H600):
#     def __init__(self, a=0.5, dtype=tf.float64):
#         super(MLP_Nanofins_2GDense1024_U350_H600, self).__init__(dtype)
#         # a is a hyper-parameter that controlls the gaussian activation function

#         a_str = str(a).replace(".", "_")
#         self.set_model_name("MLP_Nanofins_2G" + a_str + "Dense1024_U350_H600")
#         self.set_modelSavePath("trained_MLP_models/MLP_Nanofins_2G" + a_str + "Dense1024_U350_H600/")

#         # Define a new architecture
#         def activation_func(x):
#             return gaussian_activation(x, a)

#         self._arch = [
#             tf.keras.layers.Dense(
#                 1024,
#                 activation=activation_func,
#                 kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
#             ),
#             tf.keras.layers.Dense(
#                 1024,
#                 activation=activation_func,
#                 kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
#             ),
#             tf.keras.layers.Dense(6, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
#         ]

# class MLP_Nanofins_GFF2Dense_512_U350_H600_v2(MLP_Nanofins_U350_H600):
#     def __init__(self, emb_dim=256, gauss_scale=10.0, dtype=tf.float64):
#         super().__init__(dtype)

#         d_str = str(int(emb_dim))
#         gscale_str = str(gauss_scale).replace(".", "p")
#         model_name = "MLP_Nanofins_GFF" + d_str + "_" + gscale_str + "Dense512x2_U350_H600_v2"
#         self.set_model_name(model_name)
#         self.set_modelSavePath("trained_MLP_models/" + model_name)

#         # Define a new architecture
#         self._arch = [
#             GFF_Projection_layer_trained(emb_dim, gauss_scale),
#             tf.keras.layers.Dense(
#                 512,
#                 activation=leakyrelu100,
#                 kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
#             ),
#             tf.keras.layers.Dense(
#                 512,
#                 activation=leakyrelu100,
#                 kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
#             ),
#             tf.keras.layers.Dense(6, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
#         ]


# class MLP_Nanofins_GFF2Dense_1024_U350_H600_v2(MLP_Nanofins_U350_H600):
#     def __init__(self, emb_dim=256, gauss_scale=10.0, dtype=tf.float64):
#         super().__init__(dtype)

#         d_str = str(int(emb_dim))
#         gscale_str = str(gauss_scale).replace(".", "p")
#         model_name = "MLP_Nanofins_GFF" + d_str + "_" + gscale_str + "Dense1024x2_U350_H600_v2"
#         self.set_model_name(model_name)
#         self.set_modelSavePath("trained_MLP_models/" + model_name)

#         # Define a new architecture
#         self._arch = [
#             GFF_Projection_layer_trained(emb_dim, gauss_scale),
#             tf.keras.layers.Dense(
#                 1024,
#                 activation=leakyrelu100,
#                 kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
#             ),
#             tf.keras.layers.Dense(
#                 1024,
#                 activation=leakyrelu100,
#                 kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
#             ),
#             tf.keras.layers.Dense(6, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
#         ]
