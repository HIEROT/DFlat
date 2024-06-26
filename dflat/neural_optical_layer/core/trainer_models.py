import os

import tensorflow as tf
import numpy as np
import pickle

import dflat.neural_optical_layer.core.models_DNN as MLP_models
import dflat.neural_optical_layer.core.models_eRBF as eRBF_models
from sklearn.model_selection import train_test_split


def save_test_evaluation_data(model, xtest, ytest, savestring):
    # Get model errors on the test set
    modelOut = model.predict(xtest, verbose=0)
    trans_mlp, phase_mlp = model.convert_output_complex(modelOut)
    trans_test, phase_test = model.convert_output_complex(ytest)
    phase_mlp = phase_mlp.numpy()
    phase_test = phase_test.numpy()

    # Compute errors
    trans_error = trans_test - trans_mlp
    phase_error = phase_test - phase_mlp
    complex_error = np.abs(trans_mlp * np.exp(1j * phase_mlp) - trans_test * np.exp(1j * phase_test))

    # compute relative errors
    rel_trans = trans_error / trans_test
    rel_phase = phase_error / phase_test
    rel_complex = complex_error / np.abs(trans_test * np.exp(1j * phase_test))

    # Est FLOPs per evaluation
    est_FLOPs = model.profile_FLOPs()

    saveTo = model._modelSavePath + savestring + ".pickle"
    data = {
        "trans_error": trans_error,
        "rel_trans": rel_trans,
        "phase_error": phase_error,
        "rel_phase": rel_phase,
        "complex_error": complex_error,
        "rel_complex": rel_complex,
        "est_FLOPs": est_FLOPs,
    }
    with open(saveTo, "wb") as handle:
        pickle.dump(data, handle)

    return complex_error


def run_training_neural_model(model, epochs, miniEpoch=1000, batch_size=None, lr=1e-4, verbose=False, train=True):
    ### Define the model to train and associated parameters
    model.customLoadCheckpoint()

    ### Get training and testing data:
    inputData, outputData = model.returnLibraryAsTrainingData()
    xtrain, xtest, ytrain, ytest = train_test_split(inputData, outputData, test_size=0.15, random_state=13, shuffle=True)
    
    ### If no batch size is specified then use the full training dataset size
    if batch_size == None:
        batch_size = xtrain.shape[0]

    ### Call once then print summary
    model(xtrain[0:1, :])
    model.summary()

    if train:
        splitNumberSessions = np.ceil(epochs / miniEpoch).astype("int")
        optimizer = tf.keras.optimizers.Adam(lr)
        model.compile(optimizer, loss=tf.keras.losses.mean_squared_error)
        device = "GPU:0"
      
        for sessCounter in range(splitNumberSessions):
            with tf.device(device):
                trackhistory = model.fit(xtrain, ytrain, batch_size=batch_size, epochs=miniEpoch, verbose=verbose, validation_split=0.05)
            model.customSaveCheckpoint(trackhistory)

    # After Training, evaluate the performance by histogram of errors on the test set
    test_complex_error = save_test_evaluation_data(model, xtest, ytest, "training_testDataError")
    save_test_evaluation_data(model, xtrain, ytrain, "training_trainDataError")
    print("MAE Test Set: ", np.mean(np.abs(test_complex_error)))
    
    return


def train_caller(train=True, verb=True):

    # run_training_neural_model(
    #         model= MLP_models.MLP_Nanofins_Dense1024_U350_H600(),
    #         epochs=60000,
    #         miniEpoch=1000,
    #         batch_size=500000,
    #         lr=1e-4,
    #         train=train,
    #         verbose=verb
    #     )

    # run_training_neural_model(
    #         model= MLP_models.MLP_Nanofins_Dense512_U350_H600(),
    #         epochs=60000,
    #         miniEpoch=1000,
    #         batch_size=500000,
    #         lr=1e-4,
    #         train=train,
    #         verbose=verb
    #     )
    
    # run_training_neural_model(
    #         model= MLP_models.MLP_Nanofins_Dense256_U350_H600(),
    #         epochs=60000,
    #         miniEpoch=1000,
    #         batch_size=500000,
    #         lr=1e-4,
    #         train=train,
    #         verbose=verb
    #     )
    
    run_training_neural_model(
        model=MLP_models.MLP_Nanocylinders_Dense128_U650_H800(),
        epochs=100000,
        miniEpoch=1000,
        batch_size=500000,
        lr=1e-4,
        train=train,
        verbose=verb
    )
    
    # Convenient caller to train many models sequentially with one run call 
    # for a in [0.01, 0.05, 0.1, 0.2, 0.5]:
    #     use_model = MLP_models.MLP_Nanofins_2GDense1024_U350_H600(a)
    #     run_training_neural_model(
    #         model=use_model,
    #         epochs=10000,
    #         miniEpoch=1000,
    #         batch_size=None,
    #         lr=1e-3,
    #         train=train,
    #         verbose=verb,
    #     )

    # run_training_neural_model(
    #         model= MLP_models.MLP_Nanofins_Dense1024_U350_H600(),
    #         epochs=2000,
    #         miniEpoch=100,
    #         batch_size=250000,
    #         lr=1e-3,
    #         train=train,
    #         verbose=verb,
    #     )
    
    
    # run_training_neural_model(
    #     model= MLP_models.MLP_Nanofins_GFF2Dense_1024_U350_H600_v2(emb_dim=256, gauss_scale=10.0),
    #     epochs=2000,
    #     miniEpoch=100,
    #     batch_size=250000,
    #     lr=1e-3,
    #     train=train,
    #     verbose=verb,
    # )

    # run_training_neural_model(
    #     model= MLP_models.MLP_Nanofins_GFF2Dense_1024_U350_H600_v2(emb_dim=256, gauss_scale=10.0),
    #     epochs=2000,
    #     miniEpoch=100,
    #     batch_size=250000,
    #     lr=1e-3,
    #     train=train,
    #     verbose=verb,
    # )

    # run_training_neural_model(
    #     model= MLP_models.MLP_Nanofins_GFF2Dense_512_U350_H600_v2(emb_dim=256, gauss_scale=100.0),
    #     epochs=1000,
    #     miniEpoch=100,
    #     batch_size=None,
    #     lr=1e-3,
    #     train=train,
    #     verbose=verb,
    # )


    return


if __name__ == "__main__":
    # for key in os.environ:
    #     print(key, os.environ[key])
    train_caller(train=True, verb=True)

   