# #!/usr/bin/env python3

# """
# Module: predict.py
# Authors: Christian Bergler
# Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
# Last Access: 06.02.2021
# """
# import os
# import argparse

# import torch
# import torch.nn as nn

# from os import listdir
# from os.path import isfile, join

# from models.unet_model import UNet
# from data.audiodataset import DefaultSpecDatasetOps, StridedAudioDataset, SingleAudioFolder


# import data.transforms as T
# import data.signal as signal

# import scipy.io.wavfile
# from math import ceil, floor
# from utils.logging import Logger
# from collections import OrderedDict

# parser = argparse.ArgumentParser()

# parser.add_argument(
#     "-d",
#     "--debug",
#     dest="debug",
#     action="store_true",
#     help="Print additional training and model information.",
# )

# parser.add_argument(
#     "--model_path",
#     type=str,
#     default=None,
#     help="Path to a model.",
# )

# parser.add_argument(
#     "--checkpoint_path",
#     type=str,
#     default=None,
#     help="Path to a checkpoint. "
#     "If provided the checkpoint will be used instead of the model.",
# )

# parser.add_argument(
#     "--log_dir", type=str, default=None, help="The directory to store the logs."
# )

# parser.add_argument(
#     "--output_dir", type=str, default=None, help="The directory to store the output."
# )

# parser.add_argument(
#     "--sequence_len", type=float, default=2, help="Sequence length in [s]."
# )


# parser.add_argument(
#     "--batch_size", type=int, default=1, help="The number of images per batch."
# )

# parser.add_argument(
#     "--num_workers", type=int, default=4, help="Number of workers used in data-loading"
# )

# parser.add_argument(
#     "--no_cuda",
#     dest="cuda",
#     action="store_false",
#     help="Do not use cuda to train model.",
# )

# parser.add_argument(
#     "--visualize",
#     dest="visualize",
#     action="store_true",
#     help="Additional visualization of the noisy vs. denoised spectrogram",
# )

# parser.add_argument(
#     "--jit_load",
#     dest="jit_load",
#     action="store_true",
#     help="Load model via torch jit (otherwise via torch load).",
# )

# parser.add_argument(
#     "--input_file",
#     type=str,
#     default=None,
#     help="Input file could either be a directory with multiple audio files or just one single audio file"
# )
# parser.add_argument(
#     "--minimal_frequency",
#     type=int,
#     default=200,
#     help="None"
# )
# parser.add_argument(
#     "--maximal_frequency",
#     type=int,
#     default=18000,
#     help="None"
# )
# parser.add_argument(
#     "--freq_compression",
#     type=str,
#     choices=["linear", "mel", "mfcc"],
#     default="linear",
#     help="None"
# )
# parser.add_argument(
#     "--sr",
#     type=int,
#     default=44100,
#     help="None"
# )
# parser.add_argument(
#     "--fft_size",
#     type=int,
#     default=4096,
#     help="None"
# )
# parser.add_argument(
#     "--hop_length",
#     type=int,
#     default=441,
#     help="None"
# )
# parser.add_argument(
#     "--n_freq_bins",
#     type=int,
#     default=256,
#     help="None"
# )
# parser.add_argument(
#     "--window_type",
#     type=str,
#     choices=['hann', 'blackman', 'hamming', 'kaiser'],
#     default="hann",
#     help="None"
# )

# ARGS = parser.parse_args()

# log = Logger("PREDICT", ARGS.debug, ARGS.log_dir)

# """
# Main function to compute prediction by using a trained model together with the given input
# """


# def predictions(ARGS, model, sp, sr, fmin, fmax, n_fft, hop_length, dataset, concatenate, total_audio):
#     data_loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=ARGS.batch_size,
#         # num_workers=ARGS.num_workers,
#         pin_memory=True,
#     )

#     t_decompr_f = T.Decompress(f_min=fmin, f_max=fmax, n_fft=n_fft, sr=sr)

#     with torch.no_grad():
#         for i, input in enumerate(data_loader):
#             sample_spec_orig, input, spec_cmplx, filename = input

#             print("current file in process, " + str(i) +
#                   "-iterations: " + str(filename[0]))

#             if torch.cuda.is_available() and ARGS.cuda:
#                 input = input.cuda()

#             denoised_output = model(input)

#             decompressed_net_out = t_decompr_f(denoised_output)

#             spec_cmplx = spec_cmplx.squeeze(dim=0)

#             decompressed_net_out = decompressed_net_out.unsqueeze(dim=-1)

#             audio_spec = decompressed_net_out * spec_cmplx

#             if (ARGS.window_type == 'hann'):
#                 window = torch.hann_window(n_fft)
#             elif (ARGS.window_type == 'blackman'):
#                 window = torch.blackman_window(n_fft)
#             elif (ARGS.window_type == 'hamming'):
#                 window = torch.hamming_window(n_fft)
#             else:
#                 window = torch.kaiser_window(n_fft)

#             audio_spec = audio_spec.squeeze(dim=0).transpose(0, 1)

#             detected_spec_cmplx = spec_cmplx.squeeze(dim=0).transpose(0, 1)
#             print("FileName", filename[0].split(
#                 "\\")[-1].split(".")[0])
#             if sp is not None:
#                 sp.plot_spectrogram(spectrogram=input.squeeze(dim=0), title="",
#                                     output_filepath=ARGS.output_dir + "\\net_input_spec_" +
#                                     str(i) + "_" +
#                                     filename[0].split(
#                                         "\\")[-1].split(".")[0]+".pdf",
#                                     sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax, show=False, ax_title="spectrogram")

#                 sp.plot_spectrogram(spectrogram=denoised_output.squeeze(dim=0), title="",
#                                     output_filepath=ARGS.output_dir + "\\net_out_spec_" +
#                                     str(i) + "_" +
#                                     filename[0].split(
#                                         "\\")[-1].split(".")[0]+".pdf",
#                                     sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax, show=False, ax_title="spectrogram")

#             if concatenate:
#                 audio_out_denoised = torch.istft(
#                     audio_spec, n_fft, hop_length=hop_length, onesided=True, center=True, window=window)
#                 if total_audio is None:
#                     total_audio = audio_out_denoised
#                 else:
#                     total_audio = torch.cat(
#                         (total_audio, audio_out_denoised), 0)
#             else:
#                 total_audio = torch.istft(
#                     audio_spec, n_fft, hop_length=hop_length, onesided=True, center=True, window=window)
#                 scipy.io.wavfile.write(ARGS.output_dir + "\\denoised_" + str(
#                     i) + "_" + filename[0].split("\\")[-1].split(".")[0]+".wav", sr, total_audio.numpy().T)

#         # before or after writing intensity scaling to chose dB value
#         scipy.io.wavfile.write(ARGS.output_dir+"\\denoised_" + str(i) + "_" +
#                                filename[0].split("\\")[-1].split(".")[0]+".wav", sr, total_audio.numpy().T)


# if __name__ == "__main__":

#     if ARGS.checkpoint_path is not None:
#         log.info(
#             "Restoring checkpoint from {} instead of using a model file.".format(
#                 ARGS.checkpoint_path
#             )
#         )
#         checkpoint = torch.load(ARGS.checkpoint_path)
#         model = UNet(1, 1, bilinear=False)
#         model.load_state_dict(checkpoint["modelState"])
#         log.warning(
#             "Using default preprocessing options. Provide Model file if they are changed"
#         )
#         dataOpts = DefaultSpecDatasetOps
#     else:
#         if ARGS.jit_load:
#             extra_files = {}
#             extra_files['dataOpts'] = ''
#             model = torch.jit.load(ARGS.model_path, _extra_files=extra_files)
#             unetState = model.state_dict()
#             dataOpts = eval(extra_files['dataOpts'])
#             log.debug("Model successfully load via torch jit: " +
#                       str(ARGS.model_path))
#         else:
#             model_dict = torch.load(ARGS.model_path)
#             model = UNet(1, 1, bilinear=False)
#             model.load_state_dict(model_dict["unetState"])
#             model = nn.Sequential(
#                 OrderedDict([("denoiser", model)])
#             )
#             dataOpts = model_dict["dataOpts"]
#             log.debug("Model successfully load via torch load: " +
#                       str(ARGS.model_path))

#     log.info(model)

#     if ARGS.visualize:
#         sp = signal.signal_proc()
#     else:
#         sp = None

#     if torch.cuda.is_available() and ARGS.cuda:
#         model = model.cuda()

#     model.eval()

#     # sr = dataOpts['sr']
#     # fmin = dataOpts["fmin"]

#     # fmax = dataOpts["fmax"]

#     # n_fft = dataOpts["n_fft"]

#     # hop_length = dataOpts["hop_length"]
#     # n_freq_bins = dataOpts["n_freq_bins"]

#     sr = int(ARGS.sr)
#     fmin = int(ARGS.minimal_frequency)

#     fmax = int(ARGS.maximal_frequency)

#     n_fft = int(ARGS.fft_size)

#     hop_length = int(ARGS.hop_length)
#     n_freq_bins = int(ARGS.n_freq_bins)
#     freq_compression = ARGS.freq_compression

#     # log.debug("dataOpts: " + str(dataOpts))

#     sequence_len = int(ceil(ARGS.sequence_len * sr))

#     hop = sequence_len

#     input_file = ARGS.input_file

#     if os.path.isdir(input_file):

#         log.debug("Init Single Folder Audio Dataset - Predicting Files")
#         log.debug("Audio folder to process: "+str(input_file))
#         audio_files = [f for f in listdir(
#             input_file) if isfile(join(input_file, f))]

#         dataset = SingleAudioFolder(
#             file_names=audio_files,
#             working_dir=input_file,
#             sr=sr,
#             n_fft=n_fft,
#             hop_length=hop_length,
#             n_freq_bins=n_freq_bins,
#             freq_compression=freq_compression,
#             f_min=fmin,
#             f_max=fmax,
#         )

#         log.info("number of files to predict={}".format(len(audio_files)))
#         log.info(
#             "files will be entirely denoised without subsampling parts and/or padding")
#         concatenate = False
#         total_audio = True
#         predictions(ARGS, model, sp, sr, fmin, fmax, n_fft,
#                     hop_length, dataset, concatenate, total_audio)
#     elif os.path.isfile(input_file):

#         log.debug("Init Strided Audio Dataset - Predicting Files")
#         log.debug("Audio file to process: "+str(input_file))

#         dataset = StridedAudioDataset(
#             input_file.strip(),
#             sequence_len=sequence_len,
#             hop=hop,
#             sr=sr,
#             fft_size=n_fft,
#             fft_hop=hop_length,
#             n_freq_bins=n_freq_bins,
#             freq_compression=freq_compression,
#             f_min=fmin,
#             f_max=fmax,
#         )

#         log.info("size of the file(samples)={}".format(dataset.n_frames))
#         log.info("size of hop(samples)={}".format(hop))
#         stop = int(max(floor(dataset.n_frames / hop), 1))
#         log.info("stop time={}".format(stop))
#         concatenate = True
#         total_audio = None
#         predictions(ARGS, model, sp, sr, fmin, fmax, n_fft,
#                     hop_length, dataset, concatenate, total_audio)
#     else:
#         raise Exception("Not a valid data format - neither folder nor file")

#     log.debug("Finished proccessing")

#     log.close()

#!/usr/bin/env python3

"""
Module: predict.py
Authors: Christian Bergler
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 06.02.2021
"""

import os
import argparse

import torch
import torch.nn as nn
import re
from os import listdir
from os.path import isfile, join

# from models.unet_model import UNet
from .models.unet_model import UNet
from .data.audiodataset import DefaultSpecDatasetOps, StridedAudioDataset, SingleAudioFolder

from .data import transforms as T
from .data import signal

import scipy.io.wavfile
from math import ceil, floor
from .utils.logging import Logger
from collections import OrderedDict
from pathlib import Path


def predict(debug=False, model_path=os.path.join(os.getcwd(),os.path.join('ORCA_CLEAN','orca-clean.pk')), checkpoint_path=None,
            log_dir=None, output_dir=None, sequence_len=2, batch_size=1,
            num_workers=4, cuda=False, visualize=True, jit_load=False, input_file=None, min_frequency=200, max_frequency=18000, freq_compression="linear", sr=44100,
            fft_size=4096, hop_length=441, n_freq_bins=256, window_type="hann"):

    # choices=["linear", "mel", "mfcc"]
    # choices=['hann', 'blackman', 'hamming', 'kaiser'],

    # log = Logger("PREDICT", debug,  log_dir)
    # print("Hola")
    """
    Main function to compute prediction by using a trained model together with the given input
    """

    def predictions(model, sp, sr, fmin, fmax, n_fft, hop_length, window_type, dataset, concatenate, total_audio):
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            # num_workers= num_workers,
            pin_memory=True,
        )

        t_decompr_f = T.Decompress(f_min=fmin, f_max=fmax, n_fft=n_fft, sr=sr)

        with torch.no_grad():
            for i, input in enumerate(data_loader):
                sample_spec_orig, input, spec_cmplx, filename = input

                # print("current file in process, " + str(i) +
                #       "-iterations: " + str(filename[0]))

                if torch.cuda.is_available() and cuda:
                    input = input.cuda()

                denoised_output = model(input)

                decompressed_net_out = t_decompr_f(denoised_output)

                spec_cmplx = spec_cmplx.squeeze(dim=0)

                decompressed_net_out = decompressed_net_out.unsqueeze(dim=-1)

                audio_spec = decompressed_net_out * spec_cmplx

                if (window_type == 'hann'):
                    window = torch.hann_window(n_fft)
                elif (window_type == 'blackman'):
                    window = torch.blackman_window(n_fft)
                elif (window_type == 'hamming'):
                    window = torch.hamming_window(n_fft)
                else:
                    window = torch.kaiser_window(n_fft)

                audio_spec = audio_spec.squeeze(dim=0).transpose(0, 1)

                detected_spec_cmplx = spec_cmplx.squeeze(dim=0).transpose(0, 1)
                               
                # print("FileName", filename[0].split(
                #     "/")[-1].split(".")[0])
                if sp is not None:
                    
                    print("Filename: ",os.path.basename(filename[0]).split('.')[0])
                    input_path_plot = "net_input_spec_" + str(i) + "_" + os.path.basename(filename[0]).split('.')[0] + ".pdf"
                    output_path_plot = "net_out_spec_" + str(i) + "_" + os.path.basename(filename[0]).split('.')[0] + ".pdf"
                    sp.plot_spectrogram(spectrogram=input.squeeze(dim=0), title="",
                                        output_filepath= os.path.join(output_dir, input_path_plot),
                                        sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax, show=False, ax_title="spectrogram")

                    sp.plot_spectrogram(spectrogram=denoised_output.squeeze(dim=0), title="",
                                        output_filepath=os.path.join(output_dir,output_path_plot),
                                        sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax, show=False, ax_title="spectrogram")

                if concatenate:
                    audio_out_denoised = torch.istft(
                        audio_spec, n_fft, hop_length=hop_length, onesided=True, center=True, window=window)
                    if total_audio is None:
                        total_audio = audio_out_denoised
                    else:
                        total_audio = torch.cat(
                            (total_audio, audio_out_denoised), 0)
                else:
                    total_audio = torch.istft(
                        audio_spec, n_fft, hop_length=hop_length, onesided=True, center=True, window=window)
                    os.path.basename(filename[0]).split('.')[0]
                    scipy.io.wavfile.write(os.path.join(output_dir , "denoised_" + str(i) + "_" + os.path.basename(filename[0]).split('.')[0]+".wav"), 
                                           sr, 
                                           total_audio.numpy().T)
                    # scipy.io.wavfile.write(output_dir + "/denoised_" + str(i) + "_" + filename[0].split("/")[-1].split(".")[0]+".wav", sr, total_audio.numpy().T)

            # before or after writing intensity scaling to chose dB value
            #scipy.io.wavfile.write(output_dir+"/denoised_" + str(i) + "_" +filename[0].split("/")[-1].split(".")[0]+".wav", sr, total_audio.numpy().T)
            scipy.io.wavfile.write(os.path.join(output_dir , "denoised_" + str(i) + "_" + os.path.basename(filename[0]).split('.')[0]+".wav"), 
                                   sr, 
                                   total_audio.numpy().T)
    if checkpoint_path is not None:
        # log.info(
        #     "Restoring checkpoint from {} instead of using a model file.".format(
        #         checkpoint_path
        #     )
        # )
        checkpoint = torch.load(checkpoint_path)
        model = UNet(1, 1, bilinear=False)
        model.load_state_dict(checkpoint["modelState"])
        # log.warning(
        #     "Using default preprocessing options. Provide Model file if they are changed"
        # )
        dataOpts = DefaultSpecDatasetOps
    else:
        if jit_load:
            extra_files = {}
            extra_files['dataOpts'] = ''
            model = torch.jit.load(model_path, _extra_files=extra_files)
            unetState = model.state_dict()
            dataOpts = eval(extra_files['dataOpts'])
            # log.debug("Model successfully load via torch jit: " +
            #           str(model_path))
        else:
            model_dict = torch.load(model_path)
            model = UNet(1, 1, bilinear=False)
            model.load_state_dict(model_dict["unetState"])
            model = nn.Sequential(
                OrderedDict([("denoiser", model)])
            )
            dataOpts = model_dict["dataOpts"]
            # log.debug("Model successfully load via torch load: " +
            #           str(model_path))

    # log.info(model)

    if visualize:
        sp = signal.signal_proc()
    else:
        sp = None

    if torch.cuda.is_available() and cuda:
        model = model.cuda()

    model.eval()

    # sr = dataOpts['sr']
    # fmin = dataOpts["fmin"]

    # fmax = dataOpts["fmax"]

    # n_fft = dataOpts["n_fft"]

    # hop_length = dataOpts["hop_length"]
    # n_freq_bins = dataOpts["n_freq_bins"]

    sr = int(sr)
    fmin = int(min_frequency)

    fmax = int(max_frequency)

    n_fft = int(fft_size)

    hop_length = int(hop_length)
    n_freq_bins = int(n_freq_bins)
    freq_compression = freq_compression

    # log.debug("dataOpts: " + str(dataOpts))

    sequence_len = int(ceil(sequence_len * sr))

    hop = sequence_len

    input_file = input_file

    if os.path.isdir(input_file):

        # log.debug("Init Single Folder Audio Dataset - Predicting Files")
        # log.debug("Audio folder to process: "+str(input_file))
        audio_files = [f for f in listdir(
            input_file) if isfile(join(input_file, f))]

        dataset = SingleAudioFolder(
            file_names=audio_files,
            working_dir=input_file,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_freq_bins=n_freq_bins,
            freq_compression=freq_compression,
            f_min=fmin,
            f_max=fmax,
        )

        # log.info("number of files to predict={}".format(len(audio_files)))
        # log.info(
        #     "files will be entirely denoised without subsampling parts and/or padding")
        concatenate = False
        total_audio = True
        predictions(model, sp, sr, fmin, fmax, n_fft,
                    hop_length, window_type, dataset, concatenate, total_audio)
    elif os.path.isfile(input_file):

        # log.debug("Init Strided Audio Dataset - Predicting Files")
        # log.debug("Audio file to process: "+str(input_file))

        dataset = StridedAudioDataset(
            input_file.strip(),
            sequence_len=sequence_len,
            hop=hop,
            sr=sr,
            fft_size=n_fft,
            fft_hop=hop_length,
            n_freq_bins=n_freq_bins,
            freq_compression=freq_compression,
            f_min=fmin,
            f_max=fmax,
        )

        # log.info("size of the file(samples)={}".format(dataset.n_frames))
        # log.info("size of hop(samples)={}".format(hop))
        stop = int(max(floor(dataset.n_frames / hop), 1))
        # log.info("stop time={}".format(stop))
        concatenate = True
        total_audio = None
        predictions(model, sp, sr, fmin, fmax, n_fft,
                    hop_length, window_type, dataset, concatenate, total_audio)
    else:
        raise Exception("Not a valid data format - neither folder nor file")

    # log.debug("Finished proccessing")

    # log.close()
