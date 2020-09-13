import torch
import argparse
import numpy as np
import pandas as pd

from torch.nn import functional as F

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--n_samples', type=int, required=True, help='number of samples in one window')
    parser.add_argument('--distance', type=int, required=True, help='distance between successive two windows')
    parser.add_argument('--n_classes', type=int, required=False, default=12, help='number of classes: if 5, walking forward, left, right, running forward, standing')
    parser.add_argument('--dft', type=bool, required=False, default=False, help='if true, doing DFT to images')

    args = parser.parse_args()

    if args.dft:
        print('===> DFT ON')

    # Jogging Forward
    jog = pd.DataFrame(columns=['gyrX', 'gyrY', 'gyrZ', 'eulR', 'eulP', 'eulY', 'label', 'trial'])

    temp = pd.read_csv('./dataset_0818/jog_forward.csv')
    df = temp[['gyrX', 'gyrY', 'gyrZ', 'eulR', 'eulP', 'eulY', 'trial']]
    df.loc[:, 'label'] = 1
    jog = pd.concat([jog, df], axis=0)
        
    jog[['gyrX', 'gyrY', 'gyrZ']] *= 57.2985

    print('Size of Jogging Forward:', jog.shape[0])

    # Jogging Turn
    jog_turn = pd.DataFrame(columns=['gyrX', 'gyrY', 'gyrZ', 'eulR', 'eulP', 'eulY', 'label', 'trial'])

    temp = pd.read_csv('./dataset_0818/jog_turn_valid.csv')
    df = temp[['gyrX', 'gyrY', 'gyrZ', 'eulR', 'eulP', 'eulY', 'trial']]
    df.loc[:, 'label'] = 2
    jog_turn = pd.concat([jog_turn, df], axis=0)

    jog_turn[['gyrX', 'gyrY', 'gyrZ']] *= 57.2985

    print('Size of Jogging Turn:', jog_turn.shape[0])

    # Walking Forward
    walk = pd.DataFrame(columns=['gyrX', 'gyrY', 'gyrZ', 'eulR', 'eulP', 'eulY', 'label', 'trial'])

    temp = pd.read_csv('./dataset_0818/walk_forward.csv')
    df = temp[['gyrX', 'gyrY', 'gyrZ', 'eulR', 'eulP', 'eulY', 'trial']]
    df.loc[:, 'label'] = 3
    walk = pd.concat([walk, df], axis=0)
        
    walk[['gyrX', 'gyrY', 'gyrZ']] *= 57.2985

    print('Size of Walking Forward:', walk.shape[0])

    # Walking Turn
    walk_turn = pd.DataFrame(columns=['gyrX', 'gyrY', 'gyrZ', 'eulR', 'eulP', 'eulY', 'label', 'trial'])

    temp = pd.read_csv('./dataset_0818/walk_turn_valid.csv')
    df = temp[['gyrX', 'gyrY', 'gyrZ', 'eulR', 'eulP', 'eulY', 'trial']]
    df.loc[:, 'label'] = 4
    walk_turn = pd.concat([walk_turn, df], axis=0)
        
    walk_turn[['gyrX', 'gyrY', 'gyrZ']] *= 57.2985

    print('Size of Walking Turn:', walk_turn.shape[0])

    # Run
    run = pd.DataFrame(columns=['gyrX', 'gyrY', 'gyrZ', 'eulR', 'eulP', 'eulY', 'label', 'trial'])

    temp = pd.read_csv('./dataset_0818/run.csv')
    df = temp[['gyrX', 'gyrY', 'gyrZ', 'eulR', 'eulP', 'eulY', 'trial']]
    df.loc[:, 'label'] = 5
    run = pd.concat([run, df], axis=0)
        
    run[['gyrX', 'gyrY', 'gyrZ']] *= 57.2985

    print('Size of Running:', run.shape[0])

    # Stationary
    stationary = pd.DataFrame(columns=['gyrX', 'gyrY', 'gyrZ', 'eulR', 'eulP', 'eulY', 'label', 'trial'])

    temp = pd.read_csv('./dataset_0818/stationary.csv')
    df = temp[['gyrX', 'gyrY', 'gyrZ', 'eulR', 'eulP', 'eulY', 'trial']]
    df.loc[:, 'label'] = 6
    stationary = pd.concat([stationary, df], axis=0)
        
    stationary[['gyrX', 'gyrY', 'gyrZ']] *= 57.2985

    print('Size of Stationary:', stationary.shape[0])

    # concatenate all data
    all_data = pd.concat([jog, jog_turn, walk, walk_turn, run, stationary], axis=0)

    signals = all_data.to_numpy()

    # downsampling
    # signals_50hz = signals[::2, :]
    # signals_33hz = signals[::3, :]
    # signals_25hz = signals[::4, :]
    # signals_20hz = signals[::5, :]
    # signals_10hz = signals[::10, :]
    # signals_5hz = signals[::20, :]

    signal_index_sequence = [1, 2, 3, 4, 5, 6, 1, 3,
                             5, 1, 4, 6, 2, 4, 5, 2]

    # Convert raw signal(RS) to signal image(SI)
    idx = 0
    n_samples = args.n_samples
    sample_distance = args.distance
    images = np.empty((1, 16, n_samples), dtype=float)
    labels = np.empty((1, 1), dtype=int)

    while idx + n_samples < signals.shape[0]:
        
        if idx % 1000 == 0:
            print('Making dataset (100Hz): {}/{}'.format(idx, signals.shape[0]))

        # make image only in same trial
        if signals[idx][7] == signals[idx + n_samples][7]:
            # raw signal
            raw_signal = np.transpose(signals[idx:idx+n_samples, :6])

            # signal image
            signal_image = np.array([raw_signal[idx-1] for idx in signal_index_sequence], dtype=float)
            if not args.dft:
                signal_image = np.reshape(signal_image, (1, 16, n_samples))

            if args.dft:
                # activity image
                f = np.fft.fft2(signal_image)
                fshift = np.fft.fftshift(f)
                activity_image = 20 * np.log(np.abs(fshift))
                activity_image = np.reshape(activity_image, (1, 16, n_samples))

            # Label
            label = np.array([signals[idx][6] - 1], dtype=int).reshape(1, 1)

            # Images & Labels
            if not args.dft:
                images = np.append(images, signal_image, axis=0)
            else:
                images = np.append(images, activity_image, axis=0)
            labels = np.append(labels, label, axis=0)

            idx = idx + sample_distance

        else:
            idx = idx + sample_distance

    # idx = 0

    # while idx + n_samples < signals_50hz.shape[0]:
    
    #     if idx % 1000 == 0:
    #         print('Making dataset (50Hz): {}/{}'.format(idx, signals_50hz.shape[0]))

    #     # make image only in same trial
    #     if signals_50hz[idx][7] == signals_50hz[idx + n_samples][7]:
    #         # raw signal
    #         raw_signal = np.transpose(signals_50hz[idx:idx+n_samples, :6])

    #         # signal image
    #         signal_image = np.array([raw_signal[idx-1] for idx in signal_index_sequence], dtype=float)
    #         signal_image = np.reshape(signal_image, (1, 16, n_samples))

    #         # Label
    #         label = np.array([signals_50hz[idx][6] - 1], dtype=int).reshape(1, 1)

    #         # Images & Labels
    #         images = np.append(images, signal_image, axis=0)
    #         labels = np.append(labels, label, axis=0)

    #         idx = idx + sample_distance

    #     else:
    #         idx = idx + sample_distance

    # idx = 0

    # while idx + n_samples < signals_33hz.shape[0]:
    
    #     if idx % 1000 == 0:
    #         print('Making dataset (33Hz): {}/{}'.format(idx, signals_33hz.shape[0]))

    #     # make image only in same trial
    #     if signals_33hz[idx][7] == signals_33hz[idx + n_samples][7]:
    #         # raw signal
    #         raw_signal = np.transpose(signals_33hz[idx:idx+n_samples, :6])

    #         # signal image
    #         signal_image = np.array([raw_signal[idx-1] for idx in signal_index_sequence], dtype=float)
    #         signal_image = np.reshape(signal_image, (1, 16, n_samples))

    #         # Label
    #         label = np.array([signals_33hz[idx][6] - 1], dtype=int).reshape(1, 1)

    #         # Images & Labels
    #         images = np.append(images, signal_image, axis=0)
    #         labels = np.append(labels, label, axis=0)

    #         idx = idx + sample_distance

    #     else:
    #         idx = idx + sample_distance

    # idx = 0

    # while idx + n_samples < signals_25hz.shape[0]:
    
    #     if idx % 1000 == 0:
    #         print('Making dataset (25Hz): {}/{}'.format(idx, signals_25hz.shape[0]))

    #     # make image only in same trial
    #     if signals_25hz[idx][7] == signals_25hz[idx + n_samples][7]:
    #         # raw signal
    #         raw_signal = np.transpose(signals_25hz[idx:idx+n_samples, :6])

    #         # signal image
    #         signal_image = np.array([raw_signal[idx-1] for idx in signal_index_sequence], dtype=float)
    #         signal_image = np.reshape(signal_image, (1, 16, n_samples))

    #         # Label
    #         label = np.array([signals_25hz[idx][6] - 1], dtype=int).reshape(1, 1)

    #         # Images & Labels
    #         images = np.append(images, signal_image, axis=0)
    #         labels = np.append(labels, label, axis=0)

    #         idx = idx + sample_distance

    #     else:
    #         idx = idx + sample_distance

    # idx = 0

    # while idx + n_samples < signals_20hz.shape[0]:
    
    #     if idx % 1000 == 0:
    #         print('Making dataset (20Hz): {}/{}'.format(idx, signals_20hz.shape[0]))

    #     # make image only in same trial
    #     if signals_20hz[idx][7] == signals_20hz[idx + n_samples][7]:
    #         # raw signal
    #         raw_signal = np.transpose(signals_20hz[idx:idx+n_samples, :6])

    #         # signal image
    #         signal_image = np.array([raw_signal[idx-1] for idx in signal_index_sequence], dtype=float)
    #         signal_image = np.reshape(signal_image, (1, 16, n_samples))

    #         # Label
    #         label = np.array([signals_20hz[idx][6] - 1], dtype=int).reshape(1, 1)

    #         # Images & Labels
    #         images = np.append(images, signal_image, axis=0)
    #         labels = np.append(labels, label, axis=0)

    #         idx = idx + sample_distance

    #     else:
    #         idx = idx + sample_distance

    # idx = 0

    # while idx + n_samples < signals_10hz.shape[0]:
    
    #     if idx % 1000 == 0:
    #         print('Making dataset (10Hz): {}/{}'.format(idx, signals_10hz.shape[0]))

    #     # make image only in same trial
    #     if signals_10hz[idx][7] == signals_10hz[idx + n_samples][7]:
    #         # raw signal
    #         raw_signal = np.transpose(signals_10hz[idx:idx+n_samples, :6])

    #         # signal image
    #         signal_image = np.array([raw_signal[idx-1] for idx in signal_index_sequence], dtype=float)
    #         signal_image = np.reshape(signal_image, (1, 16, n_samples))

    #         # Label
    #         label = np.array([signals_10hz[idx][6] - 1], dtype=int).reshape(1, 1)

    #         # Images & Labels
    #         images = np.append(images, signal_image, axis=0)
    #         labels = np.append(labels, label, axis=0)

    #         idx = idx + sample_distance

    #     else:
    #         idx = idx + sample_distance

    # idx = 0

    # while idx + n_samples < signals_5hz.shape[0]:
    
    #     if idx % 1000 == 0:
    #         print('Making dataset (5Hz): {}/{}'.format(idx, signals_5hz.shape[0]))

    #     # make image only in same trial
    #     if signals_5hz[idx][7] == signals_5hz[idx + n_samples][7]:
    #         # raw signal
    #         raw_signal = np.transpose(signals_5hz[idx:idx+n_samples, :6])

    #         # signal image
    #         signal_image = np.array([raw_signal[idx-1] for idx in signal_index_sequence], dtype=float)
    #         signal_image = np.reshape(signal_image, (1, 16, n_samples))

    #         # Label
    #         label = np.array([signals_5hz[idx][6] - 1], dtype=int).reshape(1, 1)

    #         # Images & Labels
    #         images = np.append(images, signal_image, axis=0)
    #         labels = np.append(labels, label, axis=0)

    #         idx = idx + sample_distance

    #     else:
    #         idx = idx + sample_distance


    # Delete dummy data
    images = np.delete(images, [0, 0], axis=0)    # shape: (XXXX, 16, n_samples)
    labels = np.delete(labels, [0, 0], axis=0)    # shape: (XXXX, 1)

    # Save images and labels numpy array as npy files
    np.save('./real-data/real_images_{}_{}_{}_0826_M1+M2+M3.npy'.format(n_samples, sample_distance, args.n_classes), images)
    np.save('./real-data/real_labels_{}_{}_{}_0826_M1+M2+M3.npy'.format(n_samples, sample_distance, args.n_classes), labels)