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

    args = parser.parse_args()

    # Jogging Forward
    jog = pd.DataFrame(columns=['gyrX', 'gyrY', 'gyrZ', 'eulR', 'eulP', 'eulY', 'accX', 'accY', 'accZ', 'label', 'trial'])

    temp = pd.read_csv('./dataset_0826_LinearAcc/jog_forward.csv')
    df = temp[['gyrX', 'gyrY', 'gyrZ', 'eulR', 'eulP', 'eulY', 'accX', 'accY', 'accZ', 'trial']]
    df.loc[:, 'label'] = 1
    jog = pd.concat([jog, df], axis=0)
        
    jog[['gyrX', 'gyrY', 'gyrZ']] *= 57.2985

    print('Size of Jogging Forward:', jog.shape[0])

    # Jogging Turn
    jog_turn = pd.DataFrame(columns=['gyrX', 'gyrY', 'gyrZ', 'eulR', 'eulP', 'eulY', 'accX', 'accY', 'accZ', 'label', 'trial'])

    temp = pd.read_csv('./dataset_0826_LinearAcc/jog_turn.csv')
    df = temp[['gyrX', 'gyrY', 'gyrZ', 'eulR', 'eulP', 'eulY', 'accX', 'accY', 'accZ', 'trial']]
    df.loc[:, 'label'] = 2
    jog_turn = pd.concat([jog_turn, df], axis=0)

    jog_turn[['gyrX', 'gyrY', 'gyrZ']] *= 57.2985

    print('Size of Jogging Turn:', jog_turn.shape[0])

    # Walking Forward
    walk = pd.DataFrame(columns=['gyrX', 'gyrY', 'gyrZ', 'eulR', 'eulP', 'eulY', 'accX', 'accY', 'accZ', 'label', 'trial'])

    temp = pd.read_csv('./dataset_0826_LinearAcc/walk_forward.csv')
    df = temp[['gyrX', 'gyrY', 'gyrZ', 'eulR', 'eulP', 'eulY', 'accX', 'accY', 'accZ', 'trial']]
    df.loc[:, 'label'] = 3
    walk = pd.concat([walk, df], axis=0)
        
    walk[['gyrX', 'gyrY', 'gyrZ']] *= 57.2985

    print('Size of Walking Forward:', walk.shape[0])

    # Walking Turn
    walk_turn = pd.DataFrame(columns=['gyrX', 'gyrY', 'gyrZ', 'eulR', 'eulP', 'eulY', 'accX', 'accY', 'accZ', 'label', 'trial'])

    temp = pd.read_csv('./dataset_0826_LinearAcc/walk_turn.csv')
    df = temp[['gyrX', 'gyrY', 'gyrZ', 'eulR', 'eulP', 'eulY', 'accX', 'accY', 'accZ', 'trial']]
    df.loc[:, 'label'] = 4
    walk_turn = pd.concat([walk_turn, df], axis=0)
        
    walk_turn[['gyrX', 'gyrY', 'gyrZ']] *= 57.2985

    print('Size of Walking Turn:', walk_turn.shape[0])

    # Run
    run = pd.DataFrame(columns=['gyrX', 'gyrY', 'gyrZ', 'eulR', 'eulP', 'eulY', 'accX', 'accY', 'accZ', 'label', 'trial'])

    temp = pd.read_csv('./dataset_0826_LinearAcc/run.csv')
    df = temp[['gyrX', 'gyrY', 'gyrZ', 'eulR', 'eulP', 'eulY', 'accX', 'accY', 'accZ', 'trial']]
    df.loc[:, 'label'] = 5
    run = pd.concat([run, df], axis=0)
        
    run[['gyrX', 'gyrY', 'gyrZ']] *= 57.2985

    print('Size of Run:', run.shape[0])

    # Stationary
    stationary = pd.DataFrame(columns=['gyrX', 'gyrY', 'gyrZ', 'eulR', 'eulP', 'eulY', 'accX', 'accY', 'accZ', 'label', 'trial'])

    temp = pd.read_csv('./dataset_0826_LinearAcc/stationary.csv')
    df = temp[['gyrX', 'gyrY', 'gyrZ', 'eulR', 'eulP', 'eulY', 'accX', 'accY', 'accZ', 'trial']]
    df.loc[:, 'label'] = 6
    stationary = pd.concat([stationary, df], axis=0)
        
    stationary[['gyrX', 'gyrY', 'gyrZ']] *= 57.2985

    print('Size of Stationary:', stationary.shape[0])

    # concatenate all data
    all_data = pd.concat([jog, jog_turn, walk, walk_turn, run, stationary], axis=0)

    signals = all_data.to_numpy()

    # downsampling
    # signals = signals[::10, :]

    signal_index_sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9,
                             1, 3, 5, 7, 9, 2, 4, 6, 8,
                             1, 4, 7, 1, 5, 8, 2, 5, 9,
                             3, 6, 9, 4, 8, 3, 7, 2, 6]

    # Convert raw signal(RS) to signal image(SI)
    idx = 0
    n_samples = args.n_samples
    sample_distance = args.distance
    images = np.empty((1, 36, n_samples), dtype=float)
    labels = np.empty((1, 1), dtype=int)

    while idx + n_samples < signals.shape[0]:
        
        if idx % 1000 == 0:
            print('Making dataset: {}/{}'.format(idx, signals.shape[0]))

        # make image only in same trial
        if signals[idx][10] == signals[idx + n_samples][10]:
            # raw signal
            raw_signal = np.transpose(signals[idx:idx+n_samples, :9])

            # signal image
            signal_image = np.array([raw_signal[idx-1] for idx in signal_index_sequence], dtype=float)
            # signal_image = np.reshape(signal_image, (1, 36, n_samples))

            # activity image
            f = np.fft.fft2(signal_image)
            fshift = np.fft.fftshift(f)
            activity_image = 20 * np.log(np.abs(fshift))
            activity_image = np.reshape(activity_image, (1, 36, n_samples))

            # Label
            label = np.array([signals[idx][9] - 1], dtype=int).reshape(1, 1)

            # Images & Labels
            images = np.append(images, activity_image, axis=0)
            labels = np.append(labels, label, axis=0)

            idx = idx + sample_distance

        else:
            idx = idx + sample_distance

    # Delete dummy data
    images = np.delete(images, [0, 0], axis=0)    # shape: (XXXX, 36, n_samples)
    labels = np.delete(labels, [0, 0], axis=0)    # shape: (XXXX, 1)

    # Save images and labels numpy array as npy files
    np.save('./real-data/real_images_{}_{}_{}_0826_LinearAcc_AI.npy'.format(n_samples, sample_distance, args.n_classes), images)
    np.save('./real-data/real_labels_{}_{}_{}_0826_LinearAcc_AI.npy'.format(n_samples, sample_distance, args.n_classes), labels)