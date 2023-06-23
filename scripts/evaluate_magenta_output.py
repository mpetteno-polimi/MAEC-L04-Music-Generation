import logging
import os

from magenta.models.music_vae.maec import z_sampling
from utilities import complexity_measures
import numpy as np
import matplotlib.pyplot as plt
import pretty_midi
import glob
from tqdm import tqdm
from scipy.spatial import distance


def custom_z_plots():
    z_samples = z_sampling.grid_sampling(
        z_size=256,
        grid_width=5,
        samples_per_point=16,
        n_grid_points=50,
        rand_seed=99
    )

    x = None
    y = None
    z = None

    x, y, *_ = zip(*z_samples)

    print(np.shape(z_samples))

    '''height = np.linspace(-6, 6, 50)
    fig = plt.figure()
    d1 = fig.add_subplot()
    d1.bar(x, height)'''

    if y is not None:
        fig = plt.figure()
        d2 = fig.add_subplot()
        d2.scatter(x, y)

    if z is not None:
        fig = plt.figure()
        d3 = fig.add_subplot(projection='3d')
        d3.scatter(x, y, z)

    plt.show()



def magenta_output_folder_evaluate():
    my_dir = "D:/magenta_out/seed99_day23-06_16-15config-cat-mel_2bar_big/"  # needs the trailing slash (i.e. /root/dir/)
    midi_file_paths = []
    midi_files = []
    z_coords = []
    z_grid_points_coords = []
    complexities = []
    discarded_midi_count = 0
    idx = 0

    for idx, file_path in enumerate(tqdm(glob.iglob(my_dir + '**/*.mid', recursive=True))):
        # pretty midi obj
        midi_file = pretty_midi.PrettyMIDI(file_path)
        if len(midi_file.instruments) != 1:
            discarded_midi_count += 1
            logging.info('WARNING: Dropped 1 midi - incorrect instruments count')
            continue
        midi_files.append(midi_file)

        # file path
        midi_file_paths.append(file_path)
        file_folder_path = os.path.dirname(file_path)

        # z sample coordinates
        sample_idx = file_path.split(sep='_')[-1].split(sep='.')[0]
        z_coord_name = 'sample_' + sample_idx + '_z_coord.npy'
        z_coord_path = file_folder_path + '/' + z_coord_name
        z_coords.append((np.load(z_coord_path, allow_pickle=True, fix_imports=True, encoding='latin1')))
        print(z_coord_path)

        # z grid point coordinates
        z_grid_point_folder_path = os.path.abspath(os.path.join(file_folder_path, os.pardir))
        grid_point_idx = z_grid_point_folder_path.split(sep='_')[-1].split(sep='.')[0]
        z_grid_point_file_name = z_grid_point_folder_path + '/z_grid_pt_' + grid_point_idx + '_coord.npy'
        z_grid_points_coords.append(np.load(z_grid_point_file_name, allow_pickle=True, fix_imports=True, encoding='latin1'))
        print(z_grid_point_folder_path)

        # complexity rating
        complexities.append(complexity_measures.note_density(midi_file))

    midi_file_paths = np.asarray(midi_file_paths)
    midi_files = np.asarray(midi_files)
    z_coords = np.asarray([r[1] for r in z_coords])
    z_grid_points_coords = np.asarray([r[1] for r in z_grid_points_coords
                                       ])
    complexities = np.asarray(complexities)

    print('number of discarded midi:', discarded_midi_count, 'out of', idx)
    print('midi_files.shape', midi_files.shape)
    print('midi_file_paths.shape', midi_file_paths.shape)
    print('z_coords.shape', z_coords.shape)
    print('z_grid_points_coords.shape', z_grid_points_coords.shape)
    print('complexities.shape', complexities.shape)
    print('mean:', np.mean(complexities))
    print('min:', np.min(complexities))
    print('min:', np.min(complexities))
    max_complexity = np.max(complexities)
    min_complexity = np.max(complexities)

    results = list(zip(complexities, midi_file_paths, midi_files, z_coords, z_grid_points_coords))
    results.sort(key=lambda x: x[0])
    high_complexity = [r for r in results if r[0] > 6]
    low_complexity = [r for r in results if r[0] < 2]
    high_c_coords = [t[3] for t in high_complexity]
    mean_hc = np.mean(distance.pdist(high_c_coords, metric='euclidean', out=None))
    low_c_coords = [t[3] for t in low_complexity]
    mean_lc = np.mean(distance.pdist(low_c_coords, metric='euclidean', out=None))
    print('mean high compl', mean_hc)
    print('mean low compl', mean_lc)
    plt.figure(0)
    # make data:
    x = 0.5 + np.arange(complexities.shape[0])

    # plot
    fig, ax = plt.subplots()

    ax.bar(x, complexities, width=1, edgecolor="white", linewidth=1)

    ax.set(xlim=(-1, np.shape(complexities.shape)[0]), xticks=np.arange(0, np.shape(midi_file_paths)[0] + 1, 5),
           ylim=(0, np.max(complexities) + 0.5), yticks=np.arange(1, 9))

    plt.show()


magenta_output_folder_evaluate()
