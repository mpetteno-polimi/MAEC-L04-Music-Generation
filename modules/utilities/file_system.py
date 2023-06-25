
import os

import note_seq
import numpy as np
import tensorflow as tf


def save_grid_sampling_results(output_dir, results, z_grid, batched_gaussian_samples):
    # Create samples folder
    samples_folder_name = 'samples'
    samples_folder_path = os.path.join(output_dir, samples_folder_name)
    tf.compat.v1.gfile.MakeDirs(samples_folder_path)

    n_grid_point = z_grid.shape[0]
    for i in range(n_grid_point):
        # Create current grid point folder
        grid_point_folder_name = 'grid_point_%d' % i
        grid_point_folder_path = os.path.join(samples_folder_path, grid_point_folder_name)
        tf.compat.v1.gfile.MakeDirs(grid_point_folder_path)

        # Save current mean point coordinate file
        mean_point_coord_file_name = 'mean_pt_coord_%d.npy' % i
        mean_point_coord_file_path = os.path.join(grid_point_folder_path, mean_point_coord_file_name)
        mean_point_coord = z_grid[i, :]
        np.save(mean_point_coord_file_path, mean_point_coord)

        # Save gaussian samples results
        n_samples_per_grid_point = batched_gaussian_samples.shape[0]
        for j in range(n_samples_per_grid_point):
            # Create current sample folder
            sample_folder_name = 'sample_%d' % j
            sample_folder_path = os.path.join(grid_point_folder_path, sample_folder_name)
            tf.compat.v1.gfile.MakeDirs(sample_folder_path)

            # Save current sample coordinate file
            sample_coord_file_name = 'sample_coord_%d.npy' % j
            sample_coord_file_path = os.path.join(sample_folder_path, sample_coord_file_name)
            sample_coord = batched_gaussian_samples[j, i, :]
            np.save(sample_coord_file_path, sample_coord)

            # Save current sample MIDI output
            sample_midi_file_name = 'sample_midi_out_%d.mid' % j
            sample_midi_file_path = os.path.join(sample_folder_path, sample_midi_file_name)
            sample_note_sequence = results[i*n_samples_per_grid_point + j]
            note_seq.sequence_proto_to_midi_file(sample_note_sequence, sample_midi_file_path)

    return samples_folder_path
