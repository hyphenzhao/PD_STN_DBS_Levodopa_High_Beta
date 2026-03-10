import mne, pickle, os, traceback
import pandas as pd
import numpy as np
def split_label_z_axis(label, name_appendix_1='_upper', name_appendix_2='_lower'):
    coords = label.pos  # Vertex positions
    z_coords = coords[:, 2]  # Extract z-axis values

    # Find the median value of the z-coordinates
    median_z = np.median(z_coords)

    # Split vertices into upper and lower based on the median value of the z-coordinates
    upper_indices = label.vertices[z_coords >= median_z]
    lower_indices = label.vertices[z_coords < median_z]

    # Create new labels for upper and lower portions
    label_upper = mne.Label(vertices=upper_indices, hemi=label.hemi, name=label.name + name_appendix_1,
                            pos=coords[z_coords >= median_z], subject=label.subject)
    label_lower = mne.Label(vertices=lower_indices, hemi=label.hemi, name=label.name + name_appendix_2,
                            pos=coords[z_coords < median_z], subject=label.subject)
    return label_upper, label_lower
def get_medialcentral_labels(subjects_dir, subject):
    labels                 = mne.read_labels_from_annot(subject=subject, parc='aparc', subjects_dir=subjects_dir)
    motor_cortex_names_lh  = ['postcentral-lh', 'precentral-lh', 'paracentral-lh']
    motor_cortex_names_rh  = ['postcentral-rh', 'precentral-rh', 'paracentral-rh']
    motor_cortex_labels_lh = [label for label in labels if label.name in motor_cortex_names_lh]
    motor_cortex_labels_rh = [label for label in labels if label.name in motor_cortex_names_rh]
    medialmotorcortices_lh = [split_label_z_axis(label)[0] 
                              for label in motor_cortex_labels_lh if 'para' not in label.name]
    medialmotorcortices_rh = [split_label_z_axis(label)[0] 
                              for label in motor_cortex_labels_rh if 'para' not in label.name]
    medialmotorcortex_lh = medialmotorcortices_lh[0]
    for label in medialmotorcortices_lh[1:]:
        medialmotorcortex_lh += label
    medialmotorcortex_lh.name = 'medialcentral-lh'
    medialmotorcortex_rh = medialmotorcortices_rh[0]
    for label in medialmotorcortices_rh[1:]:
        medialmotorcortex_rh += label
    medialmotorcortex_rh.name = 'medialcentral-rh'
    return medialmotorcortex_lh, medialmotorcortex_rh
root_path     = '/Volumes/Workspace/NextCloud/Ruijin/PD EEG Analysis'
workspace     = '/Volumes/Workspace/Data/PD EEG Analysis/Dataspace'
subjects_dir  = '/Volumes/Workspace/FreeSurfer/subjects'
locator_dir   = '/Volumes/Workspace/Data/LOCATOR'
index_PD_Gait = pd.read_excel(os.path.join('Record_v2.xlsx'), index_col=1)
# index_PD_Gait = pd.read_excel(os.path.join('Record_DBS.xlsx'), index_col=1)

for subject, infos in index_PD_Gait.iterrows():
    if not infos['Useable']: continue
    if infos['Source Reconstructed']: continue
    # if infos['Same As Med-Off']: continue
    # if subject not in ['130_ZhangLan', '103_LiuZhishan']: continue
    try:
        load_name = f"{infos['Type']}-strict-raw.fif"
        load_path = os.path.join(workspace, subject, load_name)
        print('=======================================================')
        print(f"Processing {load_path}")
        raw        = mne.io.Raw(load_path, preload=True)
        raw        = raw.notch_filter(np.arange(50, 251, 50), notch_widths=4.0, picks='emg')
        epochs     = mne.read_epochs(load_path.replace('-raw', '-epo'), preload=True)
        new_epochs = mne.Epochs(raw, events=epochs.events, event_id=epochs.event_id, event_repeated='drop',
                                on_missing='ignore', tmin=-2.0, tmax=4.0, preload=True)
        medialcentral_lh, medialcentral_rh = get_medialcentral_labels(subjects_dir, subject)
        labels     = mne.read_labels_from_annot(subject=subject, parc='aparc', subjects_dir=subjects_dir)
        src        = os.path.join(subjects_dir, subject, 'mne/oct4-src.fif')
        
        # now only extract for valid_labels
        valid_labels = []
        src_spaces = mne.read_source_spaces(src)
        for label in labels:
            hemi = label.hemi  # 'lh' or 'rh'
            vertno = src_spaces[0]['vertno'] if hemi == 'lh' else src_spaces[1]['vertno']
            # see if any of this label's vertices appear in the source-space
            if np.intersect1d(label.vertices, vertno).size > 0: valid_labels.append(label)
            else: print(f"⚠️  Label {label.name} not found in src, skipping.")
        if not valid_labels: raise RuntimeError("No labels remain after filtering — nothing to extract!")
        
        for source_method in ['dspm', 'beamformer']:
            print(f"Processing mixture of {source_method} and raw for {subject}.")
            load_name = f"{infos['Type']}-{source_method}-strict-raw-stcs"
            load_path = os.path.join(workspace, subject, load_name)
            print(f"Extract signals from raw and sources of {load_path}\nFirst sample: {raw.first_samp}")
            source_estimate = mne.read_source_estimate(load_path)
            # extract_names   = ['postcentral-lh', 'postcentral-rh', 'precentral-lh', 'precentral-rh', 
            #                    'paracentral-lh', 'paracentral-rh']
            # extract_labels  = [label for label in labels if label.name in extract_names]
            extract_names   = [label.name for label in valid_labels]
            extract_labels  = valid_labels
            extract_labels += [medialcentral_lh, medialcentral_rh]
            raw_array       = np.array(source_estimate.extract_label_time_course(extract_labels,
                                                                                 mne.read_source_spaces(src)))
            source_channels = extract_names + ['medialcentral-lh', 'medialcentral-rh']
            raw_info        = mne.create_info(source_channels, raw.info['sfreq'], ch_types='eeg', verbose=None)
            stc_raw         = mne.io.RawArray(raw_array, raw_info, first_samp=raw.first_samp)
            print(raw_array.shape, raw.get_data().shape)
            mixed_raw_array = np.concatenate((raw_array, raw.get_data()), axis=0)
            mixed_raw_types = stc_raw.get_channel_types() + raw.get_channel_types()
            mixed_raw_info  = mne.create_info(stc_raw.ch_names + raw.ch_names, raw.info['sfreq'], 
                                              ch_types=mixed_raw_types, verbose=None)
            mixed_raw       = mne.io.RawArray(mixed_raw_array, mixed_raw_info, first_samp=raw.first_samp)
            save_path       = os.path.join(workspace, subject, f"{infos['Type']}-mixed-{source_method}-raw-all.fif")
            print(f"After combination: {mixed_raw.ch_names}")
            mixed_raw.set_meas_date(raw.annotations.orig_time)
            mixed_raw.set_annotations(raw.annotations)
            mixed_raw.save(save_path, overwrite=True)
            
            # stc_epochs      = mne.Epochs(stc_raw, events=epochs.events, event_id=epochs.event_id, event_repeated='drop',
            #                              on_missing='ignore', tmin=-2.0, tmax=4.0, preload=True)
            # epochs_array    = np.concatenate((new_epochs.get_data(), stc_epochs.get_data()), axis=1)
            # print(f"Data shape: {epochs_array.shape}")
            # channel_types   = new_epochs.get_channel_types() + stc_epochs.get_channel_types()
            # epochs_info     = mne.create_info(new_epochs.ch_names+stc_epochs.ch_names, raw.info['sfreq'], 
            #                                   ch_types=channel_types, verbose=None)
            # all_epochs      = mne.EpochsArray(epochs_array, epochs_info, events=epochs.events, 
            #                                   event_id=epochs.event_id, on_missing='ignore', tmin=-2.0)
            # save_path       = os.path.join(workspace, subject, f"{infos['Type']}-mixed-{source_method}-epo.fif")
            # print(f"After combination: {all_epochs.ch_names}")
            # # all_epochs.set_meas_date(epochs.annotations.orig_time)
            # all_epochs.set_annotations(epochs.annotations)
            # all_epochs.save(save_path, overwrite=True)
    except Exception as e:
        print(e)
        print(traceback.format_exc())