import mne,os,math,csv,re,json
import numpy as np
import pandas as pd
import pickle
import mne_icalabel
import shutil
import pymeshfix
from multiprocessing import Pool
import datetime
import traceback

def fix_mesh_problems(subjects_dir, subject_name):
    bem_dir = os.path.join(subjects_dir, subject_name, 'bem')
    for bem_file in os.listdir(bem_dir):
        if bem_file.endswith('.surf'):
            surf_file = os.path.join(bem_dir, bem_file)
            rr, tris  = mne.read_surface(surf_file)
            rr, tris  = pymeshfix.clean_from_arrays(rr, tris)
            mne.write_surface(surf_file, rr, tris, overwrite=True)
            
def find_sfp_file(locator_dir, patient):
    # Loop through all .sfp files in the directory
    for file_name in os.listdir(locator_dir):
        # Check if the file is a .sfp file
        if file_name.endswith('.sfp'):
            # Check if the patient name is a substring of the file name
            if patient.lower() in file_name.lower():
                sfp_fpath = os.path.join(locator_dir, file_name)
                print(f"Found matching .sfp file: {sfp_fpath}")
                return sfp_fpath

    # If no matching file is found, return None
    print(f"No matching .sfp file found for patient: {patient}")
    return None
    
def print_to_logfile(log_path, content):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content_with_timestamp = f"{content} [{timestamp}]"
    
    print(content_with_timestamp)
    with open(log_path, 'a+') as f:
        f.write(content_with_timestamp + "\n")
        
def process_subject(args):
    subject_id, infos, workspace, locator_dir, subjects_dir, bs_problems = args
    if not infos['Useable']: return
    if not infos['Preprocessed']: return
    if infos['Source Reconstructed']: return
    if '_LFP' in subject_id: return
    # if 'XiaYufang' not in subject_id: return
    epochs_beam_stcs_save_path = os.path.join(workspace, subject_id, f"{infos['Type']}-beamformer-epo-stcs.pkl")
    # if os.path.exists(epochs_beam_stcs_save_path): 
    #     print(f"Script >>>>>> {subject_id} {infos['Type']} processed, skip")
    #     return
    subject_med = f"{subject_id} {infos['Type']}"
    subject_dir = os.path.join(subjects_dir, subject_id)
    mne_dir = os.path.join(subject_dir, 'mne')
    os.makedirs(mne_dir, exist_ok=True)
    log_file   = os.path.join(workspace, subject_id, f"{infos['Type']}.log")
    error_log  = os.path.join(workspace, 'error_logs.txt')
    if os.path.exists(log_file): os.remove(log_file)
    print_to_logfile(log_file, f"======================================================")
    try:
        # Load raw and epochs data
        load_path = os.path.join(workspace, subject_id, f"{infos['Type']}-strict-raw.fif")
        raw = mne.io.read_raw_fif(load_path, preload=True)
        load_path = os.path.join(workspace, subject_id, f"{infos['Type']}-strict-epo.fif")
        epochs = mne.read_epochs(load_path, preload=True)
    
        # Load montage and apply to raw
        sfp_fpath = find_sfp_file(locator_dir, subject_id.split('_')[1])
        montage = mne.channels.read_custom_montage(sfp_fpath)
        raw.set_montage(montage, match_case=False, on_missing="ignore", verbose="WARNING")
    
        # Setup transformation
        trans_dir = os.path.join(subjects_dir, subject_id, 'trans')
        os.makedirs(trans_dir, exist_ok=True)
        trans_path = os.path.join(trans_dir, 'head_mri-trans.fif')
        if not os.path.exists(trans_path):
            trans = mne.coreg.estimate_head_mri_t(subject_id)
            mne.write_trans(trans_path, trans)
    
        # Setup source space and BEM
        print_to_logfile(log_file, f"Script >>>>>>> 1. Setup source space for subject: {subject_id}")
        src_path = os.path.join(mne_dir, 'oct4-src.fif')
        if os.path.exists(src_path): src = mne.read_source_spaces(src_path)
        else:
            src = mne.setup_source_space(subject_id, spacing='oct4', subjects_dir=subjects_dir, verbose=True)
            src.save(src_path, overwrite=True)
            
        print_to_logfile(log_file, f"Script >>>>>>> 2. Make bem model for subject: {subject_id}")
        bem_model = os.path.join(mne_dir, 'bem-model.pkl')
        if os.path.exists(bem_model):
            with open(bem_model, 'rb') as f: model = pickle.load(f)
        else:
            try:
                model = mne.make_bem_model(subject=subject_id, ico=None, 
                                           conductivity=(0.3, 0.006, 0.3), 
                                           subjects_dir=subjects_dir, verbose=True)
            except Exception as e:
                print_to_logfile(log_file, e)
                print_to_logfile(log_file, "\nTry to fix with pymeshfix first.")
                try:
                    fix_mesh_problems(subjects_dir, subject_id)
                    model = mne.make_bem_model(subject=subject_id, ico=None, 
                                           conductivity=(0.3, 0.006, 0.3), 
                                           subjects_dir=subjects_dir, verbose=True)
                except Exception as ee:
                    print_to_logfile(log_file, ee)
                    print_to_logfile(log_file, "Making bem failed.")
                    with open(log_file, 'a+') as f: f.write(traceback.format_exc())
                    print_to_logfile(error_log, f"Script !!!!!!! Error Making bem failed. [{subject_id}]")
                    with open(error_log, 'a+') as f: f.write(traceback.format_exc())
                    bs_problems[f"{subject_id} {infos['Type']}"] = ee
                    return
            with open(bem_model, 'wb') as f: pickle.dump(model, f)
        
    
        print_to_logfile(log_file, f"Script >>>>>>> 3. Making bem solution for subject: {subject_id}")
        bem_solution_path = os.path.join(mne_dir, 'bem-solution.pkl')
        if os.path.exists(bem_solution_path): 
            print_to_logfile(log_file, f"Script >>>>>>> Loading bem solution from file {bem_solution_path}")
            with open(bem_solution_path, 'rb') as f: bem = pickle.load(f)
            print_to_logfile(log_file, f"Script >>>>>>> Bem solution loaded")
        else:
            bem = mne.make_bem_solution(model, verbose=True)
            with open(bem_solution_path, 'wb') as f: pickle.dump(bem, f)
    
        # Forward solution and covariance
        print_to_logfile(log_file, f"Script >>>>>>> 4. Making forward solution for subject: {subject_id}")
        fwd_path = os.path.join(mne_dir, 'eeg-fwd.fif')
        if os.path.exists(fwd_path): fwd = mne.read_forward_solution(fwd_path)
        else:
            fwd = mne.make_forward_solution(raw.info, trans=trans_path, src=src, bem=bem, eeg=True, meg=False)
            fwd.save(fwd_path, overwrite=True)
    
        cov_method = 'empirical'
        if subject_med in ['327_XiaYufang Med-On']: cov_method = 'shrunk'
        print_to_logfile(log_file, f"Script >>>>>>> 5.1.1 Compute raw covariance for subject: {subject_id}")
        raw_cov = mne.compute_raw_covariance(raw, method=cov_method)
        print_to_logfile(log_file, f"Script >>>>>>> 5.1.2 Compute raw inverse: {subject_id}")
        invs_opt = mne.minimum_norm.make_inverse_operator(raw.info, fwd, raw_cov, loose=0.2, depth=0.8)
        raw_stcs = mne.minimum_norm.apply_inverse_raw(raw, invs_opt, 1.0 / 9.0, method='dSPM', pick_ori=None)
        raw_stcs_save_path = os.path.join(workspace, subject_id, f"{infos['Type']}-dspm-strict-raw-stcs")
        raw_stcs.save(raw_stcs_save_path, overwrite=True)
        
        # Beamformer: raw
        reduce_rank_list = ['103_LiuZhishan Med-On', '136_MaShuwen Med-On', 
                            '136_MaShuwen Med-Off', '327_XiaYufang Med-On',
                            '103_LiuZhishan Med-Off', '130_GuFengmei Med-Off']
        reduce_rank = False
        if subject_med in reduce_rank_list: reduce_rank = True
            
        print_to_logfile(log_file, f"Script >>>>>>> 5.3.1 Compute raw beamformer inverse: {subject_id}")
        filters_raw = mne.beamformer.make_lcmv(raw.info, fwd, raw_cov, reduce_rank=reduce_rank)
        raw_beam_stcs = mne.beamformer.apply_lcmv_raw(raw.pick('eeg'), filters_raw)
        raw_beam_stcs_save_path = os.path.join(workspace, subject_id, f"{infos['Type']}-beamformer-strict-raw-stcs")
        raw_beam_stcs.save(raw_beam_stcs_save_path, overwrite=True)
            
        print_to_logfile(log_file, f"Processing completed for subject: {subject_id}")
    except Exception as error:
        print(f"Script !!!!!!! Error happened when processing {subject_id} {infos['Type']}")
        with open(log_file, 'a+') as f: f.write(traceback.format_exc())
        print_to_logfile(error_log, f"Script !!!!!!! Error happened when processing {subject_id} {infos['Type']}")
        with open(error_log, 'a+') as f: f.write(traceback.format_exc())
        return
# Multiprocessing setup
if __name__ == "__main__":
    root_path = '/Volumes/Workspace/NextCloud/Ruijin/PD EEG Analysis'
    workspace = '/Volumes/Workspace/Data/PD EEG Analysis/Dataspace'
    index_PD_Gait = pd.read_excel(os.path.join('Record_v2.xlsx'), index_col=1)
    # index_PD_Gait = pd.read_excel(os.path.join('Record_DBS.xlsx'), index_col=1)
    # index_PD_Gait = index_PD_Gait.sort_values(by='Type')
    print(index_PD_Gait)
    locator_dir = '/Volumes/Workspace/Data/LOCATOR'
    subjects_dir = '/Volumes/Workspace/FreeSurfer/subjects'

    bs_problems = {}
    args_list = [(subject_id, infos, workspace, locator_dir, subjects_dir, bs_problems)
                 for subject_id, infos in index_PD_Gait.iterrows()]

    with Pool(processes=6) as pool:  # Adjust the number of processes as needed
        pool.map(process_subject, args_list)

    print("Processing completed for all subjects.")
    for key, value in bs_problems.items():
        print(f"{key}: {value}")
