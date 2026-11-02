import os
import re
import shlex
import shutil
import subprocess
from glob import glob

def download_and_rename_edf(cmd):
    parts=shlex.split(cmd)
    if len(parts)<5 or parts[:3]!=["aws","s3","cp"]:
        raise ValueError("Expected command starting with: aws s3 cp")

    s3_path=parts[3].rstrip("/")
    local_folder=parts[4].rstrip("/")
    output_folder="/home/ayush/Desktop/NESI_review_material/MDC_analysis/ContiniousNESI_RASS_EEG"

    match=re.search(r'/(sub-[^/]+)/(ses-[^/]+)$',s3_path)
    if match is None:
        raise ValueError(f"Could not extract subject/session from: {s3_path}")

    bidsfolder=match.group(1)
    session_id=match.group(2)
    bids_local_folder=os.path.join(output_folder,bidsfolder)

    os.makedirs(local_folder,exist_ok=True)
    os.makedirs(output_folder,exist_ok=True)

    print(f"\n--- Downloading data to: {local_folder} ---")
    subprocess.run(cmd,shell=True,check=True)

    eeg_folder=os.path.join(local_folder,"eeg")
    if not os.path.isdir(eeg_folder):
        print(f"No 'eeg' folder found in {local_folder}")
        shutil.rmtree(bids_local_folder,ignore_errors=True)
        return None

    edf_files=glob(os.path.join(eeg_folder,"*.edf"))
    if len(edf_files)==0:
        print(f"No EDF files found in {eeg_folder}")
        shutil.rmtree(bids_local_folder,ignore_errors=True)
        return None

    if len(edf_files)>1:
        print(f"WARNING: {len(edf_files)} EDF files found. Using first EDF.")

    final_edf=os.path.join(output_folder,f"{bidsfolder}_{session_id}.edf")
    if os.path.exists(final_edf):
        os.remove(final_edf)

    shutil.move(edf_files[0],final_edf)
    shutil.rmtree(bids_local_folder,ignore_errors=True)

    print(f"Saved EDF: {final_edf}")
    print(f"Deleted BIDS folder: {bids_local_folder}")
    return final_edf


cmd = 'aws s3 cp s3://arn:aws:s3:us-east-1:184438910517:accesspoint/bdsp-credentialed-access-point/EEG/bids/S0002/sub-S0002115452027/ses-8/ /home/ayush/Desktop/NESI_review_material/MDC_analysis/ContiniousNESI_RASS_EEG/sub-S0002115452027/ses-8/ --recursive'


edf_path=download_and_rename_edf(cmd)