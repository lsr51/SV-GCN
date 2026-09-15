import h5py
import numpy as np
import os
from tqdm import tqdm

# NTU 25 joints -> Emotion-Gait 16 joints mapping
NTU_TO_GAIT_MAPPING = {
    'root': 1,  # spine base
    'spine': 2,  # mid spine
    'neck': 21,  # neck
    'head': 4,  # head
    'left_shoulder': 5,  # left shoulder
    'left_elbow': 6,  # left elbow
    'left_hand': 7,  # left hand
    'right_shoulder': 9,  # right shoulder
    'right_elbow': 10,  # right elbow
    'right_hand': 11,  # right hand
    'left_hip': 13,  # left hip
    'left_knee': 14,  # left knee
    'left_foot': 15,  # left foot
    'right_hip': 17,  # right hip
    'right_knee': 18,  # right knee
    'right_foot': 19,  # right foot
}

# 16 joints ordered as defined in README
JOINT_ORDER = [
    'root', 'spine', 'neck', 'head',
    'left_shoulder', 'left_elbow', 'left_hand',
    'right_shoulder', 'right_elbow', 'right_hand',
    'left_hip', 'left_knee', 'left_foot',
    'right_hip', 'right_knee', 'right_foot'
]

# Maximum frame threshold (samples exceeding this will be skipped)
MAX_FRAMES = 180


def read_skeleton_filter(file):
    """Read function copied from ntu_gendata.py"""
    try:
        with open(file, 'r') as f:
            skeleton_sequence = {}
            skeleton_sequence['numFrame'] = int(f.readline())
            skeleton_sequence['frameInfo'] = []
            for t in range(skeleton_sequence['numFrame']):
                frame_info = {}
                frame_info['numBody'] = int(f.readline())
                frame_info['bodyInfo'] = []
                for m in range(frame_info['numBody']):
                    body_info = {}
                    body_info_key = [
                        'bodyID', 'clipedEdges', 'handLeftConfidence',
                        'handLeftState', 'handRightConfidence', 'handRightState',
                        'isResticted', 'leanX', 'leanY', 'trackingState'
                    ]
                    body_info = {
                        k: float(v)
                        for k, v in zip(body_info_key, f.readline().split())
                    }
                    body_info['numJoint'] = int(f.readline())
                    body_info['jointInfo'] = []
                    for v in range(body_info['numJoint']):
                        joint_info_key = [
                            'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                            'orientationW', 'orientationX', 'orientationY',
                            'orientationZ', 'trackingState'
                        ]
                        joint_info = {
                            k: float(v)
                            for k, v in zip(joint_info_key, f.readline().split())
                        }
                        body_info['jointInfo'].append(joint_info)
                    frame_info['bodyInfo'].append(body_info)
                skeleton_sequence['frameInfo'].append(frame_info)
        return skeleton_sequence
    except Exception as e:
        raise ValueError(f"Failed to read skeleton file: {e}")


def extract_single_body_xyz(file, num_joint=25):
    """
    Extract single-person xyz coordinates from a .skeleton file.
    Requirement: numBody must equal 1 for all frames (single-person action),
    otherwise returns None.
    Preserves original frame count without truncation or padding.
    Returns: (T_original, 25, 3) or None (if not single-person, frame limit exceeded, or invalid data)
    """
    try:
        seq_info = read_skeleton_filter(file)
        T = seq_info['numFrame']

        # Check if frame count exceeds threshold
        if T > MAX_FRAMES:
            print(f"  Skipped: Frame count {T} > {MAX_FRAMES}")
            return None, "frame_count_exceed"

        # Check if there are any frames
        if T == 0:
            print(f"  Warning: No frames in file")
            return None, "zero_frames"

        # First check if all frames are single-person (numBody == 1)
        for n, f in enumerate(seq_info['frameInfo']):
            if f['numBody'] != 1:
                print(f"  Skipped: Frame {n} has numBody={f['numBody']} (expected 1) -> not a single-person sample")
                return None, "not_single_person"

        data = np.zeros((T, num_joint, 3), dtype=np.float32)
        valid_frames = 0

        for n, f in enumerate(seq_info['frameInfo']):
            # numBody must be 1 here, take the first (and only) body
            body = f['bodyInfo'][0]
            # Check joint count
            if len(body['jointInfo']) < num_joint:
                print(f"  Warning: Frame {n} has only {len(body['jointInfo'])} joints (expected {num_joint})")
                continue

            for j, joint in enumerate(body['jointInfo']):
                if j < num_joint:
                    x, y, z = joint['x'], joint['y'], joint['z']
                    if np.isfinite([x, y, z]).all():
                        data[n, j, :] = [x, y, z]
                    else:
                        print(f"  Warning: Frame {n}, joint {j} has invalid coordinates: ({x}, {y}, {z})")
            valid_frames += 1

        # Check if there are any valid frames (in theory all should be valid since numBody==1 guarantees data)
        if valid_frames == 0:
            print(f"  Error: No valid frames with body data")
            return None, "no_valid_frames"

        return data, "success"

    except Exception as e:
        print(f"  Error extracting skeleton data: {e}")
        return None, "extraction_error"


def convert_ntu_to_gait(data_ntu):
    """
    Convert NTU 25-joint data to Emotion-Gait 16-joint format
    Preserves original frame count

    Input: (T, 25, 3) - NTU format, T variable
    Output: (T, 48)   - Emotion-Gait format, T unchanged
    Returns None if conversion fails
    """
    try:
        if data_ntu is None:
            return None

        T = data_ntu.shape[0]

        # Check input data dimensions
        if len(data_ntu.shape) != 3 or data_ntu.shape[1] != 25 or data_ntu.shape[2] != 3:
            print(f"  Error: Invalid data shape {data_ntu.shape}, expected (T, 25, 3)")
            return None

        # Check for all-zero frames (possible missing data)
        zero_frames = 0
        for t in range(T):
            if np.all(data_ntu[t] == 0):
                zero_frames += 1

        if zero_frames > T * 0.5:  # More than 50% all-zero frames, data quality too low
            print(f"  Warning: {zero_frames}/{T} frames are all zeros, data quality too low")

        data_gait = np.zeros((T, 48), dtype=np.float32)

        for i, joint_name in enumerate(JOINT_ORDER):
            ntu_idx = NTU_TO_GAIT_MAPPING[joint_name] - 1  # Convert to 0-based

            # Check if index is valid
            if ntu_idx < 0 or ntu_idx >= 25:
                print(f"  Error: Invalid joint index {ntu_idx} for {joint_name}")
                return None

            # Each joint occupies 3 columns: x, y, z
            data_gait[:, i * 3:(i + 1) * 3] = data_ntu[:, ntu_idx, :]

        # Check if converted data is all zeros
        if np.all(data_gait == 0):
            print(f"  Error: Converted data is all zeros")
            return None

        return data_gait

    except Exception as e:
        print(f"  Error converting data: {e}")
        return None


def validate_and_fix_gait_data(data_gait, sample_name):
    """
    Validate and fix gait data format
    Returns: (fixed_data, is_valid, error_msg)
    """
    if data_gait is None:
        return None, False, "Data is None"

    # Check shape
    if len(data_gait.shape) == 2:
        # Should be (T, 48)
        if data_gait.shape[1] != 48:
            return None, False, f"Expected 48 features, got {data_gait.shape[1]}"

        # Check for NaN or Inf
        if not np.isfinite(data_gait).all():
            # Fix: replace NaN and Inf with 0
            nan_count = np.isnan(data_gait).sum()
            inf_count = np.isinf(data_gait).sum()
            data_gait = np.nan_to_num(data_gait, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"  Fixed: Replaced {nan_count} NaN and {inf_count} Inf values in {sample_name}")

        # Check if all data is zero
        if np.all(data_gait == 0):
            return None, False, "All data is zero"

        return data_gait, True, "Valid"

    elif len(data_gait.shape) == 3:
        # Could be (T, 16, 3) format, needs flattening
        if data_gait.shape[1] == 16 and data_gait.shape[2] == 3:
            T, J, C = data_gait.shape
            data_flat = data_gait.reshape(T, J * C)
            print(f"  Fixed: Reshaped {sample_name} from {(T, J, C)} to {(T, J * C)}")

            # Validate flattened data
            if not np.isfinite(data_flat).all():
                nan_count = np.isnan(data_flat).sum()
                inf_count = np.isinf(data_flat).sum()
                data_flat = np.nan_to_num(data_flat, nan=0.0, posinf=0.0, neginf=0.0)
                print(f"  Fixed: Replaced {nan_count} NaN and {inf_count} Inf values after reshape")

            if np.all(data_flat == 0):
                return None, False, "All data is zero after reshape"

            return data_flat, True, "Valid"
        else:
            return None, False, f"Invalid 3D shape: {data_gait.shape}, expected (T, 16, 3)"

    else:
        return None, False, f"Invalid data shape: {data_gait.shape}, expected (T, 48) or (T, 16, 3)"


def save_to_h5(skeleton_dir, output_features_path, output_labels_path,
               ignored_samples_path=None, error_log_path='conversion_errors.txt',
               max_conversions=None):
    """
    Batch convert .skeleton files to .h5 format
    Each sample preserves its original frame count (T variable)
    Only keeps single-person samples (numBody == 1 for all frames)
    Automatically skips samples with frame count > MAX_FRAMES
    Displays data info for each sample during conversion
    Automatically discards samples that cannot be converted or have issues

    Parameters:
        skeleton_dir: Directory containing .skeleton files
        output_features_path: Path for output features.h5
        output_labels_path: Path for output labels.h5
        ignored_samples_path: List of missing samples (optional)
        error_log_path: Path for logging error samples
        max_conversions: Maximum number of conversions, stops early when reached (None = convert all)
    """
    # Read ignored list
    if ignored_samples_path and os.path.exists(ignored_samples_path):
        with open(ignored_samples_path, 'r') as f:
            ignored_samples = set(line.strip() + '.skeleton' for line in f.readlines())
    else:
        ignored_samples = set()

    # Get all .skeleton files
    skeleton_files = [f for f in os.listdir(skeleton_dir)
                      if f.endswith('.skeleton') and f not in ignored_samples]

    # Sort for reproducibility
    skeleton_files.sort()

    # If max_conversions is set, limit the number of files to process
    total_files = len(skeleton_files)
    if max_conversions is not None and max_conversions > 0:
        skeleton_files = skeleton_files[:max_conversions]
        print(f"Max conversions set to {max_conversions}, will process first {len(skeleton_files)} files")

    # Create HDF5 files
    f_feat = h5py.File(output_features_path, 'w')
    f_label = h5py.File(output_labels_path, 'w')

    # Statistics
    frame_lengths = []
    successful_samples = []
    failed_samples = []
    skipped_frames_samples = []  # Samples skipped due to frame count exceeding threshold
    not_single_person_samples = []  # Samples skipped due to not being single-person
    error_details = []

    # Conversion counter
    conversion_count = 0

    # Use tqdm to show conversion progress
    for idx, filename in enumerate(tqdm(skeleton_files, desc="Converting")):
        # Check if max conversion count has been reached
        if max_conversions is not None and conversion_count >= max_conversions:
            print(f"\n{'=' * 60}")
            print(f"Reached maximum conversion limit: {max_conversions}")
            print(f"Early termination (processed {idx} files, {conversion_count} successful)")
            break

        filepath = os.path.join(skeleton_dir, filename)
        sample_name = filename.replace('.skeleton', '')

        # Extract action class (A001 -> 0, A002 -> 1, ...)
        try:
            action_class_str = filename[filename.find('A') + 1:filename.find('A') + 4]
            action_class = int(action_class_str) - 1
        except (ValueError, IndexError) as e:
            error_msg = f"Failed to parse action class from {filename}: {e}"
            print(f"\n  Error: {error_msg}")
            failed_samples.append(sample_name)
            error_details.append((sample_name, error_msg))
            continue

        # Read NTU data -> (T, 25, 3), also checks if single-person sample
        data_ntu, status = extract_single_body_xyz(filepath)

        if data_ntu is None:
            if status == "frame_count_exceed":
                # Frame count exceeded, record to skipped_frames_samples
                # Note: we need the exact frame count, but extract_single_body_xyz doesn't return it, so re-read
                try:
                    seq_info = read_skeleton_filter(filepath)
                    T = seq_info['numFrame']
                    skipped_frames_samples.append((sample_name, T))
                except:
                    skipped_frames_samples.append((sample_name, -1))
                continue
            elif status == "not_single_person":
                not_single_person_samples.append(sample_name)
                error_msg = "Not a single-person sample (contains frames with numBody != 1)"
                error_details.append((sample_name, error_msg))
                continue
            else:
                error_msg = f"Failed to extract skeleton data (status: {status})"
                print(f"\n  Error: {sample_name} - {error_msg}")
                failed_samples.append(sample_name)
                error_details.append((sample_name, error_msg))
                continue

        # Convert to Emotion-Gait format -> (T, 48)
        data_gait = convert_ntu_to_gait(data_ntu)

        if data_gait is None:
            error_msg = "Failed to convert to Emotion-Gait format"
            print(f"\n  Error: {sample_name} - {error_msg}")
            failed_samples.append(sample_name)
            error_details.append((sample_name, error_msg))
            continue

        # Validate and fix converted sample
        data_gait_fixed, is_valid, reason = validate_and_fix_gait_data(data_gait, sample_name)

        if not is_valid:
            error_msg = f"Invalid sample: {reason}"
            print(f"\n  Error: {sample_name} - {error_msg}")
            failed_samples.append(sample_name)
            error_details.append((sample_name, error_msg))
            continue

        # Additional validation: check each time step has 48 features (16 joints x 3 coordinates)
        T = data_gait_fixed.shape[0]
        if data_gait_fixed.shape[1] != 48:
            error_msg = f"Feature dimension is {data_gait_fixed.shape[1]}, expected 48 after fix"
            print(f"\n  Error: {sample_name} - {error_msg}")
            failed_samples.append(sample_name)
            error_details.append((sample_name, error_msg))
            continue

        # Validate data quality for each frame
        bad_frames = 0
        zero_frames = 0
        extreme_frames = 0

        for t in range(T):
            frame_data = data_gait_fixed[t]
            # Check if entire frame is zero
            if np.all(frame_data == 0):
                zero_frames += 1
                bad_frames += 1
            # Check for extreme values (set a reasonable threshold)
            elif np.abs(frame_data).max() > 1e6:
                extreme_frames += 1
                bad_frames += 1
                if extreme_frames <= 3:  # Only print first 3 warnings
                    print(f"  Warning: {sample_name} frame {t} has extreme values: max={np.abs(frame_data).max():.2f}")

        # If more than 30% of frames are bad data, discard the sample
        if bad_frames > T * 0.3:
            error_msg = f"Too many bad frames: {bad_frames}/{T} frames are invalid (zero_frames={zero_frames}, extreme_frames={extreme_frames})"
            print(f"\n  Error: {sample_name} - {error_msg}")
            failed_samples.append(sample_name)
            error_details.append((sample_name, error_msg))
            continue

        # Save successful sample
        frame_lengths.append(T)
        successful_samples.append(sample_name)
        conversion_count += 1

        # Ensure correct data type
        data_gait_fixed = data_gait_fixed.astype(np.float32)

        # Save to HDF5
        f_feat.create_dataset(sample_name, data=data_gait_fixed, compression='gzip', compression_opts=4)
        f_label.create_dataset(sample_name, data=np.int32(action_class))

        # Display info for each sample
        print(f"  Success ({conversion_count}/{max_conversions if max_conversions else '∞'}): "
              f"{sample_name}: shape=({T}, 48), label={action_class}, "
              f"bad_frames={bad_frames}/{T} ({bad_frames / T * 100:.1f}%)")

    f_feat.close()
    f_label.close()

    # Save error log
    if error_details:
        with open(error_log_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Conversion Error Log\n")
            f.write(f"Total failed samples: {len(error_details)}\n")
            f.write("=" * 80 + "\n\n")
            for sample_name, error_msg in error_details:
                f.write(f"{sample_name}: {error_msg}\n")
        print(f"\nError log saved to {error_log_path}")

    # Print statistics
    print(f"\n{'=' * 60}")
    print(f"Conversion Summary:")
    print(f"  Max frames threshold: {MAX_FRAMES}")
    print(f"  Total files available: {total_files}")
    if max_conversions:
        print(f"  Max conversions requested: {max_conversions}")
    print(f"  Files processed: {len(successful_samples) + len(failed_samples) + len(skipped_frames_samples) + len(not_single_person_samples)}")
    print(f"  Successful: {len(successful_samples)}")
    print(f"  Failed (other reasons): {len(failed_samples)}")
    print(f"  Skipped (frames > {MAX_FRAMES}): {len(skipped_frames_samples)}")
    print(f"  Skipped (not single-person): {len(not_single_person_samples)}")
    total_attempted = len(successful_samples) + len(failed_samples) + len(skipped_frames_samples) + len(not_single_person_samples)
    if total_attempted > 0:
        print(f"  Success rate: {len(successful_samples) / total_attempted * 100:.2f}%")

    # Display frame length distribution of skipped samples (due to exceeding limit)
    if skipped_frames_samples:
        skipped_frames_values = [t for _, t in skipped_frames_samples if t != -1]
        if skipped_frames_values:
            print(f"\nSkipped samples (frame count > {MAX_FRAMES}) frame length statistics:")
            print(f"  min: {min(skipped_frames_values)}")
            print(f"  max: {max(skipped_frames_values)}")
            print(f"  mean: {np.mean(skipped_frames_values):.1f}")
            print(f"  median: {np.median(skipped_frames_values):.1f}")
            print(f"  Total skipped: {len(skipped_frames_values)}")

    if frame_lengths:
        frame_lengths = np.array(frame_lengths)
        print(f"\nFrame length statistics (successful samples, T <= {MAX_FRAMES}):")
        print(f"  Total frames across all samples: {frame_lengths.sum()}")
        print(f"  min: {frame_lengths.min()}")
        print(f"  max: {frame_lengths.max()}")
        print(f"  mean: {frame_lengths.mean():.1f}")
        print(f"  std: {frame_lengths.std():.1f}")
        print(f"  median: {np.median(frame_lengths):.1f}")

    print(f"\nOutput files:")
    print(f"  Features: {output_features_path}")
    print(f"  Labels: {output_labels_path}")

    # Verify output files
    try:
        with h5py.File(output_features_path, 'r') as check_f:
            print(f"\nVerification:")
            print(f"  Features file contains {len(check_f.keys())} datasets")
            if len(check_f.keys()) > 0:
                first_key = list(check_f.keys())[0]
                print(f"  First dataset '{first_key}': shape={check_f[first_key].shape}")
    except Exception as e:
        print(f"  Warning: Could not verify output file: {e}")

    # List some of the skipped non-single-person samples
    if not_single_person_samples:
        print(f"\nNot single-person samples ({len(not_single_person_samples)}):")
        for sample in not_single_person_samples[:20]:
            print(f"  - {sample}")
        if len(not_single_person_samples) > 20:
            print(f"  ... and {len(not_single_person_samples) - 20} more")

    # List failed samples if any
    if failed_samples:
        print(f"\nFailed samples ({len(failed_samples)}):")
        for sample in failed_samples[:20]:
            print(f"  - {sample}")
        if len(failed_samples) > 20:
            print(f"  ... and {len(failed_samples) - 20} more")
        print(f"See {error_log_path} for details")


if __name__ == '__main__':
    # Example: convert all single-person samples (stops after 56880 successful conversions)
    save_to_h5(
        skeleton_dir='./nturgb+d_skeletons/',
        output_features_path='./ntu2egait/features_NTU.h5',
        output_labels_path='./ntu2egait/labels_NTU.h5',
        ignored_samples_path='./samples_with_missing_skeletons.txt',
        error_log_path='./conversion_errors.txt',
        max_conversions=56880  # Modify as needed
    )