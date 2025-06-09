#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2025).
#----------------------------------------------------------------------------------------------------

def set_seed(seed):
    np.random.seed(seed)


def compute_xyz(depth_img, fx, fy, px, py, height, width):
    indices = np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)
    z_e = depth_img
    x_e = (indices[..., 1] - px) * z_e / fx
    y_e = (indices[..., 0] - py) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1)
    return xyz_img


def download_loftr_weights(
    weights_dir, drive_folder_id="1xu2Pq6mZT5hmFgiYMBT9Zt8h1yO-3SIp"
):
    weights_path = os.path.join(weights_dir, "outdoor_ds.ckpt")
    if os.path.exists(weights_path):
        print(f"LoFTR weights already exist at {weights_path}")
        return
    print(
        f"LoFTR weights not found at {weights_path}. Downloading from Google Drive..."
    )
    os.makedirs(weights_dir, exist_ok=True)
    folder_url = f"https://drive.google.com/drive/folders/{drive_folder_id}"
    try:
        gdown.download_folder(folder_url, output=weights_dir, quiet=False)
        if os.path.exists(weights_path):
            print(f"Successfully downloaded outdoor_ds.ckpt to {weights_path}")
        else:
            print(f"Error: outdoor_ds.ckpt not found in downloaded files.")
            sys.exit(1)
    except Exception as e:
        print(f"Error downloading LoFTR weights: {e}")
        sys.exit(1)


def read_obj_prompts(bundlesdf_dir):
    obj_prompt_mapper_file = os.path.join(bundlesdf_dir, "obj_prompt_mapper.json")
    if not os.path.exists(obj_prompt_mapper_file):
        raise Exception(f"Error: {obj_prompt_mapper_file} does not exist.")
    with open(obj_prompt_mapper_file, "r") as f:
        try:
            return pyyaml.safe_load(f)
        except pyyaml.YAMLError as e:
            raise Exception(f"Error loading YAML file {obj_prompt_mapper_file}: {e}")


def prettify_prompt(text_prompt):
    return text_prompt.replace("_", " ")


def create_symlink(target_dir, link_name):
    remove_symlink_only(link_name)
    os.symlink(target_dir, link_name)


def remove_symlink_only(symlink_path):
    if os.path.islink(symlink_path):
        os.unlink(symlink_path)
    elif os.path.exists(symlink_path):
        print(f"Warning: {symlink_path} exists but is not a symlink. Skipping removal.")


def remove_jpg_dirs(root_dir):
    logging.info(f"Removing directories containing .jpg in {root_dir}")
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path) and ".jpg" in item:
            shutil.rmtree(item_path)


def remove_ply_and_config(root_dir):
    logging.info(f"Removing .ply files and config_nerf.yml in {root_dir}")
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".ply") or file == "config_nerf.yml":
                file_path = os.path.join(root, file)
                os.remove(file_path)


def remove_unecessary_files(root_dir):
    logging.info(f"Removing unnecessary files in {root_dir}")
    remove_jpg_dirs(root_dir)
    remove_ply_and_config(root_dir)


def create_required_out_folders(out_folder):
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    pose_overlayed_rgb_dir = f"{out_folder}/pose_overlayed_rgb"
    ob_in_cam_dir = f"{out_folder}/ob_in_cam"
    for _dir in [out_folder, pose_overlayed_rgb_dir, ob_in_cam_dir]:
        os.makedirs(_dir, exist_ok=True)
        os.chmod(_dir, 0o777)
    return out_folder, pose_overlayed_rgb_dir, ob_in_cam_dir


def copy_file_if_exists(src, dst):
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
        print(f"Copied {src} to {dst}")
    else:
        print(f"{src} does not exist.")
