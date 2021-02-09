from doodad.launch import launch_api
from doodad import mode, mount
from tqdm import tqdm


def launch():
    filter_ext = (
        ".pyc",
        ".log",
        ".git",
        ".mp4",
        ".npz",
        ".ipynb",
        "lib",
        "_MACOSX",
        "__pycache__",
        "trained_models",
        "new_models",
        "images",
        ".egg-info",
        "meshes",
        "assets",
        "smpl_models",
        "smplx_npz.zip",
        "smplx",
        "assets.zip"
    )


    launch_mode = mode.AzureMode(
        azure_group_name="assistive-gym",
        azure_storage="assistivegym",
        gcp_bucket_name="assistive-gym-experiments",
        gcp_bucket_path="rss-logs",  # Folder to store logs
        instance_type="Standard_F32s_v2",
        # instance_type="Standard_D32s_v3",
        # instance_type="Standard_D16s_v3",
        # instance_type="Standard_F16s_v2",
        # instance_type="STANDARD_D4_V3",
        use_spot=False,
        # use_spot=True,
        azure_label="assistive-gym-training",
        gcp_auth_file="https://www.dropbox.com/s/swtwk3j5yf5dwmk/aerial-citron-264318-cbf5beb3284c.json",
    )
    gcp_mnt = mount.MountLocal(
        local_dir="../assistive-gym", mount_point="./assistive-gym", filter_ext=filter_ext
    )
    output_mnt = mount.MountAzure(
        azure_path="output",  # Directory on GCP
        mount_point="/gcp_output",  # Directory visible to the running job.
        output=True,
    )
    mounts = [gcp_mnt, output_mnt]

    # This will run locally
    launch_api.run_command(
        command="bash /dar_payload/assistive-gym/examples/cloud/remote_training.sh",
        mounts=mounts,
        mode=launch_mode,
    )


if __name__ == "__main__":
    launch()
