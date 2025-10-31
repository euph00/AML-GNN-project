# Guide

### Pre-requisites:

1. Apply for SoC account: https://mysoc.nus.edu.sg/~newacct/
2. Enable Cluster access using SoC account: https://mysoc.nus.edu.sg/~myacct/services.cgi
3. FortiClient VPN - download and steup: https://dochub.comp.nus.edu.sg/cf/guides/network/vpn/start
4. Create SSH Key-pair to enable easier login, in Windows/Terminal:
    ```bash
    ssh-keygen -t ed25519 -C "your_nusnet_email@nus.edu.sg"
    ```
5. To congiure your SSH client, add your SoC cluster config to `~/.ssh/config`:
    - Note: Replace `<your_soc_username>` with your SoC account name.

    ```sshconfig
    Host soc-cluster
        HostName xlogin0.comp.nus.edu.sg
        User <your_soc_username>
        IdentityFile ~/.ssh/id_ed25519
        ForwardAgent yes
        ForwardX11 no
    ```
6. (Optional) Copy Your Public Key to the cluster:

    ```bash
    ssh-copy-id soc-cluster
    ```

    If you get errors, paste the contents of `~/.ssh/id_ed25519.pub` into your account's `~/.ssh/authorized_keys` file manually on the cluster.

### Access SoC Compute Cluster

1. Turn on SoC VPN
2. In terminal, SSH to `soc-cluster` to access the Login Node:
    ```bash
    ssh soc-cluster
    ```
3. Request for a Compute node:
    ```bash
    salloc --nodes=1 --ntasks=1 --time=1:00:00
    srun --pty bash
    ```
    - Note: Modify the time-out parameter according to own needs.
4. Install VScode & Miniconda(First-time only):
    ```bash
    curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz

    tar -xf vscode_cli.tar.gz
    ```
    Refer to Miniconda Installation Documentation for more details: https://www.anaconda.com/docs/getting-started/miniconda/install#linux-terminal-installer
    ```bash
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh
    source ~/miniconda3/bin/activate
    conda init --all
    ```
5. Start the tunnel for VSCode:
    ```bash
    ./code tunnel
    ```
    - Note: For the first-time login, there will be some authentication required either via own Github/Microsoft account
6. In another VSCode window, connect to Tunnel:
    - Open the Command Palette: Press Ctrl+Shift+P (Windows/Linux) or Cmd+Shift+P (macOS).
    - Run the "Connect to Tunnel" command: Type "Remote Tunnels: Connect to Tunnel..." in the Command Palette and select the corresponding command.

#### To submit SLURM batch jobs via the Login node:

Note: Have to move over the files from github first (best if this code can be in the main repo?)
```bash
sbatch <your_slurm_job_filename.sh>
```

#### Monitor your job after submitting batch job

```bash
squeue -u <your_soc_username>
```

#### SLURM batch job file configuration (e.g. train_slurm_job.sh):

1. Modify the email address such that the compute start and end times will be sent your email account:
```bash
#SBATCH --mail-user=user@comp.nus.edu.sg
```

2. Modify the model training hyper-parameters: (Default values from original convnet.ipynb file)
```bash
# Create output directory
OUTPUT_DIR="./saved_models/"
mkdir -p $OUTPUT_DIR

# Training parameters
NUM_EPOCHS=50
LEARNING_RATE=1e-4
```