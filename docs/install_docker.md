## Install Docker with NVIDIA GPU Support

### Install the required packages for Docker
```
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release
```

### Add the official Docker GPG key
```
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
```

### Add the official Docker repository
```
echo \
"deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu \
$(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

### Update the APT package index and verify that the repository is configured
```
sudo apt-get update
apt-cache policy docker-ce | head
```

### Install Docker
```
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### Start the Docker service and enable it at boot
```
sudo systemctl enable --now docker
```

### Add the NVIDIA Container Toolkit repository
```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
```
```
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

### Install the NVIDIA Container Toolkit
```
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

### Configure the NVIDIA runtime for Docker and restart Docker
```
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Verify the installation
```
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```
