🐣 Please follow me for new updates: https://x.com/camenduru <br />
🔥 Please join our discord server: https://discord.gg/k5BwmmvJJU <br />
🥳 Please become my sponsor: https://github.com/sponsors/camenduru <br />
🍞 TostUI repo: https://github.com/camenduru/TostUI

#### 🍞 TostUI - Trellis 2

![Image](https://github.com/user-attachments/assets/33bc9dca-dd23-4c67-a40d-41cf8e19fc71)

Video: https://x.com/camenduru/status/2001478162996191317

1.  **Install Docker**\
    [Download Docker Desktop (Windows AMD64)](https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe)
    and run it.

2.  **Update the container (optional)**

    ``` bash
    docker stop tostui-trellis2; docker rm tostui-trellis2; docker pull camenduru/tostui-trellis2
    ```

3.  **Run the container**\
    Open Command Prompt / PowerShell and paste:

    ``` bash
    docker run --gpus all -p 3000:3000 --name tostui-trellis2 camenduru/tostui-trellis2
    ```

    *Requires NVIDIA GPU (Min 24GB VRAM)*

4.  **Open app**\
    Go to: http://localhost:3000
