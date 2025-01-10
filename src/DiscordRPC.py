from pypresence import Presence
import os
from .Util import log, networkCheck
from .constants import IS_FLATPAK


class DiscordRPC:
    def start_discordRPC(self):
        if networkCheck():
            """
            Attempts to connect to discord for RPC suppor
            Args:
                mode (str): The mode of the video (Interpolating, Upscaling)
                videoName (str): The name of the video
            """
            try:
                client_id = "1278176380043132961"  # ID for rpc
                if IS_FLATPAK:
                    os.system(
                        "ln -sf {app/com.discordapp.Discord,$XDG_RUNTIME_DIR}/discord-ipc-0"
                    )  # Enables discord RPC on flatpak
                    try:
                        for i in range(10):
                            ipc_path = f"{os.getenv('XDG_RUNTIME_DIR')}/discord-ipc-{i}"
                            if not os.path.exists(ipc_path) or not os.path.isfile(
                                ipc_path
                            ):
                                os.symlink(
                                    f"{os.getenv('HOME')}/.config/discord/{client_id}",
                                    ipc_path,
                                )
                    except Exception:
                        pass
                try:
                    self.RPC = Presence(client_id)  # Initialize the client class
                    self.RPC.connect()  # Start the handshake loop

                    self.RPC.update(
                        state="Enhancing Video",
                        details="Running batch process...",
                        large_image="logo-v2",
                    )
                except Exception:
                    pass

            # The presence will stay on as long as the program is running
            # Can only update rich presence every 15 seconds
            except Exception:
                log("Timed out!")

    def closeRPC(self):
        try:
            self.RPC.close()
        except Exception:
            pass
