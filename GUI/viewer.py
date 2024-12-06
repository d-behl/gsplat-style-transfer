import viser
import cv2
import os
from pathlib import Path
import time

def viewer(viewer_port=8080):
    viewer = ViserViewer(viewer_port)


class ViserViewer:
    def __init__(self, port):
        print('Viewer is running')
        self.port = port
        self.server = viser.ViserServer(host='localhost', port=port)
        self.server.request_share_url()


        with self.server.gui.add_folder("Test"):
            self.gui_vector1 = self.server.gui.add_vector2(
                "Position",
                initial_value=(0.0, 0.0),
                step=0.1,
            )
            self.gui_vector2 = self.server.gui.add_vector3(
                "Size",
                initial_value=(1.0, 1.0, 1.0),
                step=0.25,
            )

            # display mouse position upon click
            self.mouse_click = self.server.gui.add_text("Mouse click", "0, 0")



        self.server.scene.set_background_image(
            cv2.imread("../data/figurines_processed/images/frame_00001.jpg"),
            format='jpeg',
        )

        self.fg_img = self.server.scene.add_image(
            "Name of the Image",
            cv2.imread("../data/figurines_processed/images/frame_00001.jpg"),
            4.0,
            4.0,
            format='jpeg',

        )


        self.server.add_rgb("RGB", (0, 0, 0))



        self.fg_img.on_click(lambda x: self.on_fg_click(x))

        


        while True:
            time.sleep(1)


    def on_fg_click(self, handler):
        x, y = handler.screen_pos
        print('clicked')
        self.mouse_click.value = f"{round(x, 2)}, {round(y, 2)}"

        








def main():
    viewer()


if __name__ == '__main__':
    main()