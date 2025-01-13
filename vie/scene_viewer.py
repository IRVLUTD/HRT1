import os
import numpy as np
import open3d as o3d
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


class PLYViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PLY Viewer")
        self.showFullScreen()
        # self.setGeometry(200, 200, 1200, 800)  # Start with a larger window size

        # UI Elements
        self.viewer = QLabel("No PLY file loaded")
        self.viewer.setAlignment(Qt.AlignCenter)
        self.viewer.setFixedSize(1000, 700)  # Give ample space for the point cloud visualization

        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.load_button = QPushButton("Load PLY Directory")

        # Button Actions
        self.load_button.clicked.connect(self.load_directory)
        self.prev_button.clicked.connect(self.show_previous)
        self.next_button.clicked.connect(self.show_next)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.viewer)
        layout.addWidget(self.load_button)
        layout.addWidget(self.prev_button)
        layout.addWidget(self.next_button)
        self.setLayout(layout)

        # State
        self.ply_files = []
        self.current_index = -1

        # Full-screen mode
        self.setWindowState(Qt.WindowFullScreen)

    def load_directory(self):
        # Open directory dialog to select a folder
        dir_path = QFileDialog.getExistingDirectory(self, "Select PLY Directory")
        if dir_path:
            # Get all .ply files in the directory
            self.ply_files = sorted(
                [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".ply")]
            )
            if self.ply_files:
                self.current_index = 0
                self.render_ply(self.ply_files[self.current_index])
            else:
                self.viewer.setText("No PLY files found in the directory.")

    def render_ply(self, ply_path):
        try:
            # Load the PLY file as a PointCloud
            point_cloud = o3d.io.read_point_cloud(ply_path)
            if not point_cloud.is_empty():
                # Create Open3D visualization and capture an image
                vis = o3d.visualization.Visualizer()
                vis.create_window(visible=False, width=1000, height=700)  # Adjusted rendering size
                vis.add_geometry(point_cloud)
                vis.poll_events()
                vis.update_renderer()

                # Capture the screen as a float buffer
                img = vis.capture_screen_float_buffer(do_render=True)
                vis.destroy_window()

                # Convert Open3D image to NumPy and then to QPixmap
                img_np = (np.asarray(img) * 255).astype(np.uint8)
                height, width, _ = img_np.shape
                qimage = QImage(
                    img_np.data, width, height, 3 * width, QImage.Format_RGB888
                )
                pixmap = QPixmap.fromImage(qimage)

                # Display the image in the viewer
                self.viewer.setPixmap(pixmap)
            else:
                self.viewer.setText("Empty PLY file")
        except Exception as e:
            self.viewer.setText(f"Failed to render PLY file:\n{str(e)}")

    def show_previous(self):
        # Show the previous PLY file
        if self.ply_files and self.current_index > 0:
            self.current_index -= 1
            self.render_ply(self.ply_files[self.current_index])

    def show_next(self):
        # Show the next PLY file
        if self.ply_files and self.current_index < len(self.ply_files) - 1:
            self.current_index += 1
            self.render_ply(self.ply_files[self.current_index])

    def keyPressEvent(self, event):
        # Handle key press events for left and right arrows
        if event.key() == Qt.Key_Left:
            self.show_previous()
        elif event.key() == Qt.Key_Right:
            self.show_next()


if __name__ == "__main__":
    app = QApplication([])
    viewer = PLYViewer()
    viewer.show()
    app.exec_()
