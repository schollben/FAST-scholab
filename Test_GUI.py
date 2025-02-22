import datetime
import logging
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time
from collections import deque

import torch
from csbdeep.utils import normalize
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QFileDialog, QLabel, QMessageBox, \
    QWidget, QTextEdit, QHBoxLayout
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from skimage import io
from torch.utils.data import DataLoader

from datasets.dataset import ReadDatasets
from models.Unet_Lite import Unet_Lite
from utils.config import *



class EmittingStream(QObject):
    textWritten = pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))


class TestingThread(QThread):
    finished = pyqtSignal()
    imageReady = pyqtSignal(np.ndarray, str)

    def __init__(self, parameters, parent = None):
        super(TestingThread, self).__init__(parent)
        self.is_running = True
        self.parameters = parameters

    def run(self):
        if self.is_running:
            frames_to_accumulate = 100
            accumulated_raw_images = deque(maxlen = frames_to_accumulate)
            accumulated_denoised_images = deque(maxlen = frames_to_accumulate)
            args = self.parameters.args
            logging.info('Testing Start!!!')
            num_gpu = (len(args.gpu_ids.split(",")) + 1) // 2
            inputFileNames = list(os.walk(args.test_path, topdown = False))[-1][-1]
            filename = 'models_' + os.path.basename(args.test_folder)
            testSave_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'result',
                                        filename)
            print('testSave_dir = {}'.format(testSave_dir))
            if not os.path.exists(testSave_dir):
                os.makedirs(testSave_dir)
            output_channels = args.miniBatch_size

            model = self.parameters.model

            if args.local_rank == 0:
                model_path = args.checkpoint_path
                if not os.path.exists(model_path):
                    logging.error(f"Model checkpoint '{args.checkpoint}' not found.")
                    return
                checkpoint = torch.load(model_path)
                model.load_state_dict(checkpoint['state_dict'])

            logging.info(('\n' + '%15s' * 2) % ('GPU_mem', 'total_loss'))
            # pbar = tqdm(total = len(self.parameters.test_loader),
            #             bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
            start = time.perf_counter()
            with torch.no_grad():  # Disable gradient calculation during validation
                for i, data in enumerate(self.parameters.test_loader):
                    input, t = data
                    mean = torch.mean(input)
                    input = input - mean
                    del data
                    input = input[0].cuda(args.local_rank, non_blocking = True)
                    output = torch.zeros(input.shape)
                    miniBatch = args.miniBatch_size
                    timeFrame = t
                    assert miniBatch <= timeFrame, "miniBatch size must <= time frame length !!!"
                    for j in range(0, timeFrame, miniBatch):  # sub-batch
                        if j + miniBatch <= timeFrame:
                            input_batch = input[:, j:j + miniBatch, ...]
                            output_p = model(input_batch).cpu().detach()
                            if j == 0:
                                output[:, j:j + output_channels, ...] = output_p  # + input_batch_mean
                            else:
                                output[:, j + output_channels - 1, ...] = output_p[:, -1, ...]
                                output[:, j:j + output_channels - 1, ...] = 0.5 * (
                                        output_p[:, 0:-1, ...] + output[:, j:j + output_channels - 1, ...])


                            if (j) % frames_to_accumulate == 0 or j >= timeFrame:

                                start_idx = max(0, j - frames_to_accumulate)
                                end_idx = min(j, timeFrame)


                                denoised_images = (output + mean).squeeze()[start_idx:end_idx, ...].numpy().astype(
                                    np.uint16)
                                raw_images = (input + mean).squeeze().cpu().detach()[start_idx:end_idx,
                                             ...].numpy().astype(np.uint16)


                                for idx in range(denoised_images.shape[0]):
                                    self.imageReady.emit(raw_images[idx, ...], 'raw')
                                    self.imageReady.emit(denoised_images[idx, ...], 'denoised')


                        if timeFrame % frames_to_accumulate != 0:
                            remaining_frames = timeFrame % frames_to_accumulate
                            denoised_images = (output + mean).squeeze()[:, -remaining_frames:, ...].numpy().astype(
                                np.uint16)
                            raw_images = (input + mean).squeeze().cpu().detach()[:, -remaining_frames:,
                                         ...].numpy().astype(np.uint16)

                            for idx in range(remaining_frames):
                                self.imageReady.emit(raw_images[idx, ...], 'raw')
                                self.imageReady.emit(denoised_images[idx, ...], 'denoised')

                        accumulated_raw_images.clear()
                        accumulated_denoised_images.clear()

                    end = time.perf_counter()
                    print('Processing time(s): ', end - start)
                    print('FPS:', t / (end - start))
                    output = output[:, 0:t, ::]
                    output = output + mean
                    output_image = np.squeeze(output.numpy()) * 1.0
                    result_name = os.path.join(testSave_dir,
                                               datetime.datetime.now().strftime("%Y%m%d%H%M") + '_' + inputFileNames[i])
                    print(result_name)
                    print(result_name)
                    output_image = np.clip(output_image, 0, 65535)
                    io.imsave(result_name, output_image.astype(np.uint16), check_contrast = False)


            print("\nTest End")
            # pbar.close()
            self.is_running = False
            self.finished.emit()

    def stop(self):

        self.is_running = False
        print("\nTesting stopped")
        self.quit()


class TestingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.redirect_stdout()
        self.initUI()
        self.raw_image_queue = []
        self.denoised_image_queue = []
        self.display_timer = QtCore.QTimer(self)
        self.display_timer.setInterval(1000 / 30)
        self.display_timer.timeout.connect(self.display_next_image)
        self.display_timer.start()

        self.queue_info_label = QLabel("Queue: 0")
        self.central_widget.layout().addWidget(self.queue_info_label)

    def initUI(self):
        self.setWindowTitle("Testing GUI")
        self.setGeometry(100, 100, 700, 600)


        self.setStyleSheet("""
            QMainWindow {
                background-color: black;
            }
            QLabel, QPushButton, QTextEdit {
                color: white;
                background-color: black;
            }
            QPushButton {
                background-color: #333;
                border: 1px solid #555;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #555;
            }
            QTextEdit {
                border: 1px solid #555;
                padding: 5px;
            }
        """)


        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QHBoxLayout(self.central_widget)


        left_layout = QVBoxLayout()


        self.logo_label = QLabel()
        logo_pixmap = QPixmap(r'./FAST_logo.png')
        self.logo_label.setPixmap(logo_pixmap)
        self.logo_label.setAlignment(QtCore.Qt.AlignCenter)
        left_layout.addWidget(self.logo_label)


        path_and_buttons_layout = QVBoxLayout()
        self.json_path_label = QLabel("No model_json Path selected")
        path_and_buttons_layout.addWidget(self.json_path_label)

        self.select_json_btn = QPushButton('Select model_json Path')
        path_and_buttons_layout.addWidget(self.select_json_btn)

        self.config_path_label = QLabel("No Test Folder Path selected")
        path_and_buttons_layout.addWidget(self.config_path_label)

        self.select_folder_btn = QPushButton('Select Test Folder Path')
        path_and_buttons_layout.addWidget(self.select_folder_btn)

        self.start_testing_btn = QPushButton('Start testing')
        path_and_buttons_layout.addWidget(self.start_testing_btn)

        self.stop_testing_btn = QPushButton('Stop testing')
        path_and_buttons_layout.addWidget(self.stop_testing_btn)


        left_layout.addLayout(path_and_buttons_layout)


        self.log_text_edit = QTextEdit()
        left_layout.addWidget(self.log_text_edit)


        self.raw_text_label = QLabel("Raw image:")
        self.denoised_text_label = QLabel("Denoised image:")
        right_layout = QVBoxLayout()
        self.image_label_raw = QLabel()
        self.image_label_denoised = QLabel()


        for label in [self.image_label_raw, self.image_label_denoised]:
            label.setFixedSize(400, 400)
            label.setAlignment(QtCore.Qt.AlignCenter)
            black_image = QPixmap(400, 400)
            black_image.fill(QtCore.Qt.black)
            label.setPixmap(black_image)


        right_layout.addWidget(self.raw_text_label)
        right_layout.addWidget(self.image_label_raw)
        right_layout.addWidget(self.denoised_text_label)
        right_layout.addWidget(self.image_label_denoised)


        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 3)


        self.select_folder_btn.clicked.connect(self.select_folder)
        self.select_json_btn.clicked.connect(self.select_json)
        self.start_testing_btn.clicked.connect(self.start_testing)
        self.stop_testing_btn.clicked.connect(self.stop_testing)

        self.show()


    def update_image(self, image_array, figure = 'denoised'):

        if figure == 'raw':
            self.raw_image_queue.append(image_array)
        elif figure == 'denoised':
            self.denoised_image_queue.append(image_array)

    def display_next_image(self):

        if self.raw_image_queue:
            raw_image = self.raw_image_queue.pop(0)
            self.display_image(raw_image, self.image_label_raw)
        if self.denoised_image_queue:
            denoised_image = self.denoised_image_queue.pop(0)
            self.display_image(denoised_image, self.image_label_denoised)

        queue_length = len(self.raw_image_queue)
        self.queue_info_label.setText(f"Queue: {queue_length}")

    def display_image(self, image_array, image_label):

        image_array = normalize(image_array, pmin = 3, pmax = 99.8, clip = True) * 255.0
        image_array = image_array.astype(np.uint8)
        image = QImage(image_array.data, image_array.shape[1], image_array.shape[0], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        image_label.setPixmap(scaled_pixmap)

    def select_json(self):
        json_path, _ = QFileDialog.getOpenFileName(self, "Select Model Json File", "", "JSON Files (*.json)")
        if json_path:
            self.json_path_label.setText(f"json Path: {json_path}")
            config_file = json_path
            self.args = json2args(config_file)
            self.model = Unet_Lite(in_channels = self.args.miniBatch_size,
                                   out_channels = self.args.miniBatch_size,
                                   f_maps = [64, 64, 64],
                                   num_groups = 32,
                                   final_sigmoid = True).cuda()
            self.model.eval()
            print("model is loaded")

    def redirect_stdout(self):
        sys.stdout = EmittingStream(textWritten = self.normalOutputWritten)
        sys.stderr = EmittingStream(textWritten = self.normalOutputWritten)

    def normalOutputWritten(self, text):
        cursor = self.log_text_edit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.log_text_edit.setTextCursor(cursor)
        self.log_text_edit.ensureCursorVisible()

    def select_folder(self):
        self.start_testing_btn.setEnabled(False)
        test_path = QFileDialog.getExistingDirectory(self, "Select test Folder Path")
        if test_path:
            self.args.test_path = test_path

            self.config_path_label.setText(f"test Folder: {test_path}")
            print('test set')
            test_set = ReadDatasets(dataPath = self.args.test_path,
                                    dataType = self.args.data_type,
                                    dataExtension = self.args.data_extension,
                                    mode = 'test',
                                    denoising_strategy = self.args.denoising_strategy)
            test_loader = DataLoader(dataset = test_set, batch_size = self.args.batch_size,
                                     drop_last = True, num_workers = self.args.num_workers, pin_memory = True)
            self.test_loader = test_loader
            self.start_testing_btn.setEnabled(True)

    def start_testing(self):
        if hasattr(self, 'args'):
            self.start_testing_btn.setEnabled(False)
            self.testing_thread = TestingThread(self)
            self.testing_thread.finished.connect(self.on_testing_finished)
            self.testing_thread.imageReady.connect(self.update_image)
            self.testing_thread.start()
        else:
            QMessageBox.warning(self, "Configuration Required",
                                "Please load a configuration file before starting the testing.")

    def stop_testing(self):
        if hasattr(self, 'testing_thread'):
            self.testing_thread.stop()
            self.start_testing_btn.setEnabled(True)
            self.stop_testing_btn.setEnabled(True)

    def on_testing_finished(self):
        self.start_testing_btn.setEnabled(True)  # Re-enable the button after testing is finished


def main():
    app = QApplication(sys.argv)
    ex = TestingGUI()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
