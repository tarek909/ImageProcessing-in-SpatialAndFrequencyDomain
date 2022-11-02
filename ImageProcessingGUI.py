import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import cv2
from scipy import fftpack
from PyQt5 import QtCore, QtGui, QtWidgets
from PIL import Image, ImageFilter
from scipy import ndimage


def fourier_img(Image):
    dft = np.fft.fft2(Image)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = np.log(1+np.abs(dft_shift))
    return magnitude_spectrum

# reads a BGR image and returns a gray scale image
# GRAY = 0.2125 R + 0.7154 G + 0.0721 B
# Images are read as BGR which means index 0 represents blue value and 2 represents red value
def RGBtoGray(Image):
    if len(Image.shape) == 3:
        return np.dot(Image[..., :3], [0.0721, 0.7154, 0.2125])
    else:
        return Image

# reads a gray image (2Darray) and returns an array of the number of occurrences of each gray-level value
def Histogram(GrayImage):
    Bins = np.zeros(256)
    for RowCounter in range(0, len(GrayImage)):
        for ColumnCounter in range(0, len(GrayImage[0])):
            Bins[int(GrayImage[RowCounter][ColumnCounter])] += 1
    return Bins

# reads BGR or Gray Image and returns an array of the number of occurrences of each value
def HistogramColoredorGray(Image):
    Image = np.asarray(Image)
    if (len(Image.shape) == 3):
        HSVImage = cv2.cvtColor(Image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(HSVImage)   
        Bins = Histogram(v)
    else:
        Bins = Histogram(Image)
    return Bins

# reads a gray image (2Darray) and returns an array of the ratio of occurrences of each gray-level value (relative to the number of pixels)
def NormalizedHistogram(GrayImage):
    Bins = Histogram(GrayImage)
    for BinCounter in range(0, len(Bins)):
        Bins[BinCounter] /= (len(GrayImage) * len(GrayImage[0]))
    return Bins

# reads a histogram array and returns it's cumulative sum array
def CumulativeSum(Histogram):
    CumulativeSumArray = []
    for BinsCounter in range(0, len(Histogram)):
        CumulativeSumArray.append(sum(Histogram[0:BinsCounter]))
    return CumulativeSumArray

# reads a histogram array and returns it's normalized cumulative sum array (0-255)
def NormalizedCumulativeSum(Histogram):
    CumulativeSumArray = CumulativeSum(Histogram)
    CumulativeSumArray = np.asarray(CumulativeSumArray)
    nj = (CumulativeSumArray - min(CumulativeSumArray)) * 255
    N = max(CumulativeSumArray) - min(CumulativeSumArray)
    CumulativeSumArray = nj / N
    return CumulativeSumArray.astype('uint8')

# reads gray image and returns it's histogram equalized gray image
def GrayHistogramEqualize(Image):
    HistoArray = Histogram(Image)
    NCSArray = NormalizedCumulativeSum(HistoArray)
    EqualizedImage = []
    for RowCounter in range(0, len(Image)):
        EqualizedImage.append([])
        for ColumnCounter in range(0, len(Image[0])):
            EqualizedImage[RowCounter].append(NCSArray[int(Image[RowCounter][ColumnCounter])])

    return EqualizedImage

# reads colored image and returns equalized colored image
def ColoredHistogramEqualize(Image):
    HSVImage = cv2.cvtColor(Image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(HSVImage)
    v = GrayHistogramEqualize(v)
    HSVImage = np.transpose([h, s, v], (1, 2, 0))
    EqualizedImage = cv2.cvtColor(HSVImage, cv2.COLOR_HSV2BGR)
    return EqualizedImage

# reads gray image or colored image and returns equalized image of same type
def HistogramEqualizeRGBorGray(Image):
    Image = np.asarray(Image)

    if (len(Image.shape) == 3):
        EqualizedImage = ColoredHistogramEqualize(Image)
        return (EqualizedImage)

    else:
        EqualizedImage = GrayHistogramEqualize(Image)
        return (EqualizedImage)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(950,800)   #old_size(773, 588)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 771, 541))
        self.tabWidget.setObjectName("tabWidget")
        self.tab_1 = QtWidgets.QWidget(self.tabWidget)
        self.tab_1.setObjectName("tab_1")
        self.tab_1.setStyleSheet("background-image: url(backgrnd.png)")

        self.gridLayout_tab1 = QtWidgets.QGridLayout(self.tab_1)
        self.gridLayout_tab1.setObjectName("gridLayout")

        self.label_FiltersEditor = QtWidgets.QLabel(self.tab_1)
        self.label_FiltersEditor.setGeometry(QtCore.QRect(300, -1, 111, 31))
        self.label_FiltersEditor.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_FiltersEditor.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_FiltersEditor.setTextFormat(QtCore.Qt.PlainText)
        self.label_FiltersEditor.setScaledContents(True)
        self.label_FiltersEditor.setAlignment(QtCore.Qt.AlignCenter)
        self.label_FiltersEditor.setWordWrap(False)
        self.label_FiltersEditor.setObjectName("label_FiltersEditor")
        self.label_FiltersEditor.setStyleSheet("background-image: url();border-style: outset; border-width: 2px;border-radius: 10px;border-color: beige;font: bold 14px;color: white;"
                                               " min-width: 10em;padding: 6px;")
        self.gridLayout_tab1.addWidget(self.label_FiltersEditor, 0, 11, 1, 4)

        self.splitter_filters = QtWidgets.QSplitter(self.tab_1)
        self.splitter_filters.setGeometry(QtCore.QRect(20, 40, 721, 411))
        self.splitter_filters.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.splitter_filters.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.splitter_filters.setLineWidth(1)
        self.splitter_filters.setMidLineWidth(0)
        self.splitter_filters.setOrientation(QtCore.Qt.Vertical)
        self.splitter_filters.setObjectName("splitter_filters")
        self.splitter_filters.setStyleSheet(" background-image: url(); background-color: white;")

        self.gridLayout_tab1.addWidget(self.splitter_filters, 1,1,6,22)

        self.label_filters = QtWidgets.QLabel(self.tab_1)
        self.label_filters.setGeometry(QtCore.QRect(490, 470, 51, 21))
        self.label_filters.setTextFormat(QtCore.Qt.PlainText)
        self.label_filters.setScaledContents(False)
        self.label_filters.setObjectName("label_filters")
        self.label_filters.setStyleSheet(" background-image: url(); font: bold 14px; color: white;")
        self.gridLayout_tab1.addWidget(self.label_filters, 7, 17,1,1)

        self.comboBox = QtWidgets.QComboBox(self.tab_1)
        self.comboBox.setObjectName("comboBox")
        self.gridLayout_tab1.addWidget(self.comboBox, 7, 18,1,5)
        self.comboBox.addItem('median')
        self.comboBox.addItem('mean')
        self.comboBox.addItem('laplacian')
        self.comboBox.addItem('low pass filter')
        self.comboBox.addItem('high pass filter')
        self.comboBox.addItem('high pass filter spatial domain')
        self.comboBox.addItem('low pass filter spatial domain')
        self.comboBox.addItem('Gaussian')

        self.comboBox.currentIndexChanged.connect(lambda: self.DisplayFilter(self.comboBox.currentIndex()))
        self.comboBox.setStyleSheet("background-image: url();border: 1px solid gray; border-radius: 3px; padding: 1px 18px 1px 3px; min-width: 6em;")

        self.checkBox_GrayScale = QtWidgets.QCheckBox(self.tab_1)
        self.checkBox_GrayScale.setGeometry(QtCore.QRect(410, 470, 71, 21))
        self.checkBox_GrayScale.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.checkBox_GrayScale.setObjectName("checkBox_GrayScale")
        self.gridLayout_tab1.addWidget(self.checkBox_GrayScale, 7, 15,1,2)
        self.checkBox_GrayScale.setStyleSheet(" background-image: url(); font: bold 14px; color: white;")
        self.checkBox_GrayScale.stateChanged.connect(lambda: self.DisplayFilter(self.comboBox.currentIndex()))

        self.horizontalSlider_kernelsize = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider_kernelsize.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_kernelsize.setObjectName("horizontalSlider_kernelsize")
        self.gridLayout_tab1.addWidget(self.horizontalSlider_kernelsize, 7, 2, 1, 4)
        self.horizontalSlider_kernelsize.setStyleSheet("QSlider::sub-page:Horizontal { background-color: grey ; }"
              "QSlider::add-page:Horizontal { background-color: #333333; }"
              "QSlider::groove:Horizontal { background: transparent; height:4px; background-image: url(); }"
              "QSlider::handle:Horizontal { width:10px; border-radius:5px; background:grey; margin: -5px 0px -5px 0px; }")
        self.horizontalSlider_kernelsize.setMinimum(1)
        self.horizontalSlider_kernelsize.setMaximum(33)
        self.horizontalSlider_kernelsize.setValue(1)
        self.horizontalSlider_kernelsize.setTickInterval(2)
        self.horizontalSlider_kernelsize.setSingleStep(2)
        self.horizontalSlider_kernelsize.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.horizontalSlider_kernelsize.valueChanged.connect(lambda: self.change_label())
        self.horizontalSlider_kernelsize.sliderReleased.connect(lambda: self.DisplayFilter(self.comboBox.currentIndex()))

        self.label_Kernelsize = QtWidgets.QLabel(self.tab_1)
        self.label_Kernelsize.setGeometry(QtCore.QRect(490, 470, 51, 21))
        self.label_Kernelsize.setTextFormat(QtCore.Qt.PlainText)
        self.label_Kernelsize.setScaledContents(False)
        self.label_Kernelsize.setObjectName("label_Kernelsize")
        self.label_Kernelsize.setStyleSheet(" background-image: url(); font: bold 14px; color: white;")
        self.gridLayout_tab1.addWidget(self.label_Kernelsize, 7, 6, 1, 2)

        self.label_size = QtWidgets.QLabel(self.tab_1)
        self.label_size.setGeometry(QtCore.QRect(490, 470, 51, 21))
        self.label_size.setTextFormat(QtCore.Qt.PlainText)
        self.label_size.setScaledContents(False)
        self.label_size.setObjectName("label_size")
        self.label_size.setStyleSheet(" background-image: url(); font: bold 10px; color: white; border-style: outset; border-width: 2px;border-radius: 10px;border-color: beige;")
        self.gridLayout_tab1.addWidget(self.label_size, 7, 8, 1, 1)

        self.tabWidget.addTab(self.tab_1, "")
        self.tab_2 = QtWidgets.QWidget(self.tabWidget)
        self.tab_2.setObjectName("tab_2")
        self.tab_2.setStyleSheet("background-image: url(backgrnd.png)")
        self.gridLayout_tab2 = QtWidgets.QGridLayout(self.tab_2)
        self.gridLayout_tab2.setObjectName("gridLayout")

        self.splitter_histoEq = QtWidgets.QSplitter(self.tab_2)
        self.splitter_histoEq.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.splitter_histoEq.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.splitter_histoEq.setOrientation(QtCore.Qt.Vertical)
        self.splitter_histoEq.setObjectName("splitter_histoEq")
        self.splitter_histoEq.setStyleSheet(" background-image: url(); background-color: white;")
        self.gridLayout_tab2.addWidget(self.splitter_histoEq, 1, 1,6,22)

        self.label_HistoEditor = QtWidgets.QLabel(self.tab_2)
        self.label_HistoEditor.setGeometry(QtCore.QRect(300, 0, 111, 31))
        self.label_HistoEditor.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_HistoEditor.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_HistoEditor.setTextFormat(QtCore.Qt.PlainText)
        self.label_HistoEditor.setScaledContents(True)
        self.label_HistoEditor.setAlignment(QtCore.Qt.AlignCenter)
        self.label_HistoEditor.setWordWrap(False)
        self.label_HistoEditor.setObjectName("label_HistoEditor")
        self.label_HistoEditor.setStyleSheet("background-image: url();border-style: outset; border-width: 2px;border-radius: 10px;border-color: beige;font: bold 14px;color: white;"
                                               " min-width: 10em;padding: 6px;")
        self.gridLayout_tab2.addWidget(self.label_HistoEditor, 0, 10,1,4)

        self.checkBox_Equalize = QtWidgets.QCheckBox(self.tab_2)
        self.checkBox_Equalize.setGeometry(QtCore.QRect(670, 480, 71, 21))
        self.checkBox_Equalize.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.checkBox_Equalize.setObjectName("checkBox_Equalize")
        self.checkBox_Equalize.setStyleSheet(" background-image: url(); font: bold 14px; color: white;")
        self.gridLayout_tab2.addWidget(self.checkBox_Equalize,7,21,1,2)
        self.checkBox_Equalize.stateChanged.connect(lambda: self.Equalize())

        self.checkBox_greyScale2 = QtWidgets.QCheckBox(self.tab_2)
        self.checkBox_greyScale2.setGeometry(QtCore.QRect(670, 480, 71, 21))
        self.checkBox_greyScale2.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.checkBox_greyScale2.setObjectName("checkBox_greyScale2")
        self.checkBox_greyScale2.setStyleSheet(" background-image: url(); font: bold 14px; color: white;")
        self.gridLayout_tab2.addWidget(self.checkBox_greyScale2, 7, 18, 1, 2)
        self.checkBox_greyScale2.stateChanged.connect(lambda: self.Equalize())

        self.tabWidget.addTab(self.tab_2, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 773, 21))
        self.menubar.setObjectName("menubar")
        self.menufile = QtWidgets.QMenu(self.menubar)
        self.menufile.setObjectName("menufile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionopen = QtWidgets.QAction(MainWindow)
        self.actionopen.setObjectName("actionopen")
        self.menufile.addAction(self.actionopen)
        self.menubar.addAction(self.menufile.menuAction())
        self.actionopen.triggered.connect(lambda: self.Open_file())

        self.value_kernels = 1
        self.filter_bool = 0
        self.equalize_bool = 0
        self.isOpened = 0

        self.gridLayout.addWidget(self.tabWidget, 15, 15)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        self.splitter_filters.setFixedHeight(450)
        self.splitter_histoEq.setFixedHeight(450)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Image Editor"))
        self.label_filters.setText(_translate("MainWindow", "Filters :"))
        self.label_Kernelsize.setText(_translate("MainWindow", "Kernel Size :"))
        self.label_size.setText(_translate("MainWindow", "1"))
        self.checkBox_GrayScale.setText(_translate("MainWindow", "Gray Scale"))
        self.label_FiltersEditor.setText(_translate("MainWindow", "Filters Editor"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_1), _translate("MainWindow", "Filters"))
        self.label_HistoEditor.setText(_translate("MainWindow", "Histogram Editor"))
        self.checkBox_Equalize.setText(_translate("MainWindow", "Equalize"))
        self.checkBox_greyScale2.setText(_translate("MainWindow", "Grey Scale"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Histogram"))
        self.menufile.setTitle(_translate("MainWindow", "file"))
        self.actionopen.setText(_translate("MainWindow", "open"))

    def change_label(self):
        if self.horizontalSlider_kernelsize.value()%2 == 0:
            return
        else:
            string_value = str(self.horizontalSlider_kernelsize.value())
            self.value_kernels = self.horizontalSlider_kernelsize.value()
            self.label_size.setText(string_value)


    def Equalize (self):
        if self.isOpened == 0:
            return
        else:
            #update original image to if changes are made "rgb or grey scale"
            image = self.OriginalHisto(self.fileName, self.checkBox_greyScale2.isChecked())
            self.ax1_or_hist.imshow(image, cmap='gray')
            histo = HistogramColoredorGray(image)
            self.ax2_or_hist.clear()
            self.ax2_or_hist.bar(np.arange(len(histo)), histo)
            self.OriginalHisto_Fig.draw()

        if self.equalize_bool == 0:
            self.fig_eq_histo, (self.ax1_eq_histo, self.ax2_eq_histo) = plt.subplots(figsize=(11, 6), ncols=2)
            image = self.OriginalHisto(self.fileName, self.checkBox_greyScale2.isChecked())
            equalized_image = HistogramEqualizeRGBorGray(image)
            self.ax1_eq_histo.imshow(equalized_image, cmap='gray')
            self.ax1_eq_histo.set(title='equalized_image')
            self.ax1_eq_histo.set_axis_off()
            self.ax1_eq_histo.grid()
            equalized_histo = HistogramColoredorGray(equalized_image)
            self.ax2_eq_histo.clear()
            self.ax2_eq_histo.bar(np.arange(len(equalized_histo)), equalized_histo)
            self.ax2_eq_histo.set(title='Equalized Histogram')
            self.ax2_eq_histo.grid()
            self.EqualizedHistoFig = FigureCanvasQTAgg(self.fig_eq_histo)
            self.splitter_histoEq.addWidget(self.EqualizedHistoFig)
            self.equalize_bool = 1

        if self.checkBox_Equalize.isChecked():
            if self.equalize_bool == 1:
                image = self.OriginalHisto(self.fileName, self.checkBox_greyScale2.isChecked())
                equalized_image = HistogramEqualizeRGBorGray(image)
                self.ax1_eq_histo.imshow(equalized_image, cmap='gray')
                equalized_histo = HistogramColoredorGray(equalized_image)
                self.ax2_eq_histo.clear()
                self.ax2_eq_histo.bar(np.arange(len(equalized_histo)), equalized_histo)
                self.EqualizedHistoFig.draw()
                self.EqualizedHistoFig.show()

        else:
            if self.equalize_bool == 1:
                self.EqualizedHistoFig.hide()


    def Open_file(self):
        self.fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, 'Open jpg', QtCore.QDir.rootPath(), 'jpg(*.jpg)')
        if self.isOpened == 1:
            magnitude_spectrum, image_original = self.OriginalImage(self.fileName, self.checkBox_GrayScale.isChecked())
            isGray = self.checkBox_GrayScale.isChecked()
            if (isGray):
                self.ax1_OriginalImage.imshow(image_original, cmap='gray')
            else:
                self.ax1_OriginalImage.imshow(cv2.cvtColor(image_original, cv2.COLOR_HSV2RGB))
            self.ax2_OriginalImage.imshow(magnitude_spectrum, cmap='gray')
            self.OriginalImage_Fig.draw()
            image = self.OriginalHisto(self.fileName, self.checkBox_greyScale2.isChecked())
            self.ax1_or_hist.imshow(image, cmap='gray')
            histo = HistogramColoredorGray(image)
            self.ax2_or_hist.clear()
            self.ax2_or_hist.bar(np.arange(len(histo)), histo)
            self.OriginalHisto_Fig.draw()
            if self.filter_bool == 1:
                self.DisplayFilter(self.comboBox.currentIndex())
            if self.equalize_bool == 1:
                self.Equalize()
        else:
            self.fig_OriginalImage, (self.ax1_OriginalImage,self.ax2_OriginalImage) = plt.subplots(figsize=(11, 6),ncols=2)
            magnitude_spectrum , image_original = self.OriginalImage(self.fileName,self.checkBox_GrayScale.isChecked())
            isGray = self.checkBox_GrayScale.isChecked()
            if (isGray):
                self.ax1_OriginalImage.imshow(image_original, cmap='gray')
            else:
                self.ax1_OriginalImage.imshow(cv2.cvtColor(image_original, cv2.COLOR_HSV2RGB))
            self.ax1_OriginalImage.set(title='Original')
            self.ax2_OriginalImage.imshow(magnitude_spectrum, cmap='gray')
            self.ax2_OriginalImage.set(title='Magnitude Spectrum ')
            self.OriginalImage_Fig = FigureCanvasQTAgg(self.fig_OriginalImage)
            self.splitter_filters.addWidget(self.OriginalImage_Fig)
            #  original image histogram
            self.fig_orHisto, (self.ax1_or_hist, self.ax2_or_hist) = plt.subplots(figsize=(11, 6), ncols=2)
            image = self.OriginalHisto(self.fileName, self.checkBox_greyScale2.isChecked())
            self.ax1_or_hist.imshow(image, cmap='gray')
            self.ax1_or_hist.set(title='image')
            self.ax1_or_hist.set_axis_off()
            histo = HistogramColoredorGray(image)
            self.ax2_or_hist.clear()
            self.ax2_or_hist.bar(np.arange(len(histo)), histo)
            self.ax2_or_hist.set(title='Histogram')
            self.OriginalHisto_Fig = FigureCanvasQTAgg(self.fig_orHisto)
            self.splitter_histoEq.addWidget(self.OriginalHisto_Fig)
            self.isOpened = 1


    def OriginalImage (self,path,isGray):
        original = cv2.imread(path)  # reads the image

        if (isGray):
            original_image = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)  # convert to GRAY
            magnitude_spectrum = fourier_img(original_image)
            # self.ax1.imshow(original_image, cmap='gray')

        else:
            original_image = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)  # convert to HSV
            h, s, v = cv2.split(original_image)
            magnitude_spectrum = fourier_img(v)
            # self.ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_HSV2RGB))
        return magnitude_spectrum,original_image

    def OriginalHisto(self, path , GRAYFLAG):
        image = cv2.imread(path)
        if (GRAYFLAG == 1):
            image = RGBtoGray(image)
        elif (GRAYFLAG == 0):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def DisplayFilter(self,index):
        magnitude_spectrum, image_original = self.OriginalImage(self.fileName, self.checkBox_GrayScale.isChecked())
        isGray = self.checkBox_GrayScale.isChecked()
        self.ax2_OriginalImage.imshow(magnitude_spectrum, cmap='gray')

        if (isGray):
            self.ax1_OriginalImage.imshow(image_original, cmap='gray')
        else:
            self.ax1_OriginalImage.imshow(cv2.cvtColor(image_original, cv2.COLOR_HSV2RGB))

        self.OriginalImage_Fig.draw()
        # filters
        if self.filter_bool == 0:
            self.fig_filters, (self.ax1_filters, self.ax2_filters) = plt.subplots(figsize=(11, 6), ncols=2)

        isGray = self.checkBox_GrayScale.isChecked()
        if index == 0:
            filtered_image,magnitude_spectrum = self.median(self.fileName,self.checkBox_GrayScale.isChecked(),self.value_kernels)
            ax1title='Median Filter'
            ax2title='Magnitude Spectrum'
        elif index == 1:
            filtered_image,magnitude_spectrum = self.mean(self.fileName,self.checkBox_GrayScale.isChecked(),self.value_kernels)
            ax1title = 'Mean Filter'
            ax2title = 'Magnitude Spectrum'
        elif index == 2:
            filtered_image,magnitude_spectrum = self.laplacian(self.fileName,self.checkBox_GrayScale.isChecked(),self.value_kernels)
            ax1title = 'Laplacian Filter'
            ax2title = 'Magnitude Spectrum'
        elif index == 3:
            filtered_image,magnitude_spectrum = self.LPF(self.fileName,self.checkBox_GrayScale.isChecked(),self.value_kernels)
            ax1title = 'Low Pass Filter'
            ax2title = 'Magnitude Spectrum'
        elif index == 4:
            filtered_image, magnitude_spectrum = self.HPF(self.fileName, self.checkBox_GrayScale.isChecked(),self.value_kernels)
            ax1title = 'High Pass Filter'
            ax2title = 'Magnitude Spectrum'
        elif index == 5:
            filtered_image, magnitude_spectrum = self.HPF_kernels(self.fileName, self.checkBox_GrayScale.isChecked(),self.value_kernels)
            ax1title = 'High Pass Filter'
            ax2title = 'Magnitude Spectrum'
        elif index == 6:
            filtered_image, magnitude_spectrum = self.LPF_kernels(self.fileName, self.checkBox_GrayScale.isChecked(),self.value_kernels)
            ax1title = 'Low Pass Filter'
            ax2title = 'Magnitude Spectrum'
        elif index == 7:
            filtered_image, magnitude_spectrum = self.gaussian(self.fileName,self.checkBox_GrayScale.isChecked(),self.value_kernels)
            ax1title = 'Gaussian Filter'
            ax2title = 'Magnitude Spectrum'

        self.ax2_filters.imshow(magnitude_spectrum, cmap='gray')
        if (isGray):
            self.ax1_filters.imshow(filtered_image, cmap='gray')
        else:
            self.ax1_filters.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_HSV2RGB))

        self.ax1_filters.set(title=ax1title)
        self.ax2_filters.set(title=ax2title)
        if self.filter_bool == 0:
            self.FilterImage_Fig = FigureCanvasQTAgg(self.fig_filters)
            self.splitter_filters.addWidget(self.FilterImage_Fig)
        else:
            self.FilterImage_Fig.draw()

        self.filter_bool = 1

    def LPF_kernels(self,path,isGray,KernelSize):
        original = cv2.imread(path)  # reads the image

        if (isGray):
            original_image_GS = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            kernel = np.ones((5, 5), np.float32) / 25
            filtered_image = cv2.filter2D(original_image_GS, -1, kernel)
            magnitude_spectrum = fourier_img(filtered_image)
            return filtered_image,magnitude_spectrum
        else:
            original_image_HSV = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)  # convert to HSV
            kernel = np.ones((5, 5), np.float32) / 25
            filtered_image = cv2.filter2D(original_image_HSV, -1, kernel)
            h, s, v = cv2.split(filtered_image)
            magnitude_spectrum = fourier_img(v)
            return filtered_image,magnitude_spectrum

    def HPF_kernels(self,path,isGray,KernelSize):
        original = cv2.imread(path)  # reads the image
        kernel = np.array([[-1, -1, -1, -1, -1],
                           [-1, 1, 2, 1, -1],
                           [-1, 2, 4, 2, -1],
                           [-1, 1, 2, 1, -1],
                           [-1, -1, -1, -1, -1]])

        if (isGray):
            original_image_GS = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            filtered_image = cv2.filter2D(original_image_GS, -1, kernel)
            magnitude_spectrum = fourier_img(filtered_image)
            return filtered_image,magnitude_spectrum

        else:
            original_image_HSV = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(original_image_HSV)
            filtered_image = cv2.filter2D(v, -1, kernel)
            merged = cv2.merge([h, s, filtered_image])
            magnitude_spectrum = fourier_img(filtered_image)
            return merged,magnitude_spectrum
            # return cv2.cvtColor(merged, cv2.COLOR_HSV2RGB),magnitude_spectrum

    def HPF(self,path,isGray,KernelSize):
        original = cv2.imread(path)  # reads the image

        #####for GrayScale
        GrayScale_imgae = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        Forier = fftpack.fft2((GrayScale_imgae).astype(float))
        Forier = fftpack.fftshift(Forier)
        (w, h) = GrayScale_imgae.shape
        half_w, half_h = int(w / 2), int(h / 2)
        # high pass filter
        n = KernelSize  # to change the area of the cube filter
        Forier[half_w - n:half_w + n + 1, half_h - n:half_h + n + 1] = 0
        Filtered_image = fftpack.ifft2(fftpack.ifftshift(Forier)).real
        Filtered_image = cv2.normalize(Filtered_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)

        #####for RGB
        original_image = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)  # convert to HSV
        h, s, v = cv2.split(original_image)
        V_coloredImage = v
        dft = cv2.dft(np.float32(V_coloredImage), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)  # shift the zero frequncy component to the center
        w, l = V_coloredImage.shape
        CenterW = w // 2
        CenterL = l // 2
        filter = np.ones((w, l, 2), np.uint8)
        filter[CenterW - KernelSize:CenterW + KernelSize, CenterL - KernelSize:CenterL + KernelSize] = 0
        filter = cv2.GaussianBlur(filter, (19, 19), 0)
        shift = dft_shift * filter
        ishift = np.fft.ifftshift(shift)
        filtered_image = cv2.idft(ishift)
        filtered_image = cv2.magnitude(filtered_image[:, :, 0], filtered_image[:, :, 1])
        filtered_image = cv2.normalize(filtered_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
        magnitude_spectrum = fourier_img(Filtered_image)
        if (isGray):
            return filtered_image, magnitude_spectrum
        else:
            RGB_filtered_image = cv2.merge([h, s, filtered_image])
            return RGB_filtered_image, magnitude_spectrum

    def LPF (self, path,isGray,KernelSize):
        original = cv2.imread(path)  # reads the image

        if (isGray):
            image2 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        else:
            original_image2 = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)  # convert to HSV
            h, s, v = cv2.split(original_image2)
            image2 = v
        dft = cv2.dft(np.float32(image2), flags=cv2.DFT_COMPLEX_OUTPUT)
        # shift the zero-frequncy component to the center of the spectrum
        dft_shift = np.fft.fftshift(dft)
        rows, cols = image2.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols, 2), np.uint8)
        cv2.circle(mask, (ccol, crow), KernelSize, (255, 255, 255), -1)
        mask = cv2.GaussianBlur(mask, (19, 19), 0)

        # apply mask and inverse DFT

        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        img_back = cv2.normalize(img_back, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        magnitude_spectrum = fourier_img(img_back)
        if (isGray):
            return img_back,magnitude_spectrum

        else:
            merged = cv2.merge([h, s, img_back])
            return merged, magnitude_spectrum

    def laplacian(self,path,isGray,KernelSize):
        original = cv2.imread(path)  # reads the image
        if KernelSize < 31:
            figure_size = KernelSize
        else:
            figure_size = 31

        if (isGray):
            original_image_GS = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)  # convert to GRAY
            source = cv2.GaussianBlur(original_image_GS, (figure_size, figure_size), 0)
            dest = cv2.Laplacian(source, cv2.CV_16S, ksize=figure_size)
            filtered_image = cv2.convertScaleAbs(dest)
            # self.ax1.imshow(filtered_image, cmap='gray')
            magnitude_spectrum = fourier_img(filtered_image)
            # self.ax2.imshow(magnitude_spectrum, cmap='gray')

        else:
            original_image_HSV = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)  # convert to HSV
            source = cv2.GaussianBlur(original_image_HSV, (figure_size, figure_size), 0)
            dest = cv2.Laplacian(source, cv2.CV_16S, ksize=figure_size)
            filtered_image = cv2.convertScaleAbs(dest)
            # self.ax1.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_HSV2RGB))
            h, s, v = cv2.split(filtered_image)
            magnitude_spectrum = fourier_img(v)
            # self.ax2.imshow(magnitude_spectrum, cmap='gray')
        return filtered_image,magnitude_spectrum

    def gaussian (self,path,isGray,KernelSize):
        original = cv2.imread(path)  # reads the image
        if KernelSize < 31:
            figure_size = KernelSize
        else:
            figure_size = 31

        if (isGray):
            original_image_GS = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)  # convert to GRAY
            source = cv2.GaussianBlur(original_image_GS, (figure_size, figure_size), 5)
            # self.ax1.imshow(filtered_image, cmap='gray')
            filtered_image=source
            magnitude_spectrum = fourier_img(source)
            # self.ax2.imshow(magnitude_spectrum, cmap='gray')

        else:
            original_image_HSV = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)  # convert to HSV
            source = cv2.GaussianBlur(original_image_HSV, (figure_size, figure_size), 0)
            # self.ax1.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_HSV2RGB))
            filtered_image=source
            h, s, v = cv2.split(source)
            magnitude_spectrum = fourier_img(v)
            # self.ax2.imshow(magnitude_spectrum, cmap='gray')
        return filtered_image,magnitude_spectrum

    def mean (self,path,isGray,KernelSize):
        original = cv2.imread(path)  # reads the image

        if (isGray):
            original_image_GS = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)  # convert to GRAY
            filtered_image = cv2.blur(original_image_GS, (KernelSize, KernelSize))
            # self.ax1.imshow(filtered_image, cmap='gray')
            magnitude_spectrum = fourier_img(filtered_image)
            # self.ax2.imshow(magnitude_spectrum, cmap='gray')

        else:
            original_image_HSV = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)  # convert to HSV
            filtered_image = cv2.blur(original_image_HSV, (KernelSize, KernelSize))
            # self.ax1.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_HSV2RGB))
            h, s, v = cv2.split(filtered_image)
            magnitude_spectrum = fourier_img(v)
        return filtered_image,magnitude_spectrum

    def median (self, path,isGray , KernelSize):
        figure_size = KernelSize
        original = cv2.imread(path)  # reads the image

        if (isGray):
            original_image_GS = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)  # convert to GRAY
            filtered_image = cv2.medianBlur(original_image_GS, figure_size)
            # self.ax1.imshow(filtered_image, cmap='gray')
            magnitude_spectrum = fourier_img(filtered_image)
            # self.ax2.imshow(magnitude_spectrum, cmap='gray')

        else:
            original_image_HSV = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)  # convert to HSV
            filtered_image = cv2.medianBlur(original_image_HSV, figure_size)
            # self.ax1.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_HSV2RGB))
            h, s, v = cv2.split(filtered_image)
            magnitude_spectrum = fourier_img(v)
            # self.ax2.imshow(magnitude_spectrum, cmap='gray')
        return filtered_image , magnitude_spectrum

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
