import sys
from PySide2 import QtCore, QtGui
from calib import main
from utils.ui_fonctions import *
from utils.segmentation import seg_img
from PySide2.QtWidgets import *
import random
import glob
# from PySide2.QtWebEngineWidgets import QWebEngineView

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialisation de la fenêtre
        self.setWindowTitle("UnderSea")
        self.setGeometry(550, 200, 512, 512)
        # Ajouter le logo pour l'application comme une icône
        icons = ["./data/logo/logo.ico", "./data/logo/logo1.ico"]
        icon = random.choice(icons)
        ico = QtGui.QIcon(icon)
        self.setWindowIcon(ico)

        self.label = QLabel(self)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.images = glob.glob("./data/fond_ecran/*.jpeg")
        self.current_image = 0
        self.change_image()
        self.setCentralWidget(self.label)

        # Ajouter un rectangle gris transparent pour le texte de bienvenue
        self.welcome_rect = QLabel(self)
        self.welcome_rect.setAlignment(QtCore.Qt.AlignCenter)
        self.welcome_rect.setGeometry(QtCore.QRect(self.width()/2 - 50, 0, 340, 70))
        self.welcome_rect.setStyleSheet("background-color: rgba(60, 60, 60, 0.9); border-radius:15px;")

        # Ajouter le texte de bienvenue
        self.welcome_label = QLabel("Bienvenue sur UnderSea", self)
        self.welcome_label.setStyleSheet("font: bold 20px; color: rgba(255, 255, 255, 0.97);")
        self.welcome_label.setAlignment(QtCore.Qt.AlignCenter)
        self.welcome_label.setGeometry(self.width()/2 - 125, 0, 500, 100)
        
        # Ajouter un rectangle gris transparent pour le texte de bienvenue
        self.button_rect_g = QLabel(self)
        self.button_rect_g.setAlignment(QtCore.Qt.AlignCenter)
        self.button_rect_g.setGeometry(QtCore.QRect(self.width()/2 - 125, 400, 500, 300))
        self.button_rect_g.setStyleSheet("background-color: rgba(60, 60, 60, 0.9); border-radius:15px;")
        
        # Ajouter le texte de que voullez vous faire
        self.welcome_label = QLabel("Que voullez vous faire ?", self)
        self.welcome_label.setStyleSheet("font: bold 15px; color: rgba(255, 255, 255, 0.9);")
        self.welcome_label.setAlignment(QtCore.Qt.AlignCenter)
        self.welcome_label.setGeometry(self.width()/2 - 125, 280, 500, 300)
        
        # Ajouter les boutons d'action
        self.button1 = QPushButton("Calibration", self)
        self.button1.setGeometry(170, 500, 200, 50)
        self.button1.setStyleSheet("background-color: lightblue; border-radius: 10px; font: bold 15px;")
        self.button1.clicked.connect(self.calibration_func_windows)

        self.button2 = QPushButton("Tracking", self)
        self.button2.setGeometry(400, 500, 200, 50)
        self.button2.setStyleSheet("background-color: lightblue; border-radius: 10px; font: bold 15px;")
        self.button2.clicked.connect(self.tracking)

        self.button3 = QPushButton("Image segmantion", self)
        self.button3.setGeometry(170, 600, 200, 50)
        self.button3.setStyleSheet("background-color: lightblue; border-radius: 10px; font: bold 15px;")
        self.button3.clicked.connect(self.segmentation)

        self.button4 = QPushButton("Video segmentation", self)
        self.button4.setGeometry(400, 600, 200, 50)
        self.button4.setStyleSheet("background-color: lightblue; border-radius: 10px; font: bold 15px;")
        self.button4.clicked.connect(self.video_segmentation)

        # I. Initialisation de la barre d'outils
        self.toolbar = QToolBar("Bar d'outils")
        self.addToolBar(self.toolbar)
        
        # -------------- Fichiers --------------
        # I.0 Ajout des actions à la barre d'outils
        self.fichier_action = QAction("Fichier")
        # I.1 Ajout des sous menu à la barre d'outils
        self.fichier_menu = QMenu()
        # Ajouter un bouton au sous menu
        self.fichier_tool_button = QToolButton()
        self.fichier_tool_button.setMenu(self.fichier_menu)
        self.fichier_tool_button.setPopupMode(QToolButton.InstantPopup)
        self.fichier_tool_button.setDefaultAction(self.fichier_action)
        # I.2 definir les actions du sous menu
        self.nouveau_action = QAction("Nouveau", self)
        self.ouvrir_action = QAction("Ouvrir", self)
        self.enregistrer_action = QAction("Enregistrer", self)
        self.exit_action = QAction("Quitter", self)
        # I.3 Ajouter les actions au sous menu
        self.fichier_menu.addAction(self.nouveau_action)
        self.fichier_menu.addAction(self.ouvrir_action)
        self.fichier_menu.addAction(self.enregistrer_action)
        self.fichier_menu.addAction(self.exit_action)
        # I.4 Associer des fonction au action des sous menu
        self.nouveau_action.triggered.connect(lambda: nouveau_fichier(self))
        self.ouvrir_action.triggered.connect(lambda : ouvrir_fichier(self))
        self.enregistrer_action.triggered.connect(lambda : enregistrer_fichier(self))
        self.exit_action.triggered.connect(lambda: exit(self))
        
        # --------------- Outils-------------------
        # I.0 Ajout des actions à la barre d'outils
        self.outils_action = QAction("Outils")
        # I.1 Ajout des sous menu à la barre d'outils
        self.outils_menu = QMenu()
        # Ajouter un bouton au sous menu
        self.outils_tool_button = QToolButton()
        self.outils_tool_button.setMenu(self.outils_menu)
        self.outils_tool_button.setPopupMode(QToolButton.InstantPopup)
        self.outils_tool_button.setDefaultAction(self.outils_action)
        # I.2 definir les actions du sous menu
        self.calib_action = QAction("Calibration", self)
        self.seg_action = QAction("Image segmentation", self)
        self.vid_seg_action = QAction("Video segmentation", self)
        self.tracking_action = QAction("Tracking", self)
        # I.3 Ajouter les actions au sous menu
        self.outils_menu.addAction(self.calib_action)
        self.outils_menu.addAction(self.seg_action)
        self.outils_menu.addAction(self.vid_seg_action)
        self.outils_menu.addAction(self.tracking_action)
        # I.4 Associer des fonction au action des sous menu
        self.calib_action.triggered.connect(self.calibration_func_windows)
        self.seg_action.triggered.connect(self.segmentation)
        self.vid_seg_action.triggered.connect(self.video_segmentation)
        self.tracking_action.triggered.connect(self.tracking)
        
        # I.5 Ajouter un bouton parametres
        self.params_action = QAction("Parametres")
        self.params_action.triggered.connect(lambda : params(self))
        self.params_button = QToolButton()
        self.params_button.setDefaultAction(self.params_action)
        # I.6 Ajouter un bouton Aide
        self.aide_action = QAction("Aide")
        self.aide_action.triggered.connect(self.aide)
        self.aide_button = QToolButton()
        self.aide_button.setDefaultAction(self.aide_action)
        # I.6 Ajouter un bouton A propos
        self.a_propos_action = QAction("A propos")
        self.a_propos_action.triggered.connect(lambda: a_propos(self))
        self.a_propos_button = QToolButton()
        self.a_propos_button.setDefaultAction(self.a_propos_action)
        # I.7 integrer les bouton dans la barre d'outils
        self.toolbar.addWidget(self.fichier_tool_button)
        self.toolbar.addWidget(self.outils_tool_button)
        self.toolbar.addWidget(self.params_button)
        self.toolbar.addWidget(self.aide_button)
        self.toolbar.addWidget(self.a_propos_button)
        # self.toolbar.setStyleSheet("background-color: rgb(200,200,200);")
        self.toolbar.setStyleSheet("background-color: lightgray;")
            
    def change_image(self):
        # Créer un QPixmap à partir de l'image
        pixmap = QtGui.QPixmap(self.images[self.current_image])
        self.label.setPixmap(pixmap)
        # Planifier l'appel de la fonction de changement d'image suivante après un délai de 1000ms
        QtCore.QTimer.singleShot(3000, self.next_image)

    def next_image(self):
        # Incrémenter l'index de l'image courante
        self.current_image += 1
        if self.current_image >= len(self.images):
            self.current_image = 0
        # Connecter la fin de l'animation à la fonction de changement d'image suivante*
        QtCore.QTimer.singleShot(3000, self.change_image)
        
    def segmentation(self):
        self.img_seg = img_seg_window()
        self.img_seg.show()    
        
    def video_segmentation(self):
        self.vid_seg = vid_seg_window()
        self.vid_seg.show()    
        
    def calibration_func_windows(self):
        self.calib = calib_window()
        self.calib.show()
        
    def tracking(self):
        self.track = tracking_window()
        self.track.show()
    
    def aide(self):
        # Create an instance of the PDFWindow class and pass the path to the PDF file
        pdf_window = PDFWindow("rapport/Rapport.pdf")
        pdf_window.show()
        
class calib_window(QWidget):
    def __init__(self):
        super(calib_window, self).__init__()
        
        self.setGeometry(650, 400, 600, 400)
        self.setWindowTitle("Calibration")
        
        # Default path
        self.folder_path1 = './data/camera0'
        self.folder_path2 = './data/camera1'
        
        # Create labels to display "Sélectionner un dossier 1" and "Sélectionner un dossier 2"
        self.label1 = QLabel("Sélectionner le dossier la caméra 1")
        self.label2 = QLabel("Sélectionner le dossier la caméra 2")
        # Create labels to display "Sélectionner un dossier 1" and "Sélectionner un dossier 2"
        self.label3 = QLabel("Taille d'un carré de la mire (cm)")
        self.label4 = QLabel("Nombre de lignes de la mire")
        self.label5 = QLabel("Nombre de colones de la mire")
        
        # Create the browse buttons
        self.browse_button1 = QPushButton('Parcourir', self)
        self.browse_button1.clicked.connect(lambda: browse_folder(self, 1))
        self.browse_button2 = QPushButton('Parcourir', self)
        self.browse_button2.clicked.connect(lambda: browse_folder(self, 2))
        
        # Create a double spin box with a default value
        self.checkerboard_box_size_scale_box = QDoubleSpinBox()
        self.checkerboard_box_size_scale_box.setSingleStep(0.1)
        self.checkerboard_box_size_scale_box.setValue(2.19)
        # Create a double spin box with a default value
        self.checkerboard_rows_box = QDoubleSpinBox()
        self.checkerboard_rows_box.setValue(6)
        # Create a double spin box with a default value
        self.checkerboard_columns_box = QDoubleSpinBox()
        self.checkerboard_columns_box.setValue(9)
        
        self.calib_button = QPushButton('Calibrer', self)
        self.calib_button.clicked.connect(lambda: main(self.path1, 
                                                       self.path2,
                                                       checkerboard_box_size_scale = self.checkerboard_box_size_scale_box.isChecked(), 
                                                       checkerboard_rows = self.checkerboard_rows_box.isChecked(), 
                                                       checkerboard_columns = self.checkerboard_columns_box.isChecked()))
        
        # Create a grid layout
        self.grid = QGridLayout()
        self.grid.setSpacing(10)

        # Add the label and button to the grid layout
        self.grid.addWidget(self.label1, 0, 0)
        self.grid.addWidget(self.browse_button1, 0, 1)
        self.grid.addWidget(self.label2, 1, 0)
        self.grid.addWidget(self.browse_button2, 1, 1)
        self.grid.addWidget(self.label3, 2, 0)
        self.grid.addWidget(self.checkerboard_box_size_scale_box, 2, 1)
        self.grid.addWidget(self.label4, 3, 0)
        self.grid.addWidget(self.checkerboard_rows_box, 3, 1)
        self.grid.addWidget(self.label5, 4, 0)
        self.grid.addWidget(self.checkerboard_columns_box, 4, 1)
        
        
        
        self.grid.addWidget(self.calib_button, 5, 0, 1, 2, QtCore.Qt.AlignCenter)
        
        # Create a vertical layout
        self.layout = QVBoxLayout(self)
        self.layout.addLayout(self.grid)
        self.setLayout(self.layout)
        
class tracking_window(QWidget):
    def __init__(self):
        super(tracking_window, self).__init__()
        
        self.setGeometry(650, 400, 600, 400)
        self.setWindowTitle("Tracking")
        
        # Default path
        self.folder_path1 = 'data/imgs_c1'
        self.folder_path2 = 'data/imgs_c2'
        
        # Create labels to display "Sélectionner un dossier 1" and "Sélectionner un dossier 2"
        self.label1 = QLabel("Sélectionner le dossier la caméra 1")
        self.label2 = QLabel("Sélectionner le dossier la caméra 2")
        
        # Create the browse buttons
        self.browse_button1 = QPushButton('Parcourir', self)
        self.browse_button1.clicked.connect(lambda: browse_folder(self, 1))
        self.browse_button2 = QPushButton('Parcourir', self)
        self.browse_button2.clicked.connect(lambda: browse_folder(self, 2))
        
        self.calib_button = QPushButton('Demarrer', self)
        # self.calib_button.clicked.connect(lambda: main(self.path1, self.path2))
        
        # Create a grid layout
        self.grid = QGridLayout()
        self.grid.setSpacing(10)

        # Add the label and button to the grid layout
        self.grid.addWidget(self.label1, 0, 0)
        self.grid.addWidget(self.browse_button1, 0, 1)
        self.grid.addWidget(self.label2, 1, 0)
        self.grid.addWidget(self.browse_button2, 1, 1)
        self.grid.addWidget(self.calib_button, 2, 0, 1, 2, QtCore.Qt.AlignCenter)
        
        # Create a vertical layout
        self.layout = QVBoxLayout(self)
        self.layout.addLayout(self.grid)
        self.setLayout(self.layout)

class PDFWindow(QWidget):
    def __init__(self, pdf_path):
        super().__init__()
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle("Aide")

        # Create a QWebEngineView widget to display the PDF
        self.web_view = QWebEngineView(self)

        # Load the PDF file into the widget
        pdf_file = QtCore.QFile(pdf_path)
        pdf_file.open(QtCore.QIODevice.ReadOnly)
        self.web_view.setHtml(pdf_file.readAll().data().decode('latin-1'))

        # Create a vertical layout for the window
        layout = QVBoxLayout(self)
        # Add the QWebEngineView widget to the layout
        layout.addWidget(self.web_view)

class img_seg_window(QWidget):
    def __init__(self):
        super(img_seg_window, self).__init__()
        
        self.setGeometry(650, 400, 600, 400)
        self.setWindowTitle("Image segmentation")
        # Initialiser de la variable erreur
        self.error = error()
        
        # Default path
        self.path1 = 'data/imgs_c0/GOPR1471.JPG'
        self.path2 = 'data/imgs_c1/GOPR1489.JPG'
        
        # Create labels to display "Sélectionner un dossier 1" and "Sélectionner un dossier 2"
        self.label1 = QLabel("Sélectionner l'image depuis la caméra 1")
        self.label2 = QLabel("Sélectionner l'image depuis la caméra 2")
        self.label3 = QLabel("Definir le seuil de detection")
        self.label4 = QLabel("Afficher la segmentation")
        self.label5 = QLabel("Afficher la presentation en 3D")
        self.label6 = QLabel("Afficher l'image avec les dimentions")
        
        # Create the browse buttons
        self.browse_button1 = QPushButton('Parcourir', self)
        self.browse_button1.clicked.connect(lambda: browse_file(self, 1))
        self.browse_button2 = QPushButton('Parcourir', self)
        self.browse_button2.clicked.connect(lambda: browse_file(self, 2))
        
        self.seg_button = QPushButton('Detecter et segmenter', self)
        self.seg_button.clicked.connect(self.segmentation)
        
        # Create a grid layout
        self.grid = QGridLayout()
        self.grid.setSpacing(10)
        
        # Create a checkbox
        self.checkbox = QCheckBox("Avant d'afficher la distance", self)
        self.checkbox1 = QCheckBox("en 3D", self)
        self.checkbox2 = QCheckBox("FInal", self, checked=True)

        # Add the label and button to the grid layout
        self.grid.addWidget(self.label1, 0, 0)
        self.grid.addWidget(self.browse_button1, 0, 1)
        self.grid.addWidget(self.label2, 1, 0)
        self.grid.addWidget(self.browse_button2, 1, 1)
        self.grid.addWidget(self.seg_button, 6, 0, 1, 2, QtCore.Qt.AlignCenter)
        
        # Create a double spin box with a default value of 0.5
        self.double_spin_box = QDoubleSpinBox()
        self.double_spin_box.setMinimum(0.1)
        self.double_spin_box.setMaximum(1)
        self.double_spin_box.setSingleStep(0.1)
        self.double_spin_box.setValue(0.8)

        # Add the double spin box to the grid layout
        self.grid.addWidget(self.label3, 2, 0)
        self.grid.addWidget(self.double_spin_box, 2, 1, 1, 1)
        self.grid.addWidget(self.label4, 3, 0)
        self.grid.addWidget(self.checkbox, 3, 1)
        self.grid.addWidget(self.label5, 4, 0)
        self.grid.addWidget(self.checkbox1, 4, 1)
        self.grid.addWidget(self.label6, 5, 0)
        self.grid.addWidget(self.checkbox2, 5, 1)

        # Create a vertical layout
        self.layout = QVBoxLayout(self)
        self.layout.addLayout(self.grid)
        self.setLayout(self.layout)
        
    def segmentation(self):
        seg_img(self, SCORE_THRESH_TEST = self.double_spin_box.value(), 
                        show_inf = self.checkbox.isChecked(), 
                        show_3d = self.checkbox1.isChecked(), 
                        show_final = self.checkbox2.isChecked() )
        
        if self.error.check_error:
            dlg = QMessageBox(self)
            dlg.setWindowTitle("Erreur lors de la segmentation")
            dlg.setText(self.error.error_msg)
            button = dlg.exec_()
            if button == QMessageBox.Ok:
                self.error.check_error = False
                self.error.error_msg = ''
                
class error():
    def __init__(self):
        self.error_msg = ''
        self.check_error = False

class vid_seg_window(QWidget):
    def __init__(self):
        super(vid_seg_window, self).__init__()
        
        self.setGeometry(650, 400, 600, 400)
        self.setWindowTitle("Video segmentation")
        # Initialiser de la variable erreur
        self.error = error()
        
        # Default path
        self.path1 = 'data/imgs_c1/image1842_jpg.rf.fc8794fa0e45edf088fbd294b4b66188.jpg'
        self.path2 = 'data/imgs_c1/image1856_jpg.rf.7ced282377b524c9ed4e9c301f4ce73c.jpg'
        
        # Create labels to display "Sélectionner un dossier 1" and "Sélectionner un dossier 2"
        self.label1 = QLabel("Sélectionner la video depuis la caméra 1")
        self.label2 = QLabel("Sélectionner la video depuis la caméra 2")
        self.label3 = QLabel("Definir le seuil de detection")
        self.label4 = QLabel("Afficher la segmentation")
        self.label5 = QLabel("Afficher la presentation en 3D")
        self.label6 = QLabel("Afficher l'image avec les dimentions")
        
        # Create the browse buttons
        self.browse_button1 = QPushButton('Parcourir', self)
        self.browse_button1.clicked.connect(lambda: browse_file(self, 1))
        self.browse_button2 = QPushButton('Parcourir', self)
        self.browse_button2.clicked.connect(lambda: browse_file(self, 2))
        
        self.seg_button = QPushButton('Segmenter les videos', self)
        self.seg_button.clicked.connect(self.vid_segmentation)
        
        # Create a grid layout
        self.grid = QGridLayout()
        self.grid.setSpacing(10)
        
        # Create a checkbox
        self.checkbox = QCheckBox("Avant d'afficher la distance", self)
        self.checkbox1 = QCheckBox("en 3D", self)
        self.checkbox2 = QCheckBox("Final", self, checked=True)

        # Add the label and button to the grid layout
        self.grid.addWidget(self.label1, 0, 0)
        self.grid.addWidget(self.browse_button1, 0, 1)
        self.grid.addWidget(self.label2, 1, 0)
        self.grid.addWidget(self.browse_button2, 1, 1)
        self.grid.addWidget(self.seg_button, 6, 0, 1, 2, QtCore.Qt.AlignCenter)
        
        # Create a double spin box with a default value of 0.5
        self.double_spin_box = QDoubleSpinBox()
        self.double_spin_box.setMinimum(0.1)
        self.double_spin_box.setMaximum(1)
        self.double_spin_box.setSingleStep(0.1)
        self.double_spin_box.setValue(0.8)

        # Add the double spin box to the grid layout
        self.grid.addWidget(self.label3, 2, 0)
        self.grid.addWidget(self.double_spin_box, 2, 1, 1, 1)
        self.grid.addWidget(self.label4, 3, 0)
        self.grid.addWidget(self.checkbox, 3, 1)
        self.grid.addWidget(self.label5, 4, 0)
        self.grid.addWidget(self.checkbox1, 4, 1)
        self.grid.addWidget(self.label6, 5, 0)
        self.grid.addWidget(self.checkbox2, 5, 1)

        # Create a vertical layout
        self.layout = QVBoxLayout(self)
        self.layout.addLayout(self.grid)
        self.setLayout(self.layout)
        
    def vid_segmentation(self):
        seg_img(self, SCORE_THRESH_TEST = self.double_spin_box.value(), 
                        show_inf = self.checkbox.isChecked(), 
                        show_3d = self.checkbox1.isChecked(), 
                        show_final = self.checkbox2.isChecked() )
        
        if self.error.check_error:
            dlg = QMessageBox(self)
            dlg.setWindowTitle("Erreur lors de la segmentation")
            dlg.setText(self.error.error_msg)
            button = dlg.exec_()
            if button == QMessageBox.Ok:
                self.error.check_error = False
                self.error.error_msg = ''
         
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())