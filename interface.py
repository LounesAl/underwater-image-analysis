import sys
from PySide2 import QtCore, QtGui
from utils.ui_fonctions import *
from temporaire.main_copy import seg_img
from PySide2.QtWidgets import *
from PySide2.QtWebEngineWidgets import QWebEngineView


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialisation de la fenêtre
        self.setWindowTitle("UnderSor")
        self.setGeometry(550, 300, 800, 600)

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
        # I.3 Ajouter les actions au sous menu
        self.outils_menu.addAction(self.calib_action)
        self.outils_menu.addAction(self.seg_action)
        # I.4 Associer des fonction au action des sous menu
        self.calib_action.triggered.connect(self.calibration_func_windows)
        self.seg_action.triggered.connect(self.segmentation)
        
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
        
    def segmentation(self):
        self.img_seg = img_seg_window()
        self.img_seg.show()    
        
    def calibration_func_windows(self):
        self.calib = calib_window()
        self.calib.show()
    
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
        
        self.calib_button = QPushButton('Calibrer', self)
        self.calib_button.clicked.connect(lambda: calibration_func(self))
        
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
        
        # Default path
        self.img_path1 = 'data/imgs_c1/gibbula_07_16-1097-_jpg.rf.632cd8777f2b205c4fcd37545105890d.jpg'
        self.img_path2 = 'data/imgs_c2/image1226_jpg.rf.5c88e00836cd73eefb40ade61f5ea77a.jpg'
        
        # Create labels to display "Sélectionner un dossier 1" and "Sélectionner un dossier 2"
        self.label1 = QLabel("Sélectionner l'image depuis la caméra 1")
        self.label2 = QLabel("Sélectionner l'image depuis la caméra 2")
        
        # Create the browse buttons
        self.browse_button1 = QPushButton('Parcourir', self)
        self.browse_button1.clicked.connect(lambda: browse_file(self, 1))
        self.browse_button2 = QPushButton('Parcourir', self)
        self.browse_button2.clicked.connect(lambda: browse_file(self, 2))
        
        self.seg_button = QPushButton('detecter et segmenter', self)
        self.seg_button.clicked.connect(lambda: seg_img(self))
        
        # Create a grid layout
        self.grid = QGridLayout()
        self.grid.setSpacing(10)

        # Add the label and button to the grid layout
        self.grid.addWidget(self.label1, 0, 0)
        self.grid.addWidget(self.browse_button1, 0, 1)
        self.grid.addWidget(self.label2, 1, 0)
        self.grid.addWidget(self.browse_button2, 1, 1)
        self.grid.addWidget(self.seg_button, 2, 0, 1, 2, QtCore.Qt.AlignCenter)
        
        # Create a vertical layout
        self.layout = QVBoxLayout(self)
        self.layout.addLayout(self.grid)
        self.setLayout(self.layout)
        
         
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())