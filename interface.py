import sys
from PySide2 import QtCore, QtGui
from PySide2.QtWidgets import (QApplication, QMainWindow, QAction, QToolBar, QGridLayout,
                               QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QMenu, QToolButton, QFileDialog, QWidget, QMessageBox, QFileDialog)

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
        self.nouveau_action.triggered.connect(self.nouveau_fichier)
        self.ouvrir_action.triggered.connect(self.ouvrir_fichier)
        self.enregistrer_action.triggered.connect(self.enregistrer_fichier)
        self.exit_action.triggered.connect(self.exit)
        
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
        self.seg_action = QAction("Segmentation", self)
        # I.3 Ajouter les actions au sous menu
        self.outils_menu.addAction(self.calib_action)
        self.outils_menu.addAction(self.seg_action)
        # I.4 Associer des fonction au action des sous menu
        self.calib_action.triggered.connect(self.calibration_func_windows)
        self.seg_action.triggered.connect(self.segmentation)
        
        # I.5 Ajouter un bouton parametres
        self.params_action = QAction("Parametres")
        self.params_action.triggered.connect(self.params)
        self.params_button = QToolButton()
        self.params_button.setDefaultAction(self.params_action)
        # I.6 Ajouter un bouton Aide
        self.aide_action = QAction("Aide")
        self.aide_action.triggered.connect(self.aide)
        self.aide_button = QToolButton()
        self.aide_button.setDefaultAction(self.aide_action)
        # I.6 Ajouter un bouton A propos
        self.a_propos_action = QAction("A propos")
        self.a_propos_action.triggered.connect(self.a_propos)
        self.a_propos_button = QToolButton()
        self.a_propos_button.setDefaultAction(self.a_propos_action)
        # I.7 integrer les bouton dans la barre d'outils
        self.toolbar.addWidget(self.fichier_tool_button)
        self.toolbar.addWidget(self.outils_tool_button)
        self.toolbar.addWidget(self.params_button)
        self.toolbar.addWidget(self.aide_button)
        self.toolbar.addWidget(self.a_propos_button)
    
    def nouveau_fichier(self):
        print("Call nouveau")

    def ouvrir_fichier(self):
        print("Call ouviir")

    def enregistrer_fichier(self):
        print("Call enregistrer")
        
    def exit(self):
        QApplication.exit()
    
    def calibration_func_windows(self):
        self.calib = calib_window()
        self.calib.show()
        
    def segmentation(self):
        print("Segmentation")

    def demarrer(self):
        print("Call demarrer")
    
    def params(self):
        print("Call params")
    
    def aide(self):
        print("Call aide")

    def a_propos(self):
        dlg = QMessageBox(self)
        dlg.setWindowTitle("A propos de l'application")
        dlg.setText("Cette application est créer par un groupe d'étudiants de \n                         Sorbonne Sniversité.\n                         Tous droits reservés")
        button = dlg.exec_()
        if button == QMessageBox.Ok:
            print("OK!")
        
class calib_window(QWidget):
    def __init__(self):
        super(calib_window, self).__init__()
        
        self.setGeometry(650, 400, 600, 400)
        self.setWindowTitle("Calibration")
        
        # Default path
        self.folder_path1 = 'data/imgs_c1'
        self.folder_path2 = 'data/imgs_c2'
        
        # Create labels to display "Sélectionner un dossier 1" and "Sélectionner un dossier 2"
        self.label1 = QLabel("Sélectionner un dossier 1")
        self.label2 = QLabel("Sélectionner un dossier 2")
        
        # Create the browse buttons
        self.browse_button1 = QPushButton('Parcourir 1', self)
        self.browse_button1.clicked.connect(lambda: self.browse_folder(1))
        self.browse_button2 = QPushButton('Parcourir 2', self)
        self.browse_button2.clicked.connect(lambda: self.browse_folder(2))
        
        self.calib_button = QPushButton('Calibrer', self)
        self.calib_button.clicked.connect(self.calibration_func)
        
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


    def browse_folder(self, button_id):
        folder_path = QFileDialog.getExistingDirectory(self, 'Sélectionner un dossier', '', QFileDialog.ShowDirsOnly)
        if folder_path:
            if button_id == 1:
                self.folder_path1 = folder_path
            elif button_id == 2:
                self.folder_path2 = folder_path
                
    def calibration_func(self):
        # calib(self.folder_path1, self.folder_path2)
        print(f"path1 : {self.folder_path1} \npath2 : {self.folder_path2}")

       
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())