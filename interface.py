import sys
from PySide2 import QtCore, QtGui
from PySide2.QtWidgets import (QApplication, QMainWindow, QAction, QToolBar, 
                               QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QMenu, QToolButton, QStackedLayout, QWidget, QMessageBox, QFileDialog)

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
        self.calib_action.triggered.connect(self.calibration_main)
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
    
    def calibration_main(self):
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
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """
    def __init__(self):
        super().__init__()
        self.setGeometry(650, 400, 600, 400)
        layout = QVBoxLayout()
        self.label = QLabel("Another Window")
        layout.addWidget(self.label)
        self.setLayout(layout)
        button1 = QPushButton("Calibrer")
        button1.clicked.connect(self.select_intput_file)
        layout.addWidget(button1)
        
    def calibration(self):
        print("En train de calibrer ... ")
        path=str(self.setOpenFileName('Shapefile (*.shp)'))
        
        
    def select_intput_file(self):
            """
    
            :return:
            """
            self.filename = str(QFileDialog.getExistingDirectory(
                self.dlg, "Select input file "))
            self.dlg.lineEdit.setText(self.filename)
            self.filename =  self.filename + "/"
       
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())