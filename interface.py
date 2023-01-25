import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QAction, QToolBar,
                             QHBoxLayout, QVBoxLayout, QPushButton, QLabel)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialisation de la fenêtre
        self.setWindowTitle("Mon application")
        self.setGeometry(50, 50, 800, 600)
        self.toolbar = QToolBar("Bar d'outils")
        self.setWindowFlags(self.windowFlags() # reuse initial flags
            & ~Qt.WindowContextHelpButtonHint) # negate the flag you want to unset)
        #self.setWindowFlags(Qt.WindowCloseButtonHint)
        # Initialisation de la barre d'outils
        
        self.addToolBar(self.toolbar)

        # Ajout des actions à la barre d'outils
        self.fichier_action = QAction("Fichier", self)
        self.parametres_action = QAction("Paramètres", self)
        self.aide_action = QAction("Aide", self)
        self.toolbar.addAction(self.fichier_action)
        self.toolbar.addAction(self.parametres_action)
        self.toolbar.addAction(self.aide_action)
        
        # # Initialisation des boutons pour Calibration ou utiliser l'application
        # self.calibration_button = QPushButton("Calibration", self)
        # self.demarrer_button = QPushButton("Démarrer", self)
        
        # # Initialisation des layout pour disposer les boutons
        # self.h_layout = QHBoxLayout()
        # self.v_layout = QVBoxLayout()
        
        # # Ajout des boutons dans le layout
        # self.h_layout.addWidget(self.calibration_button)
        # self.h_layout.addWidget(self.demarrer_button)
        # self.v_layout.addLayout(self.h_layout)

        # self.setLayout(self.v_layout)
        
        # # Connecter les boutons avec les fonctions
        # self.calibration_button.clicked.connect(self.calibration)
        # self.demarrer_button.clicked.connect(self.demarrer)
        
    def calibration(self):
        """Fonction pour la calibration de l'application"""
        pass

    def demarrer(self):
        """Fonction pour démarrer l'application"""
        pass
    
if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
    
    
# from PyQt5.QtWidgets import QMainWindow, QApplication, QAction, QToolBar
# from PyQt5 import QtGui

# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Example PyQt Application")
#         self.setGeometry(100, 100, 800, 600)

#         # Create toolbar
#         self.toolbar = QToolBar("Toolbar")
#         self.addToolBar(self.toolbar)

#         # Create actions for toolbar buttons
#         self.file_action = QAction(QtGui.QIcon("file.png"), "Fichier", self)
#         self.settings_action = QAction(QtGui.QIcon("settings.png"), "Paramètres", self)
#         self.help_action = QAction(QtGui.QIcon("help.png"), "Aide", self)

#         # Add actions to toolbar
#         self.toolbar.addAction(self.file_action)
#         self.toolbar.addAction(self.settings_action)
#         self.toolbar.addAction(self.help_action)






