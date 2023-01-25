import sys
from PySide2.QtWidgets import (QApplication, QMainWindow, QAction, QToolBar, 
                               QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QMenu, QToolButton)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialisation de la fenêtre
        self.setWindowTitle("Mon application")
        self.setGeometry(50, 50, 800, 600)

        # Initialisation de la barre d'outils
        self.toolbar = QToolBar("Bar d'outils")
        self.addToolBar(self.toolbar)

        # # Ajout des actions à la barre d'outils
        self.fichier_action = QAction("Fichier")
        # Ajout des sous menu à la barre d'outils
        self.fichier_menu = QMenu()
        # Ajouter un bouton au sous menu
        self.fichier_tool_button = QToolButton()
        self.fichier_tool_button.setMenu(self.fichier_menu)
        self.fichier_tool_button.setPopupMode(QToolButton.InstantPopup)
        self.fichier_tool_button.setDefaultAction(self.fichier_action)
        # definir les actions du sous menu
        self.nouveau_action = QAction("Nouveau", self)
        self.ouvrir_action = QAction("Ouvrir", self)
        self.enregistrer_action = QAction("Enregistrer", self)
        # Ajouter les actions au sous menu
        self.fichier_menu.addAction(self.nouveau_action)
        self.fichier_menu.addAction(self.ouvrir_action)
        self.fichier_menu.addAction(self.enregistrer_action)
        # Associer des fonction au action des sous menu
        self.nouveau_action.triggered.connect(self.nouveau_fichier)
        self.ouvrir_action.triggered.connect(self.ouvrir_fichier)
        self.enregistrer_action.triggered.connect(self.enregistrer_fichier)

        # Ajouter un bouton parametres
        self.params_action = QAction("Parametres")
        self.params_action.triggered.connect(self.params)
        self.params_button = QToolButton()
        self.params_button.setDefaultAction(self.params_action)
        
        # Ajouter un bouton Aide
        self.aide_action = QAction("Aide")
        self.aide_action.triggered.connect(self.aide)
        self.aide_button = QToolButton()
        self.aide_button.setDefaultAction(self.aide_action)
        
        # Ajouter un bouton A propos
        self.a_propos_action = QAction("A propos")
        self.a_propos_action.triggered.connect(self.a_propos)
        self.a_propos_button = QToolButton()
        self.a_propos_button.setDefaultAction(self.a_propos_action)
        
        # integrer les bouton dans la barre d'outils
        self.toolbar.addWidget(self.fichier_tool_button)
        self.toolbar.addWidget(self.params_button)
        self.toolbar.addWidget(self.aide_button)
        self.toolbar.addWidget(self.a_propos_button)

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
    
    def nouveau_fichier(self):
        pass

    def ouvrir_fichier(self):
        pass

    def enregistrer_fichier(self):
        pass
    
    def calibration(self):
        pass

    def demarrer(self):
        pass
    
    def params(self):
        pass
    
    def aide(self):
        pass

    def a_propos(self):
        pass
       
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())






