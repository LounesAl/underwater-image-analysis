import sys
from PySide2 import QtCore, QtGui
from PySide2.QtWidgets import (QApplication, QMainWindow, QAction, QToolBar, 
                               QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QMenu, QToolButton, QStackedLayout, QWidget)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialisation de la fenêtre
        self.setWindowTitle("UnderSor")
        self.setGeometry(50, 50, 800, 600)

        # I. Initialisation de la barre d'outils
        self.toolbar = QToolBar("Bar d'outils")
        self.addToolBar(self.toolbar)

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
        
        # I.0 Ajout des actions à la barre d'outils
        self.outils_action = QAction("Outils")
        # I.1 Ajout des sous menu à la barre d'outils
        self.outils_menu = QMenu()
        # Ajouter un bouton au sous menu
        self.outils_tool_button = QToolButton()
        self.fichier_tool_button.setMenu(self.outils_menu)
        self.fichier_tool_button.setPopupMode(QToolButton.InstantPopup)
        self.fichier_tool_button.setDefaultAction(self.outils_action)
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
        self.toolbar.addWidget(self.params_button)
        self.toolbar.addWidget(self.aide_button)
        self.toolbar.addWidget(self.a_propos_button)
        
        pagelayout = QVBoxLayout()
        button_layout = QHBoxLayout()
        # self.stacklayout = QStackedLayout()

        pagelayout.addLayout(button_layout)
        # pagelayout.addLayout(self.stacklayout)

        btn = QPushButton("Calibration")
        btn.pressed.connect(self.calibration)
        button_layout.addWidget(btn)

        btn = QPushButton("Demarrer le programme")
        btn.pressed.connect(self.demarrer)
        button_layout.addWidget(btn)

        widget = QWidget()
        widget.setLayout(pagelayout)
        self.setCentralWidget(widget)

        # # II. Initialisation des boutons pour Calibration ou utiliser l'application
        # self.calibration_button = QPushButton("Calibration", self)
        # self.demarrer_button = QPushButton("Démarrer l'application", self)
        # # II.1 Agrandir les boutons
        # self.calibration_button.setFixedSize(200, 50)
        # self.demarrer_button.setFixedSize(200, 50)
        # # II.2 Positionner les boutons au milieu de la fenêtre
        # #self.calibration_button.move(self.width()/2 - self.calibration_button.width()/2, self.height()/2 - self.calibration_button.height()/2)
        # #self.demarrer_button.move(self.width()/2 - self.demarrer_button.width()/2, self.height()/2 - self.demarrer_button.height()/2 + self.calibration_button.height())
        # # II.3 Initialisation des layout pour disposer les boutons
        # self.h_layout = QHBoxLayout()
        # self.v_layout = QVBoxLayout()
        # # II.4 Ajout des boutons dans le layout
        # self.h_layout.addWidget(self.calibration_button)
        # self.h_layout.addWidget(self.demarrer_button)
        # self.h_layout.setAlignment(QtCore.Qt.AlignCenter)
        # self.h_layout.setSpacing(10)
        # self.v_layout.addLayout(self.h_layout)
        # self.setLayout(self.v_layout)
        # # II.5 Espacement vertical entre les boutons
        # # II.6 Connecter les boutons avec les fonctions
        # self.calibration_button.clicked.connect(self.calibration)
        # self.demarrer_button.clicked.connect(self.demarrer)
    
    def nouveau_fichier(self):
        print("Call nouveau")

    def ouvrir_fichier(self):
        print("Call ouviir")

    def enregistrer_fichier(self):
        print("Call enregistrer")
        
    def exit(self):
        print("Call enregistrer")
    
    def calibration(self):
        print("Call calibration")

    def demarrer(self):
        print("Call demarrer")
    
    def params(self):
        print("Call params")
    
    def aide(self):
        print("Call aide")

    def a_propos(self):
        print("Call a propos")
       
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())






