# ur5_project

Questo pacchetto ROS 2 contiene dei nodi per la pianificazione del movimento di un braccio robotico UR5 e per l'acquisizione di dati da una sorgente visiva. È stato progettato per essere integrato nel container della repository [`ros2_ur5_interface`](https://github.com/pla10/ros2_ur5_interface), che fornisce l'ambiente di simulazione.

## Struttura del progetto

ur5_project/<br>
├── CMakeLists.txt<br>
├── package.xml<br>
├── src/<br>
│ ├── motion_planner_node.cpp # Nodo ROS 2 per pianificare e inviare traiettorie all’UR5<br>
│ ├── high_planning_node.cpp # Nodo con logica di alto livello per il controllo<br>
└── vision/<br>
└── vision_node.py # Nodo Python per l'elaborazione visiva<br>

## Requisiti

- Versione di numpy minore alla 2.0 nel container<br>
  Se necessario eseguire:<br>
  ```bash
  pip install "numpy<2.0"
  pip install ultralytics 
  ```
  O se fallisce:
   ```bash
  pip install "numpy<2.0" --force-reinstall --break-system-packages
  pip install ultralytics --break-system-packages
  ```
  

## Setup del progetto

1. **Clona la repository del container**:
     ```bash
     git clone https://github.com/pla10/ros2_ur5_interface.git
     cd ros2_ur5_interface
      ```
2. Clona questo pacchetto all'interno della cartella  ```src/ ```:<br><br>

   ```bash
    cd src
    git clone https://github.com/eliazov/ur5_project.git
    cd ..
   ```
3. Esegui:

   ```bash
   source /opt/ros/jazzy/setup.bash
   ```
    poi:
  
     ```bash
    colcon build --packages-select ur5_project
     source install/setup.bash
     ```
## Esecuzione:

Lancia il simulatore e controller UR5 (seguendo le istruzioni nella repo ros2_ur5_interface).

Avvia i nodi di questo pacchetto:

Nodo per la pianificazione del movimento:

   ```bash
    ros2 run ur5_project motion_planner_node
   ```
Nodo di controllo di alto livello:
  
   ```bash
    ros2 run ur5_project high_planning_node
   ```
Nodo visivo:

   ```bash
    ros2 run ur5_project vision_node.py
   ```

## Funzionamento
Una volta attivato i nodi eseguire in un nuovo terminale 
   ```bash
   ros2 service call /start_planning std_srvs/srv/Trigger "{}"
   ```
Per avviare il movimento del robot
