cmake_minimum_required(VERSION 3.8)
project(ur5_eigen_motion)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(trajectory_msgs REQUIRED) # Questo definisce trajectory_msgs_LIBRARIES e trajectory_msgs_INCLUDE_DIRS
find_package(Eigen3 REQUIRED CONFIG)
find_package(control_msgs REQUIRED)      # NUOVO
find_package(rclcpp_action REQUIRED)   # NUOVO
find_package(sensor_msgs REQUIRED)     # NUOVO
find_package(std_srvs REQUIRED)  
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(motion_planner_node src/motion_planner_node.cpp)

# Includi le directory necessarie
# Eigen3::Eigen e rclcpp::rclcpp (se linkati come target) dovrebbero aggiungere i loro include.
# Per trajectory_msgs, lo aggiungiamo esplicitamente.
target_include_directories(motion_planner_node
  PRIVATE
  # ${Eigen3_INCLUDE_DIRS} # Se Eigen3::Eigen non aggiungesse gli include, ma di solito lo fa
  # ${rclcpp_INCLUDE_DIRS} # Se rclcpp::rclcpp non aggiungesse gli include, ma di solito lo fa
  ${trajectory_msgs_INCLUDE_DIRS} # Aggiungiamo esplicitamente gli include per trajectory_msgs
  ${control_msgs_INCLUDE_DIRS}      # NUOVO
  ${sensor_msgs_INCLUDE_DIRS}       # NUOVO
  ${std_srvs_INCLUDE_DIRS}          
  ${geometry_msgs_INCLUDE_DIRS}
  ${std_msgs_INCLUDE_DIRS}
)

# Linka tutte le librerie
target_link_libraries(motion_planner_node
  PRIVATE
  Eigen3::Eigen                     # Target importato da Eigen3
  rclcpp::rclcpp                    # Target importato da rclcpp
  rclcpp_action::rclcpp_action
  ${trajectory_msgs_LIBRARIES}      # Variabile CMake per le librerie di trajectory_msgs
  ${control_msgs_LIBRARIES}         # NUOVO
  ${sensor_msgs_LIBRARIES}          # NUOVO
  ${std_srvs_LIBRARIES}  
  ${geometry_msgs_LIBRARIES}
  ${std_msgs_LIBRARIES}
)

# Commentiamo ancora ament_target_dependencies per questo target
# ament_target_dependencies(motion_planner_node
#   rclcpp
#   trajectory_msgs
# )

install(TARGETS
  motion_planner_node
  DESTINATION lib/${PROJECT_NAME}
)

# --- NUOVO: Eseguibile per High-Level Planner ---
add_executable(high_level_planner_node src/high_level_planner_node.cpp)

# Le directory di include sono probabilmente le stesse del motion_planner_node
# dato che usano tipi di messaggi simili. Se ci sono differenze, specifica.
target_include_directories(high_level_planner_node
  PRIVATE
  ${geometry_msgs_INCLUDE_DIRS}
  ${std_msgs_INCLUDE_DIRS}
  ${std_srvs_INCLUDE_DIRS}
  # Aggiungi altri include se necessari specificamente per questo nodo
)

target_link_libraries(high_level_planner_node
  PRIVATE
  rclcpp::rclcpp # rclcpp è sempre necessario
  ${geometry_msgs_LIBRARIES}
  ${std_msgs_LIBRARIES}
  ${std_srvs_LIBRARIES}
  # Aggiungi altre librerie se necessarie specificamente per questo nodo
)

install(TARGETS
  high_level_planner_node     # Installa il nuovo eseguibile
  DESTINATION lib/${PROJECT_NAME}
)
# --- FINE NUOVO ---

install(PROGRAMS
  vision/vision_node.py
  DESTINATION lib/${PROJECT_NAME}
)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()