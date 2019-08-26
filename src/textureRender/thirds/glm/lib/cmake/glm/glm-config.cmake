get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

add_library(glm::glm INTERFACE IMPORTED)

set_target_properties(glm::glm PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${PACKAGE_PREFIX_DIR}/include"
)

