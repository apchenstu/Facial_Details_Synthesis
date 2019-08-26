SET(SOURCE
    src/ImageCapture.cpp
	src/RecorderCSV.cpp
    src/RecorderHOG.cpp
	src/RecorderOpenFace.cpp
    src/RecorderOpenFaceParameters.cpp
	src/SequenceCapture.cpp
	src/stdafx_ut.cpp
	src/VisualizationUtils.cpp
	src/Visualizer.cpp
)

SET(HEADERS
    include/ImageCapture.h	
    include/RecorderCSV.h
	include/RecorderHOG.h
    include/RecorderOpenFace.h
	include/RecorderOpenFaceParameters.h
	include/SequenceCapture.h
	include/stdafx_ut.h
	include/VisualizationUtils.h
	include/Visualizer.h
	include/ConcurrentQueue.h
)

add_library( Utilities ${SOURCE} ${HEADERS})
add_library( OpenFace::Utilities ALIAS Utilities)

target_include_directories(Utilities PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/OpenFace>)

target_include_directories(Utilities PRIVATE ${UTILITIES_SOURCE_DIR}/include)
target_include_directories(Utilities PUBLIC ${OpenCV_INCLUDE_DIRS})

target_link_libraries(Utilities PUBLIC ${OpenCV_LIBS})
target_link_libraries(Utilities PUBLIC dlib::dlib)

if(${Boost_FOUND})
	target_include_directories(Utilities PUBLIC ${Boost_INCLUDE_DIRS})
	target_link_libraries(Utilities PUBLIC ${Boost_LIBRARIES})
else()
	target_link_libraries(Utilities PUBLIC stdc++fs)
endif()

install (TARGETS Utilities EXPORT OpenFaceTargets LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install (FILES ${HEADERS} DESTINATION include/OpenFace)