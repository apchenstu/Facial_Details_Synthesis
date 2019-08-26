#LandmarkDetector library
include_directories(../../local/LandmarkDetector/include)

#Utilities library
include_directories(../../local/Utilities/include)

SET(SOURCE
    src/Face_utils.cpp
	src/FaceAnalyser.cpp
	src/FaceAnalyserParameters.cpp
	src/stdafx_fa.cpp
	src/SVM_dynamic_lin.cpp
	src/SVM_static_lin.cpp
	src/SVR_dynamic_lin_regressors.cpp
	src/SVR_static_lin_regressors.cpp
)

SET(HEADERS
    include/Face_utils.h	
	include/FaceAnalyser.h
	include/FaceAnalyserParameters.h
	include/stdafx_fa.h
	include/SVM_dynamic_lin.h
	include/SVM_static_lin.h
	include/SVR_dynamic_lin_regressors.h
	include/SVR_static_lin_regressors.h
)


add_library( FaceAnalyser ${SOURCE} ${HEADERS})
add_library( OpenFace::FaceAnalyser ALIAS FaceAnalyser)

target_include_directories(FaceAnalyser PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/OpenFace>)

target_include_directories(FaceAnalyser PRIVATE ${FACEANALYSER_SOURCE_DIR}/include)
target_include_directories(FaceAnalyser PUBLIC ${OpenCV_INCLUDE_DIRS})

target_link_libraries(FaceAnalyser PUBLIC ${OpenCV_LIBS} ${OpenBLAS_LIB})
target_link_libraries(FaceAnalyser PUBLIC dlib::dlib)

if(${Boost_FOUND})
	target_include_directories(FaceAnalyser PUBLIC ${Boost_INCLUDE_DIRS})
	target_link_libraries(FaceAnalyser PUBLIC ${Boost_LIBRARIES})
else()
	target_link_libraries(FaceAnalyser PUBLIC stdc++fs)
endif()

target_include_directories(FaceAnalyser PRIVATE ${OpenBLAS_INCLUDE_DIR})

install (TARGETS FaceAnalyser EXPORT OpenFaceTargets LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install (FILES ${HEADERS} DESTINATION include/OpenFace)