SOURCES += \
    main.cpp
unix:{
LIBS += -lopencv_highgui -lopencv_core -lopencv_imgproc
}
win32:{
LIBS += "C:\opencv\build\x86\mingw\lib\libopencv_core242.dll.a"
LIBS += "C:\opencv\build\x86\mingw\lib\libopencv_imgproc242.dll.a"
LIBS += "C:\opencv\build\x86\mingw\lib\libopencv_highgui242.dll.a"
INCLUDEPATH += "C:\opencv\build\include"
}
