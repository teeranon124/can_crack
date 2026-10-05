@echo off
setlocal EnableDelayedExpansion

echo ===================================================================
echo   Building Canned Food Crack Inspector (C++ / OpenCV 4)
echo ===================================================================

:: Check for CMake
where cmake >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] Searching for Visual Studio CMake...
    set "VS_CMAKE=C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
    if exist "!VS_CMAKE!" (
        set "CMAKE_EXE=!VS_CMAKE!"
    ) else (
        echo [ERROR] CMake not found. Please install CMake or Visual Studio C++.
        exit /b 1
    )
) else (
    set "CMAKE_EXE=cmake"
)

echo [1/3] Configuring project with CMake...
"!CMAKE_EXE!" -B build -S . -DOpenCV_DIR="C:/opencv/build/x64/vc16/lib"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] CMake configuration failed.
    exit /b 1
)

echo.
echo [2/3] Compiling Release executable...
"!CMAKE_EXE!" --build build --config Release
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Build compilation failed.
    exit /b 1
)

echo.
echo [3/3] Deploying OpenCV runtime DLL...
if exist "C:\opencv\build\x64\vc16\bin\opencv_world4120.dll" (
    copy /Y "C:\opencv\build\x64\vc16\bin\opencv_world4120.dll" "build\bin\Release\" >nul
    echo Copied opencv_world4120.dll to build\bin\Release\
)

echo.
echo ===================================================================
echo [SUCCESS] Build completed!
echo Executable located at: build\bin\Release\can_crack_inspector.exe
echo Run example: build\bin\Release\can_crack_inspector.exe data\samples\crack1.jpg models\model.xml
echo ===================================================================
