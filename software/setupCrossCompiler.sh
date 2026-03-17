set -e
#sudo mkdir -p /opt/de10-nano
#sudo chown $USER /opt/de10-nano
#sshfs root@192.168.0.109:/ /opt/de10-nano # might need to change this

cd ~/

cat > ds5-toolchain.cmake <<EOF
# takes NMPI_ROOT and NLC_ROOT from environment
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_FIND_ROOT_PATH  /opt/de10-nano)

set(tools $HOME/gcc-linaro-7.5.0-2019.12-x86_64_arm-linux-gnueabihf/bin/arm-linux-gnueabihf)

set(CMAKE_C_COMPILER \${tools}-gcc CACHE STRING "")
set(CMAKE_CXX_COMPILER \${tools}-g++ CACHE STRING "")
set(CMAKE_ASM_COMPILER \${tools}-as CACHE STRING "")
set(CMAKE_LINKER \${tools}-ld CACHE STRING "")
set(CMAKE_STRIP \${tools}-strip CACHE STRING "")
set(CMAKE_AR \${tools}-ar CACHE STRING "")
set(CMAKE_RANLIB \${tools}-ranlib CACHE STRING "")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
EOF


wget https://releases.linaro.org/components/toolchain/binaries/7.5-2019.12/arm-linux-gnueabihf/gcc-linaro-7.5.0-2019.12-x86_64_arm-linux-gnueabihf.tar.xz
tar xvf gcc-linaro-7.5.0-2019.12-x86_64_arm-linux-gnueabihf.tar.xz

