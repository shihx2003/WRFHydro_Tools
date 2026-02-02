#!/bin/bash
# 可选版本安装 GRIB2 支持库：zlib、libpng、JasPer
# 安装路径
INSTALL_DIR=/data0/home/shihx/software/grib2_intel
BUILD_DIR=$INSTALL_DIR/build

# 可选版本（根据需要修改）
ZLIB_VER=1.3.1
LIBPNG_VER=1.6.54
JASPER_VER=4.2.8

# 编译器环境（Intel OneAPI）
source ~/SetEnv/intel_oneapi_2024.0.2.sh

# 创建安装和构建目录
mkdir -p $INSTALL_DIR
mkdir -p $BUILD_DIR

cp /data0/home/shihx/software/grib2_intel/*gz $BUILD_DIR
# -------------------------
# 1. 安装 zlib
# -------------------------
cd $BUILD_DIR
if [ ! -f zlib-$ZLIB_VER.tar.gz ]; then
    wget https://www.zlib.net/zlib-$ZLIB_VER.tar.gz
fi
rm -rf zlib-$ZLIB_VER
tar -xf zlib-$ZLIB_VER.tar.gz
cd zlib-$ZLIB_VER
./configure --prefix=$INSTALL_DIR
make -j4
make install

# -------------------------
# 2. 安装 libpng
# -------------------------
cd $BUILD_DIR
if [ ! -f libpng-$LIBPNG_VER.tar.gz ]; then
    wget https://download.sourceforge.net/libpng/libpng-$LIBPNG_VER.tar.gz
fi
rm -rf libpng-$LIBPNG_VER
tar -xf libpng-$LIBPNG_VER.tar.gz
cd libpng-$LIBPNG_VER
./configure --prefix=$INSTALL_DIR --with-zlib-prefix=$INSTALL_DIR
make -j4
make install

# -------------------------
# 3. 安装 JasPer (CMake, out-of-source)
# -------------------------

SRC_DIR=$INSTALL_DIR/build/jasper-$JASPER_VER
BLD_DIR=$INSTALL_DIR/build/jasper-$JASPER_VER-build

cd $INSTALL_DIR/build

tar -xf jasper-$JASPER_VER.tar.gz

rm -rf $BLD_DIR
mkdir $BLD_DIR
cd $BLD_DIR

cmake $SRC_DIR \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
  -DJAS_ENABLE_SHARED=ON \
  -DJAS_ENABLE_STATIC=ON \
  -DJAS_ENABLE_OPENGL=OFF \
  -DJAS_ENABLE_DOC=OFF \
  -DJAS_ENABLE_PROGRAMS=OFF \
  -DZLIB_ROOT=$INSTALL_DIR

make -j4
make install


# -------------------------
# 4. 输出环境变量设置（csh/tcsh）
# -------------------------
echo ""
echo "GRIB2 库安装完成！编译 WPS 前，请执行以下 csh/tcsh 命令："
echo "setenv JASPERLIB $INSTALL_DIR/lib"
echo "setenv JASPERINC $INSTALL_DIR/include"
echo "setenv LDFLAGS -L$INSTALL_DIR/lib"
echo "setenv CPPFLAGS -I$INSTALL_DIR/include"
