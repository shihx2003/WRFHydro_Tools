#!/bin/sh
#
# install_netcdf.sh
# Copyright (C) 2018 Daniel Peláez-Zapata <daniel.pelaez-zapata@ucdconnect.ie>
#
# Distributed under terms of the GNU/GPL license.
#
set -e


# ============================================================================
#  Installation of NetCDF4 Fortran libraries
# ----------------------------------------------------------------------------
#  
#  Purpose:
#    This script get the given versions of the NetCD4 libreries and its
#    dependencies and install them in the MAINDIR=/usr/local/netcdf/ directory
# 
#  Usage:
#    [sudo] CC=gcc FC=gfortran MAINDIR=/usr/local/netcdf ./install_netcdf.sh
# 
#  Autor:
#    Daniel Peláez-Zapata
#    github/dspelaez
#
# ============================================================================

## define compilers
echo $CC
echo $CXX
echo $FC
echo $F77
echo $F90
MAINDIR=/data0/home/shihx/software/netcdf_intel
# main directory
MAINDIR=${MAINDIR:-./netcdf}
MAINDIR=$(realpath $MAINDIR)
mkdir -p $MAINDIR
echo " --->> Creating directory $MAINDIR"

# version of libs
CLTAG="8.10.0"
ZLTAG="1.3.1"
H5TAG="1.14.5"
NCTAG="4.9.2"
NFTAG="4.6.1"

# ## donwload source code of depencies
wget -nc -nv "https://curl.haxx.se/download/curl-$CLTAG.tar.gz"
wget -nc -nv "https://zlib.net/fossils/zlib-$ZLTAG.tar.gz"
wget -nc -nv "https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-${H5TAG%.*}/hdf5-$H5TAG/src/hdf5-$H5TAG.tar"
wget -nc -nv "https://downloads.unidata.ucar.edu/netcdf-c/$NCTAG/netcdf-c-$NCTAG.tar.gz"
wget -nc -nv "https://downloads.unidata.ucar.edu/netcdf-fortran/$NFTAG/netcdf-fortran-$NFTAG.tar.gz"

# ## create config.log
# touch config_curl.log
# touch config_zlib.log
# touch config_hdf5.log
# touch config_nc.log
# touch config_nf.log

## curl
tar -xf curl-$CLTAG.tar.gz
cd curl-$CLTAG/
CLDIR=$MAINDIR
echo " --->> Compiling curl-$CLTAG"
./configure --prefix=${CLDIR} --without-ssl --without-libpsl > ../config_curl_1.log 2>&1
make -j4 > ../config_curl_2.log 2>&1
make install > ../config_curl_3.log 2>&1
cd ..
# rm -rf curl-$CLTAG


## zlib 
tar -xf zlib-$ZLTAG.tar.gz
cd zlib-$ZLTAG/
ZDIR=$MAINDIR
echo " --->> Compiling zlib-$ZLTAG"
./configure --prefix=${ZDIR} > ../config_zlib_1.log 2>&1
make -j4 > ../config_zlib_2.log 2>&1
make install > ../config_zlib_3.log 2>&1
cd ..
# rm -rf zlib-$ZLTAG
echo " --->> Finish Compile zlib-$ZLTAG"


## hdf5
tar -xf hdf5-$H5TAG.tar.gz
cd hdf5-$H5TAG/
H5DIR=$MAINDIR
echo " --->> Compiling hdf5-$H5TAG"

export CPPFLAGS="-I${ZDIR}/include"
export LDFLAGS="-L${ZDIR}/lib"

./configure --prefix=${H5DIR} --with-zlib=${ZDIR} > ../config_hdf5_1.log 2>&1
make -j4 >> ../config_hdf5_2.log 2>&1
make install >> ../config_hdf5_3.log 2>&1
cd ..
# rm -rf hdf5-$H5TAG
echo " --->> Finish Compile hdf5-$H5TAG"


## netcdf4-c
tar -xf netcdf-c-$NCTAG.tar.gz
cd netcdf-c-$NCTAG/
NCDIR=$MAINDIR
echo " --->> Compiling netcdf-c-$NCTAG"
CPPFLAGS=-I${H5DIR}/include LDFLAGS=-L${H5DIR}/lib ./configure --prefix=${NCDIR} --disable-libxml2> ../config_nc_1.log 2>&1
make -j4 > ../config_nc_2.log 2>&1
make install > ../config_nc_3.log 2>&1
cd ..
# rm -rf netcdf-c-$NCTAG
echo " --->> Finish Compile netcdf-c-$NCTAG"

## netcdf4-fortran
tar -xf netcdf-fortran-$NFTAG.tar.gz
cd netcdf-fortran-$NFTAG/
echo " --->> Compiling netcdf-fortran-$NFTAG"

export MAINDIR=/data0/home/shihx/software/netcdf_intel
export LD_LIBRARY_PATH=$MAINDIR/lib:$LD_LIBRARY_PATH
export CPPFLAGS="-I$MAINDIR/include"
export LDFLAGS="-L$MAINDIR/lib"

$FC --version
env | grep -E 'FC|LD_LIBRARY_PATH|CPPFLAGS|LDFLAGS'


CPPFLAGS=-I${NCDIR}/include LDFLAGS=-L${NCDIR}/lib ./configure --prefix=${NCDIR} > ../config_nf_1.log 2>&1
make -j4 > ../config_nf_2.log 2>&1
make install > ../config_nf_3.log 2>&1
cd ..
# rm -rf netcdf-fortran-$NFTAG
echo " --->> Finish Compile netcdf-fortran-$NFTAG"

## show compilation options
$NCDIR/bin/nf-config --all

echo ""
echo ===============================================================================
echo "Finally, you must add this to the .profile (or .bashrc or .zshrc) file"
echo "  Linux --\>" export LD_LIBRARY_PATH=$NCDIR/lib:'$LD_LIBRARY_PATH'
echo "  OSX   --\>" export DYLD_LIBRARY_PATH=$NCDIR/lib:'$DYLD_LIBRARY_PATH'
echo ===============================================================================
echo ""
