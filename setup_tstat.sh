#!/bin/bash

rm -fr tstat-3.1.1
sudo apt install libpcap-dev
wget http://tstat.polito.it/download/tstat-3.1.1.tar.gz
tar xvf tstat-3.1.1.tar.gz
rm tstat-3.1.1.tar.gz
cp tstat.patch tstat-3.1.1
cd tstat-3.1.1 || exit
patch -p0 <tstat.patch
./autogen.sh
./configure --enable-libtstat --disable-dependency-tracking --enable-shared CC="gcc-9" CXX="g++-9"
make
cd tstat-conf
VAR=$(pwd)
sed -i "s,net.all,$VAR/net.all,g" tstat.conf
sed -i "s,#-s outdir,-s /tmp,g" tstat.conf
