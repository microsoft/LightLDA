# build lightlda

git clone -b multiverso-initial git@github.com:Microsoft/multiverso.git

cd multiverso
cd third_party
sh install.sh
cd ..
make -j4 all

cd ..
make -j4
