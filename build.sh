# build lightlda

git clone git@github.com:Microsoft/multiverso.git

cd multiverso
cd third_party
sh install.sh
cd ..
make -j4 all

cd ..
make -j4
