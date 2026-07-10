cd tools/LCP
mkdir build 
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=. ..
make lcp
make install
cd ../../..
python -m pip install -r requirements.txt