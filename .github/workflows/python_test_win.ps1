python -m pip install --pre finufft -f .\wheelhouse\
if (-not $?) {throw "Failed to pip install finufft"}

c:\msys64\usr\bin\ldd "C:\hostedtoolcache\windows\Python\3.8.10\x64\Lib\site-packages\finufft\libfinufft.dll"

python python/finufft/test/run_accuracy_tests.py
if (-not $?) {throw "Tests failed"}
python python/finufft/examples/simple1d1.py
if (-not $?) {throw "Simple1d1 test failed"}
