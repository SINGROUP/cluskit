all:
	gcc -fPIC -O3 -shared -o lib/libclusgeo3 src/clusGeo3.c -lm 
cpp:
	g++  *.cpp
