all:
	gcc -fPIC -O3 -shared -o libclusgeo3 clusGeo3.c -lm 
cpp:
	g++  *.cpp
