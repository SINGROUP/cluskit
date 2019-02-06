all:
	gcc -fPIC -O3 -shared -o lib/libcluskit src/cluskit.c -lm 
cpp:
	g++  *.cpp
