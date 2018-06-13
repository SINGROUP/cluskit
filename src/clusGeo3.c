#include<stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdlib.h>


#define PI     3.14159265358979
#define PI2    9.86960440108936
#define PIHALF 1.57079632679490
//-----------------------------------------------------------
double findMin(double* x, int size);
double findMin(double* x, int size){
  double minVal = x[0];
  for(int i = 1; i < size; i++){
      if(x[i] < minVal){
         minVal = x[i];
      }
  }

 return minVal;

}
//-----------------------------------------------------------
double findMax(double* x, int size);
double findMax(double* x, int size){
  double maxVal = x[0];
  for(int i = 1; i < size; i++){
      if(x[i] > maxVal){
         maxVal = x[i];
      }
  }

 return maxVal;

}
//-----------------------------------------------------------
int findSurf(double* x, double* y, double* z, int* surfAtoms,  int size, double bubblesize);
int findSurf(double* x, double* y, double* z, int* surfAtoms,  int size, double bubblesize){

  double vacuumRad = bubblesize * bubblesize;
  int gridSize = 100;

  double xMin = findMin(x,size); double xMax = findMax(x,size);
  double yMin = findMin(y,size); double yMax = findMax(y,size);
  double zMin = findMin(z,size); double zMax = findMax(z,size);

  int* surfAtomsBuff = (int*) malloc(sizeof(int)*gridSize*gridSize*gridSize);
  double rBuff2Double;  int marcher = 0;   int marcherSurf = 0;

  double xPos = xMin - 3.5; double dx = (xMax - xMin + 7.0)/(double) gridSize;
  double yPos = yMin - 3.5; double dy = (yMax - yMin + 7.0)/(double) gridSize;
  double zPos = zMin - 3.5; double dz = (zMax - zMin + 7.0)/(double) gridSize;
  double xPosInit = xPos; double yPosInit = yPos; double zPosInit = zPos;

  int* isSurf = (int*) malloc(sizeof(int)*size);
  int maxAtomIndx;  double minVal;  int minAtomsI;

  while( xPos < xMax + 3.5){ yPos = yPosInit;
    while( yPos < yMax + 3.5){ zPos = zPosInit;
      while( zPos < zMax + 3.5){ minVal = 10000000;
        for(int i = 0; i < size; i++){
          rBuff2Double =(xPos-x[i])*(xPos-x[i])+(yPos-y[i])*(yPos-y[i])+(zPos-z[i])*(zPos-z[i]);
          if(rBuff2Double < minVal ){
            minAtomsI = i; minVal = rBuff2Double;
          }
        }
        if( minVal > vacuumRad){
            surfAtomsBuff[marcher] = minAtomsI; 
            marcher++; 
        } zPos += dz;
      } yPos += dy;
    } xPos += dx;
  }

  for(int i = 0; i < size; i++){
    for(int j = 0; j < marcher; j++){
      if(surfAtomsBuff[j] == i){
        surfAtoms[marcherSurf] = i; marcherSurf++;
        break;
      }
    }
  }

 return marcherSurf;

}
//-----------------------------------------------------------
//-----------------------------------------------------------
void getEta1(double* surfH1, double* x,double* y, double* z, int* surfAtoms, int surfNum, int size, double distH);
void getEta1(double* surfH1, double* x,double* y, double* z, int* surfAtoms, int surfNum, int size, double distH){
  int gridT = 100;
  double distBuf;
  double* newX = (double*) malloc(sizeof(double)*gridT*gridT);
  double* newY = (double*) malloc(sizeof(double)*gridT*gridT);
  double* newZ = (double*) malloc(sizeof(double)*gridT);
  double newS;
  double indxT;
  double indxP;
  double forceBuf=0;
  double oOgS = 1/(double) gridT;
  double diffX;
  double diffY;
  double diffZ;

  int indxTgT;

  double* bestX = (double*) malloc(sizeof(double)*surfNum);
  double* bestY = (double*) malloc(sizeof(double)*surfNum);
  double* bestZ = (double*) malloc(sizeof(double)*surfNum);
  for(int i = 0; i < surfNum; i++){
    bestX[i]=0;
    bestY[i]=0;
    bestZ[i]=0;
  }
  double  bestF;

  for(int t=0; t < gridT; t++){
    indxT = oOgS*PI*(double) t;
    newS = distH*sin(indxT);
    newZ[t] = distH*cos(indxT);
    for(int p=0; p < gridT; p++){
      indxP = 2*oOgS*PI*(double) p;
      newX[t*gridT + p] = newS*cos(indxP);
      newY[t*gridT + p] = newS*sin(indxP);

    }
  }

  for(int i = 0; i < surfNum; i++){
        bestF = 500000;
    for(int t=0; t < gridT; t++){
      for(int p=0; p < gridT; p++){
        forceBuf = 0;
        indxTgT = t*gridT + p;
        for(int j = 0; j < size; j++){
          diffX = x[surfAtoms[i]] + newX[indxTgT] - x[j];
          diffY = y[surfAtoms[i]] + newY[indxTgT] - y[j];
          diffZ = z[surfAtoms[i]] + newZ[t      ] - z[j];
          forceBuf += 1/(diffX*diffX + diffY*diffY + diffZ*diffZ);
        }
        if(bestF > forceBuf){
          bestX[i] = x[surfAtoms[i]] + newX[indxTgT];
          bestY[i] = y[surfAtoms[i]] + newY[indxTgT];
          bestZ[i] = z[surfAtoms[i]] + newZ[t];
          bestF = forceBuf;
        }
      }
    }
    surfH1[3*i + 0] = bestX[i];
    surfH1[3*i + 1] = bestY[i];
    surfH1[3*i + 2] = bestZ[i]; // 
  }

  free(bestX);
  free(bestY);
  free(bestZ);
  free(newX);
  free(newY);
  free(newZ);

}
//-----------------------------------------------------------
//-----------------------------------------------------------
int getEta2(double* surfH2, double* x,double* y, double* z, int* surfAtoms, int surfNum, int size, double distH);
int getEta2(double* surfH2, double* x,double* y, double* z, int* surfAtoms, int surfNum, int size, double distH){

  int flag;
  int gridT = 100;
  double newS;
  double indxT;
  double indxP;
  double forceBuf=0;
  double oOgS = 1/(double) gridT;
  double diffX;
  double diffY;
  double diffZ;
  double* newX = (double*) malloc(sizeof(double)*gridT*gridT);
  double* newY = (double*) malloc(sizeof(double)*gridT*gridT);
  double* newZ = (double*) malloc(sizeof(double)*gridT);
  int NEta2 = 0;
  double* bestX = (double*) malloc(sizeof(double)*surfNum*surfNum);
  double* bestY = (double*) malloc(sizeof(double)*surfNum*surfNum);
  double* bestZ = (double*) malloc(sizeof(double)*surfNum*surfNum);
  for(int i = 0; i < surfNum*surfNum; i++){
    bestX[i]=0; bestY[i]=0; bestZ[i]=0;
  }
  double r;
  double bestF;
  int indxTgT;
  double r2;

  for(int t=0; t < gridT; t++){
    indxT = oOgS*PI*(double) t;
    newS = distH*sin(indxT);
    newZ[t] = distH*cos(indxT);
    for(int p=0; p < gridT; p++){
      indxP = 2*oOgS*PI*(double) p;
      newX[t*gridT + p] = newS*cos(indxP);
      newY[t*gridT + p] = newS*sin(indxP);

    }
  }

  for(int i = 0; i < surfNum; i++){
    for(int j = i+1; j < surfNum; j++){
      flag = 0;
      bestF = 50000000;
      for(int t=0; t < gridT; t++){
        for(int p=0; p < gridT; p++){
          indxTgT = t*gridT + p;
          diffX = x[surfAtoms[i]] + newX[indxTgT] - x[surfAtoms[j]];
          diffY = y[surfAtoms[i]] + newY[indxTgT] - y[surfAtoms[j]];
          diffZ = z[surfAtoms[i]] + newZ[t      ] - z[surfAtoms[j]];
          r2 = diffX*diffX + diffY*diffY + diffZ*diffZ;
          if( distH*distH - 0.1 < r2 && r2  < distH*distH + 0.1 ){
            forceBuf = 0;
            for(int k = 0; k < size; k++){
              diffX = x[surfAtoms[i]] + newX[indxTgT] - x[k];
              diffY = y[surfAtoms[i]] + newY[indxTgT] - y[k];
              diffZ = z[surfAtoms[i]] + newZ[t      ] - z[k];
              r = sqrt(diffX*diffX + diffY*diffY + diffZ*diffZ);
              forceBuf += 1.0/r;
            }
            if(bestF > forceBuf){
              if(flag==0){ flag = 1;}
              bestX[NEta2] = x[surfAtoms[i]] + newX[indxTgT];
              bestY[NEta2] = y[surfAtoms[i]] + newY[indxTgT];
              bestZ[NEta2] = z[surfAtoms[i]] + newZ[t      ];
              bestF = forceBuf;
            }
          }
        }
            
      }
       if(flag==1) NEta2++;
        
    }
  }
  for(int i = 0; i < NEta2; i++){
    surfH2[3*i + 0] = bestX[i];
    surfH2[3*i + 1] = bestY[i];
    surfH2[3*i + 2] = bestZ[i]; // 
  }

return NEta2;
}
//-----------------------------------------------------------
int getEta3(double* surfH3, double* x,double* y, double* z, int* surfAtoms, int surfNum, int size, double distH);
int getEta3(double* surfH3, double* x,double* y, double* z, int* surfAtoms, int surfNum, int size, double distH){

  int flag;
  int gridT = 200;
  double newS;
  double indxT;
  double indxP;
  double forceBuf=0;
  double oOgS = 1/(double) gridT;
  double diffX;
  double diffY;
  double diffZ;
  double* newX = (double*) malloc(sizeof(double)*gridT*gridT);
  double* newY = (double*) malloc(sizeof(double)*gridT*gridT);
  double* newZ = (double*) malloc(sizeof(double)*gridT);
  int NEta3 = 0;
  double* bestX = (double*) malloc(sizeof(double)*surfNum*surfNum*surfNum);
  double* bestY = (double*) malloc(sizeof(double)*surfNum*surfNum*surfNum);
  double* bestZ = (double*) malloc(sizeof(double)*surfNum*surfNum*surfNum);
  for(int i = 0; i < surfNum*surfNum*surfNum; i++){
    bestX[i]=0; bestY[i]=0; bestZ[i]=0;
  }
  double r;
  double bestF;
  int indxTgT;
  double r2;

  for(int t=0; t < gridT; t++){
    indxT = oOgS*PI*(double) t;
    newS = distH*sin(indxT);
    newZ[t] = distH*cos(indxT);
    for(int p=0; p < gridT; p++){
      indxP = 2*oOgS*PI*(double) p;
      newX[t*gridT + p] = newS*cos(indxP);
      newY[t*gridT + p] = newS*sin(indxP);

    }
  }

  for(int i = 0; i < surfNum; i++){
    for(int j = i+1; j < surfNum; j++){
      for(int k = j+1; k < surfNum; k++){
        flag = 0;
        bestF = 50000000;
        for(int t=0; t < gridT; t++){
          for(int p=0; p < gridT; p++){
            forceBuf = 0;
            indxTgT = t*gridT + p;
            diffX = x[surfAtoms[i]] + newX[indxTgT] - x[surfAtoms[j]];
            diffY = y[surfAtoms[i]] + newY[indxTgT] - y[surfAtoms[j]];
            diffZ = z[surfAtoms[i]] + newZ[t      ] - z[surfAtoms[j]];
            r2 = diffX*diffX + diffY*diffY + diffZ*diffZ;
            if( distH*distH - 0.2 < r2 && r2  < distH*distH + 0.2 ){
              indxTgT = t*gridT + p;
              diffX = x[surfAtoms[i]] + newX[indxTgT] - x[surfAtoms[k]];
              diffY = y[surfAtoms[i]] + newY[indxTgT] - y[surfAtoms[k]];
              diffZ = z[surfAtoms[i]] + newZ[t      ] - z[surfAtoms[k]];
              r2 = diffX*diffX + diffY*diffY + diffZ*diffZ;
              if( distH*distH - 0.2 < r2 && r2  < distH*distH + 0.2 ){
                for(int l = 0; l < size; l++){
                  diffX = x[surfAtoms[i]] + newX[indxTgT] - x[l];
                  diffY = y[surfAtoms[i]] + newY[indxTgT] - y[l];
                  diffZ = z[surfAtoms[i]] + newZ[t      ] - z[l];
                  r = sqrt(diffX*diffX + diffY*diffY + diffZ*diffZ);
                  forceBuf += 1.0/r;
                }
                if(bestF > forceBuf){
                  if(flag==0){ flag = 1;}
                  bestX[NEta3] = x[surfAtoms[i]] + newX[indxTgT];
                  bestY[NEta3] = y[surfAtoms[i]] + newY[indxTgT];
                  bestZ[NEta3] = z[surfAtoms[i]] + newZ[t      ];
                  bestF = forceBuf;
                }
              }
            }
          }
              
        }
         if(flag==1) NEta3++;
          
      }
    }
  }
  for(int i = 0; i < NEta3; i++){
    surfH3[3*i + 0] = bestX[i];
    surfH3[3*i + 1] = bestY[i];
    surfH3[3*i + 2] = bestZ[i]; // 
  }
  return NEta3;
}
//-----------------------------------------------------------
int x2_to_x(double* xH,double* yH, double* zH,double* x,double* y, double* z, int* types, int* typeNs, double bondlength);
int x2_to_x(double* xH,double* yH, double* zH,double* x,double* y, double* z, int* types, int* typeNs, double bondlength){
  double dist2, xD, yD, zD;
  int flag, count = 0;
  if(types[0]==1){
    for(int i = 0; i < typeNs[0]; i++){
      flag = 0;
      for(int j = i + 1; j < typeNs[0]; j++){
        xD = x[i] - x[j]; yD = y[i] - y[j]; zD = z[i] - z[j];
        dist2 = xD*xD + yD*yD + zD*zD;
        if(dist2 < bondlength*bondlength){
//          xH[count] = x[i] + xD*0.5; 
//          yH[count] = y[i] + yD*0.5; 
//          zH[count] = z[i] + zD*0.5; 
          flag = 1;
//          count++;
        }
      } //done j
      if(flag==0){
        xH[count] = x[i];
        yH[count] = y[i];
        zH[count] = z[i];
        count++;
      }
    }
  }
  return count;
}
//-----------------------------------------------------------
