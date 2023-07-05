import sys
import csv
import copy
import numpy as np

def printMat(mat):
    print('======================')
    for row in mat:
        print( row)
    print('======================')
   
def extrartMap(filename):
    terrainMap =[]
    with open(filename, newline='\n',) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
       
        for row in spamreader:
            rowList =[]

            for elem in row:
                floatNumber=elem.replace(',','.')
                rowList.append(float(floatNumber))

            terrainMap.append(rowList)
    return terrainMap

def exportMap(terrainMap,filename):
    npArray = np.array(terrainMap)
    np.savetxt(filename,npArray,fmt='%1.10f', delimiter=';')
    

def createPositionMap(terrainMap=[]):
    n = len(terrainMap)
    positionMap = []
    for row in range(n):
        positionMap.append([0] * n)
    return  positionMap

def getUpperLeftPositionMatrix(centerPosition,radius):
    (row,column) = centerPosition
    
    possibleRow = row - radius
    newRow = possibleRow if possibleRow >= 0 else 0
    
    possibleColumn = column - radius
    newColumn = possibleColumn if possibleColumn >=0 else 0 

    return(newRow,newColumn)
    
def getUpperRightPositionMatrix(lenMatrix,centerPosition,radius):
    (row,column) = centerPosition
    
    possibleRow = row - radius
    newRow = possibleRow if possibleRow >= 0 else 0
    
    possibleColumn = column + radius
    newColumn = possibleColumn if possibleColumn < lenMatrix else lenMatrix-1

    return(newRow,newColumn)
    
def getBottomRightPositionMatrix(lenMatrix,centerPosition,radius):
    (row,column) = centerPosition
    
    possibleRow = row + radius
    newRow = possibleRow if possibleRow < lenMatrix else lenMatrix-1
    
    possibleColumn = column + radius
    newColumn = possibleColumn if possibleColumn < lenMatrix else lenMatrix-1

    return(newRow,newColumn)

def getBottomLeftPositionMatrix(lenMatrix,centerPosition,radius):
    (row,column) = centerPosition
    
    possibleRow = row + radius
    newRow = possibleRow if possibleRow < lenMatrix else lenMatrix-1
    
    possibleColumn = column - radius
    newColumn = possibleColumn if possibleColumn >= 0 else 0

    return(newRow,newColumn)

def getSubMatrix(terrainMap,centerPosition,radius):
    try:
        lenMatrix = len(terrainMap)

        upperLeftPosition = getUpperLeftPositionMatrix(centerPosition, radius)
        bottomRightPosition = getBottomRightPositionMatrix(lenMatrix, centerPosition, radius)
    
        (rowUpperLeft,columnUpperLeft) = upperLeftPosition
        (rowBottomRight,columnbottomRight) = bottomRightPosition
        subMatrix =[]
        for row in range(rowUpperLeft,rowBottomRight+1):
            columnList=[]
            for column in range(columnUpperLeft,columnbottomRight+1):
                value = float(terrainMap[row][column])
                columnList.append(value)
            subMatrix.append(columnList)
        return subMatrix
    except:
        print("Erro insperado")
        print('upperLeftPosition',upperLeftPosition)
        print('bottomRightPosition',bottomRightPosition)
        print('centerPosition',centerPosition)
        print('lenMatrix',lenMatrix)

def getMatrixSum(matrix):
    try:
        rowLen = len(matrix)
        columnLen = len(matrix[0])
        
        matrixSum = 0
        for row in range(0,rowLen):
            for column in range(0,columnLen):
                matrixSum+= matrix[row][column]
        return matrixSum
    except:
        print("Error at getMatrixSum")
        print("rowLen",rowLen)
        print("columnLen",columnLen)
        printMat(matrix)

def resetSubMatrix(terrainMap,position,radius):    
    upperLeftPosition = getUpperLeftPositionMatrix(position,radius)
    bottomRightPosition = getBottomRightPositionMatrix(len(terrainMap),position,radius)
    (rowUpperLeft,columnUpperLeft) = upperLeftPosition
    (rowBottomRight,columnbottomRight) = bottomRightPosition
    for row in range(rowUpperLeft,rowBottomRight+1):
        for column in range(columnUpperLeft,columnbottomRight+1):
            terrainMap[row][column]=0
  

def markPosition(positionMap,terrainMap, position,radius):
    (row,column) = position
    positionMap[row][column] = 1
    resetSubMatrix(terrainMap,position,radius)
 

def GRASP(terrainMap,numberSensors,radius,alpha,limit):
    numberSensorsUse = 0
    currentRun = 0
    currentTerrainMap = copy.deepcopy(terrainMap)
    positionMap = createPositionMap(currentTerrainMap)
    while(numberSensorsUse < numberSensors and currentRun < limit ):
        row = np.random.randint(0,len(currentTerrainMap))
        column = np.random.randint(0,len(currentTerrainMap))

        subMatrix = getSubMatrix(currentTerrainMap,(row,column),radius)
        subMatrixSum = getMatrixSum(subMatrix)

        if ( subMatrixSum >= alpha ):
            markPosition(positionMap,currentTerrainMap,(row,column),radius)
            numberSensorsUse += 1
        
        currentRun+=1
    sumMap = getMatrixSum(currentTerrainMap)
    coeficiente = sumMap/numberSensorsUse

    return (coeficiente,positionMap,currentTerrainMap)

def main():
    argv = sys.argv
    filename = argv[1]
    numberSensors = int(argv[2])
    radius = int(argv[3])
    alpha = 2
    limit = 40 
    bestCoeficiente=10000000
    bestPositionMap=[]
    bestMap=[]

    midtarm =[]
    terrainMap = extrartMap(filename)

    for i in range(10000):
        (coeficiente,positionMap,currentTerrainMap) = GRASP(terrainMap,numberSensors,radius,alpha,limit)

        if(bestCoeficiente>coeficiente):
            bestCoeficiente = coeficiente
            bestPositionMap = copy.deepcopy(positionMap)
            bestMap = copy.deepcopy(currentTerrainMap)
        if(i % 1000 ==0):
            midtarm.append(bestCoeficiente)
            print(str(i/100)+"%")
    printMat(bestPositionMap)
    printMat(bestMap)
    print(bestCoeficiente)
    print(midtarm)
    exportMap(bestMap,'bestmap.csv')
    exportMap(midtarm,'midtarm.csv')
    

    



main()