import sys
import csv
import copy
import numpy as np
from numpy.random import Generator, PCG64

def printMat(mat):
    print('======================')
    for row in mat:
        print( row)
    print('======================')

def getRandomNumber(init:int, end:int ):
    randomFunction = Generator(PCG64())
    return randomFunction.integers(init,end)

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
    return terrainMap
  

def markPosition(positionMap,terrainMap, position,radius):
    (row,column) = position
    positionMap[row][column] = 1
    resetSubMatrix(terrainMap,position,radius)


def checkSubMatrix(upperLeftPosition,bottomRightPosition,lenMatrix):
    if (upperLeftPosition[0] < 0 ):
        return False
    if(upperLeftPosition[1] < 0):
        return False
    if(bottomRightPosition[0] >= lenMatrix):
        return False
    if(bottomRightPosition[1] >= lenMatrix):
        return False
    return True

def getSubMatrixPosition(originalMatrix,radius):
    isSubMatrixCorrect= False
    lenMatrix = len(originalMatrix)
    while (isSubMatrixCorrect == False):
      
        row = getRandomNumber(radius,lenMatrix-radius)
        column = getRandomNumber(radius,lenMatrix-radius)
        centerPosition = (row,column)

        upperLeftPosition = getUpperLeftPositionMatrix(centerPosition, radius)
        upperRightPosition = getUpperRightPositionMatrix(lenMatrix,centerPosition, radius)
        bottomRightPosition = getBottomRightPositionMatrix(lenMatrix, centerPosition, radius) 
        bottomLeftPosition = getBottomLeftPositionMatrix(lenMatrix, centerPosition, radius) 

        isSubMatrixCorrect = checkSubMatrix(upperLeftPosition,bottomRightPosition,lenMatrix)
    return {
        "upperLeft": upperLeftPosition,
        "upperRight":upperRightPosition,
        "bottomRight":bottomRightPosition,
        "bottomLeft":bottomLeftPosition,
        "centerPosition":centerPosition
    }


def checkCollision(originMatrix,targetMatrix):
    pointsToCheck =[]
    pointsToCheck.append(targetMatrix['upperLeft'])
    pointsToCheck.append(targetMatrix['upperRight'])
    pointsToCheck.append(targetMatrix['bottomRight'])
    pointsToCheck.append(targetMatrix['bottomLeft'])
   
    collided = False
   
    upperRow = originMatrix['upperLeft'][0]
    leftColumn = originMatrix['upperLeft'][1]
    bottomRow = originMatrix['bottomRight'][0]
    rightColumn = originMatrix['bottomRight'][1]

    for position in pointsToCheck:
        positionRow = position[0]
        positionColumn = position[1]
        if ( positionRow >= upperRow ) and ( positionRow <= bottomRow ):
            if (positionColumn >= leftColumn) and (positionColumn <= rightColumn):
                return True

       
    return collided


def getVNSSubMatrixs(originalMatrix,radius):
    collision =True
    while collision:
        subMatrix1 = getSubMatrixPosition(originalMatrix,radius)
        subMatrix2 = getSubMatrixPosition(originalMatrix,radius)
        collision = checkCollision(subMatrix1,subMatrix2)

    return (subMatrix1,subMatrix2)

def GRASP(terrainMap,numberSensors,radius,alpha,limit):
    numberSensorsUse = 0
    currentRun = 0
    currentTerrainMap = copy.deepcopy(terrainMap)
    positionMap = createPositionMap(currentTerrainMap)
    while(numberSensorsUse < numberSensors and currentRun < limit ):
        row = getRandomNumber(0,len(currentTerrainMap))
        column = getRandomNumber(0,len(currentTerrainMap))

        subMatrix = getSubMatrix(currentTerrainMap,(row,column),radius)
        subMatrixSum = getMatrixSum(subMatrix)

        if ( subMatrixSum >= alpha ):
            markPosition(positionMap,currentTerrainMap,(row,column),radius)
            numberSensorsUse += 1
        
        currentRun+=1
    sumMap = getMatrixSum(currentTerrainMap)
    if numberSensorsUse == 0:
       return GRASP(terrainMap,numberSensors,radius,alpha,limit)
    coefficient = sumMap/numberSensorsUse
    

    return (coefficient,positionMap,currentTerrainMap)

def copySubMatrix(subMatrix1,subMatrix2,positionMap):
    lenSubMat = subMatrix1["upperRight"][1] - subMatrix1["upperLeft"][1] + 1
    (initailRow1,initailColumn1) = subMatrix1["upperLeft"]
    (initailRow2,initailColumn2) = subMatrix2["upperLeft"]
    
    for row in range(lenSubMat):
        for column in range(lenSubMat):
            positionMap[initailRow1+row][initailColumn1+column] = positionMap[initailRow2+row][initailColumn2+column] 
    
    return positionMap

def swapSubMatrix(subMatrix1,subMatrix2,positionMap):
    lenSubMat = subMatrix1["upperRight"][1] - subMatrix1["upperLeft"][1] + 1
    (initailRow1,initailColumn1) = subMatrix1["upperLeft"]
    (initailRow2,initailColumn2) = subMatrix2["upperLeft"]
    for row in range(lenSubMat):
        for column in range(lenSubMat):
            temp = positionMap[initailRow1+row][initailColumn1+column]
            positionMap[initailRow1+row][initailColumn1+column] = positionMap[initailRow2+row][initailColumn2+column] 
            positionMap[initailRow2+row][initailColumn2+column] = temp

    return positionMap

def markPositionsInAMatrix(positionMap,terrainMap,radius):
    lenMatrix = len(positionMap)
    numberSensorsUse = 0

    for row in range(lenMatrix):
        for column in range(lenMatrix):
            if(positionMap[row][column]==1):
                terrainMap = resetSubMatrix(terrainMap,(row,column),radius)
                numberSensorsUse += 1

    return (terrainMap,numberSensorsUse)


def heuristicSwap(positionMap,radius):
    (subMatrix1,subMatrix2) = getVNSSubMatrixs(positionMap,radius)
    newPositionMap = swapSubMatrix(subMatrix1,subMatrix2,positionMap)
    return newPositionMap

def heuristicCopy(positionMap,radius):
    (subMatrix1,subMatrix2) = getVNSSubMatrixs(positionMap,radius)
    newPositionMap = copySubMatrix(subMatrix1,subMatrix2,positionMap)
    return newPositionMap



def VNS(positionMap,coefficient,originalMap,radius):
    
    currentTerrainMap = copy.deepcopy(originalMap)
    currentPositionMap = copy.deepcopy(positionMap)

    bestTerrainMap = copy.deepcopy(currentTerrainMap)
    bestPositionMap = copy.deepcopy(currentPositionMap)
    heuristics = [heuristicSwap,heuristicCopy]
    k = 0
    while k <= 1:
        shakedPositionMap = heuristics[k](currentPositionMap,radius)
        newPositionMap = heuristics[k](shakedPositionMap,radius)

        currentTerrainMap = copy.deepcopy(originalMap)
        (currentTerrainMap,numberSensorsUse) = markPositionsInAMatrix(newPositionMap,currentTerrainMap,radius)

        sumMap = getMatrixSum(currentTerrainMap)
        newCoefficient = sumMap/numberSensorsUse
        if newCoefficient < coefficient:
            currentPositionMap = copy.deepcopy(newPositionMap)
            bestPositionMap = copy.deepcopy(newPositionMap)
            bestTerrainMap = copy.deepcopy(currentTerrainMap)
            coefficient = newCoefficient
            k = 0
        else:
            k += 1
    return (coefficient, bestPositionMap, bestTerrainMap)

   


def main():
    argv = sys.argv
    filename = argv[1]
    numberSensors = int(argv[2])
    radius = int(argv[3])
    alpha = 2
    limit = 40 
    bestCoefficient=10000000
    bestPositionMap=[]
    bestMap=[]

    midtarm =[]
    terrainMap = extrartMap(filename)

    for i in range(10000):
        (coefficient,positionMap,currentTerrainMap) = GRASP(terrainMap,numberSensors,radius,alpha,limit)

        (coefficient,positionMap,currentTerrainMap) = VNS(positionMap,coefficient,terrainMap,radius)

        if(bestCoefficient>coefficient):
            bestCoefficient = coefficient
            bestPositionMap = copy.deepcopy(positionMap)
            bestMap = copy.deepcopy(currentTerrainMap)
        if(i % 1000 ==0):
            midtarm.append(bestCoefficient)
            print(str(i/100)+"%")

    printMat(bestPositionMap)
    printMat(bestMap)
    print(bestCoefficient)
    print(midtarm)
    exportMap(bestMap,'bestmap.csv')
    exportMap(midtarm,'midtarm.csv')
    

    



main()