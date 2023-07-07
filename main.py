import sys
import csv
import copy
import numpy as np
from numpy.random import Generator, PCG64
from time import time
import math
class Main:

    def __init__(self):
        self.randomFunction = Generator(PCG64())

    def printMat(self,mat):
        print('======================')
        for row in mat:
            print( row)
        print('======================')

    def getRandomNumber(self,init:int, end:int ):
        initScale = init/end
        return  math.floor( self.randomFunction.uniform(initScale,1) * end)

    def extrartMap(self, filename):
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

    def exportMap(self,terrainMap,filename):
        npArray = np.array(terrainMap)
        np.savetxt(filename,npArray,fmt='%1.10f', delimiter=';')
        

    def createPositionMap(self,terrainMap=[]):
        n = len(terrainMap)
        positionMap = []
        for row in range(n):
            positionMap.append([0] * n)
        return  positionMap

    def getUpperLeftPositionMatrix(self,centerPosition,radius):
        (row,column) = centerPosition
        
        possibleRow = row - radius
        newRow = possibleRow if possibleRow >= 0 else 0
        
        possibleColumn = column - radius
        newColumn = possibleColumn if possibleColumn >=0 else 0 

        return(newRow,newColumn)
        
    def getUpperRightPositionMatrix(self,lenMatrix,centerPosition,radius):
        (row,column) = centerPosition
        
        possibleRow = row - radius
        newRow = possibleRow if possibleRow >= 0 else 0
        
        possibleColumn = column + radius
        newColumn = possibleColumn if possibleColumn < lenMatrix else lenMatrix-1

        return(newRow,newColumn)
        
    def getBottomRightPositionMatrix(self,lenMatrix,centerPosition,radius):
        (row,column) = centerPosition
        
        possibleRow = row + radius
        newRow = possibleRow if possibleRow < lenMatrix else lenMatrix-1
        
        possibleColumn = column + radius
        newColumn = possibleColumn if possibleColumn < lenMatrix else lenMatrix-1

        return(newRow,newColumn)

    def getBottomLeftPositionMatrix(self,lenMatrix,centerPosition,radius):
        (row,column) = centerPosition
        
        possibleRow = row + radius
        newRow = possibleRow if possibleRow < lenMatrix else lenMatrix-1
        
        possibleColumn = column - radius
        newColumn = possibleColumn if possibleColumn >= 0 else 0

        return(newRow,newColumn)

    def getSubMatrix(self,terrainMap,centerPosition,radius):
        try:
            lenMatrix = len(terrainMap)

            upperLeftPosition = self.getUpperLeftPositionMatrix(centerPosition, radius)
            bottomRightPosition = self.getBottomRightPositionMatrix(lenMatrix, centerPosition, radius)
        
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

    def getMatrixSum(self,matrix):
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
            self.printMat(matrix)

    def resetSubMatrix(self,terrainMap,position,radius):    
        upperLeftPosition = self.getUpperLeftPositionMatrix(position,radius)
        bottomRightPosition = self.getBottomRightPositionMatrix(len(terrainMap),position,radius)
        (rowUpperLeft,columnUpperLeft) = upperLeftPosition
        (rowBottomRight,columnbottomRight) = bottomRightPosition
        for row in range(rowUpperLeft,rowBottomRight+1):
            for column in range(columnUpperLeft,columnbottomRight+1):
                terrainMap[row][column]=0
        return terrainMap
    

    def markPosition(self,positionMap,terrainMap, position,radius):
        (row,column) = position
        positionMap[row][column] = 1
        self.resetSubMatrix(terrainMap,position,radius)


    def checkSubMatrix(self,upperLeftPosition,bottomRightPosition,lenMatrix):
        if (upperLeftPosition[0] < 0 ):
            return False
        if(upperLeftPosition[1] < 0):
            return False
        if(bottomRightPosition[0] >= lenMatrix):
            return False
        if(bottomRightPosition[1] >= lenMatrix):
            return False
        return True

    def getSubMatrixPosition(self,originalMatrix,radius):
        isSubMatrixCorrect= False
        lenMatrix = len(originalMatrix)
        while (isSubMatrixCorrect == False):
        
            row = self.getRandomNumber(radius,lenMatrix-radius)
            column = self.getRandomNumber(radius,lenMatrix-radius)
            centerPosition = (row,column)

            upperLeftPosition = self.getUpperLeftPositionMatrix(centerPosition, radius)
            upperRightPosition = self.getUpperRightPositionMatrix(lenMatrix,centerPosition, radius)
            bottomRightPosition = self.getBottomRightPositionMatrix(lenMatrix, centerPosition, radius) 
            bottomLeftPosition = self.getBottomLeftPositionMatrix(lenMatrix, centerPosition, radius) 

            isSubMatrixCorrect = self.checkSubMatrix(upperLeftPosition,bottomRightPosition,lenMatrix)
        return {
            "upperLeft": upperLeftPosition,
            "upperRight":upperRightPosition,
            "bottomRight":bottomRightPosition,
            "bottomLeft":bottomLeftPosition,
            "centerPosition":centerPosition
        }


    def checkCollision(self,originMatrix,targetMatrix):
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


    def getVNSSubMatrixs(self,originalMatrix,radius):
        collision =True
        while collision:
            subMatrix1 = self.getSubMatrixPosition(originalMatrix,radius)
            subMatrix2 = self.getSubMatrixPosition(originalMatrix,radius)
            collision = self.checkCollision(subMatrix1,subMatrix2)

        return (subMatrix1,subMatrix2)

    def GRASP(self,terrainMap,numberSensors,radius,alpha,limit):
        numberSensorsUse = 0
        currentRun = 0
        currentTerrainMap = copy.deepcopy(terrainMap)
        positionMap = self.createPositionMap(currentTerrainMap)
        while(numberSensorsUse < numberSensors and currentRun < limit ):
 
            row = self.getRandomNumber(0,len(currentTerrainMap))
            column = self.getRandomNumber(0,len(currentTerrainMap))

            subMatrix = self.getSubMatrix(currentTerrainMap,(row,column),radius)
            subMatrixSum = self.getMatrixSum(subMatrix)

            if ( subMatrixSum >= alpha ):
                self.markPosition(positionMap,currentTerrainMap,(row,column),radius)
                numberSensorsUse += 1
            
            currentRun+=1
        
        sumMap = self.getMatrixSum(currentTerrainMap)
        if numberSensorsUse == 0:
            return self.GRASP(terrainMap,numberSensors,radius,alpha,limit)
        coefficient = sumMap/numberSensorsUse
        

        return (coefficient,positionMap,currentTerrainMap)

    def copySubMatrix(self,subMatrix1,subMatrix2,positionMap):
        lenSubMat = subMatrix1["upperRight"][1] - subMatrix1["upperLeft"][1] + 1
        (initailRow1,initailColumn1) = subMatrix1["upperLeft"]
        (initailRow2,initailColumn2) = subMatrix2["upperLeft"]
        
        for row in range(lenSubMat):
            for column in range(lenSubMat):
                positionMap[initailRow1+row][initailColumn1+column] = positionMap[initailRow2+row][initailColumn2+column] 
        
        return positionMap

    def swapSubMatrix(self, subMatrix1,subMatrix2,positionMap):
        lenSubMat = subMatrix1["upperRight"][1] - subMatrix1["upperLeft"][1] + 1
        (initailRow1,initailColumn1) = subMatrix1["upperLeft"]
        (initailRow2,initailColumn2) = subMatrix2["upperLeft"]
        for row in range(lenSubMat):
            for column in range(lenSubMat):
                temp = positionMap[initailRow1+row][initailColumn1+column]
                positionMap[initailRow1+row][initailColumn1+column] = positionMap[initailRow2+row][initailColumn2+column] 
                positionMap[initailRow2+row][initailColumn2+column] = temp

        return positionMap

    def markPositionsInAMatrix(self,positionMap,terrainMap,radius):
        lenMatrix = len(positionMap)
        numberSensorsUse = 0

        for row in range(lenMatrix):
            for column in range(lenMatrix):
                if(positionMap[row][column]==1):
                    terrainMap = self.resetSubMatrix(terrainMap,(row,column),radius)
                    numberSensorsUse += 1

        return (terrainMap,numberSensorsUse)


    def heuristicSwap(self,positionMap,radius):
        (subMatrix1,subMatrix2) = self.getVNSSubMatrixs(positionMap,radius)
        newPositionMap = self.swapSubMatrix(subMatrix1,subMatrix2,positionMap)
        return newPositionMap

    def heuristicCopy(self,positionMap,radius):
        (subMatrix1,subMatrix2) = self.getVNSSubMatrixs(positionMap,radius)
        newPositionMap = self.copySubMatrix(subMatrix1,subMatrix2,positionMap)
        return newPositionMap


    def checkMaxSensors(self,positionMap,numberSensors):
        sensosrsPosition =[]
        lenPosition = len(positionMap)
        for row in range(lenPosition):
            for column in range(lenPosition):
                if( positionMap[row][column] == 1):
                    sensosrsPosition.append((row,column))

        while len(sensosrsPosition) > numberSensors:
            index = self.getRandomNumber(0, len(sensosrsPosition))
            (row,column) = sensosrsPosition.pop(index)
            positionMap[row][column] = 0
        
        return positionMap


    def VNS(self,positionMap,coefficient,originalMap,radius,numberSensors):
        
        currentTerrainMap = copy.deepcopy(originalMap)
        currentPositionMap = copy.deepcopy(positionMap)

        bestTerrainMap = copy.deepcopy(currentTerrainMap)
        bestPositionMap = copy.deepcopy(currentPositionMap)
        heuristics = [self.heuristicSwap,self.heuristicCopy]
        k = 0
        while k <= 1:
            shakedPositionMap = heuristics[k](currentPositionMap,radius)
            shakedPositionMap = self.checkMaxSensors(shakedPositionMap,numberSensors)

            newPositionMap = heuristics[k](shakedPositionMap,radius)
            newPositionMap = self.checkMaxSensors(newPositionMap,numberSensors)
          
            currentTerrainMap = copy.deepcopy(originalMap)
            (currentTerrainMap,numberSensorsUse) = self.markPositionsInAMatrix(newPositionMap,currentTerrainMap,radius)
      

            sumMap = self.getMatrixSum(currentTerrainMap)
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

    
    def getAlha(self,terreinMatrix,radius):
        distance = (radius*2 )+ 1
        lenMatrix = len(terreinMatrix)
        workTerreinMatrix = copy.deepcopy(terreinMatrix)
        row = radius
        column = radius
        sumList = []
        lastRow = False

        while (  lastRow == False and row  < lenMatrix ):
            subMatrix = self.getSubMatrix(workTerreinMatrix,(row,column),radius)
            sumSubMatrix = self.getMatrixSum(subMatrix)

            self.resetSubMatrix(workTerreinMatrix,(row,column),radius)

            sumList.append(sumSubMatrix)
            column += distance 

            if(column >= lenMatrix):
                if(row >= lenMatrix and column >= lenMatrix ):
                    continue
                column = lenMatrix-1
                subMatrix = self.getSubMatrix(workTerreinMatrix,(row,column),radius)
                sumSubMatrix = self.getMatrixSum(subMatrix)
                self.resetSubMatrix(workTerreinMatrix,(row,column),radius)
                sumSubMatrix 
                sumList.append(sumSubMatrix)
                column = radius
                row += distance
                if (row >= (lenMatrix-1)  and lastRow == False):
                    if(row >= lenMatrix):
                        row = lenMatrix - 1
                    lastRow =True    
        
        totalSum = 0
        for sumItem in sumList:
            totalSum += sumItem

        lenSumList = len(sumList)
        avgSum = totalSum/lenSumList  
        variance = 0
        for sumItem in sumList:
            variance += ( (avgSum - sumItem) ** 2)/lenSumList 
        
        standardDeviation = math.sqrt(variance)
        
        return avgSum - standardDeviation
            
                


            


    def main(self):
        argv = sys.argv
        filename = argv[1]
        numberSensors = int(argv[2])
        radius = int(argv[3])
      
        limit = 1000 
        bestCoefficient=10000000
        bestPositionMap=[]
        bestMap=[]

        midtarm =[]
        terrainMap = self.extrartMap(filename)
        
        alpha = self.getAlha(terrainMap,radius)
        print("alpha",alpha)
        for i in range(10000):
            secound=self.getRandomNumber(0,round(time()))
            seed = round(time()*10000)+secound
            self.randomFunction = Generator(PCG64(seed))
            (coefficient,positionMap,currentTerrainMap) = self.GRASP(terrainMap,numberSensors,radius,alpha,limit)

            (coefficientVNS,positionMapVNS,currentTerrainMapVNS) = self.VNS(positionMap,coefficient,terrainMap,radius,numberSensors)

            if(coefficientVNS < coefficient):
                coefficient = coefficientVNS
                positionMap = positionMapVNS 
                currentTerrainMap = currentTerrainMapVNS
            if(i % 100 ==0):
                print("currentCoefficient",coefficient)
            if(bestCoefficient>coefficient):
                print("oldCoefficient",bestCoefficient)
                print("coefficient",coefficient)
                bestCoefficient = coefficient
                bestPositionMap = copy.deepcopy(positionMap)
                bestMap = copy.deepcopy(currentTerrainMap)
            if(i % 1000 ==0):
                midtarm.append(bestCoefficient)
                print(str(i/100)+"%")

        self.printMat(bestPositionMap)
        self.printMat(bestMap)
        print(bestCoefficient)
        print(midtarm)
        self.exportMap(bestMap,'bestmap.csv')
        self.exportMap(midtarm,'midtarm.csv')
        

        


mainClass = Main()
mainClass.main()