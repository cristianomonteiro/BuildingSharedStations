import networkx as nx
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import psycopg2 as pg
import numpy as np
from math import floor
from datetime import datetime, timedelta
from pathlib import Path
import json

def loadMultiGraph():
    params = {'host':'localhost', 'port':'5432', 'database':'afterqualifying', 'user':'cristiano', 'password':'cristiano'}
    conn = pg.connect(**params)

    sqlQuery = '''	select	EDGE.IDVERTEXORIG_FK::text,
                            EDGE.IDVERTEXDEST_FK::text,
                            EDGE.IDEDGE::text,
                            EDGE.LENGTH,
                            0 as PARKINGEXPENSES
                    from	STREETSEGMENT as EDGE
                    
                    union all
                    
                    select  EDGE.IDVERTEXORIG_FK::text as IDVERTEXORIG_FK,
                            'station_'||STATION.IDSTATION::text as IDVERTEXDEST_FK,
                            EDGE.IDVERTEXORIG_FK::text||'_to_station_'||STATION.IDSTATION::text as IDEDGE,
                            STATION.POSITIONINEDGE * EDGE.LENGTH as LENGTH,
                            PARKINGEXPENSES
                    from    STATION, STREETSEGMENT as EDGE
                    where   EDGE.IDEDGE = STATION.IDEDGE_FK
                            
                    union all
                    
                    select  'station_'||STATION.IDSTATION::text as IDVERTEXORIG_FK,
                            EDGE.IDVERTEXDEST_FK::text as IDVERTEXDEST_FK,
                            STATION.IDSTATION::text||'_station_to_vertex_'||EDGE.IDVERTEXDEST_FK::text as IDEDGE,
                            EDGE.LENGTH - (STATION.POSITIONINEDGE * EDGE.LENGTH) as LENGTH,
                            PARKINGEXPENSES
                    from    STATION, STREETSEGMENT as EDGE
                    where   EDGE.IDEDGE = STATION.IDEDGE_FK
                              
                    union all
                    
                    select  EDGE.IDVERTEXORIG_FK::text as IDVERTEXORIG_FK,
                            'place_'||PLACE.IDPLACE::text as IDVERTEXDEST_FK,
                            EDGE.IDVERTEXORIG_FK::text||'_to_place_'||PLACE.IDPLACE::text as IDEDGE,
                            PLACE.POSITIONINEDGE * EDGE.LENGTH as LENGTH,
                            0 as PARKINGEXPENSES
                    from    PLACE, STREETSEGMENT as EDGE
                    where   EDGE.IDEDGE = PLACE.IDEDGE_FK
                            
                    union all
                    
                    select  'place_'||PLACE.IDPLACE::text as IDVERTEXORIG_FK,
                            EDGE.IDVERTEXDEST_FK::text as IDVERTEXDEST_FK,
                            PLACE.IDPLACE::text||'_place_to_vertex_'||EDGE.IDVERTEXDEST_FK::text as IDEDGE,
                            EDGE.LENGTH - (EDGE.LENGTH * PLACE.POSITIONINEDGE) as LENGTH,
                            0 as PARKINGEXPENSES
                    from    PLACE, STREETSEGMENT as EDGE
                    where   EDGE.IDEDGE = PLACE.IDEDGE_FK
                            '''
    dataFrameEdges = pd.read_sql_query(sqlQuery, conn)
    conn.close()

    #It must be loaded as a directed graph and then (last line of this method) convert it to undirected.
    #By doing so, the edges order is maintained. That makes it easier to test solutions in another scripts.
    G = nx.MultiDiGraph()
    for row in dataFrameEdges.itertuples():
        dictRow = row._asdict()
        keyAndIdEdge = dictRow['idvertexorig_fk'] + '-' + dictRow['idedge'] + '-' + dictRow['idvertexdest_fk']

        G.add_edge(dictRow['idvertexorig_fk'], dictRow['idvertexdest_fk'], key=keyAndIdEdge,
                    idedge=keyAndIdEdge, length=dictRow['length'], parkingexpenses=dictRow['parkingexpenses'])

    print(G.number_of_edges(), G.number_of_nodes())

    return G.to_undirected()

def loadTrips(stations, places, tripsAmountServed, model):
    params = {'host':'localhost', 'port':'5432', 'database':'afterqualifying', 'user':'cristiano', 'password':'cristiano'}
    conn = pg.connect(**params)

    sqlQuery = '''	select	'trip_'||TRIP.IDTRIP::text as IDTRIP,
                            TRIP.TRIPEXPANSIONFACTOR,
                            'place_'||TRIP.IDPLACEDEPARTURE::text as IDPLACEDEPARTURE,
                            'place_'||TRIP.IDPLACEDESTINATION::text as IDPLACEDESTINATION,
                            TRIP.TIMESTAMPDEPARTURE,
                            TRIP.TIMESTAMPARRIVAL,
                            TRIP.DRIVINGDISTANCE,
                            TRIP.DRIVINGDURATION
                    from	TRIP
                    where   TRIP.DRIVINGDISTANCE > 500
                    '''
    dataFrameEdges = pd.read_sql_query(sqlQuery, conn)
    conn.close()

    trips = dict()
    for row in dataFrameEdges.itertuples():
        dictRow = row._asdict()
        placeStart=places[dictRow['idplacedeparture']]
        placeEnd=places[dictRow['idplacedestination']]
        trip = Trip(idTrip=dictRow['idtrip'],
                    expansionFactor=dictRow['tripexpansionfactor'],
                    placeStart=placeStart,
                    placeEnd=placeEnd,
                    timestampDeparture=dictRow['timestampdeparture'],
                    timestampArrival=dictRow['timestamparrival'],
                    drivingDistance=dictRow['drivingdistance'],
                    drivingDuration=dictRow['drivingduration'],
                    model=model,
                    stations=stations,
                    tripsAmountServed=tripsAmountServed)
        trips[trip.idTrip] = trip

    return trips

def getVariable(varName, model):
    variable = None
    try:
        variable = model.getVarByName(varName)
    except:
        print('ERROR: VARIABLE', varName, 'NOT FOUND')
    finally:
        return variable

def timeToStr(timestamp):
    return timestamp.strftime('%m/%d/%Y, %H:%M:%S')

class Place:
    def __init__(self, G, idVertex, distanceCutOff):
        self.idVertex = idVertex

        self.reachedStations = list()
        reachedVertices = nx.single_source_dijkstra_path_length(G, self.idVertex, cutoff=distanceCutOff, weight='length')
        for item in reachedVertices.keys():
            if isinstance(item, str) and item.startswith('station'):
                self.reachedStations.append(item)

def createVariable(nameVar, model, lowerBound=None, upperBound=None):
    if lowerBound is None:
        if upperBound is None:
            createdVar = model.addVar(lb=0.0, vtype=GRB.INTEGER, name=nameVar)
        else:
            createdVar = model.addVar(lb=0.0, ub=upperBound, vtype=GRB.INTEGER, name=nameVar)

    else:
        if upperBound is None:
            createdVar = model.addVar(lb=lowerBound, vtype=GRB.INTEGER, name=nameVar)
        else:
            createdVar = model.addVar(lb=lowerBound, ub=upperBound, vtype=GRB.INTEGER, name=nameVar)
    
    createdVar.VTag = nameVar

    return createdVar

class Station:
    def __init__(self, idVertex, monthlyParkingExpenses, stationsInvolved, model):
        self.idVertex = idVertex
        self.monthlyParkingExpenses = monthlyParkingExpenses

        lowerBound = None
        if self.idVertex in stationsInvolved:
            lowerBound = 1
        
        self.variableStart = createVariable(nameVar=self.idVertex + '_start', lowerBound=lowerBound, model=model)
        self.tripsTimeStart = list()
        self.tripsTimeEnd = list()
        self.variableFlowBack = createVariable(nameVar=self.idVertex + '_flow_back', lowerBound=lowerBound, model=model)
    
    def defineIdleEdges(self, model):
        self.tripsTimeStart = sorted(self.tripsTimeStart)
        self.tripsTimeEnd = sorted(self.tripsTimeEnd)

        self.variablesIdle = list()
        for tupleTrip in self.tripsTimeStart:
            createdVar = createVariable(nameVar='idle_' + tupleTrip[1] + '_' + self.idVertex, model=model)
            self.variablesIdle.append(createdVar)
        
class Trip:
    #Values for defining the new timestamps
    ADJUSTED_MONTH = 1
    ADJUSTED_YEAR = 2017

    def adjustDay(timeDepartureOld, drivingDuration):
        #The weekday() starts from 0 but it does not exist a day 0 in the calendar. Then, a +1 solves this issue
        adjustedDayDeparture = timeDepartureOld.weekday() + 1

        timeDepartureNew = timeDepartureOld.replace(day=adjustedDayDeparture, month=Trip.ADJUSTED_MONTH, year=Trip.ADJUSTED_YEAR)
        timeArrivalNew = timeDepartureNew + timedelta(minutes=drivingDuration)
        
        return timeDepartureNew, timeArrivalNew

    def __init__(self, idTrip, expansionFactor, placeStart, placeEnd, timestampDeparture, timestampArrival, drivingDistance, drivingDuration, model, stations, tripsAmountServed):
        self.idTrip = idTrip
        self.expansionFactor = expansionFactor
        self.placeStart = placeStart
        self.placeEnd = placeEnd
        self.timestampDepartureOld = timestampDeparture
        self.timestampArrivalOld = timestampArrival
        self.drivingDistance = drivingDistance
        self.drivingDuration = drivingDuration

        self.timestampDeparture, self.timestampArrival = Trip.adjustDay(timestampDeparture, self.drivingDuration)

        lowerBound = 0
        if self.idTrip in tripsAmountServed:
            lowerBound = tripsAmountServed[self.idTrip]
        
        self.startStationsVars = list()
        for startStation in self.placeStart.reachedStations:
            createdVar = createVariable(nameVar=self.idTrip + '_start_' + startStation, upperBound=floor(self.expansionFactor), model=model)
            self.startStationsVars.append(createdVar)

            stations[startStation].tripsTimeStart.append((self.timestampDeparture, self.idTrip))
        
        self.endStationsVars = list()
        for endStation in self.placeEnd.reachedStations:
            createdVar = createVariable(nameVar=self.idTrip + '_end_' + endStation, upperBound=floor(self.expansionFactor), model=model)
            self.endStationsVars.append(createdVar)

            stations[endStation].tripsTimeEnd.append((self.timestampArrival, self.idTrip))

        self.variableFilter = createVariable(nameVar=self.idTrip + '_filter', lowerBound=lowerBound, upperBound=floor(self.expansionFactor), model=model)

#Discover which trips and stations were served and used in the Free-floating with parameters 500 meters and 0.7 multiplier
def selectTripsStationsInvolved():
    fileName = 'Optimal Solutions/Free Floating/4000/500/70.json'
    tripsAmountServed = {}
    stationsInvolved = set()
    with open(fileName) as jsonFile:
        optimalSolution = json.load(jsonFile)

        for var in optimalSolution["Vars"]:
            varVTag = var["VTag"][0]
            varVTagSplit = varVTag.split('_')

            if varVTag.startswith('station') and varVTag.endswith('start'):
                amountVehicles = int(var["X"])
                if amountVehicles > 0:
                    idStation = 'station_' + varVTagSplit[1]
                    stationsInvolved.add(idStation)

            elif varVTag.startswith('trip') and varVTag.endswith('filter'):
                idTrip = 'trip_' + varVTagSplit[1]
                amountServed = int(var["X"])
                if amountServed > 0:
                    tripsAmountServed[idTrip] = amountServed

            #Storing situations where the Free-Floating served a trip in a station with no allocated vehicles.
            elif varVTag.startswith('trip') and varVTagSplit[3] == 'station':
                amountServed = int(var["X"])
                if amountServed > 0:
                    idStation = 'station_' + varVTagSplit[4]
                    stationsInvolved.add(idStation)
                    
    return tripsAmountServed, stationsInvolved

#Calculate the profits difference between the optimal solution for the current parameters and using Free-Floating with 500 meters and 0.7 multiplier
def calculateProfitDifference(MINIMUM_FARE, RESERVATION_FEE, BASE_FARE, MINUTE_COST, KM_COST, priceMultiplier, trips, tripsServed):
    freeFloatingProfitDifference = 0
    for idTrip in tripsServed.keys():
        trip = trips[idTrip]
        freeFloatingProfitDifference += (0.7 - priceMultiplier) * max(MINIMUM_FARE,
                    RESERVATION_FEE + BASE_FARE + MINUTE_COST * trip.drivingDuration + KM_COST * trip.drivingDistance/1000)
    
    return freeFloatingProfitDifference

def buildGurobiModel(G, distanceCutOff=100, priceMultiplier=1, MONTHLY_CAR_RENTAL_COSTS=2815.01):
    #Defining Uber costs
    MINIMUM_FARE = 5.28
    RESERVATION_FEE = 0.75
    BASE_FARE = 2.67
    MINUTE_COST = 0.3
    KM_COST = 1.03

    #Defining Gas prices. Price from https://precodoscombustiveis.com.br/pt-br/city/brasil/sao-paulo/sao-paulo/3830 got day 2022/11/04.
    GAS_LITER = 7.07
    DISTANCE_PER_LITER = 10
    GAS_PER_KM = GAS_LITER / DISTANCE_PER_LITER

    #Discovering which trips were served and which stations were used in the Free Floating optimization
    tripsAmountServed, stationsInvolved = selectTripsStationsInvolved()

    #Create a Gurobi Model
    model = gp.Model("IEEE T-ITS")

    objective = 0
    #Create the variables and define the objective
    print("Creating the variables")
    stations = dict()
    places = dict()
    for idVertex in G.nodes():
        if isinstance(idVertex, str):
            if idVertex.startswith('station'):
                #All edges connected to a station has the same parkingExpenses attribute. Then [0][2] will get the attribute from the first edge
                monthlyParkingExpenses = list(G.edges(idVertex, data='parkingexpenses'))[0][2]
                station = Station(idVertex, monthlyParkingExpenses, stationsInvolved, model)
                stations[station.idVertex] = station

            elif idVertex.startswith('place'):
                place = Place(G, idVertex, distanceCutOff)
                places[place.idVertex] = place
    
    trips = loadTrips(stations, places, tripsAmountServed, model)

    for station in stations.keys():
        if len(stations[station].tripsTimeStart) > 0:
            stations[station].defineIdleEdges(model)

    #Calculate the profit difference between the optimizations
    profitDifference = calculateProfitDifference(   MINIMUM_FARE=MINIMUM_FARE,
                                                    RESERVATION_FEE=RESERVATION_FEE,
                                                    BASE_FARE=BASE_FARE,
                                                    MINUTE_COST=MINUTE_COST,
                                                    KM_COST=KM_COST,
                                                    priceMultiplier=priceMultiplier,
                                                    trips=trips,
                                                    tripsServed=tripsAmountServed)

    #Defining the objective and constraints
    print("Defining the objective and constraints")
    model.update()
    objective = profitDifference
    for idStation, station in stations.items():
        #The parking expenses data is monthly but the simulation is weekly, then it is divided by four.
        #Also, it is a cost. Then the parking expenses is multiplied by minus one. The same for the rental cost.
        objective += -1 * ((MONTHLY_CAR_RENTAL_COSTS + station.monthlyParkingExpenses)/4) * station.variableFlowBack

        #Adding constraint about the first station vertex
        model.addConstr(station.variableFlowBack - station.variableStart == 0, idStation + '_first_vertex')

        #Defining objective and adding constraints regarding the "middle" of the period assessed
        #Initialize with a small timestamp
        earlierTimeStart = datetime(1, 1, 1, 0, 0)
        for i, (timestampDeparture, idTrip) in enumerate(station.tripsTimeStart):
            trip = trips[idTrip]

            tripStationVar = getVariable(idTrip + '_start_' + idStation, model)
            objective += priceMultiplier * tripStationVar * max(MINIMUM_FARE,
                        RESERVATION_FEE + BASE_FARE + MINUTE_COST * trip.drivingDuration + KM_COST * trip.drivingDistance/1000) - tripStationVar * GAS_PER_KM * trip.drivingDistance/1000
            
            idleTripStationVar = getVariable('idle_' + idTrip + '_' + idStation, model)

            returnedVehiclesVars = list()
            flagBreak = False
            for timeArrivalOtherTrip, otherIdTrip in station.tripsTimeEnd:
                #Checking if the otherTrip happens after the earlier trip made in this station and before the current trip starts
                #If the current trip is the first one, earlierTimeStart has a small timestamp value
                if timeArrivalOtherTrip > earlierTimeStart and timeArrivalOtherTrip < timestampDeparture:
                    otherTripVar = getVariable(otherIdTrip + '_end_' + idStation, model)
                    returnedVehiclesVars.append(otherTripVar)

                    flagBreak = True
                #Since the timestamps are sorted, when the if above stop being satisfied, it will never be satisfied again.
                elif flagBreak:
                    break

            if i == 0:
                model.addConstr(station.variableStart + sum(returnedVehiclesVars) - tripStationVar - idleTripStationVar == 0, 'cnstr_idle_' + idTrip + '_' + idStation)

            elif i < len(station.tripsTimeStart):
                lastIdleTripVar = getVariable('idle_' + station.tripsTimeStart[i-1][1] + '_' + idStation, model)
                model.addConstr(lastIdleTripVar + sum(returnedVehiclesVars) - tripStationVar - idleTripStationVar == 0, 'cnstr_idle_' + idTrip + '_' + idStation)
            
            earlierTimeStart = timestampDeparture

        #Adding constraint tying the first and the last station vertex
        returnedVehiclesVars = list()
        if len(station.tripsTimeStart) > 0:
            timeLastTripStation = station.tripsTimeStart[-1][0]
            lastIdleTripVar = getVariable('idle_' + station.tripsTimeStart[-1][1] + '_' + idStation, model)
        else:
            timeLastTripStation = datetime(1, 1, 1, 0, 0)
            lastIdleTripVar = station.variableStart
            
        for timeArrivalOtherTrip, otherIdTrip in reversed(station.tripsTimeEnd):
            #Checking if the otherTrip happens after the last trip made in this station
            if timeArrivalOtherTrip > timeLastTripStation:
                otherTripVar = getVariable(otherIdTrip + '_end_' + idStation, model)
                returnedVehiclesVars.append(otherTripVar)
            else:
                break

        model.addConstr(sum(returnedVehiclesVars) + lastIdleTripVar - station.variableFlowBack == 0, idStation + '_last_vertex')
    
    #Defining the constraints regarding the green vertices (filters)
    for idTrip, trip in trips.items():
        tripFilterVar = getVariable(idTrip + '_filter', model)
        model.addConstr(sum(trip.startStationsVars) - tripFilterVar == 0, 'cstr_start_' + idTrip)
        model.addConstr(tripFilterVar - sum(trip.endStationsVars) == 0, 'cstr_end_' + idTrip)

    #Set objective: maximize the utility value by allocating stations on edges
    model.setObjective(objective, GRB.MAXIMIZE)

    print("MODEL BUILT!!")
    return model

sc = gp.StatusConstClass
gurobiStatus = {sc.__dict__[k]: k for k in sc.__dict__.keys() if k[0] >= 'A' and k[0] <= 'Z'}

folderSaveSolutions = 'Optimal Solutions/Mixed Partial Floating'
flagError = False
G = loadMultiGraph()
#Defining Localiza Meoo car rental costs for 3.000 km monthly mileage limit and contract of 12 months in São Paulo
#THIS PRICE DOES NOT INCLUDES PROTECTION FOR GLASSES AND TIRES. PRICE AS OF APRIL 10th.
LOCALIZA_MONTHLY_RENTAL_3000_NO_GLASSES_TIRES = 2521
#Defining Movida Mensal Flex car rental costs for 3.000 km monthly mileage limit and contract of 1 month in São Paulo for Mobi Like or similar (AX Group).
#THIS PRICE DOES NOT INCLUDES PROTECTION FOR GLASSES AND TIRES. PRICE AS OF APRIL 10th. ADDITIONAL KMs WILL COST R$0,49.
MOVIDA_MONTHLY_RENTAL_3000_NO_GLASSES_TIRES = 2065.73
#Defining Movida Mensal Flex car rental costs for 3.000 km monthly mileage limit and contract of 1 month in São Paulo for Mobi Like or similar (AX Group).
#THIS PRICE INCLUDES PROTECTION FOR GLASSES AND TIRES. PRICE AS OF APRIL 10th, 2022. ADDITIONAL KMs WILL COST R$0,49.
MOVIDA_MONTHLY_RENTAL_3000_WITH_GLASSES_TIRES = 2468.93
#Defining Movida Mensal Flex car rental costs for 4.000 km monthly mileage limit and contract of 1 month in São Paulo for Mobi Like or similar (AX Group).
#THIS PRICE DOES NOT INCLUDES PROTECTION FOR GLASSES AND TIRES. PRICE AS OF APRIL 10th, 2022. ADDITIONAL KMs WILL COST R$0,49.
MOVIDA_MONTHLY_RENTAL_4000_NO_GLASSES_TIRES = 2411.81
#Defining Movida Mensal Flex car rental costs for 4.000 km monthly mileage limit and contract of 1 month in São Paulo for Mobi Like or similar (AX Group).
#THIS PRICE INCLUDES PROTECTION FOR GLASSES AND TIRES. PRICE AS OF APRIL 10th, 2022. ADDITIONAL KMs WILL COST R$0,49.
MOVIDA_MONTHLY_RENTAL_4000_WITH_GLASSES_TIRES = 2815.01

pricesMileageLimits = { #3000: MOVIDA_MONTHLY_RENTAL_3000_WITH_GLASSES_TIRES,
                        4000: MOVIDA_MONTHLY_RENTAL_4000_WITH_GLASSES_TIRES}

for mileageLimit in pricesMileageLimits.keys():
    #It is only simulated for distance of 500 meters because the Free-Floating used this, and smaller distances do not reach the same stations
    for distanceCutOff in [500]:#[100, 200, 300, 400, 500]:
        for priceMultiplier in np.arange(2, 0, -0.1):
            adjustedPriceMultiplier = round(priceMultiplier, 1)
            model = buildGurobiModel(G, distanceCutOff, adjustedPriceMultiplier, pricesMileageLimits[mileageLimit])
            try:
                #model.Params.Presolve = 0
                #model.Params.Method = 0
                model.Params.LogToConsole = 0
                model.optimize()

                print('MODEL STATUS IS', gurobiStatus[model.status])
                print('MILEAGE:', mileageLimit, 'DISTANCE:', distanceCutOff, 'MULTIPLIER:', adjustedPriceMultiplier)

                if model.status == GRB.OPTIMAL:
                    print("OBJECTIVE VALUE:", model.objVal)
                    for variable in model.getVars():
                        if abs(variable.X - round(variable.X)) > 0.001:
                            print("ERROR:\tNOT AN INTEGER VALUE!!!", variable.VarName, variable.X)
                            flagError = True
                            break

                    if model.objVal > 0.1:
                        folderPath = Path('./' + folderSaveSolutions + '/' + str(mileageLimit) + '/' + str(distanceCutOff))
                        folderPath.mkdir(parents=True, exist_ok=True)
                        #Multiplier as percentage is easier to explain, then it is multiplied by 100. The round function was needed to fix numeric errors.
                        percentagePrice = int(round(adjustedPriceMultiplier*100, 0))
                        fileName = folderPath / (str(percentagePrice) + '.json')
                        model.write(str(fileName.resolve()))
                    else:
                        break
                
                elif model.status == GRB.INFEASIBLE:
                    model.computeIIS()
                    model.write("modelInfeasible.ilp")
                    break

            except gp.GurobiError as e:
                print("ERROR:", str(e))
                break

            if flagError:
                break
        if flagError:
            break