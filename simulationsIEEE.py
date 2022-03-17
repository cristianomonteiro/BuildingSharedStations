import networkx as nx
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import psycopg2 as pg

def loadMultiGraph():
    params = {'host':'localhost', 'port':'5432', 'database':'afterqualifying', 'user':'cristiano', 'password':'cristiano'}
    conn = pg.connect(**params)

    sqlQuery = '''	select	EDGE.IDVERTEXORIG_FK::text,
                            EDGE.IDVERTEXDEST_FK::text,
                            EDGE.IDEDGE::text,
                            EDGE.LENGTH,
                            EDGE.UTILITYVALUE
                    from	STREETSEGMENT as EDGE
                    
                    union all
                    
                    select  EDGE.IDVERTEXORIG_FK::text as IDVERTEXORIG_FK,
                            'station_'||STATION.IDSTATION::text as IDVERTEXDEST_FK,
                            EDGE.IDVERTEXORIG_FK::text||'_to_station_'||STATION.IDSTATION::text as IDEDGE,
                            STATION.POSITIONINEDGE as LENGTH,
                            0 as UTILITYVALUE
                    from    STATION, STREETSEGMENT as EDGE
                    where   EDGE.IDEDGE = STATION.IDEDGE_FK
                            
                    union all
                    
                    select  'station_'||STATION.IDSTATION::text as IDVERTEXORIG_FK,
                            EDGE.IDVERTEXDEST_FK::text as IDVERTEXDEST_FK,
                            STATION.IDSTATION::text||'_station_to_vertex_'||EDGE.IDVERTEXDEST_FK::text as IDEDGE,
                            case    when    (EDGE.LENGTH - STATION.POSITIONINEDGE) <= 1 then EDGE.LENGTH
                                    else   (EDGE.LENGTH - STATION.POSITIONINEDGE)
                            end as LENGTH,
                            0 as UTILITYVALUE
                    from    STATION, STREETSEGMENT as EDGE
                    where   EDGE.IDEDGE = STATION.IDEDGE_FK
                              
                    union all
                    
                    select  EDGE.IDVERTEXORIG_FK::text as IDVERTEXORIG_FK,
                            'place_'||PLACE.IDPLACE::text as IDVERTEXDEST_FK,
                            EDGE.IDVERTEXORIG_FK::text||'_to_place_'||PLACE.IDPLACE::text as IDEDGE,
                            PLACE.POSITIONINEDGE as LENGTH,
                            0 as UTILITYVALUE
                    from    PLACE, STREETSEGMENT as EDGE
                    where   EDGE.IDEDGE = PLACE.IDEDGE_FK
                            
                    union all
                    
                    select  'place_'||PLACE.IDPLACE::text as IDVERTEXORIG_FK,
                            EDGE.IDVERTEXDEST_FK::text as IDVERTEXDEST_FK,
                            PLACE.IDPLACE::text||'_place_to_vertex_'||EDGE.IDVERTEXDEST_FK::text as IDEDGE,
                            case    when    (EDGE.LENGTH - PLACE.POSITIONINEDGE) <= 1 then EDGE.LENGTH
                                    else   (EDGE.LENGTH - PLACE.POSITIONINEDGE)
                            end as LENGTH,
                            0 as UTILITYVALUE
                    from    PLACE, STREETSEGMENT as EDGE
                    where   EDGE.IDEDGE = PLACE.IDEDGE_FK
                            '''
    dataFrameEdges = pd.read_sql_query(sqlQuery, conn)
    conn.close()

    G = nx.MultiGraph()
    for row in dataFrameEdges.itertuples():
        dictRow = row._asdict()
        keyAndIdEdge = dictRow['idvertexorig_fk'] + '-' + dictRow['idedge'] + '-' + dictRow['idvertexdest_fk']

        G.add_edge(dictRow['idvertexorig_fk'], dictRow['idvertexdest_fk'], key=keyAndIdEdge,
                    idedge=keyAndIdEdge, length=dictRow['length'], utilityvalue=dictRow['utilityvalue'])

    print(G.number_of_edges(), G.number_of_nodes())

    return G

def loadTrips(places, model):
    params = {'host':'localhost', 'port':'5432', 'database':'afterqualifying', 'user':'cristiano', 'password':'cristiano'}
    conn = pg.connect(**params)

    sqlQuery = '''	select	TRIP.IDTRIP,
                            TRIP.TRIPEXPANSIONFACTOR,
                            'place_'||TRIP.IDPLACEDEPARTURE::text as IDPLACEDEPARTURE,
                            'place_'||TRIP.IDPLACEDESTINATION::text as IDPLACEDESTINATION,
                            TRIP.TIMESTAMPDEPARTURE,
                            TRIP.TIMESTAMPARRIVAL,
                            TRIP.DRIVINGDISTANCE,
                            TRIP.DRIVINGDURATION
                    from	TRIP
                    '''
    dataFrameEdges = pd.read_sql_query(sqlQuery, conn)
    conn.close()

    trips = list()
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
                    model=model)
        trips.append(trip)

    return trips

def getVariable(model, varName):
    variable = None
    try:
        variable = model.getVarByName(varName)
    finally:
        return variable

class Place:
    def __init__(self, G, idVertex, distanceCutOff, model):
        self.idVertex = idVertex

        self.reachedStationsVars = list()
        reachedStations = nx.single_source_dijkstra_path_length(G, self.idVertex, cutoff=distanceCutOff, weight='length')
        for item in reachedStations.keys():
            if isinstance(item, str) and item.startswith('station'):
                self.reachedStationsVars.append(getVariable(model, item))

class Station:
    def __init__(self, idVertex, model):
        self.idVertex = idVertex
        self.variable = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=self.idVertex)

class Trip:
    def __init__(self, idTrip, expansionFactor, placeStart, placeEnd, timestampDeparture, timestampArrival, drivingDistance, drivingDuration, model):
        self.idTrip = idTrip
        self.expansionFactor = expansionFactor
        self.placeStart = placeStart
        self.placeEnd = placeEnd
        self.timestampDeparture = timestampDeparture
        self.timestampArrival = timestampArrival
        self.drivingDistance = drivingDistance
        self.drivingDuration = drivingDuration
        self.variable = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='trip_' + str(self.idTrip))

def buildGurobiModel(distanceCutOff=100):
    G = loadMultiGraph()

    #Create a Gurobi Model
    model = gp.Model("IEEE T-ITS")

    objective = 0
    #Create the variables and define the objective
    print("Creating the variables")
    stations = list()
    for idVertex in G.nodes():
        if isinstance(idVertex, str) and idVertex.startswith('station'):
            station = Station(idVertex, model)
            stations.append(station)

    model.update()
    places = dict()
    for idVertex in G.nodes():
        if isinstance(idVertex, str) and idVertex.startswith('place'):
            place = Place(G, idVertex, distanceCutOff, model)
            places[place.idVertex] = place
        
    trips = loadTrips(places, model)
    model.update()
    
    objective = 0
    #Defining the constraints
    print("Defining the constraints")
            
    #Set objective: maximize the utility value by allocating stations on edges
    model.setObjective(objective, GRB.MAXIMIZE)

    print("MODEL BUILT!!")
    return model

for distanceCutOff in [100, 200, 300, 400, 500]:
    model = buildGurobiModel(distanceCutOff)

    try:
        model.optimize()

    except gp.GurobiError as e:
        print("ERROR: " + str(e))
        break