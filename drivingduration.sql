update	CARSHARING.TRIP
set		DRIVINGDURATION = ((DRIVINGDISTANCE/1000) /
						  		(	select	avg(TRIP2.DRIVINGDISTANCE/1000) / avg(TRIP2.DRIVINGDURATION/60)
								 	from	CARSHARING.TRIP as TRIP2
									where	TRIP2.DRIVINGDURATION is not NULL and
											extract(dow from TRIP2.TIMESTAMPDEPARTURE) = extract(dow from TIMESTAMPDEPARTURE) and
											extract(hour from TRIP2.TIMESTAMPDEPARTURE) = extract(hour from TIMESTAMPDEPARTURE))
						  )*60
where	DRIVINGDURATION is NULL 
