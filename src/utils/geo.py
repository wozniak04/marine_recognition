from numpy import radians, sin, cos, sqrt, asin, atan2, degrees

# Calculace traveled distance
def haversine(lat1, lon1, lat2, lon2): 
    R = 6371000 # Earth's radius in meters

    lat1, lat2 = radians(lat1), radians(lat2)
    dlat = lat2 - lat1
    
    lon1, lon2 = radians(lon1), radians(lon2)
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance = 2 * R * asin(sqrt(a))
    
    return distance

def calc_course(lat1, lon1, lat2, lon2):
    lat1, lat2 = radians(lat1), radians(lat2)
    dlat = lat2 - lat1
    
    lon1, lon2 = radians(lon1), radians(lon2)
    dlon = lon2 - lon1
    
    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    
    course = degrees( atan2(x, y) )
    return (course + 360) % 360 # normalization, e.g. -10* => 350*