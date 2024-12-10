from Define import *
import statistics as stat
from Generate import GeneratePlan


"""
def GetArrInfo(ship_id):
    #  return '2024-10-04 21:03:10', '34.833', '128.417'
# 
    #get AIS data
    url = f"https://svmp.seavantage.com/api/v1/ship/search?keyword={ship_id}"
    headers = {"Authorization": "Basic ZHNtZV9zZXJ2aWNlQGRzbWUudXNlcjpzbWFydHNoaXAxIQ=="}
    response = requests.get(url=url, headers=headers, data={}, timeout=3)
    response_value = response.json().get('response', None)
    
    #load port_data 
    port_data = json.load(open('port_data.json', 'r'))

    arr_time = response_value[0]['eta']
    #destination parsing 
    arr_port = response_value[0]['position']['aisDestination'].split('>')[-1]
    arr_lat, arr_lon = port_data.get(arr_port, [None, None])

    return arr_time, arr_lat, arr_lon


def VoyageDetect(ship_id, latest_5hr_data):
    #type casting
    for k, vlist in latest_5hr_data.items():
        if k != 'time' : 
            latest_5hr_data[k] = [float(v) for v in vlist]

    if len(latest_5hr_data) < 5 : 
        return None
    
    InitPlan = Data_VoyagePlan()
    if (stat.mean(latest_5hr_data['sog'][0:2]) < 1 and
        latest_5hr_data['sog'][2] >= 5 and
        stat.mean(latest_5hr_data['sog'][2:]) >= 5 
        ):
        
        if (stat.mean(latest_5hr_data['foc'][0:2]) < 30 and
            latest_5hr_data['foc'][2] >= 30 and
            stat.mean(latest_5hr_data['foc'][2:] )>= 30 
            ):
            
            dep_time = latest_5hr_data['time'][2]
            dep_lat = latest_5hr_data['lat'][2]
            dep_lon = latest_5hr_data['lon'][2]
            dep_draft = latest_5hr_data['draft'][2]
            arr_time, arr_lat, arr_lon = GetArrInfo(ship_id)

            dep_wp = Data_WayPoint()
            dep_wp.no = "1"
            dep_wp.time = dep_time
            dep_wp.lat = dep_lat
            dep_wp.lon = dep_lon


            arr_wp = Data_WayPoint()
            arr_wp.no = "2"
            arr_wp.time= arr_time
            arr_wp.lat = arr_lat
            arr_wp.lon = arr_lon

            InitPlan.Set_WP([dep_wp, arr_wp])
            print(InitPlan.waypts_dict)
            waypoints =[]
            for wp in InitPlan.waypts_dict:
                waypoints.append(str(wp))
            print(waypoints)
            waypoints = GeneratePlan(ship_id, waypoints, dep_draft)['waypoints']
            return {'status': True, 'waypoints' : waypoints}
        
    return {'status':False, 'waypoints' : []}
"""


def VoyageDetect(p_conn:psycopg.Connection, p_ship_id:str, p_latest_5hr_data:dict,
                 p_arr_time:str, p_arr_lat:float, p_arr_lon:float, p_graph_data:dict, p_weather_pickle_path:str):
    #type casting
    for k, vlist in p_latest_5hr_data.items():
        if k != 'time' : 
            p_latest_5hr_data[k] = [float(v) for v in vlist]

    if len(p_latest_5hr_data) < 5 : 
        return {'status':False, 'waypoints': None}
    
    InitPlan = Data_VoyagePlan()
    if (stat.mean(p_latest_5hr_data['sog'][0:2]) < 1 and
        p_latest_5hr_data['sog'][2] >= 5 and
        stat.mean(p_latest_5hr_data['sog'][2:]) >= 5 
        ):
        
        if (stat.mean(p_latest_5hr_data['foc'][0:2]) < 30 and
            p_latest_5hr_data['foc'][2] >= 30 and
            stat.mean(p_latest_5hr_data['foc'][2:] )>= 30):
            
            dep_time = p_latest_5hr_data['time'][2]
            dep_lat = p_latest_5hr_data['lat'][2]
            dep_lon = p_latest_5hr_data['lon'][2]
            dep_draft = p_latest_5hr_data['draft'][2]
            arr_time = p_arr_time
            arr_lat = p_arr_lat
            arr_lon = p_arr_lon

            dep_wp = Data_WayPoint()
            dep_wp.no = "1"
            dep_wp.time = dep_time
            dep_wp.lat = dep_lat
            dep_wp.lon = dep_lon


            arr_wp = Data_WayPoint()
            arr_wp.no = "2"
            arr_wp.time= arr_time
            arr_wp.lat = arr_lat
            arr_wp.lon = arr_lon

            InitPlan.Set_WP([dep_wp, arr_wp])
            print(InitPlan.waypts_dict)
            waypoints =[]
            for wp in InitPlan.waypts_dict:
                waypoints.append(str(wp))
            print(waypoints)
            waypoints = GeneratePlan(p_conn, p_ship_id, waypoints, dep_draft, p_graph_data, p_weather_pickle_path)['waypoints']
            return {'status': True, 'waypoints' : waypoints}
        
    return {'status':False, 'waypoints' : []}

