from Define import *
from voyage.RefRoute import FindRefRoute
from voyage.node_dict import coordlist
from voyage.Manager import VoyageManager
from voyage.Ship.Ship import cShip
from voyage.Ship.Dynamics import Hull, Propeller, Rudder
from voyage.Ship.Env import Current, Wave, Wind
from voyage.Ship.Machinery import MainEngine
import copy 
import requests
import pickle as pkl


#운항 중 주기적으로 현재 위치와 시간을 고려해 최적 경로를 리턴하는 함수
#1시간 단위로 실행, 현재 운항중인 plan 과 현재 위치, 시간 정보 입력

def Make_RefPlan(waypts_list:dict, lat, lon, time, p_graph_data):
    vp = Data_VoyagePlan()
    vp.Set_Data(waypts_list)

    #현재 위치에 대한 waypoint 객체 생성
    input_wp = Data_WayPoint()
    input_wp.lat = lat
    input_wp.lon = lon
    input_wp.time = str(datetime.strptime(time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())

    #기존 경로의 timeseries 추출
    WPList = vp.waypoints
    timeList = [wp.ToDict()['time'] for wp in WPList]

    #시간 기준으로 현재 시간의 index 추출
    last_idx = sorted(timeList+[time]).index(time)-1

    #현재의 waypoint를 main waypoint로 설정 (no를 정수로 변경)
    input_wp.no = int(float(WPList[last_idx].no))

    WPList = WPList[last_idx+1:]
    WPList.insert(0, input_wp)
    MainWP = []
    #Fine Main Waypoint
    for idx, wp in enumerate(WPList): 
        if float(wp.no) % 1 != 0 : continue
        else:
            MainWP.append([idx, int(wp.no), float(wp.lon), float(wp.lat)])
    
    #Make Ref Waypoint
    RefRoutes = []
    for i in range(len(MainWP)-1):
        dep = MainWP[i][2:]
        arr = MainWP[i+1][2:]
        dep_i = MainWP[i][0]
        arr_i = MainWP[i+1][0]
        dep_no = MainWP[i][1]
        RefRoutes.append(WPList[dep_i])
        refroute = FindRefRoute(dep, arr, p_graph_data)
        if refroute is not None:
            for j, refPT in enumerate(refroute):
                subWP = Data_WayPoint()
                subWP.no = str(dep_no +((j+1)/1000))
                subWP.lat = refPT[1]
                subWP.lon = refPT[0]
                RefRoutes.append(subWP)
                
            RefRoutes.append(WPList[arr_i])
            
        else : return vp

    vp.Set_WP(RefRoutes)

    return vp

def Make_ShipModel(p_conn:psycopg.Connection, p_ship_id:str):
    #region load json string

    '''this needs to be replaced with API call responses (json string format)'''

    # forecast = json.dumps(json.load(open('../../../json_data/forecast.json', 'r')))
    eng_data_dict = get_engineering_data(p_conn, p_ship_id)
    principal = eng_data_dict["ShipPrincipal"]
    propulsion =  eng_data_dict["ShipSelfPropulsion"]
    hull_res =  eng_data_dict["HullResistance"]
    main_engine =  eng_data_dict["EnginePerformance"]
    propeller =  eng_data_dict["PropellerPOW"]
    wind_coef =  eng_data_dict["WindLoadCoefficient"]
    wave_qtf =  eng_data_dict["WaveDriftSurgeQTF"]
    forecast = pkl.load(open('/home/tapp/weather.pickle', 'rb'))    
    #end region load json string


    #region set engineering data
    ship_principal = Data_ShipPrincipal
    ship_principal.Set_Data(jsonstr=principal)

    ship_propulsion = Data_ShipPropulsion()
    ship_propulsion.Set_Data(jsonstr=propulsion)

    ship_hull = Data_HullResistance()
    ship_hull.Set_Data(jsonstr=hull_res)
   
    ship_ME = Data_MainEngine()
    ship_ME.Set_Data(jsonstr=main_engine)

    ship_Propeller = Data_PropellerPOW()
    ship_Propeller.Set_Data(jsonstr=propeller)

    ship_windcoef = Data_WindLoadCoefficienct()
    ship_windcoef.Set_Data(wind_coef)

    ship_waveqtf = Data_WaveDriftSurgeQTF()
    ship_waveqtf.Set_Data(wave_qtf)

    ship_wind = Data_Forecast_Wind()
    ship_wave = Data_Forecast_Wave()
    ship_current = Data_Forecast_Current()
    ship_wind.Set_Data(forecast)
    ship_wave.Set_Data(forecast)
    ship_current.Set_Data(forecast)
    #end region set engineering data

    #region make ship model
    ship = cShip(ship_principal, ship_propulsion)
    hull = Hull.cHull(ship_hull)
    propeller1 = Propeller.cPropeller(ship_Propeller)
    propeller2 = Propeller.cPropeller(ship_Propeller)
    current = Current.cCurrent()
    current.setData(ship_current)
    wave = Wave.cWave(ship_wave, ship_waveqtf)
    wind = Wind.cWind()
    wind.setData(ship_wind, ship_windcoef)
    main_eng1 = MainEngine.cMainEngine(ship_ME)
    main_eng2 = MainEngine.cMainEngine(ship_ME)

    ship.setHull(hull)
    ship.setProp(propeller1)
    ship.setProp(propeller2)
    ship.setCurrent(current)
    ship.setWave(wave)
    ship.setWind(wind)
    ship.setME(main_eng1)
    ship.setME(main_eng2)
    #end region make ship model

    return ship

def AutoOptimizePlan(p_conn:psycopg.Connection, p_ship_id:str, waypts_list, lat, lon, time, draft, Options:dict, p_graph_data:dict):
    RefPlan = Make_RefPlan(waypts_list, lat, lon, time, p_graph_data)
    Ship = Make_ShipModel(p_conn, p_ship_id)
    draft = float(draft)
    vm = VoyageManager(Ship)
    refwpts= vm.Generate(RefPlan.waypoints, draft)
    candid = dict()
    for idx, wp in enumerate(refwpts):
        if idx == 0 or idx == len(refwpts) : continue
        candid[wp.no] = []
        for c in coordlist:
            c_lon = c[0]
            c_lat = c[1]
            if (float(wp.lon)- 3 <= c_lon <= float(wp.lon)+3) and (float(wp.lat) - 3 <= c_lat <= float(wp.lat) + 3):
                c_wp = Data_WayPoint()
                c_wp.lat = c_lat
                c_wp.lon = c_lon
                c_wp.no = wp.no
                candid[wp.no].append(c_wp)
    
    waypoints = vm.Optimize(refwpts, candid, draft, Options)
    OptPlan = Data_VoyagePlan()
    OptPlan.Set_WP(waypoints)

    return  {'waypoints' : OptPlan.waypts_dict}

