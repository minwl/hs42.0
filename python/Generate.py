import json

from Define import *
from voyage.RefRoute import FindRefRoute
from voyage.Manager import VoyageManager
from voyage.Ship.Ship import cShip
from voyage.Ship.Dynamics import Hull, Propeller, Rudder
from voyage.Ship.Env import Current, Wave, Wind
from voyage.Ship.Machinery import MainEngine
import copy 
import pickle as pkl


def Make_RefPlan(waypts_list, p_graph_data:dict):
    vp =  Data_VoyagePlan()
    vp.Set_Data(waypts_list)
    FullPlan = []
    n_wpts = len(vp.waypoints)
    i = 0
    j = 1
    while i < n_wpts and j < n_wpts:
        
        currWP = vp.waypoints[i]
        nextWP = vp.waypoints[j]
        if(float(nextWP.no) - float(currWP.no) == 1):
            if (j-i == 1):  #sub wp 없는 구간 : ref plan 생성 
                dep = [float(currWP.lon), float(currWP.lat)]
                arr = [float(nextWP.lon), float(nextWP.lat)]
                refroute = FindRefRoute(dep, arr, p_graph_data)
                if refroute is not None:
                    refsection = [currWP]
                    for k, refPT in enumerate(refroute):
                        subWP = Data_WayPoint()
                        subWP.no = str(float(currWP.no) +((k+1)/1000))
                        subWP.lat = refPT[1]
                        subWP.lon = refPT[0]
                        refsection.append(subWP)
                    FullPlan.extend(refsection)
                    i=j
                else : 
                    print("no route found")
                    return vp                 
            else: #sub wp 있는 구간 : Plan 유지
                FullPlan.extend(vp.waypoints[i:j])
                i=j
        else: #다음 main wp 찾아가기
            j+=1
    FullPlan.append(vp.waypoints[-1])
    vp.Set_WP(FullPlan)
    
    return vp

def Make_ShipModel(p_conn:psycopg.Connection, p_ship_id:str, p_weather_pickle_path:str):
    #region load data
    eng_data_dict = get_engineering_data(p_conn, p_ship_id)
    principal = json.loads(eng_data_dict["ShipPrincipal"])
    propulsion =  json.loads(eng_data_dict["ShipSelfPropulsion"])
    hull_res =  json.loads(eng_data_dict["HullResistance"])
    main_engine =  json.loads(eng_data_dict["EnginePerformance"])
    propeller =  json.loads(eng_data_dict["PropellerPOW"])
    wind_coef =  json.loads(eng_data_dict["WindLoadCoefficient"])
    wave_qtf =  json.loads(eng_data_dict["WaveDriftSurgeQTF"])
    forecast = pkl.load(open(p_weather_pickle_path, 'rb'))
    #end region load data

    #region set engineering data
    ship_principal = Data_ShipPrincipal
    ship_principal.Set_Data(principal)

    ship_propulsion = Data_ShipPropulsion()
    ship_propulsion.Set_Data(propulsion)

    ship_hull = Data_HullResistance()
    ship_hull.Set_Data(hull_res)
   
    ship_ME = Data_MainEngine()
    ship_ME.Set_Data(main_engine)

    ship_Propeller = Data_PropellerPOW()
    ship_Propeller.Set_Data(propeller)

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

def GeneratePlan(p_conn:psycopg.Connection, shipid, waypts_list, draft, p_graph_data:dict, p_weather_pickle_path:str):
    RefPlan = Make_RefPlan(waypts_list, p_graph_data)
    Ship = Make_ShipModel(p_conn, shipid, p_weather_pickle_path)
    draft = float(draft)
    vm = VoyageManager(Ship)
    waypoints = vm.Generate(RefPlan.waypoints, draft)
    EvalPlan = Data_VoyagePlan()
    EvalPlan.Set_WP(waypoints)
    return {'waypoints' : EvalPlan.waypts_dict}
