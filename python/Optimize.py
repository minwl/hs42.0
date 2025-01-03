import sys, os

from Define import *
from voyage.node_dict import coordlist
from voyage.Manager import VoyageManager
from voyage.Ship.Ship import cShip
from voyage.Ship.Dynamics import Hull, Propeller, Rudder
from voyage.Ship.Env import Current, Wave, Wind
from voyage.Ship.Geo import Waterdepth
from voyage.Ship.Machinery import MainEngine
import copy 
import requests
import psycopg
import pickle as pkl

def Make_ShipModel(p_conn:psycopg.Connection, p_ship_id:str, p_weather_pickle_path:str, p_depth_pickle_path:str):
    #region load json string
    eng_data_dict = get_engineering_data(p_conn, p_ship_id)
    principal = json.loads(eng_data_dict["ShipPrincipal"])
    propulsion =  json.loads(eng_data_dict["ShipSelfPropulsion"])
    hull_res =  json.loads(eng_data_dict["HullResistance"])
    main_engine =  json.loads(eng_data_dict["EnginePerformance"])
    propeller =  json.loads(eng_data_dict["PropellerPOW"])
    wind_coef =  json.loads(eng_data_dict["WindLoadCoefficient"])
    wave_qtf =  json.loads(eng_data_dict["WaveDriftSurgeQTF"])
    forecast = pkl.load(open(p_weather_pickle_path, 'rb'))
    waterdepth = pkl.load(open(p_depth_pickle_path, 'rb'))
    #end region load json string

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
    
    water_depth = Data_WaterDepth()
    water_depth.Set_Data(waterdepth)
    
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

    depth = Waterdepth.cWaterdepth(water_depth)
    
    ship.setHull(hull)
    ship.setProp(propeller1)
    ship.setProp(propeller2)
    ship.setCurrent(current)
    ship.setWave(wave)
    ship.setWind(wind)
    ship.setME(main_eng1)
    ship.setME(main_eng2)
    ship.setWaterDepth(depth)
    #end region make ship model
    return ship


def OptimizePlan(p_conn:psycopg.Connection, shipid, waypts_list, draft, Options:dict, p_weather_pickle_path, p_depth_pickle_path):
    vp = Data_VoyagePlan()
    vp.Set_Data(waypts_list)
    Ship = Make_ShipModel(p_conn, shipid, p_weather_pickle_path, p_depth_pickle_path)
    draft = float(draft)
    vm = VoyageManager(Ship)
    candid = dict()
    for idx, wp in enumerate(vp.waypoints):
        if idx == 0 or idx == len(vp.waypoints) : continue
        candid[float(wp.no)] = []
        for c in coordlist:
            c_lon = c[0]
            c_lat = c[1]
            if (float(wp.lon) - 2 <= c_lon <= float(wp.lon)+2) and (float(wp.lat) - 2 <= c_lat <= float(wp.lat) + 2):
                c_wp = Data_WayPoint()
                c_wp.lat = c_lat
                c_wp.lon = c_lon
                c_wp.no = wp.no
                candid[float(wp.no)].append(c_wp)
    Options = Data_OptimizeOption(Options)
    waypoints = vm.Optimize(vp.waypoints, candid, draft, Options)
    OptPlan = Data_VoyagePlan()
    OptPlan.Set_WP(waypoints)

    return  {'waypoints' : OptPlan.waypts_dict}
