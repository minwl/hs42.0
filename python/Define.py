import math, json, copy
from datetime import datetime, timezone

import psycopg
import numpy as np

from hs4da import EngineeringData


G_ENG_DATA_DICT:dict = None


'''''
Utility
'''''
#region Utility
class Constants:
    rho : float = 1.025             # Density of sea water [kg/m^3]
    rho_a : float = 0.001225;		# Density of air [kg/m^3]
    g : float = 9.81;			    # Gravitational acceleration
    R_equator : float = 6378.137;   # Radius of earth


class hMath:
    def Normalize_0to180(degree:float):
        while degree < 0.0:            degree += 360.0
        while degree >= 360.0:         degree -= 360.0
        degree = (360.0 - degree) if (180.0 < degree) else (degree)
        return degree

    def Normalize_0to360(degree:float):
        while degree < 0.0:            degree += 360.0
        while degree >= 360.0:         degree -= 360.0
        return degree

    def get_slope(axis:list, target_val:float):
        if target_val<=axis[0]:      left_val, right_val = axis[0], axis[0]      # Check Left end
        elif target_val>= axis[-1]:  left_val, right_val = axis[-1], axis[-1]    # Check Right end
        else:
            for i in range(len(axis)-1):    # Find Index & Value
                if (axis[i] <= target_val) and (target_val < axis[i+1]):
                    left_val = axis[i]
                    right_val= axis[i+1]
                    break        
        slope = 1 if (left_val == right_val) else ((target_val-left_val) / (right_val-left_val))    # Calculate Slope
        return left_val, right_val, slope
    
    def LineInterpol_1D(table1d: dict, target_x):
        range_x = sorted(table1d.keys())
        xl, xr, dx = hMath.get_slope(range_x, target_x)
        p0 = table1d[xl]
        p1 = table1d[xr]
        result = np.multiply(p0,(1-dx)) + np.multiply(p1,dx)
        try : return float(result)
        except : return [float(x) for x in result]     
    
    def LineInterpol_2D(table2d: dict, target_x, target_y):
        range_x = sorted(table2d.keys())
        range_y = sorted(table2d[range_x[0]].keys())
        xl, xr, dx = hMath.get_slope(range_x, target_x)
        yl, yr, dy = hMath.get_slope(range_y, target_y)
        p00 = table2d[xl][yl]
        p10 = table2d[xr][yl]
        p01 = table2d[xl][yr]
        p11 = table2d[xr][yr]
        p0 = np.multiply(p00,(1-dx)) + np.multiply(p10,dx)
        p1 = np.multiply(p01,(1-dx)) + np.multiply(p11,dx)
        result = np.multiply(p0,(1-dy)) + np.multiply(p1,dy)
        try : return float(result)
        except : return [float(x) for x in result]
    
    def LineInterpol_3D(table3d: dict, target_x, target_y, target_z):
        range_x = sorted(table3d.keys())
        range_y = sorted(table3d[range_x[0]].keys())
        range_z = sorted(table3d[range_x[0]][range_y[0]].keys())
        xl, xr, dx = hMath.get_slope(range_x, target_x)
        yl, yr, dy = hMath.get_slope(range_y, target_y)
        zl, zr, dz = hMath.get_slope(range_z, target_z)
        p000 = table3d[xl][yl][zl]
        p100 = table3d[xr][yl][zl]
        p010 = table3d[xl][yr][zl]
        p001 = table3d[xl][yl][zr]
        p101 = table3d[xr][yl][zr]
        p110 = table3d[xr][yr][zl]
        p011 = table3d[xl][yr][zr]
        p111 = table3d[xr][yr][zr]
        p00 = np.multiply(p000,(1-dx)) + np.multiply(p100,dx)
        p01 = np.multiply(p001,(1-dx)) + np.multiply(p101,dx)
        p10 = np.multiply(p010,(1-dx)) + np.multiply(p110,dx)
        p11 = np.multiply(p011,(1-dx)) + np.multiply(p111,dx)
        p0 = np.multiply(p00,(1-dy)) + np.multiply(p10,dy)
        p1 = np.multiply(p01,(1-dy)) + np.multiply(p11,dy)
        result = np.multiply(p0,(1-dz)) + np.multiply(p1,dz)
        try : return float(result)
        except : return [float(x) for x in result]
    
    def CalDistance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        # Earth radius in kilometers
        R = 6371.0  # Use 6378.137 for equatorial radius, 6356.752 for polar radius if needed

        # Convert degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        # Differences in coordinates
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        # Haversine formula
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        # Distance in kilometers
        distance = R * c
        return round(distance, 2)
    
    def CalBearing(lat1:float, lon1:float, lat2:float, lon2:float):      # Harversine [deg]
        P1 = math.radians(lat1)
        P2 = math.radians(lat2)
        L1 = math.radians(lon1)
        L2 = math.radians(lon2)
        dL = L2 - L1
        Bearing = math.degrees( math.atan2( ( math.sin(dL)*math.cos(P2) ) , ( math.cos(P1)*math.sin(P2) - math.sin(P1)*math.cos(P2)*math.cos(dL) ) ) )
        return round(Bearing, 2)
    
    def CalPos(lat:float, lon:float, distance:float, bearing:float):
        P = math.radians(lat)
        L = math.radians(lon)
        brg = math.radians(bearing)
        TLat = math.asin( math.sin(P)*math.cos(distance/Constants.R_equator) + math.cos(P)*math.sin(distance/Constants.R_equator)*math.cos(brg) )
        TLon = L + math.atan2( (math.sin(brg)*math.sin(distance/Constants.R_equator)*math.cos(P)) , (math.cos(distance/Constants.R_equator)-math.sin(P)*math.sin(TLat)) )
        return math.degrees(TLat), math.degrees(TLon)


    def get_data_with_point(p_latitude:float, p_longitude:float, p_data:list) -> dict:
        LATITUDE_GUIDE:list = list()
        for x in range(140, -141, -1):
            LATITUDE_GUIDE.append(x/2)
        
        LONGITUDE_GUIDE:list = list()
        for x in range(0, 720):
            LONGITUDE_GUIDE.append(x/2)
        ret_list:list = list()
        # if -70.0 <= p_latitude <= 70.0:
        #     lat_index = int((70.0 - p_latitude) / 0.5)
        if p_latitude < -70.0:
            p_latitude = -70.0
        if p_latitude > 70.0:
            p_latitude = 70.0
        lat_index = int((70.0 - p_latitude) / 0.5)

        # if 0 <= p_longitude <=359.5:
        #     lon_index = int(p_longitude / 0.5)
        if p_longitude > 359.5:
            p_longitude = p_longitude - 360
        if p_longitude < 0:
            p_longitude = 360 + p_longitude
        lon_index = int(p_longitude / 0.5)

        value_index = lat_index * 720 + lon_index
        ret_list.append({"latitude": LATITUDE_GUIDE[lat_index], "longitude": LONGITUDE_GUIDE[lon_index], "value": p_data[value_index]})
        if lat_index < 280:
            value_index = (lat_index + 1) * 720 + lon_index
            ret_list.append({"latitude": LATITUDE_GUIDE[lat_index + 1], "longitude": LONGITUDE_GUIDE[lon_index], "value": p_data[value_index]})
        if lon_index < 719:
            value_index = lat_index * 720 + (lon_index + 1)
            ret_list.append({"latitude": LATITUDE_GUIDE[lat_index], "longitude": LONGITUDE_GUIDE[lon_index + 1], "value": p_data[value_index]})
        if lat_index < 280 and lon_index < 719:
            value_index = (lat_index + 1) * 720 + (lon_index + 1)
            ret_list.append({"latitude": LATITUDE_GUIDE[lat_index + 1], "longitude": LONGITUDE_GUIDE[lon_index + 1], "value": p_data[value_index]})
        return ret_list
    
    def weather_interpolate(base_dict : dict, Time : float, Lat:float, Lon:float):
        timeseries = list(base_dict.keys())
        left_t, right_t, _ = hMath.get_slope(timeseries, Time)
        left_list = base_dict[left_t]
        right_list = base_dict[right_t]
        left_vals= hMath.get_data_with_point(Lat, Lon, left_list)
        right_vals = hMath.get_data_with_point(Lat, Lon, right_list)
        min_left_val = min([x['value'] for x in left_vals if x['value']!=-9999])
        min_right_val = min([x['value'] for x in right_vals if x['value']!=-9999])
        dir_table = dict()
        
        for v in left_vals:
            v_lat = v['latitude']
            v_lon = v['longitude']
            value = v['value']
            if value == -9999 : value = min_left_val
            dir_table[v_lat] = dict()
            dir_table[v_lat][v_lon] = dict()
            dir_table[v_lat][v_lon][left_t] = value
        
        for v in right_vals:
            v_lat = v['latitude']
            v_lon = v['longitude']
            value = v['value']
            if value == -9999 : value = min_right_val
            dir_table[v_lat] = dict()
            dir_table[v_lat][v_lon] = dict()
            dir_table[v_lat][v_lon][right_t] = value            
        
        return float(hMath.LineInterpol_3D(dir_table, Lat, Lon, Time))
#endregion

#region Voyage Plan
class Data_OptimizeOption:
    def __init__(self, data:dict):
        self.target = data.get('target', 0)             # Fuel Consumption / Just In Time arrival / CII / Fastest / Safest
        self.wind =data.get('wind_threshold', 0)           
        self.wave =data.get('wave_threshold', 0)
        self.current =data.get('current_threshold',0)
        self.thresholds = data.get('thresholds', dict())    # min_waterdepth / max_waveheight
    
class Data_WayPoint:
    def __init__(self):
        self.no = '-inf'
        self.code = ""
        self.time = '-inf'
        self.lat  = "-inf"            
        self.lon  ="-inf"
        self.dur ="-inf"
        self.sog ="-inf"
        self.sog_avg ="-inf"
        self.brg ="-inf"
        self.dist ="-inf"
        self.dist_total ="-inf"
        self.shaft_rpm ="-inf"
        self.equiv_mdo ="-inf"
        self.equiv_mdo_total ="-inf"
        self.winDir="-inf"
        self.winSpd="-inf"
        self.wavDir="-inf"
        self.wavHgt="-inf"
        self.curDir="-inf"
        self.curSpd="-inf"

    def Set_Data(self,
                no:str, code:str, lat:str, lon:str, time:str = '-inf',
                dur:str = '-inf', sog:str = '-inf', sog_avg:str = '-inf',
                brg:str = '-inf',
                dist:str = '-inf', dist_total:str = '-inf',
                shaft_rpm:str = '-inf',
                equiv_mdo:str = '-inf', equiv_mdo_total:str = '-inf',
                winDir:str = '-inf', winSpd:str = '-inf',
                wavDir:str = '-inf', wavHgt:str = '-inf',
                curDir:str = '-inf', curSpd:str = '-inf'):
        if '-' in no:
           self.no = str(int(no.split('-')[0]) + float(no.split('-')[1])/1000)
        else: self.no = no
        if time != '-inf' and len(time) > 10:
            self.time=str(datetime.strptime(time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())        
        else: self.time = time
        self.code = code
        self.lat = lat              
        self.lon = lon
        self.dur = dur
        self.sog = sog
        self.sog_avg = sog_avg
        self.brg = brg
        self.dist =dist
        self.dist_total = dist_total
        self.shaft_rpm = shaft_rpm
        self.equiv_mdo = equiv_mdo
        self.equiv_mdo_total = equiv_mdo_total
        self.winDir = winDir
        self.winSpd = winSpd
        self.wavDir = wavDir
        self.wavHgt = wavHgt
        self.curDir = curDir
        self.curSpd = curSpd
    
    def ToDict(self):
        self.waypts_dict = dict()
        if float(self.no) % 1 == 0: self.waypts_dict['no'] = str(int(self.no))
        else : 
            n = round(float(self.no) * 1000)
            self.waypts_dict['no'] = '-'.join([str(n // 1000), str(n % 1000)])
        self.waypts_dict['code'] = self.code
        if type(self.time) == float:
            self.waypts_dict['time'] = datetime.utcfromtimestamp(self.time).strftime("%Y-%m-%d %H:%M:%S")
        else : self.waypts_dict['time'] = self.time
        self.waypts_dict['lat'] = self.lat
        self.waypts_dict['lon'] = self.lon
        self.waypts_dict['dur'] = self.dur
        self.waypts_dict['sog'] = self.sog
        self.waypts_dict['sog_avg'] = self.sog_avg
        self.waypts_dict['brg'] = self.brg
        self.waypts_dict['dist'] = self.dist
        self.waypts_dict['dist_total'] = self.dist_total
        self.waypts_dict['shaft_rpm'] = self.shaft_rpm
        self.waypts_dict['equiv_mdo'] = self.equiv_mdo
        self.waypts_dict['equiv_mdo_total'] = self.equiv_mdo_total
        self.waypts_dict['winDir'] = self.winDir
        self.waypts_dict['winSpd'] = self.winSpd
        self.waypts_dict['wavHgt'] = self.wavHgt
        self.waypts_dict['wavDir'] = self.wavDir
        self.waypts_dict['curDir'] = self.curDir
        self.waypts_dict['curSpd'] = self.curSpd
        return self.waypts_dict
    
    def ToFloat(self):
        self.no = float(self.no)
        self.lat = float(self.lat)              
        self.lon = float(self.lon)
        self.time = float(self.time)

        self.dur = float(self.dur)
        self.sog = float(self.sog)
        self.sog_avg =  float(self.sog_avg)
        self.brg = float(self.brg)
        self.dist =float(self.dist)
        self.dist_total = float(self.dist_total)
        self.shaft_rpm = float(self.shaft_rpm)
        self.equiv_mdo = float(self.equiv_mdo)
        self.equiv_mdo_total = float(self.equiv_mdo_total)
        self.winDir = float(self.winDir)
        self.winSpd = float(self.winSpd)
        self.wavDir = float(self.wavDir)
        self.wavHgt = float(self.wavHgt)
        self.curDir = float(self.curDir)
        self.curSpd = float(self.curSpd)

        return self
    
class Data_VoyagePlan:
    def __init__(self):
        self.waypts_dict = dict()
        self.waypoints = []

    def Set_Data(self, wapts_list):
        self.waypts_dict = wapts_list
        self.waypoints = []
        for pt in self.waypts_dict:
            # pt = eval(pt)
            wp = Data_WayPoint()
            wp.Set_Data(**pt)
            self.waypoints.append(wp)
            
    def Set_WP(self, waypoints):
        self.waypoints = waypoints
        self.waypts_dict = [wp.ToDict() for wp in waypoints]



# class Data_Options:
#     def Set_Data(self, jsonstr):
#         self.receive  = json.loads(jsonstr)
#         self.option_dict = self.receive['Options']
#         self.target = self.option_dict['target']
#         self.wind = self.option_dict['wind']
#         self.wave = self.option_dict['wave']
#         self.current = self.option_dict['current']
#         self.thresholds = json.loads(self.option_dict['thresholds'])

#endregion

#region Data Structure - Trim optimization
class Data_TrimEfficiency:        # Ship_TrimEfficiencyData.csv
    data = dict()      # Key1: Speed, Key2: Draft, Key3: Trim / Value : Efficiency

    def Set_Data(self, RawData):
        TrimEfficiency = dict()      # Key1: Speed, Key2: Draft, Key3: Trim / Value : Efficiency
        TrimData =copy.deepcopy(RawData) 
        for speed, speed_data in TrimData["Speed"].items():
            TrimEfficiency[float(speed)] = {}
            for draft, draft_data in speed_data["Draft"].items():
                TrimEfficiency[float(speed)][float(draft)] = {}
                for trim, trim_data in draft_data["Trim"].items():
                    TrimEfficiency[float(speed)][float(draft)][float(trim)] = trim_data
        self.data = TrimEfficiency

    def Get_TrimEfficiency(self, Speed:float, Draft:float, Trim:float):
        return hMath.LineInterpol_3D(self.data, Speed, Draft, Trim)
#endregion

#region Data Structure - Voyage
class Data_ShipPrincipal:           # Ship_PrincipalData
    Lpp : float = 0.0               # Length between perpendicular (=LBP)
    Loa : float = 0.0               # Length overall
    B : float = 0.0                 # Breadth    
    Number_Propulsion : int = 1     # Number of Main Engine, Propeller, Rudder, --
    Dpop : float = 0.0              # Propeller Diameter
    Draft : list = [0.0, 0.0]       # [Ballast, Design] Mean draft (=(fwd+aft)/2)
    AT : list = [0.0, 0.0]          # [Ballast, Design] Projected area in the Transverse direction (above water surface)
    AL : list = [0.0, 0.0]          # [Ballast, Design] Projected area in the Longitudinal direction (above water surface)
    
    def Set_Data(RawData):
        PrincipalData=copy.deepcopy(RawData)
        Data_ShipPrincipal.Lpp = PrincipalData["LBP"]
        Data_ShipPrincipal.Loa = PrincipalData["LOA"]
        Data_ShipPrincipal.B = PrincipalData["Breadth"]
        Data_ShipPrincipal.Number_Propulsion = PrincipalData["Number of Propulsion System"]
        Data_ShipPrincipal.Dpop = PrincipalData["Propeller Diameter"]
        Data_ShipPrincipal.Draft = [PrincipalData["Draft"]['Ballast'], PrincipalData["Draft"]['Design']]
        Data_ShipPrincipal.AT = [PrincipalData["AT_air"]['Ballast'], PrincipalData["AT_air"]['Design']]
        Data_ShipPrincipal.AL = [PrincipalData["AL_air"]['Ballast'], PrincipalData["AL_air"]['Design']]

    def Get_AT(Draft:float):
        table1d:dict = {Data_ShipPrincipal.Draft[0]:Data_ShipPrincipal.AT[0], Data_ShipPrincipal.Draft[1]:Data_ShipPrincipal.AT[1]}
        return hMath.LineInterpol_1D(table1d, Draft)    
    def Get_AL(Draft:float):
        table1d:dict = {Data_ShipPrincipal.Draft[0]:Data_ShipPrincipal.AL[0], Data_ShipPrincipal.Draft[1]:Data_ShipPrincipal.AL[1]}
        return hMath.LineInterpol_1D(table1d, Draft)

class Data_ShipPropulsion:          # Ship_SelfPropulsionData.csv
    Factor_Speed = dict()           # Key1: Speed, Key2: Draft(float, LoadingCondition) / Value: [t, w, etaR]

    def Set_Data(self, RawData):
        FactorSpeed = copy.deepcopy(RawData)
        FactorSpeed = FactorSpeed["Speed"]
        DRAFT_B = Data_ShipPrincipal.Draft[0]
        DRAFT_D = Data_ShipPrincipal.Draft[1]
        Speedkeys = list(FactorSpeed.keys())
        for key in Speedkeys:
            FactorSpeed[float(key)] = FactorSpeed.pop(key)
            ballast_value = FactorSpeed[float(key)].pop('Ballast')
            design_value = FactorSpeed[float(key)].pop('Design')
            FactorSpeed[float(key)][DRAFT_B] = [ballast_value['t'], ballast_value['w'], ballast_value['etaR']]
            FactorSpeed[float(key)][DRAFT_D] = [design_value['t'], design_value['w'], design_value['etaR']]
        self.Factor_Speed = FactorSpeed

    def Get_t(self, Speed:float, Draft:float):
        return hMath.LineInterpol_2D(self.Factor_Speed, Speed, Draft)[0]
    def Get_w(self, Speed:float, Draft:float):
        return hMath.LineInterpol_2D(self.Factor_Speed, Speed, Draft)[1]
    def Get_etaR(self, Speed:float, Draft:float):
        return hMath.LineInterpol_2D(self.Factor_Speed, Speed, Draft)[2]

class Data_HullResistance:          # Hull_ResistanceData.csv
    Resistance_Speed = dict()       # Key1: Speed, Key2: Draft(float, LoadingCondition) / Value: Resistance

    def Set_Data(self, RawData):
        ResSpeed = copy.deepcopy(RawData)
        ResSpeed = ResSpeed["Speed"]              # delete string key
        DRAFT_B = Data_ShipPrincipal.Draft[0]   # get loading condition
        DRAFT_D = Data_ShipPrincipal.Draft[1]   # get loading condition
        speedkeys = list(ResSpeed.keys())
        for key in speedkeys:
            ResSpeed[float(key)] = ResSpeed.pop(key)
            ResSpeed[float(key)][DRAFT_B] = ResSpeed[float(key)].pop('Ballast')
            ResSpeed[float(key)][DRAFT_D] = ResSpeed[float(key)].pop('Design')        
        self.Resistance_Speed = ResSpeed

    def Get_Resistance(self, Speed:float, Draft:float):
        return hMath.LineInterpol_2D(self.Resistance_Speed, Speed, Draft)

class Data_PropellerPOW:            # Propeller_POWData.csv
    POWdata_J = dict()              # Key: J (Advanced Raio) / Value: [KT, 10KQ, eta0]

    def Set_Data(self, RawData):
        POWData = copy.deepcopy(RawData)
        J = list(POWData.keys())[0]     # get key "J"
        POWData = POWData[J]            # delete key "J"
        Jkeys = list(POWData.keys())    # get J range
        for key in Jkeys:
            values = POWData.pop(key)
            POWData[float(key)] = [values['KT'], values['10KQ'], values['eta0']] #string key to float    
        self.POWdata_J = POWData
    
    def Get_KT(self, J:float):
        return hMath.LineInterpol_1D(self.POWdata_J, J)[0]
    def Get_KQ(self, J:float):
        return hMath.LineInterpol_1D(self.POWdata_J, J)[1]/10.0
    def Get_eta0(self, J:float):
        return hMath.LineInterpol_1D(self.POWdata_J, J)[2]

class Data_MainEngine:                      # Engine_PerformanceData.csv    
    ISO_ScavAirTemp : list = [0.0, 0.0, 0.0]     # [Oil, Gas]
    ISO_AmbientTemp : list = [0.0, 0.0, 0.0]
    ISO_AmbientPress : list = [0.0, 0.0, 0.0]
    ISO_TCbackPress : list = [0.0, 0.0, 0.0]
    LoadRate_Power = dict()      # Key1: FuelType('Oil', 'Gas'), Key2: Tier('Tier2', 'Tier3'), Key3: Power / Value: LoadRate
    Perform_LoadRate = dict()    # Key1: FuelType('Oil', 'Gas'), Key2: Tier('Tier2', 'Tier3'), Key3: LoadRate / Value: [Power, RPM, Speed, PriseLimit, PmaxLimit, SFOC(SPOC, SFGC)]
    
    def Set_Data(self, RawData):
        PerformDict =copy.deepcopy(RawData) #load rate : performance table
        # count = len(self.ISO_ScavAirTemp)
        # param = list(PerformDict["ISO Parameter"]["Scav air temp"].keys())

        # for n in range(count):
        #     self.ISO_ScavAirTemp[n] = PerformDict["ISO Parameter"]["Scav air temp"][param[n]]
        #     self.ISO_AmbientTemp[n] = PerformDict["ISO Parameter"]["Ambient temp"][param[n]]
        #     self.ISO_AmbientPress[n] = PerformDict["ISO Parameter"]["Ambient press"][param[n]]
        #     self.ISO_TCbackPress[n] = PerformDict["ISO Parameter"]["TC Back press"][param[n]]

        O , G = "Oil", "Gas"        
        LRDict = {}
        LRDict[O], LRDict[G] = {}, {}
        for O_TIER in PerformDict[O].keys():
            if ' ' in O_TIER : ''.join(O_TIER.split(' '))
            O_LR = list(PerformDict[O][O_TIER].keys())[0] #key for load rate dictionary
            PerformDict[O][O_TIER] = PerformDict[O][O_TIER][O_LR] #delete string key
            LRDict[O][O_TIER] = {}
            o_loadrates = list(PerformDict[O][O_TIER].keys()) #load rate values in string
            for o_lr in o_loadrates:
                perform_data_o = PerformDict[O][O_TIER].pop(o_lr)
                PerformDict[O][O_TIER][float(o_lr)] = [perform_data_o['Power'], perform_data_o.get('RPM', perform_data_o.get('rpm')), perform_data_o['Speed'], perform_data_o['PriseLimit'], perform_data_o['PmaxLimit'], perform_data_o['SFOC']]
                PwrVal = PerformDict[O][O_TIER][float(o_lr)][0]
                LRDict[O][O_TIER][PwrVal] = float(o_lr)
        for G_TIER in PerformDict[G].keys():
            if ' ' in O_TIER : ''.join(O_TIER.split(' '))
            G_LR =list(PerformDict[G][G_TIER].keys())[0]
            PerformDict[G][G_TIER] = PerformDict[G][G_TIER][G_LR]
            LRDict[G][G_TIER] = {}
            g_loadrates = list(PerformDict[G][G_TIER].keys())
            for g_lr in g_loadrates:
                perform_data_g = PerformDict[G][G_TIER].pop(g_lr)
                PerformDict[G][G_TIER][float(g_lr)] = [perform_data_g['Power'], perform_data_g.get('RPM', perform_data_g.get('rpm')), perform_data_g['Speed'], perform_data_g['PriseLimit'], perform_data_g['PmaxLimit'], perform_data_g['SPOC'], perform_data_g['SFGC']]
                PwrVal = PerformDict[G][G_TIER][float(g_lr)][0]
                LRDict[G][G_TIER][PwrVal] = float(g_lr)
        self.LoadRate_Power = LRDict
        self.Perform_LoadRate = PerformDict
        

    def Get_LoadRate_Power(self, FuelType:str, Tier:str, Power:float):
        return hMath.LineInterpol_1D(self.LoadRate_Power[FuelType][Tier], Power)    
    def Get_Power_LoadRate(self, FuelType:str, Tier:str, LoadRate:float):
        return hMath.LineInterpol_1D(self.Perform_LoadRate[FuelType][Tier], LoadRate)[0]
    def Get_RPM_LoadRate(self, FuelType:str, Tier:str, LoadRate:float):
        return hMath.LineInterpol_1D(self.Perform_LoadRate[FuelType][Tier], LoadRate)[1]
    def Get_Speed_LoadRate(self, FuelType:str, Tier:str, LoadRate:float):
        return hMath.LineInterpol_1D(self.Perform_LoadRate[FuelType][Tier], LoadRate)[2]
    def Get_PriseLimit_LoadRate(self, FuelType:str, Tier:str, LoadRate:float):
        return hMath.LineInterpol_1D(self.Perform_LoadRate[FuelType][Tier], LoadRate)[3]
    def Get_PmaxLimit_LoadRate(self, FuelType:str, Tier:str, LoadRate:float):
        return hMath.LineInterpol_1D(self.Perform_LoadRate[FuelType][Tier], LoadRate)[4]
    def Get_SFOC_LoadRate(self, FuelType:str, Tier:str, LoadRate:float):
        return hMath.LineInterpol_1D(self.Perform_LoadRate[FuelType][Tier], LoadRate)[5]
    def Get_SPOC_LoadRate(self, FuelType:str, Tier:str, LoadRate:float):
        if(FuelType == 'Oil'):  return False
        else:                   return hMath.LineInterpol_1D(self.Perform_LoadRate[FuelType][Tier], LoadRate)[5]
    def Get_SFGC_LoadRate(self, FuelType:str, Tier:str, LoadRate:float):
        if(FuelType == 'Oil'):  return False
        else:                   return hMath.LineInterpol_1D(self.Perform_LoadRate[FuelType][Tier], LoadRate)[6]

class Data_ShipRAO:              # Ship_RAOData.csv
    RollRAO_Ballast = dict()
    RollRAO_Design = dict()     # Key1: Speed, Key2: Heading, Key3: Frequency / Value : Roll RAO
    PitchRAO_Ballast = dict()
    PitchRAO_Design = dict()    # Key1: Speed, Key2: Heading, Key3: Frequency / Value : Pitch RAO
    HeaveRAO_Ballast = dict()
    HeaveRAO_Design = dict()    # Key1: Speed, Key2: Heading, Key3: Frequency / Value : Heave RAO    

    def Set_Data(self, RawData):
        RollRAO_Ballast = dict()
        RollRAO_Design = dict()     # Key1: Speed, Key2: Heading, Key3: Frequency / Value : Roll RAO
        PitchRAO_Ballast = dict()
        PitchRAO_Design = dict()    # Key1: Speed, Key2: Heading, Key3: Frequency / Value : Pitch RAO
        HeaveRAO_Ballast = dict()
        HeaveRAO_Design = dict()    # Key1: Speed, Key2: Heading, Key3: Frequency / Value : Heave RAO    
        
        # RawData = json.loads(jsonstr)
        ShipRAO =copy.deepcopy(RawData) 
        # B, D = list(ShipRAO.keys())
        BallastRAO = ShipRAO["Ballast"]
        DesignRAO = ShipRAO["Design"]
        B_SPEED, D_SPEED = 'Speed', 'Speed'
        for speed, speed_data in BallastRAO[B_SPEED].items():
            HeaveRAO_Ballast[float(speed)] = dict()
            RollRAO_Ballast[float(speed)] = dict()
            PitchRAO_Ballast[float(speed)] = dict()         
            B_HDG = 'HDG'
            for hdg, hdg_data in speed_data[B_HDG].items():
                HeaveRAO_Ballast[float(speed)][float(hdg)] = dict()    
                RollRAO_Ballast[float(speed)][float(hdg)] = dict()
                PitchRAO_Ballast[float(speed)][float(hdg)] = dict()
                B_FREQ = 'Frequency'
                for freq, freq_data in hdg_data[B_FREQ].items():
                    H, R, P = sorted(list(freq_data.keys()))
                    HeaveRAO_Ballast[float(speed)][float(hdg)][float(freq)] = freq_data[H]   
                    RollRAO_Ballast[float(speed)][float(hdg)][float(freq)] = freq_data[R]
                    PitchRAO_Ballast[float(speed)][float(hdg)][float(freq)] = freq_data[P]
        for speed, speed_data in DesignRAO[D_SPEED].items():
            HeaveRAO_Design[float(speed)] = dict()
            RollRAO_Design[float(speed)] = dict()
            PitchRAO_Design[float(speed)] = dict()         
            D_HDG = list(speed_data.keys())[0]
            for hdg, hdg_data in speed_data[D_HDG].items():
                HeaveRAO_Design[float(speed)][float(hdg)] = dict()    
                RollRAO_Design[float(speed)][float(hdg)] = dict()
                PitchRAO_Design[float(speed)][float(hdg)] = dict()  
                D_FREQ = list(hdg_data.keys())[0]
                for freq, freq_data in hdg_data[D_FREQ].items():
                    H, R, P = list(freq_data.keys())
                    HeaveRAO_Design[float(speed)][float(hdg)][float(freq)] = freq_data[H]   
                    RollRAO_Design[float(speed)][float(hdg)][float(freq)] = freq_data[R]
                    PitchRAO_Design[float(speed)][float(hdg)][float(freq)] = freq_data[P]
        self.RollRAO_Ballast = RollRAO_Ballast
        self.HeaveRAO_Ballast = HeaveRAO_Ballast
        self.PitchRAO_Ballast = PitchRAO_Ballast
        self.RollRAO_Design = RollRAO_Design
        self.HeaveRAO_Design = HeaveRAO_Design
        self.PitchRAO_Design = PitchRAO_Design

    def Get_RollRAO(self, Speed:float, Heading:float, Frequency:float, Draft:float):
        table1D:dict = { Data_ShipPrincipal.Draft[0]: hMath.LineInterpol_3D(self.RollRAO_Ballast, Speed, Heading, Frequency) , Data_ShipPrincipal.Draft[1]: hMath.LineInterpol_3D(self.RollRAO_Design, Speed, Heading, Frequency)}
        return hMath.LineInterpol_1D(table1D, Draft)
    def Get_PitchRAO(self, Speed:float, Heading:float, Frequency:float, Draft:float):
        table1D:dict = { Data_ShipPrincipal.Draft[0]: hMath.LineInterpol_3D(self.PitchRAO_Ballast, Speed, Heading, Frequency) , Data_ShipPrincipal.Draft[1]: hMath.LineInterpol_3D(self.PitchRAO_Design, Speed, Heading, Frequency)}
        return hMath.LineInterpol_1D(table1D, Draft)
    def Get_HeaveRAO(self, Speed:float, Heading:float, Frequency:float, Draft:float):
        table1D:dict = { Data_ShipPrincipal.Draft[0]: hMath.LineInterpol_3D(self.HeaveRAO_Ballast, Speed, Heading, Frequency) , Data_ShipPrincipal.Draft[1]: hMath.LineInterpol_3D(self.HeaveRAO_Design, Speed, Heading, Frequency)}
        return hMath.LineInterpol_1D(table1D, Draft)

class Data_WindLoadCoefficienct: # Wind_LoadCoefficienct.csv
    Cx_Bearing = dict()          # Key1: Draft(float, LoadingCondition), Key2: Heading / Value: Cx
    Cy_Bearing = dict()          # Key1: Draft(float, LoadingCondition), Key2: Heading / Value: Cy
    Cn_Bearing = dict()          # Key1: Draft(float, LoadingCondition), Key2: Heading / Value: Cn

    def Set_Data(self, RawData):
        Cx_Bearing = dict()          # Key1: Draft(float, LoadingCondition), Key2: Heading / Value: Cx
        Cy_Bearing = dict()          # Key1: Draft(float, LoadingCondition), Key2: Heading / Value: Cy
        Cn_Bearing = dict()          # Key1: Draft(float, LoadingCondition), Key2: Heading / Value: Cn
        # RawData = json.loads(jsonstr)
        WindCoeff =copy.deepcopy(RawData)
        
        B, D = list(WindCoeff.keys())
        BRG_B, BRG_D = list(WindCoeff[B].keys())[0], list(WindCoeff[D].keys())[0]
        DIR_B, DIR_D = list(WindCoeff[B][BRG_B].keys())[0], list(WindCoeff[D][BRG_D].keys())[0]
        DRAFT_D = Data_ShipPrincipal.Draft[0] #get loading condition
        DRAFT_B = Data_ShipPrincipal.Draft[1] #get loading condition
        Cx_Bearing[DRAFT_B], Cx_Bearing[DRAFT_D] = {}, {}
        Cy_Bearing[DRAFT_B], Cy_Bearing[DRAFT_D] = {}, {}
        Cn_Bearing[DRAFT_B], Cn_Bearing[DRAFT_D] = {}, {}

        for brg, data in WindCoeff[B][BRG_B][DIR_B].items():
            CN, CX, CY = sorted(list(data.keys()))
            Cx_Bearing[DRAFT_B].update({float(brg):float(data[CX])})
            Cy_Bearing[DRAFT_B].update({float(brg):float(data[CY])})
            Cn_Bearing[DRAFT_B].update({float(brg):float(data[CN])})
        for brg, data in WindCoeff[D][BRG_D][DIR_D].items():
            CN, CX, CY = sorted(list(data.keys()))
            Cx_Bearing[DRAFT_D].update({float(brg):float(data[CX])})
            Cy_Bearing[DRAFT_D].update({float(brg):float(data[CY])})
            Cn_Bearing[DRAFT_D].update({float(brg):float(data[CN])})
        
        self.Cx_Bearing = Cx_Bearing
        self.Cy_Bearing = Cy_Bearing
        self.Cn_Bearing = Cn_Bearing

    def Get_Cx(self, Draft:float, Heading:float):
        return hMath.LineInterpol_2D(self.Cx_Bearing, Draft, Heading)
    def Get_Cy(self, Draft:float, Heading:float):
        return hMath.LineInterpol_2D(self.Cy_Bearing, Draft, Heading)
    def Get_Cn(self, Draft:float, Heading:float):
        return hMath.LineInterpol_2D(self.Cn_Bearing, Draft, Heading)

class Data_WaveDriftSurgeQTF:    # Wave_DriftSurgeQTFData.csv 
    QTF_Ballast = dict()         # Key1: Speed, Key2: Heading, Key3: Frequency / Value : QTF
    QTF_Design = dict()

    def Set_Data(self, RawData):
        QTF_Ballast = dict()         # Key1: Speed, Key2: Heading, Key3: Frequency / Value : QTF
        QTF_Design = dict()

        # RawData = json.loads(jsonstr)
        WaveQTF =copy.deepcopy(RawData) 
        B, D = sorted(list(WaveQTF.keys()))

        B_SPEED, D_SPEED ="Ship Speed", "Ship Speed"
        
        for speed, speed_data in WaveQTF[B][B_SPEED].items():
            QTF_Ballast[float(speed)] = {}
            B_HDG = "Direction"
            for hdg, hdg_data in speed_data[B_HDG].items():
                QTF_Ballast[float(speed)][float(hdg)] = {}
                B_FREQ = 'Frequency'
                for freq, freq_data in hdg_data[B_FREQ].items():
                    QTF_Ballast[float(speed)][float(hdg)].update({float(freq): float(freq_data)})

        for speed, speed_data in WaveQTF[D][D_SPEED].items():
            QTF_Design[float(speed)] = {}
            D_HDG = "Direction"
            for hdg, hdg_data in speed_data[D_HDG].items():
                QTF_Design[float(speed)][float(hdg)] = {}
                D_FREQ = 'Frequency'
                for freq, freq_data in hdg_data[D_FREQ].items():
                    QTF_Design[float(speed)][float(hdg)].update({float(freq): float(freq_data)})

        self.QTF_Ballast = QTF_Ballast
        self.QTF_Design = QTF_Design

    def Get_QTF(self, Speed:float, Heading:float, Frequency:float, Draft:float):
        table1D:dict = { Data_ShipPrincipal.Draft[0]: hMath.LineInterpol_3D(self.QTF_Ballast, Speed, Heading, Frequency) , Data_ShipPrincipal.Draft[1]: hMath.LineInterpol_3D(self.QTF_Design, Speed, Heading, Frequency)}
        return hMath.LineInterpol_1D(table1D, Draft)

class Data_Forecast_Wind:
    WindDirection = dict()
    WindSpeed = dict()
    def Set_Data(self, forecast):
        WindDirection = copy.deepcopy(forecast['wind_direction'])
        WindSpeed = copy.deepcopy(forecast['wind_speed'])
        wind_dir_keys = list(WindDirection.keys())
        wind_spd_keys = list(WindSpeed.keys())
        for key in wind_dir_keys:
            unix_key = datetime.strptime(key, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
            WindDirection[unix_key] = WindDirection.pop(key)
        for key in wind_spd_keys:
            unix_key = datetime.strptime(key, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
            WindSpeed[unix_key] = WindSpeed.pop(key)
        
        self.WindSpeed = WindSpeed
        self.WindDirection = WindDirection

    def Get_Direction(self,Time:float, Lat:float, Lon:float):
        return hMath.weather_interpolate(self.WindDirection, Time, Lat, Lon)
        
    def Get_Speed(self,Time:float, Lat:float, Lon:float):
        return hMath.weather_interpolate(self.WindSpeed, Time, Lat, Lon)

class Data_Forecast_Wave:
    WaveDirection = dict()
    WaveHeight = dict()
    WavePeriod = dict()
    def Set_Data(self, forecast):
        WaveDirection = forecast['wave_direction']
        WaveHeight = forecast['wave_height']
        WavePeriod = forecast['wave_period']
        wave_dir_keys = list(WaveDirection.keys())
        wave_hgt_keys = list(WaveHeight.keys())
        wave_prd_keys = list(WavePeriod.keys())
        for key in wave_dir_keys:
            unix_key = datetime.strptime(key, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
            WaveDirection[unix_key] = WaveDirection.pop(key)
        for key in wave_hgt_keys:
            unix_key = datetime.strptime(key, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
            WaveHeight[unix_key] = WaveHeight.pop(key)
        for key in wave_prd_keys:
            unix_key = datetime.strptime(key, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
            WavePeriod[unix_key] = WavePeriod.pop(key)
        
        self.WaveDirection = WaveDirection
        self.WaveHeight = WaveHeight
        self.WavePeriod = WavePeriod

    def Get_Direction(self,Time:float, Lat:float, Lon:float):
        return hMath.weather_interpolate(self.WaveDirection, Time, Lat, Lon)
    def Get_Height(self,Time:float, Lat:float, Lon:float):
        return hMath.weather_interpolate(self.WaveHeight, Time, Lat, Lon)
    def Get_Period(self,Time:float, Lat:float, Lon:float):
        return hMath.weather_interpolate(self.WavePeriod, Time, Lat, Lon)

        
class Data_Forecast_Current:
    CurrentDirection = dict()
    CurrentSpeed = dict()
    def Set_Data(self, forecast):
        CurrentDirection = forecast['current_direction']
        CurrentSpeed = forecast['current_speed'] 
        curr_dir_keys = list(CurrentDirection.keys())
        curr_spd_keys = list(CurrentSpeed.keys())
        for key in curr_dir_keys:
            unix_key = datetime.strptime(key, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
            CurrentDirection[unix_key] = CurrentDirection.pop(key)
        for key in curr_spd_keys:
            unix_key = datetime.strptime(key, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
            CurrentSpeed[unix_key] = CurrentSpeed.pop(key)
        
        self.CurrentDirection = CurrentDirection
        self.CurrentSpeed = CurrentSpeed

    def Get_Direction(self,Time:float, Lat:float, Lon:float):
        return hMath.weather_interpolate(self.CurrentDirection, Time, Lat, Lon)

    def Get_Speed(self,Time:float, Lat:float, Lon:float):
        return hMath.weather_interpolate(self.CurrentSpeed, Time, Lat, Lon)

class Data_WaterDepth:             # External_MinWaterDepth.csv
    Grids = dict()          # Key1: Latitude, Key2: Longitude / Value : MinDepth

    def Set_Data(self, RawData):
        self.Grids = copy.deepcopy(RawData)

    def Get_MinDepth(self, Lat:float, Lon:float):
        return hMath.LineInterpol_2D(self.Grids, Lat, Lon)

class Data_ECA:                  # External_ECA.csv
    Areas = dict()                 # Key1: Name / Value: list[Latitude, Longitude]

    def Set_Data(self, RawData):
        ECA = dict()                 # Key1: Name / Value: list[(Latitude, Longitude),(Latitude, Longitude),(Latitude, Longitude)... ]
        # RawData = json.loads(jsonstr)
        ecaData =copy.deepcopy(RawData) 
        names = list(ecaData.keys())
        for name in names:
            coords = [tuple(x.values()) for x in ecaData[name]]
            ECA[name] = coords    
        self.Areas = ECA

    def Get_Area(self):
        return self.Areas

class Data_HRA:                  # External_HRA.csv
    Areas = dict()                 # Key1: Name / Value: list[Latitude, Longitude]
    
    def Set_Data(self, RawData):
        HRA = dict()                 # Key1: Name / Value: list[(Latitude, Longitude),(Latitude, Longitude),(Latitude, Longitude)... ]
        # RawData = json.loads(jsonstr)
        hraData =copy.deepcopy(RawData) 
        names = list(hraData.keys())
        for name in names:
            coords = [tuple(x.values()) for x in hraData[name]]
            HRA[name] = coords   
        self.Areas = HRA
    
    def Get_Area(self):
        return self.HRA
    
#endregion

# ------------------------------------------------------------------------------------------
# Data functions
# ------------------------------------------------------------------------------------------
def get_engineering_data_fron_db(p_conn:psycopg.Connection, p_ship_id: str) -> dict:
    ret_dict:dict = dict()
    cipher = EngineeringData.EncDec(p_ship_id)
    db = EngineeringData.EngineeringDataDB(p_conn)
    db.begin_transaction()
    result_list = db.get_list(p_ship_id)
    for row in result_list:
        data_name = row[0]
        result = db.get_data(p_ship_id, data_name)
        if result[0][1]:
            ret_dict[data_name] = cipher.decrypt(result[0][3])
        else:
            ret_dict[data_name] = result[0][3]
    db.end_transaction()
    return ret_dict


def get_engineering_data(p_conn:psycopg.Connection, p_ship_id: str) -> dict:
    global G_ENG_DATA_DICT
    if G_ENG_DATA_DICT is None:
        G_ENG_DATA_DICT = get_engineering_data_fron_db(p_conn, p_ship_id)
    return G_ENG_DATA_DICT
