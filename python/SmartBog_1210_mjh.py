import pyproj
import numpy as np
import pandas as pd

from geopy import distance
from datetime import timedelta, datetime, timezone
from scipy.interpolate import interp1d
from math import sin, cos, sqrt, atan2, radians, degrees

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import SinglePointCrossover
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
from pymoo.optimize import minimize


import warnings
warnings.filterwarnings("ignore")



########################################################
### 호선마다 변경 필요 ###
hicom_no1 = 5300
hicom_no2 = 5300
condensate_max_no1 = 1600
condensate_max_no2 = 1600
fgc_ge_average = 625

values = {
    'speed_coef_rpm' : 0.287560,
    'speed_coef_totalswell_h1_d1' : -0.227536,
    'speed_coef_Vcurrent_long' : -0.966745,
    'speed_coef_laden' :  -0.618376,

    'coef_const_fgc' : 599.3836207351836,
    'coef_rpm_3' : 0.008452771102643378,
    'coef_Vwind_long_2' : 0.3420719770786013,
    'coef_totalswell_h1_d1' : -0.37375612965668026,
    'coef_totalswell_h2_d2' : 17.289473860050627,
    'coef_rpm_cat' : 144.66742860344684,
    'coef_laden' : 253.62148997979608,

    'coef_const_vessel' : 0.9144981156644572,
    'coef_bog_consumed' : -0.00026143701959057276,
    'coef_roll_swell' : 0.039907395389524605,
    'coef_roll_ww' : 0.7789904244437367,
    'coef_pitch_swell' : 0.534764391216928,
    'coef_pitch_ww' : 0.5263461642742484,
    'coef_operation_days' : 0.005226300074939119,
    'ge_flow_frs' : 820,

    'p0' : 136.05
}
########################################################

########################################################
### Tank Pressure Boundary Setting ###
# tk_press_set = [100, 120, 230, 250, 150]
########################################################
def get_coef_speed_model() :  # Predict vessel speed in knots
    speed_coef_rpm              = values['speed_coef_rpm']
    speed_coef_totalswell_h1_d1 = values['speed_coef_totalswell_h1_d1']
    speed_coef_Vcurrent_long    = values['speed_coef_Vcurrent_long']
    speed_coef_laden            = values['speed_coef_laden']
        
    return speed_coef_rpm,speed_coef_totalswell_h1_d1,speed_coef_Vcurrent_long, speed_coef_laden
    
def get_coef_fgc_me_ge_model() :
    coef_const            = values['coef_const_fgc']
    coef_rpm_3            = values['coef_rpm_3']
    coef_Vwind_long_2     = values['coef_Vwind_long_2']
    coef_totalswell_h1_d1 = values['coef_totalswell_h1_d1']
    coef_totalswell_h2_d2 = values['coef_totalswell_h2_d2']
    coef_rpm_cat          = values['coef_rpm_cat']
    coef_laden            = values['coef_laden']
        
    return coef_const, coef_rpm_3, coef_Vwind_long_2, coef_totalswell_h1_d1, coef_totalswell_h2_d2, coef_rpm_cat, coef_laden

def get_vessel_model():
    coef_const          = values['coef_const_vessel']
    coef_bog_consumed   = values['coef_bog_consumed']
    coef_roll_swell     = values['coef_roll_swell']
    coef_roll_ww        = values['coef_roll_ww']
    coef_pitch_swell    = values['coef_pitch_swell']
    coef_pitch_ww       = values['coef_pitch_ww']
    coef_operation_days = values['coef_operation_days']
    ge_flow_frs         = values['ge_flow_frs']   # frs 
    return coef_const, coef_bog_consumed, coef_roll_swell, coef_roll_ww, coef_pitch_swell, coef_pitch_ww, coef_operation_days, ge_flow_frs

def get_pressure_conditions(tk_press_set):
    p0 = values['p0']
    p_ll, p_l, p_h, p_hh, p_unload = tk_press_set
    
    return p0, p_h, p_l, p_hh, p_ll, p_unload

def get_wdir(df):
    if (df['v10'] == 'None') or (df['u10'] == 'None'):
        result = 'None'
    else :
        result = 90 - degrees(atan2(df['v10'],df['u10']))
    return result 

def get_wsp(df):    
    if (df['v10'] == 'None') or (df['u10'] == 'None'):
        result = 'None'
    else :
        result = sqrt(df['v10']**2 + df['u10']**2)
    return result


def get_heading(lat1, lon1, lat2, lon2):

    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    # 경도 차이 계산
    d_lon = lon2_rad - lon1_rad

    # 방위각 계산
    x = sin(d_lon) * cos(lat2_rad)
    y = cos(lat1_rad) * sin(lat2_rad) - sin(lat1_rad) * cos(lat2_rad) * cos(d_lon)

    initial_bearing = atan2(x, y)

    # 방위각을 0도에서 360도 사이로 변환
    initial_bearing_deg = (degrees(initial_bearing) + 360) % 360

    return initial_bearing_deg



def roll_pitch_acc(result1, df_Aroll, df_Apitch, speed, wave_height, wave_period, wave_dir):
    
    if speed >0 and wave_height >0  and wave_period >0 :
        speed_list = df_Aroll.speed.unique()
        wave_height_list = df_Aroll.wave_height.unique()
        wave_period_list = df_Aroll.wave_period.unique()
        wave_dir_dir_list = df_Aroll.wave_dir.unique()

        absolute_difference_function = lambda list_value : abs(list_value - speed)
        speed = min(speed_list, key=absolute_difference_function)

        absolute_difference_function = lambda list_value : abs(list_value - wave_height)
        wave_height = min(wave_height_list, key=absolute_difference_function)

        absolute_difference_function = lambda list_value : abs(list_value - wave_period)
        wave_period = min(wave_period_list, key=absolute_difference_function)

        absolute_difference_function = lambda list_value : abs(list_value - wave_dir)
        wave_dir = min(wave_dir_dir_list, key=absolute_difference_function)

        return df_Aroll['MostProbableMax'][df_Aroll['speed'] == speed][df_Aroll['wave_height'] == wave_height][df_Aroll['wave_period'] == wave_period][df_Aroll['wave_dir'] == wave_dir].values[0], df_Apitch['MostProbableMax'][df_Apitch['speed'] == speed][df_Apitch['wave_height'] == wave_height][df_Apitch['wave_period'] == wave_period][df_Apitch['wave_dir'] == wave_dir].values[0]
    else:
        return 0.0, 0.0

def get_speed_fgc_prediction(df_input_resample_1h, lat1, lon1, lat2, lon2, time1, rpm, laden ) :
       
    # df = get_env_forecast((lat1+lat2)/2, (lon1+lon2)/2, time1)
    
    #df_input_resample columns : ['time','lat', 'lon', 'shaft_rpm', 'sog, 'winDir',	'winSpd','wavHgt','wavDir','curDir','curSpd',  'laden', 'gcu mode']
    
    df = df_input_resample_1h.to_frame().T

    df.replace({'': 0.0}, inplace=True)
    df.replace({np.nan: 0.0}, inplace=True)

    
    df['mpww'] = 0 #wave period 0 처리
    
    #swell관련 값 모두 0처리
    df['shts'] = 0
    df['mdts'] = 0
    df['mpts'] = 0

    df.columns = ['time', 'lat', 'lon', 'shaft_rpm', 'sog', 'wdir', 'wsp', 'shww', 'mdww', 'cdir', 'csp', 'laden', 'gcu mode',  'mpww', 'shts', 'mdts', 'mpts'] 

    df['longitude_departure'] = lon1
    df['latitude_departure'] = lat1
    df['longitude_arrival'] = lon2
    df['latitude_arrival'] = lat2

    df['heading'] = get_heading(lat1, lon1, lat2, lon2)

    df['laden'] = laden
    df['rpm']   = rpm
    df['rpm_3'] = rpm**3
    # Current
    df['current_dir'] = np.cos(np.radians(df['heading'] - df['csp']))   # heading은 선박의 진행방향
    df['Vcurrent_long'] = df['csp']*np.cos(np.radians(df['heading'] - df['cdir'])) * 1.943844  # 1 m/s = 1.943844 kn

    # total swell
    # df['totalswell_dir'] = np.cos(np.radians(df['heading'] - df['WavDir']))   # wave가 선수파일때 1, 선미파일때 -1, 옆파도일때 0)
    # df['totalswell_h1_d1'] = df['shts'] * df['totalswell_dir'] 
    # df['totalswell_h2_d2'] = df['totalswell_h1_d1'] ** 2
    df['totalswell_dir'] = 0
    df['totalswell_h1_d1'] = 0
    df['totalswell_h2_d2'] = 0
    
    # windwave
    df['windwave_dir'] = np.cos(np.radians(df['heading'] - df['mdww']))   # wave가 선수파일때 1, 선미파일때 -1, 옆파도일때 0)

    # wind
    df['wind_dir'] = np.cos(np.radians(df['heading'] - df['wdir']))
    df['Vwind_long'] = df['wsp']*np.cos(np.radians(df['heading'] - df['wdir'])) * 1.943844  # 1 m/s = 1.943844 kn
    df['Vwind_long_2'] = df["Vwind_long"] ** 2 * df["Vwind_long"] / np.abs(df["Vwind_long"])
    
    speed_coef_rpm, speed_coef_totalswell_h1_d1, speed_coef_Vcurrent_long, speed_coef_laden = get_coef_speed_model()
    df['speed'] = speed_coef_rpm * rpm + speed_coef_totalswell_h1_d1 * df['totalswell_h1_d1'] + speed_coef_Vcurrent_long * df['Vcurrent_long'] + speed_coef_laden * df['laden']
    df['speed'] = np.maximum(df['sog'], 0)
    # if df['speed'].iloc[0] == 0:  df.loc[0, 'speed'] = df.loc[0, 'sog']

    coordinates_from = [lat1, lon1]
    coordinates_to = [lat2, lon2]    
    df['distance_nm'] = distance.great_circle(coordinates_from, coordinates_to).nm

    df['time1'] = pd.to_datetime(time1, format='%Y-%m-%d %H:%M:%S')
    df['sailing_hours'] = df['distance_nm'].iloc[0] / df['speed'].iloc[0]

    # df['time2'] = df['time1']
    df['time2'] = df['time1'] + timedelta(seconds=df['distance_nm'].iloc[0] / df['speed'].iloc[0] * 3600)

    rpm_bins = [39, 50]
    df['rpm_cat'] = np.digitize(df['rpm'], rpm_bins)    
    coef_const, coef_rpm_3, coef_Vwind_long_2, coef_totalswell_h1_d1, coef_totalswell_h2_d2, coef_rpm_cat, coef_laden = get_coef_fgc_me_ge_model()
    
    df['fgc_me'] = coef_const + coef_rpm_3 * df['rpm_3'] +coef_Vwind_long_2 * df['Vwind_long_2'] + coef_totalswell_h1_d1 * df['totalswell_h1_d1'] + coef_totalswell_h2_d2 * df['totalswell_h2_d2'] + coef_rpm_cat * df['rpm_cat'] + coef_laden * df['laden'] 

    return df[['time1','time2','sailing_hours', 'longitude_departure','latitude_departure','longitude_arrival','latitude_arrival','heading', 'shaft_rpm', 'speed', 'distance_nm', 'fgc_me', 'shts', 'mdts', 'mpts', 'shww','mdww', 'mpww', 'wsp', 'wdir', 'csp', 'cdir','totalswell_dir', 'windwave_dir']]



def smart_bog_fgc(df_input):   

    df_input['time'] = pd.to_datetime(df_input['time'])
    df_input.set_index("time", inplace=True)

    df_input = df_input.astype(float)

    resample_hour= 1
    resample_time = str(resample_hour) + "h"
    df_input_resamp = df_input.resample(resample_time).mean()
    
    df_input_resamp.interpolate(method='time', inplace=True)
    df_input_resamp.bfill(inplace=True)
   
    df_input_resamp.reset_index(inplace=True)

    num = len(df_input_resamp.index)-1 
    for i in range(num):
        lat1 = df_input_resamp['lat'].iloc[i]
        lon1 = df_input_resamp['lon'].iloc[i]
        lat2 = df_input_resamp['lat'].iloc[i+1]
        lon2 = df_input_resamp['lon'].iloc[i+1]    
        time1 = df_input_resamp['time'].iloc[i]
        rpm = df_input_resamp['shaft_rpm'].iloc[i]
        laden = df_input_resamp['laden'].iloc[i]

        df_tmp = get_speed_fgc_prediction(df_input_resamp.iloc[i], lat1, lon1,lat2, lon2, time1, rpm, laden)
        if i <= num :    
            df_input_resamp.loc[i+1, 'time']  = df_tmp['time2'].values[0]
        else :
            pass

        if i == 0 :
            result = df_tmp.copy()
            del df_tmp
            
        else :    
            if df_tmp.isna().sum().sum()==0:
                result = pd.concat([result, df_tmp], axis=0, ignore_index=True)
                del df_tmp
            else:
                pass
        
    #최종 도착점의 시각, 위경도, 정보가 빠짐
    df_speed_fgc = result.drop(['time2','sailing_hours','longitude_arrival','latitude_arrival', 'distance_nm'], axis=1).copy()        
    df_speed_fgc.columns = ['ds_timeindex','longitude','latitude','heading','rpm','speed','fgc_me', 
                            'shts', 'mdts', 'mpts', 'shww','mdww', 'mpww', 'wsp', 'wdir', 'csp', 'cdir',
                            'totalswell_dir','windwave_dir']
    df_speed_fgc['gcu_mode'] = df_input_resamp['gcu_mode']
    df_speed_fgc.loc[len(df_speed_fgc.index)] = [result.iloc[-1]['time2'], result.iloc[-1]['longitude_arrival'], result.iloc[-1]['latitude_arrival'], result.iloc[-1]['heading']] + [0.0] * 15 +  [df_speed_fgc.iloc[-1]['gcu_mode']]    
    df_speed_fgc.ffill(inplace=True)
    
    return df_speed_fgc
    

def smart_bog_opt(dff_guide, continous_operating_hours, tk_press_set):
    
    dff_guide = pd.DataFrame(dff_guide)
    
    dff_guide = dff_guide[:len(dff_guide) - len(dff_guide)%continous_operating_hours] 
    dff_guide.set_index("ds_timeindex", inplace=True)
    dff_guide = dff_guide.resample("1h").mean()

    way_point_interval = int(len(dff_guide) / continous_operating_hours)
    
     # 최적화 알고리즘 적용
    problem = ElementwiseExample(dff_guide, way_point_interval, continous_operating_hours, tk_press_set)

    initial_generation = generate_population(way_point_interval)
    
    algorithm = NSGA2(
    pop_size = way_point_interval,
    n_offsprings = 20,
    sampling  = initial_generation, 
    crossover= SinglePointCrossover(prob=0.5, vtype=float, repair=RoundingRepair()), 
    mutation=PM(prob= 0.1, eta=3.0, vtype=float, repair=RoundingRepair()),
    eliminate_duplicates = True
    )
    termination = get_termination("n_gen", 1500)

    res = minimize(problem,
                  algorithm,
    #             termination,
                  seed=1,
                  save_history=True,
                  return_least_infeasible=True,
                  verbose=False)
   
    way_point_interval = int(len(dff_guide) / continous_operating_hours)

    p0, p_h, p_l, p_hh, p_ll, p_unload = get_pressure_conditions(tk_press_set)
  
    if np.array(res.X.tolist()).ndim>1 :
        modes = res.X.tolist()[0]
    else :
        modes = res.X.tolist()

    original_modes = modes.copy()
    shifted_modes = modes.copy()

    switching = []
    for original_mode, shifted_mode in zip(original_modes[:-1], shifted_modes):
        item = original_mode - shifted_mode
        switching.append(item)
    number_switching = np.sum(np.abs(switching))

    dp = []
    condensate= [] 
    bog_consumed_optimum = []
    total_fgc_me_ge = []
    
    dp, condensate, bog_consumed_optimum, total_fgc_me_ge, modes = f_pressure(dff_guide, modes, continous_operating_hours)
    dp_cumsum = np.cumsum(dp) 
    tank_p = [x + p0 for x in dp_cumsum]

    result = []
    for i in np.arange(24, way_point_interval*continous_operating_hours, 24):
        daily_p = tank_p[i-24 : i]
        result.append(np.mean(daily_p))

    dff_guide.reset_index(inplace=True)
    
    data_t = {
        'press_predicted': tank_p,
        'condensate_predicted': condensate,
        'bog_consumed_predicted': bog_consumed_optimum,
        'frs_mode_recommended': modes,
        'fg_consumption_predicted': total_fgc_me_ge
    }

    # DataFrame 생성
    df_cc = pd.DataFrame(data_t)
    dff_guide_cc = pd.concat([dff_guide, df_cc], axis=1)

    return dff_guide_cc


class ElementwiseExample(ElementwiseProblem):
    def __init__(self, dff_guide, way_point_interval, continous_operating_hours, tk_press_set):
        super().__init__(            
            n_var= way_point_interval, 
            n_obj=1, 
            n_ieq_constr= 3, 
            xl=np.array([0] * way_point_interval), 
            xu=np.array([1] * way_point_interval),
            vtype=int
        )
        self.continous_operating_hours = continous_operating_hours
        self.way_point_interval = way_point_interval
        self.dff_guide = dff_guide
        self.tk_press_set = tk_press_set
                
    def _evaluate(self, modes, out, *args, **kwargs):
        
        modes = modes.tolist()     
        f1 = np.sum(modes)
        
        original_modes = modes.copy()
        shifted_modes = modes.copy()
      
        switching = []
        for original_mode, shifted_mode in zip(original_modes[:-1], shifted_modes):
            item = original_mode - shifted_mode
            switching.append(item)
        f2 = np.sum(np.abs(switching))        

        tank_dp, _, __, total_fgc_me_ge, modes = f_pressure(self.dff_guide, modes, self.continous_operating_hours)
        
        p0, p_h, p_l, p_hh, p_ll, p_unload = get_pressure_conditions(self.tk_press_set)

        tank_dp_cumsum = np.cumsum(tank_dp) 
        tank_p = [x + p0 for x in tank_dp_cumsum]

        result = []
        for i in np.arange(24, self.way_point_interval*self.continous_operating_hours, 24):
            daily_p = tank_p[i-24 : i]
            result.append(np.mean(daily_p))
        f3 = len([i for i in result if i < p_l])    # p_l 보다 daily average tank pressure가 작았던 시간
        f4 = len([i for i in result if i > p_h])    # p_h 보다 daily average tank pressure가 컸던 시간
        f5 = np.max(p_unload - tank_p[-1], 0)

        f_combined = f1 + f2**3 + (f3 + f4)  + f5
        
        g0 = tank_p[-1] - p_unload    # tank pressure가 terminal requirement를 만족시키는지 확인
        g1 = -np.min(tank_p) +p_ll    # p_ll 보다 tank pressure가 작은지 확인
        g2 =  np.max(tank_p) -p_hh    # p_hh 보다 tank pressure가 큰지 확인

        out["F"] = [f_combined]        
        out["G"] = [g0, g1, g2]


def f_pressure(dff_guide, modes, continous_operating_hours): 
    coef_const, coef_bog_consumed, coef_roll_swell, coef_roll_ww, coef_pitch_swell, coef_pitch_ww, coef_operation_days, ge_flow_frs = get_vessel_model()
    
    X_ = [[x] * continous_operating_hours for x in modes]
    modes = [y for x in X_ for y in x]
    fgc_frs = [x * ge_flow_frs for x in modes]    
    hicom_max = [x * hicom_no2 + hicom_no1 for x in modes]        
    fgc_me = dff_guide.fgc_me.values[:len(modes)]
    fgc_gcu = dff_guide.gcu_mode.values[:len(modes)]
    
    condensate_max = [x * condensate_max_no2 + condensate_max_no1 for x in modes]   
    for i in np.argwhere(np.array(fgc_me)<500).flatten().tolist():
        condensate_max[i] = 500
    
    fgc_ge =[fgc_ge_average] * len(modes)
    
    dp_env =  coef_const + coef_roll_swell* dff_guide['roll_acc_swell'].values[:len(modes)] + coef_roll_ww * dff_guide['roll_acc_ww'].values[:len(modes)] 
    + coef_pitch_swell * dff_guide['pitch_acc_swell'].values[:len(modes)] + coef_pitch_ww * dff_guide['pitch_acc_ww'].values[:len(modes)] 
    + coef_operation_days* dff_guide['operation_days'].values[:len(modes)]  
    
    condensate = list(map(min, hicom_max - fgc_me - fgc_frs - fgc_ge , condensate_max))

    total_fgc_me_ge = fgc_me + fgc_ge + fgc_gcu + fgc_frs
    
    bog_consumed_optimum = total_fgc_me_ge + [x * 0.85 for x in condensate]
    
    dp = (coef_bog_consumed  * bog_consumed_optimum + dp_env)

    return dp, condensate, bog_consumed_optimum, total_fgc_me_ge, modes


def generate_population(way_point_interval):
    initial_generation = []
    for ratio in [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        for start in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]:
            gen1, gen2 = gen_population(way_point_interval,ratio)
            initial_generation.append(gen1)
            initial_generation.append(gen2)
            try : 
                gen3, gen4 = gen_population_middle(way_point_interval,ratio, start)
                initial_generation.append(gen3)
                initial_generation.append(gen4)   
            except:
                pass

    initial_generation = np.array(initial_generation)    
    return initial_generation   

def gen_population(way_point_interval, ratio):
    return ([0]*int(round(way_point_interval*ratio,0))) + ([1]*(way_point_interval-int(round(way_point_interval*ratio,0)))),([1]*int(round(way_point_interval*ratio,0))) + ([0]*(way_point_interval-int(round(way_point_interval*ratio,0))))

def gen_population_middle(way_point_interval, ratio, start):
    if ratio + start < 0.9 :
        return ([0]*int(round(way_point_interval*start,0))) + ([1]*(int(round(way_point_interval*ratio,0)))) + ([0]*(way_point_interval-int(round(way_point_interval*start,0))-int(round(way_point_interval*ratio,0)))), ([1]*int(round(way_point_interval*start,0))) + ([0]*(int(round(way_point_interval*ratio,0)))) + ([1]*(way_point_interval-int(round(way_point_interval*start,0))-int(round(way_point_interval*ratio,0)))), 
    else:
        pass                                 
                                 
def main_flow(df_input, tk_press_set, departure_time, p_pitch_file:str, p_roll_file:str):
    
    # p_min_ll, p_min, p_max, p_max_hh, tank_pressure_departure = tk_press_set
    if 'GCU Run' not in df_input.columns : 
        df_input['gcu_mode'] = 0.0
        df_input = df_input.drop("gcu", axis=1)
        
    else : df_input.rename({'GCU Run : gcu_mode'}, inplace=True, axis = 1)
    

    result1 = smart_bog_fgc(df_input)

    wave_height_min = result1[['shts', 'shww']].min().min() - 0.2
    wave_height_max = result1[['shts', 'shww']].max().max() + 0.2
    speed_min = result1[['speed']].min().min() - 1.0
    speed_max = result1[['speed']].max().max() + 1.0
    
    ####### Roll, Pitch DB 저장 #######
    query_roll = pd.read_csv(p_roll_file)
    query_pitch = pd.read_csv(p_pitch_file)

    df_Aroll = query_roll[(query_roll['wave_height'] >= wave_height_min) & (query_roll['wave_height'] <= wave_height_max) & (query_roll['speed'] >= speed_min) & (query_roll['speed'] <= speed_max)]
    df_Apitch = query_pitch[(query_pitch['wave_height'] >= wave_height_min) & (query_pitch['wave_height'] <= wave_height_max) & (query_pitch['speed'] >= speed_min) & (query_pitch['speed'] <= speed_max)]
    
    result1[['roll_acc_swell', 'pitch_acc_swell']] = list(
        map(
            lambda speed, shts, mpts, totalswell_dir : 
                roll_pitch_acc(result1, df_Aroll, df_Apitch, speed, shts, mpts, totalswell_dir), 
                result1['speed'], 
                result1['shts'], 
                result1['mpts'], 
                result1['totalswell_dir']
        )
    )
    
    result1[['roll_acc_ww', 'pitch_acc_ww']] = list(
        map(
            lambda speed, shww, mpww, windwave_dir: 
            roll_pitch_acc(result1, df_Aroll, df_Apitch, speed, shww, mpww, windwave_dir),   
                result1['speed'], 
                result1['shww'], 
                result1['mpww'], 
                result1['windwave_dir']
           )
    )

    result1.set_index("ds_timeindex", inplace=True)
    resample_hour= 1        
    resample_time = str(resample_hour) + "h"        
    time_of_arrival = result1.index[-1]

    ############################################################################            
    ######### 계산하여 지저분하게 온 시간축을 1시간 단위로 정렬함  #############
    ############################################################################
    result1 = result1.resample(resample_time).mean()
    result1.interpolate(method='time', inplace=True)
    result1.bfill(inplace=True)        
    result1 = result1[:time_of_arrival]

    result1['operation_time'] = result1.index - departure_time
    result1['operation_days']  = list(map(lambda x: x.days, result1['operation_time']))

    result1.reset_index(inplace=True)

    # New 최적화 로직
    continous_operating_hours = 6
    df = smart_bog_opt(result1, continous_operating_hours, tk_press_set)

    # 컬럼명 지정
    df.columns = ['ds_timeindex', 'longitude', 'latitude', 'heading', 'rpm', 'speed_predicted', 'fgc_me', 
                  'swell_height', 'swell_direction', 'swell_period','wave_height', 'wave_direction', 'wave_period', 'wind_speed', 'wind_direction',
                  'current_speed', 'current_direction', 'totalswell_dir', 'windwave_dir', 'GCU_Running',
                  'swell_roll_acc', 'swell_pitch_acc', 'wave_roll_acc', 'wave_pitch_acc', 'operation_time', 'operation_days', 
                  'press_predicted', 'condensate_predicted', 'bog_consumed_predicted', 'frs_mode_recommended', 'fg_consumption_predicted']

    bog_simul = df[['ds_timeindex', 'latitude', 'longitude', 'rpm', 'speed_predicted', 'frs_mode_recommended', 'GCU_Running', 'press_predicted', 'condensate_predicted', 'bog_consumed_predicted']]

    # bog_simul.loc[:, bog_simul.columns != 'ds_timeindex'] = bog_simul.loc[:, bog_simul.columns != 'ds_timeindex'].astype(float).round(3) # float type & 반올림
    # bog_simul.columns = ['/ship////1h_dt/time', '/vdr////plan_deg/lat', '/vdr////plan_deg/lon', '/aprs////aver_simul/run', '/aprs////aver_simul_mbar/press', '/aprs/septr/gas//aver_simul_out_kgph/flow', '/aprs//bog//sum_simul_t/cons']
    bog_simul.columns = ['Time','Lat','Lon','SOG','Shaft RPM','GCU Run','APRS Run','LNG tank press','APRS Relique flow','BOG']

    return bog_simul

    
def bog_real(bog_simul, now):
    start_time = bog_simul['Time'][0]
    bog_simul.set_index('Time', inplace=True)
    bog_actual = pd.DataFrame(index=bog_simul.index, columns = ['Lat','Lon','SOG','Shaft RPM','GCU Run', 'APRS Run','LNG tank press','APRS Relique flow','BOG'])
    bog_actual = bog_actual[start_time:now]
    bog_actual.reset_index(inplace=True)
    # bog_actual.columns = ['Time','Lat','Lon','SOG','Shaft RPM','GCU Run', 'APRS Run','LNG tank press','APRS Relique flow','BOG']
    API_TAGS = ['/vdr////deg/lat', '/vdr////deg/lon', '/vdr////kn/sog','/me/shaft///rpm/speed', '/aprs/////run', '/gcu/fuel/gas///mode', '/ct03////mbar/press', '/aprs/septr/gas//out_kgph/flow', '/aprs//bog//t/cons']

    for n in range(len(bog_actual)):
        time = bog_actual.loc[n, 'Time'].strftime("%Y-%m-%d %H:%M:%S")
        bog_actual.loc[n, 'Lat'] = 1.0                    # sd  API call - '/ship////1h_dt/time' 시간에 해당하는 선박의 lat tag : /vdr////deg/lat (범위 x)
        bog_actual.loc[n, 'Lon'] = 1.0                    # sd  API call - '/ship////1h_dt/time' 시간에 해당하는 선박의 lon tag : /vdr////deg/lon (범위 x)
        bog_actual.loc[n, 'SOG'] = 1.0                    # sd  API call - '/ship////1h_dt/time' 시간에 해당하는 선박의 lon tag : '/vdr////kn/sog'(범위 x)
        bog_actual.loc[n, 'Shaft RPM'] = 1.0              # rda API call - ['/ship////1h_dt/time' 시간 - 1시간] 부터 ['/ship////1h_dt/time' 시간] 까지의 ME Shaft RPM ((1+2) / 2) 평균 tag : /me/shaft///rpm/speed
        bog_actual.loc[n, 'GCU Run'] = 0.0               # rda API call - '/ship////1h_dt/time' 시간에 해당하는 gcu 작동 여부 (범위 x) tag : '/gcu/fuel/gas///mode'
        bog_actual.loc[n, 'APRS Run'] = 0.0               # rda API call - '/ship////1h_dt/time' 시간에 해당하는 aprs 장비의 작동 여부 (범위 x) tag : '/aprs/////run'
        bog_actual.loc[n, 'LNG tank press'] = 1.0        # sd  API call - ['/ship////1h_dt/time' 시간 - 1시간] 부터 ['/ship////1h_dt/time' 시간] 까지의 LNG tank pressure 평균 tag : '/ct03////mbar/press'
        bog_actual.loc[n, 'APRS Relique flow'] = 1.0     # sd  API call - ['/ship////1h_dt/time' 시간 - 1시간] 부터 ['/ship////1h_dt/time' 시간] 까지의 Gas outlet flowmeter 평균 tag : '/aprs/septr/gas//out_kgph/flow'
        bog_actual.loc[n, 'BOG'] = 1.0                   # rda API call - ['/ship////1h_dt/time' 시간 - 1시간] 부터 ['/ship////1h_dt/time' 시간] 까지의 Boil Of Gas [ton] 평균 tag : '/aprs//bog//t/cons'

    return bog_actual


def estimate(p_tank_pressure:list, p_waypoints:list, p_pitch_filename:str, p_roll_filename:str) -> dict:
    '''
    Parameters
    - p_tank_pressure (list) : [ll, l, h, hh, target], user setting value
    - p_waypoints(list) : [{no:1, ...}, {no: 1-1, ...}], from voyage plan DB
    
    Returns 
    - bog_simul_dict(dict) : voyage plan의 initial simulation 값
    { 
        'Time' : [... ...], 
        'Lat' : [..., ....], 
        'Lon' : [..., ... ],
        'SOG' : [..., ...],
        'Shaft RPM': [..., ...], 
        'GCU Run': [..., ....],
        'APRS Run' : [..., ....],
        'LNG tank press': [...., .....],
        'APRS Relique flow':[....,....],
        'BOG': [..., ...]
    }
    
    - re_simul_dict (dict) : now time까지의 실적을 반영한 simulation 값
    
   { 
        'Time' : [... ...], 
        'Lat' : [..., ....], 
        'Lon' : [..., ... ],
        'SOG' : [..., ...],
        'Shaft RPM': [..., ...], 
        'GCU Run': [..., ....],
        'APRS Run' : [..., ....],
        'LNG tank press': [...., .....],
        'APRS Relique flow':[....,....],
        'BOG': [..., ...]
    }
    

    - now_time_str (str) : current time 
    
    '''
    
    now_time = datetime.now()
    if p_tank_pressure == [] : tk_press_set = [100, 120, 230, 250, 150]
    else : tk_press_set = p_tank_pressure
    if p_waypoints == [] : return
    
    columns = ['time', 'lat', 'lon', 'shaft_rpm', 'sog', 'winDir',	'winSpd','wavHgt','wavDir','curDir','curSpd']
    waypoints_df = pd.DataFrame(p_waypoints)[columns]
    
    df_input = waypoints_df.copy()
    df_input['gcu'] = 'off'
    df_input['laden'] = 1

    dp_start = df_input['time'][0]
    start_time = datetime.strptime(dp_start, '%Y-%m-%d %H:%M:%S')

    bog_simul_init = main_flow(df_input, tk_press_set, start_time, p_pitch_filename, p_roll_filename)

    bog_actual = bog_real(bog_simul_init, now_time)

    bog_simul = bog_simul_init[now_time:]
    bog_simul.reset_index(inplace=True)

    bog_actual_dict = bog_actual.to_dict('records')
    bog_simul_dict = bog_simul.to_dict('records')

    now_time_str = now_time.strftime("%Y-%m-%d %H:%M:%S")

    return bog_actual_dict, bog_simul_dict, now_time_str
