import argparse
import datetime
import json
import logging
import pickle
import sys
from pathlib import Path

import psycopg
import psycopg_pool
from flask import request
from flask_openapi3 import Info, Tag, OpenAPI
from pydantic import BaseModel, Field
from typing import List
import waitress

from hs4da import SFWBase, DataAccess, Voyage

import HS4Wrapper
import Generate
import Optimize
import TrimOptimizer
import SmartBogSupport
import VoyageStat

# -------------------------------------------------------------------------------------------------
# OpenAPI 3 Documentation Setup
# -------------------------------------------------------------------------------------------------
info = Info(title="Service Framework: Voyage Planner API", version="0.7.0")
app:OpenAPI = OpenAPI(__name__, info=info)
TAGS:dict = {
  "voyage": Tag(name="voyage and playback", description="Voyage trim"),
  "voyage-plan": Tag(name="voyage plan", description="Voyage plan"),
  "emission": Tag(name="emission and cii", description="Emission and cii data"),
  "bog": Tag(name="smart bog", description="Smart BoG data"),
}


# -------------------------------------------------------------------------------------------------
# Global Configurations
# -------------------------------------------------------------------------------------------------
G_CONFIG:dict = None


# -------------------------------------------------------------------------------------------------
# pre-load data
# -------------------------------------------------------------------------------------------------
def load_graph_data(p_graph_pickle_filename: str) -> dict:
  ret_dict: dict = None
  with open(p_graph_pickle_filename, "rb") as fp:
    unpickler = pickle.Unpickler(fp)
    ret_dict = unpickler.load()
  return ret_dict


def update_cii_base_data(p_ship_id:str, p_conn:psycopg.Connection):
  global G_CONFIG
  year = datetime.datetime.now(datetime.timezone.utc).year
  if "ship_data" not in G_CONFIG:
    basedb = DataAccess.ShipBaseData(p_conn)
    basedb.begin_transaction()
    # s.id, s.name, s.mmsi, s.size, s.length, s.width, s.gt, s.nt, s.dwt, st.name
    ship_data = basedb.get_ship_data(p_ship_id)
    cii_grade_lines = basedb.get_cii_grade(p_ship_id, year)
    basedb.end_transaction()
    G_CONFIG["ship_data"] = {
      "shipid": ship_data[0], "name": ship_data[1], "mmsi": ship_data[2], 
      "size": ship_data[3], "length": ship_data[4], "width": ship_data[5],
      "gt": ship_data[6], "nt": ship_data[7], "dwt": ship_data[8], "type": ship_data[9],
      "cii_year": year, "cii_grade_lines": cii_grade_lines}
  else:
    if G_CONFIG["ship_data"]["cii_year"] != year:
      basedb = DataAccess.ShipBaseData(p_conn)
      cii_grade_lines = basedb.get_cii_grade(p_ship_id, year)
      G_CONFIG["ship_data"]["cii_year"] = year
      G_CONFIG["ship_data"]["cii_grade_lines"] = cii_grade_lines


# -------------------------------------------------------------------------------------------------
# Shared functions
# -------------------------------------------------------------------------------------------------
def access_log(p_status:int, p_content_length:int):
  """
  Access logging for statistics

  :param p_status: HTTP status code on response
  :param p_content_legnth: content length
  """
  formatted_date:str = datetime.datetime.now(datetime.timezone.utc).isoformat()
  access_logger = logging.getLogger("access")
  access_logger.info(f'{request.remote_addr} - - [{formatted_date}] "{request.method} {request.url}" {p_status} {p_content_length}')


def setup_api_server(p_app_config:dict):
  """
  Global configuration assigns from application configuration read on file format

  :param p_app_config: application configuration
  """
  global G_CONFIG
  G_CONFIG = p_app_config

def result_to_dict(p_results: list) -> dict:
  if p_results is not None and len(p_results) > 0:
    ret_dict: dict = dict()
    ret_dict["rows"] = dict()
    for cursor in p_results:
      if cursor[0] not in ret_dict["rows"]:
        ret_dict["rows"][cursor[0]] = list()
      ret_dict["rows"][cursor[0]].append({
        "ts": SFWBase.to_iso8601(cursor[2]),
        "value": cursor[DataAccess.ResultTimeseries.RESULTS_DICT[cursor[7]]]
      })
    return ret_dict
  return None


def convert_scoreboard_to_dict(p_row:list) -> dict:
  #  0 ship_id, voyage_seq, voyage_id, departure_code, arrival_code, departure_time, arrival_time, distance_from_next_port,
  #  8 loading, all_cnt_sog, all_tot_sog, all_cnt_stw, all_tot_stw, all_tot_dist,
  # 14 all_cargo_carried, all_tot_transport_work,  all_tot_vlsfo_foc, all_tot_lsmgo_foc, all_tot_hfo_foc, all_tot_mdo_foc,
  # 20 all_cnt_foc, all_tot_foc, all_cnt_fgc, all_tot_fgc, all_cnt_co2, all_tot_co2, all_cnt_ch4, all_tot_ch4, all_cnt_co2e, all_tot_co2e,
  # 30 me_cnt_load, me_tot_load, me_cnt_foc, me_tot_foc, me_cnt_fgc, me_tot_fgc, me_cnt_co2, me_tot_co2, me_cnt_ch4, me_tot_ch4, me_cnt_co2e, me_tot_co2e,
  # 42 ge_cnt_foc, ge_tot_foc, ge_cnt_fgc, ge_tot_fgc, ge_cnt_co2, ge_tot_co2, ge_cnt_ch4, ge_tot_ch4, ge_cnt_co2e, ge_tot_co2e,
  # 52 ab_cnt_co2, ab_tot_co2, ab_cnt_ch4, ab_tot_ch4, ab_cnt_co2e, ab_tot_co2e,
  # 58 cb_cnt_co2, cb_tot_co2, cb_cnt_ch4, cb_tot_ch4, cb_cnt_co2e, cb_tot_co2e,
  # 64 gcu_cnt_co2, gcu_tot_co2, gcu_cnt_ch4, gcu_tot_ch4, gcu_cnt_co2e, gcu_tot_co2e
  ret_dict = {
    "ship_id": p_row[0],
    "voyage_seq": p_row[1],
    "voyage_id": p_row[2],
    "/ship//depar//voy/code": p_row[3],
    "/ship//arriv//voy/code": p_row[4],
    "/ship//depar//voy_dt/time": p_row[5].isoformat(),
    "/ship//arriv//voy_dt/time": p_row[6].isoformat(),
    "/vdr////nport_sum_km/dist": p_row[7],
    "/ship/handl///voy/stat": p_row[8]}
  
  ret_dict["/vdr////voy_aver_kn/sog"] = p_row[10] / p_row[9] if p_row[9] > 0 else 0.0
  ret_dict["/vdr////voy_aver_kn/stw"] = p_row[12] / p_row[11] if p_row[11] > 0 else 0.0
  ret_dict["/vdr////voy_sum_km/dist"] = p_row[13]
  
  cii_value, cii_grade = Voyage.calc_cii_value_and_grade(p_row[25], p_row[13], G_CONFIG["ship_data"]["cii_grade_lines"], G_CONFIG["ship_data"]["dwt"])
  ret_dict["/ship/cii///voy_aver_gptmile/value"] = cii_value
  ret_dict["/ship/cii///voy_aver/grade"] = cii_grade
  
  ret_dict["/ct//liq//t/vol"] = p_row[14]
  ret_dict["/ship////voy_sum_tpmile/trans"] = p_row[15]
  ret_dict["/ship/fuel/oil/vlsfo/voy_sum_t/foc"] = p_row[16]
  ret_dict["/ship/fuel/oil/lsmgo/voy_sum_t/foc"] = p_row[17]
  ret_dict["/ship/fuel/oil/hfo/voy_sum_t/foc"] = p_row[18]
  ret_dict["/ship/fuel/oil/mdo/voy_sum_t/foc"] = p_row[19]
  
  ret_dict["/ship/fuel/oil//voy_aver_kgph/foc"] = p_row[21] / p_row[20] if p_row[20] > 0 else 0.0
  ret_dict["/ship/fuel/oil//voy_sum_t/foc"] = p_row[21] * 0.001
  ret_dict["/ship/fuel/gas//voy_aver_kgph/fgc"] = p_row[23] / p_row[22] if p_row[22] > 0 else 0.0
  ret_dict["/ship/fuel/gas//voy_sum_t/fgc"] = p_row[23] * 0.001
  ret_dict["/ship////voy_aver_kgph/co2"] = p_row[25] / p_row[24] if p_row[24] > 0 else 0.0
  ret_dict["/ship////voy_sum_t/co2"] = p_row[25] * 0.001
  ret_dict["/ship////voy_aver_kgph/ch4"] = p_row[27] / p_row[26] if p_row[26] > 0 else 0.0
  ret_dict["/ship////voy_sum_t/ch4"] = p_row[27] * 0.001
  ret_dict["/ship////voy_aver_kgph/co2e"] = p_row[29] / p_row[28] if p_row[28] > 0 else 0.0
  ret_dict["/ship////voy_sum_t/co2e"] = p_row[29] * 0.001

  ret_dict["/me////aver_per/load"] = (p_row[31] / p_row[30]) if p_row[30] > 0 else 0.0
  ret_dict["/me/fuel/oil//voy_aver_kgph/foc"] = p_row[33] / p_row[32] if p_row[32] > 0 else 0.0
  ret_dict["/me/fuel/oil//voy_sum_t/foc"] = p_row[33] * 0.001
  ret_dict["/me/fuel/gas//voy_aver_kgph/fgc"] = p_row[35] / p_row[34] if p_row[34] > 0 else 0.0
  ret_dict["/me/fuel/gas//voy_sum_t/fgc"] = p_row[35] * 0.001
  ret_dict["/me////voy_aver_kgph/co2"] = p_row[37] / p_row[36] if p_row[36] > 0 else 0.0
  ret_dict["/me////voy_sum_t/co2"] = p_row[37] * 0.001
  ret_dict["/me////voy_aver_kgph/ch4"] = p_row[39] / p_row[38] if p_row[38] > 0 else 0.0
  ret_dict["/me////voy_sum_t/ch4"] = p_row[39] * 0.001
  ret_dict["/me////voy_aver_kgph/co2e"] = p_row[41] / p_row[40] if p_row[40] > 0 else 0.0
  ret_dict["/me////voy_sum_t/co2e"] = p_row[41] * 0.001
  
  ret_dict["/ge/fuel/oil//voy_aver_kgph/foc"] = p_row[43] / p_row[42] if p_row[42] > 0 else 0.0
  ret_dict["/ge/fuel/oil//voy_sum_t/foc"] = p_row[43] * 0.001
  ret_dict["/ge/fuel/gas//voy_aver_kgph/fgc"] = p_row[45] / p_row[44] if p_row[44] > 0 else 0.0
  ret_dict["/ge/fuel/gas//voy_sum_t/fgc"] = p_row[45] * 0.001
  ret_dict["/ge////voy_aver_kgph/co2"] = p_row[47] / p_row[46] if p_row[46] > 0 else 0.0
  ret_dict["/ge////voy_sum_t/co2"] = p_row[47] * 0.001
  ret_dict["/ge////voy_aver_kgph/ch4"] = p_row[49] / p_row[48] if p_row[48] > 0 else 0.0
  ret_dict["/ge////voy_sum_t/ch4"] = p_row[49] * 0.001
  ret_dict["/ge////voy_aver_kgph/co2e"] = p_row[51] / p_row[50] if p_row[50] > 0 else 0.0
  ret_dict["/ge////voy_sum_t/co2e"] = p_row[51] * 0.001

  ret_dict["/ab////voy_aver_kgph/co2"] = p_row[53] / p_row[52] if p_row[52] > 0 else 0.0
  ret_dict["/ab////voy_sum_t/co2"] = p_row[53] * 0.001
  ret_dict["/ab////voy_aver_kgph/ch4"] = p_row[55] / p_row[54] if p_row[54] > 0 else 0.0
  ret_dict["/ab////voy_sum_t/ch4"] = p_row[55] * 0.001
  ret_dict["/ab////voy_aver_kgph/co2e"] = p_row[57] / p_row[56] if p_row[56] > 0 else 0.0
  ret_dict["/ab////voy_sum_t/co2e"] = p_row[57] * 0.001

  ret_dict["/cb////voy_aver_kgph/co2"] = p_row[59] / p_row[58] if p_row[58] > 0 else 0.0
  ret_dict["/cb////voy_sum_t/co2"] = p_row[59] * 0.001
  ret_dict["/cb////voy_aver_kgph/ch4"] = p_row[61] / p_row[60] if p_row[60] > 0 else 0.0
  ret_dict["/cb////voy_sum_t/ch4"] = p_row[61] * 0.001
  ret_dict["/cb////voy_aver_kgph/co2e"] = p_row[63] / p_row[62] if p_row[62] > 0 else 0.0
  ret_dict["/cb////voy_sum_t/co2e"] = p_row[63] * 0.001

  ret_dict["/gcu////voy_aver_kgph/co2"] = p_row[65] / p_row[64] if p_row[64] > 0 else 0.0
  ret_dict["/gcu////voy_sum_t/co2"] = p_row[65] * 0.001
  ret_dict["/gcu////voy_aver_kgph/ch4"] = p_row[67] / p_row[66] if p_row[66] > 0 else 0.0
  ret_dict["/gcu////voy_sum_t/ch4"] = p_row[67] * 0.001
  ret_dict["/gcu////voy_aver_kgph/co2e"] = p_row[69] / p_row[68] if p_row[68] > 0 else 0.0
  ret_dict["/gcu////voy_sum_t/co2e"] = p_row[69] * 0.001

  return ret_dict


def convert_stat_row_to_dict(p_row:list) -> dict:
  columns:list = [
    "ship_id",
    "base_time",
    "/vdr////1h_deg/lat",
    "/vdr////1h_deg/lon",
    "/vdr////1h_aver_kn/sog",
    "/vdr////1h_aver_kn/stw",
    "/vdr////1h_sum_km/dist",
    "/ship/fuel/oil//1h_aver_kgph/foc",
    "/ship/fuel/oil//1h_sum_t/foc",	
    "/ship/fuel/gas//1h_aver_kgph/fgc",
    "/ship/fuel/gas//1h_sum_t/fgc",
    "/ship////1h_aver_kgph/co2",
    "/ship////1h_sum_t/co2",
    "/ship////1h_aver_kgph/ch4",
    "/ship////1h_sum_t/ch4",
    "/ship////1h_aver_kgph/co2e",
    "/ship////1h_sum_t/co2e",
    "/me////per/load",
    "/me/fuel/oil//1h_aver_kgph/foc",
    "/me/fuel/oil//1h_sum_t/foc",
    "/me/fuel/gas//1h_aver_kgph/fgc",
    "/me/fuel/gas//1h_sum_t/fgc",
    "/me////1h_aver_kgph/co2",
    "/me////1h_aver_kgph/ch4",
    "/me////1h_aver_kgph/co2e",
    "/ge/fuel/oil//1h_aver_kgph/foc",
    "/ge/fuel/oil//1h_sum_t/foc",
    "/ge/fuel/gas//1h_aver_kgph/fgc",
    "/ge/fuel/gas//1h_sum_t/fgc",
    "/ge////1h_aver_kgph/co2",
    "/ge////1h_aver_kgph/ch4",
    "/ge////1h_aver_kgph/co2e",
    "/ab////1h_aver_kgph/co2",
    "/ab////1h_aver_kgph/ch4",
    "/ab////1h_aver_kgph/co2e",
    "/cb////1h_aver_kgph/co2",
    "/cb////1h_aver_kgph/ch4",
    "/cb////1h_aver_kgph/co2e",
    "/gcu////1h_aver_kgph/co2",
    "/gcu////1h_aver_kgph/ch4",
    "/gcu////1h_aver_kgph/co2e"
  ]
  ret_dict:dict = dict()
  for i in range(len(columns)):
    if columns[i] == "base_time":
      ret_dict[columns[i]] = p_row[i].isoformat()
    else:
      ret_dict[columns[i]] = p_row[i]
  return ret_dict


# -------------------------------------------------------------------------------------------------
# API - Shared models for OpenAPI3 (Swagger)
# -------------------------------------------------------------------------------------------------
class ResponseBodyError(BaseModel):
  """
  Error response body class
  """
  status:str = Field(None, description="Status")
  reason:str = Field(None, description="Error reason")


class RequestPathShipID(BaseModel):
  shipid:str = Field(None, description="Ship ID")


class RequestQueryDuration(BaseModel):
  start:str = Field(None, alias="from", description="시작 일 (yyyy-MM-ddTHH:mm:ssz). ex) 2019-08-08T00:00:00+09:00, 2019-08-08T00:00:00Z")
  end:str = Field(None, alias="to",description="종료 일 (yyyy-MM-ddTHH:mm:ssz). ex) 2019-08-08T00:00:00+09:00, 2019-08-08T00:00:00Z")


# -------------------------------------------------------------------------------------------------
# API - voyage and playback
# -------------------------------------------------------------------------------------------------
@app.get('/api/v1/ships/<string:shipid>/playback',
    summary="Get playback data",
    tags=[TAGS["voyage"]],
    description="Get playback data from database")
def api_get_playback_data(path: RequestPathShipID, query: RequestQueryDuration):
  stat_channel_ids:list = ["/ship/fuel/oil//equiv_kgph/flow", "/ship/fuel/gas//equiv_kgph/flow"]
  with G_CONFIG["db_conn_pool"].connection() as conn:
    pb = Voyage.PlaybackBase(conn)
    pb.begin_transaction()
    ret_list:list = pb.select(path.shipid, query.start, query.end)
    pb.end_transaction()
    sdb = DataAccess.ResultStatDB(conn)
    stat_list:list = sdb.get_hourly_stat(path.shipid, stat_channel_ids, query.start, query.end)
  playback = Voyage.Playback(path.shipid, int(G_CONFIG["playback"]["refresh-duration"]), True)
  for row in ret_list:
    playback.set_data(row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13], row[14])
  playback.flush()
  stat_dict:dict = dict()
  for row in stat_list:
    # data_channel_id, created_time, min, max, sum, avg, count
    if row[1] not in stat_dict:
      stat_dict[row[1]] = dict()  
    stat_dict[row[1]][row[0]] = row[5]
  if len(playback.results) > 0:
    ret_dict = {"result": "success", "rows": list()}
    for row in playback.results:
      # ship_id, base_time, latitude, longitude, sog, voyage_status, loading, shaft_rpm, oil_type, gas_type, foc, fgc, total_foc, total_fgc
      if len(row) == 0:
        continue
      if row[1] is None:
        continue
      stat_time = datetime.datetime(row[1].year, row[1].month, row[1].day, row[1].hour, 0, 0)
      ret_dict["rows"].append({
        "ts": row[1].isoformat(), "latitude": row[2], "longitude": row[3], "sog": row[4], "status": row[5], "loading": row[6], "shaft_rpm": row[7],
        "oil_type": row[8], "gas_type": row[9], "foc": row[10], "fgc": row[11], "total_foc": row[12], "total_fgc": row[13],
        "aveg_foc": stat_dict[stat_time]["/ship/fuel/oil//equiv_kgph/flow"] if stat_time in stat_dict else 0.0,
        "aveg_fgc": stat_dict[stat_time]["/ship/fuel/gas//equiv_kgph/flow"] if stat_time in stat_dict else 0.0
      })
    if len(ret_dict["rows"]) > 0:
      ret_str = json.dumps(ret_dict)
    else:
      ret_str = json.dumps({"code": 104, "message": "no data"})
  else:
    ret_str = json.dumps({"code": 104, "message": "no data"})
  access_log(200, len(ret_str))
  return ret_str


@app.get('/api/v1/ships/<string:shipid>/voyage/scoreboard',
    summary="Get voyage scoreboard",
    tags=[TAGS["voyage"]],
    description="Get voyage scoreboard")
def api_get_voyage_scoreboard(path: RequestPathShipID, query: RequestQueryDuration):
  ret_dict:dict = None
  with G_CONFIG["db_conn_pool"].connection() as conn:
    db = Voyage.VoyageServiceDB(conn)
    result = db.select_voyage_scoreboard_list(path.shipid, query.start, query.end)
    update_cii_base_data(path.shipid, conn)

  if result is not None and len(result) > 0:
    ret_dict = {"result": "success", "rows": list()}
    for row in result:
      ret_dict["rows"].append(convert_scoreboard_to_dict(row))
    ret_str = json.dumps(ret_dict)
    status = 200
  else:
    ret_str = json.dumps({"code": 104, "message": "no data"})
    status = 400
  access_log(status, len(ret_str))
  if status == 200:
    return ret_str
  return ret_str, status


@app.get('/api/v1/ships/<string:shipid>/voyage/current',
    summary="Get current",
    tags=[TAGS["voyage"]],
    description="Get current voyage info")
def api_get_current_voyage(path: RequestPathShipID):
  ret_dict:dict = None
  # with G_CONFIG["db_conn_pool"].connection() as conn:
  #   db = Voyage.VoyageServiceDB(conn)
  #   db.begin_transaction()
  #   voyage_seq_info = db.get_max_sequence(path.shipid)
  #   if voyage_seq_info[0] > 0 and voyage_seq_info[1] is not None:
  #     max_sequence = voyage_seq_info[1]
  #     current_voyage_scoreboard = db.select_voyage_scoreboard(path.shipid, max_sequence)
  #     current_voyage = db.get_current_voyage(path.shipid)
  #     voyage_name = 
  #   else:
      

  #   result = db.select_voyage_scoreboard_list(path.shipid, query.start, query.end)
  #   update_cii_base_data(path.shipid, conn)
  result = ["test", "2024-12-31 00:00:00"]
  if result is not None and len(result) > 0:
    ret_dict = {"result": "success", "rows": result}
    ret_str = json.dumps(ret_dict)
    status = 200
  else:
    ret_str = json.dumps({"code": 104, "message": "no data"})
    status = 400
  access_log(status, len(ret_str))
  if status == 200:
    return ret_str
  return ret_str, status


@app.get('/api/v1/ships/<string:shipid>/voyage/stat',
    summary="Get voyage statistics",
    tags=[TAGS["voyage"]],
    description="Get voyage statistics")
def api_get_voyage_hourly_stat(path: RequestPathShipID, query: RequestQueryDuration):
  ret_dict:dict = None
  with G_CONFIG["db_conn_pool"].connection() as conn:
    db = VoyageStat.VoyageStatDB(conn)
    result = db.select(path.shipid, query.start, query.end)

  if result is not None and len(result) > 0:
    ret_dict = {"result": "success", "rows": list()}
    for row in result:
      ret_dict["rows"].append(convert_stat_row_to_dict(row))
    ret_str = json.dumps(ret_dict)
    status = 200
  else:
    ret_str = json.dumps({"code": 104, "message": "no data"})
    status = 400
  access_log(status, len(ret_str))
  if status == 200:
    return ret_str
  return ret_str, status


class RequestOptimalTrim(BaseModel):
  """
  api_voyage_trim_optimal options request class
  """
  speed: float = Field(None, description="speed")
  draft: float = Field(None, description="draft")

class ResponseOptimalTrim(BaseModel):
  """
  api_voyage_plan_generate response body class
  """
  opt_trim: float = Field(None, description="optimal trim")
  max_eff: float = Field(None, description="max effieciency")

@app.get(
    "/api/v1/ships/<string:shipid>/voyage/trim/optimize",
    summary="Get optimial trim from speed, draft",
    tags=[TAGS["voyage"]],
    description="Get optimial trim from speed, draft",
    responses={200: ResponseOptimalTrim, 400:ResponseBodyError})
def api_voyage_trim_optimal(path:RequestPathShipID, query:RequestOptimalTrim):
  logging.info("Input data = %s %s", path.shipid)

  with G_CONFIG["db_conn_pool"].connection() as conn:
    output_list = TrimOptimizer.GetOptimalTrim(conn, path.shipid, query.speed, query.draft)
  
  if output_list is None:
    logging.warning("Data is null")
    ret_str = json.dumps({"status": "failed", "reason": "data is null"})
    status = 400
  else:
    ret_str = json.dumps({"optimal_trim": output_list[0], "maximum_efficiency": output_list[1]})
    status = 200
  access_log(status, len(ret_str))
  return ret_str, status


# -------------------------------------------------------------------------------------------------
# API - voyage plan
# -------------------------------------------------------------------------------------------------
class RequestVoyagePlan(BaseModel):
  """
  api_voyage_plan_generate request class
  """
  waypoints: list = Field(None, description="Waypoints: [{no:\"1\", code:\"USSAB\", lat:\"29.717\", lon:\"-93.87\"}]")

class ResponseBodyVoyagePlan(BaseModel):
  """
  api_voyage_plan_generate response body class
  """
  waypoints: list = Field(None, description="list of result data")

@app.post(
    "/api/v1/ships/<string:shipid>/voyage-plans/generate",
    summary="Get voyage-plan generate data",
    tags=[TAGS["voyage-plan"]],
    description="Get voyage-plan generate data of requested voyage-plan",
    responses={200: ResponseBodyVoyagePlan, 400:ResponseBodyError})
def api_voyage_plan_generate(path:RequestPathShipID, body:RequestVoyagePlan):
  logging.info("Input data = %s %s", path.shipid, body)
  output_dict:dict = None
  with G_CONFIG["db_conn_pool"].connection() as conn:
    access = DataAccess.ResultTimeseries(conn)
    ret_dict = result_to_dict(access.get_latest_results(path.shipid,["/hs4rda_v1/ship////atf_fwd_m/draft"] ),)
    if ret_dict is not None:
      draft = list(ret_dict['rows'].values())[0][0]['value']
    # call voyage plan generate function
    output_dict = Generate.GeneratePlan(conn, path.shipid, body.waypoints, draft, G_CONFIG["graph_data"], G_CONFIG["weather"]["pickle-file"])

  if output_dict is None:
    logging.warning("Data is null")
    return json.dumps({"status": "failed", "reason": "data is null"}), 400
  return json.dumps(output_dict), 200


class RequestOptimalOptions(BaseModel):
  """
  api_voyage_plan_optimal options request class
  """
  target: str = Field(None, description="target")
  wind_threshold: int = Field(None, description="wind threshold (0 or 1)")
  wave_threshold: int = Field(None, description="wave threshold (0 or 1)")
  current_threshold: int = Field(None, description="current threshold (0 or 1)")
  thresholds: dict = Field(None, description="threshold list")

class RequestVoyagePlanOptimal(BaseModel):
  """
  api_voyage_plan_optimal request class
  """
  optimizeOptions: RequestOptimalOptions = Field(None, description="OptimizeOptions")
  waypoints: list = Field(None, description="waypoints: [{no:\"1\", code:\"USSAB\", lat:\"29.717\", lon:\"-93.87\"}]")

@app.post(
    "/api/v1/ships/<string:shipid>/voyage-plans/optimize",
    summary="Get voyage-plan optimal data",
    tags=[TAGS["voyage-plan"]],
    description="Get voyage-plan optimal data of requested voyage-plan",
    responses={200: ResponseBodyVoyagePlan, 400:ResponseBodyError})

def api_voyage_plan_optimal(path:RequestPathShipID, body:RequestVoyagePlanOptimal):
  logging.info("Input data = %s %s", path.shipid, body)
  output_dict:dict = None
  # call voyage plan optimal function
  with G_CONFIG["db_conn_pool"].connection() as conn:
    access = DataAccess.ResultTimeseries(conn)
    ret_dict = result_to_dict(access.get_latest_results(path.shipid,["/hs4rda_v1/ship////atf_fwd_m/draft"] ))
    if ret_dict is not None:
      draft = list(ret_dict['rows'].values())[0][0]['value']
    output_dict = Optimize.OptimizePlan(conn, path.shipid, body.waypoints, draft, dict(body.optimizeOptions), G_CONFIG["weather"]["pickle-file"], G_CONFIG["weather"]["waterdepth-file"])

  if output_dict is None:
    logging.warning("Data is null")
    return json.dumps({"status": "failed", "reason": "data is null"}), 400
  return json.dumps(output_dict), 200


# -------------------------------------------------------------------------------------------------
# API - emission and cii
# -------------------------------------------------------------------------------------------------
class ResponseBodyEmission(BaseModel):
  """
  response body class for emission apis
  """
  result: str = Field(None, description="succes or faild")
  rows: dict = Field(None, description="result data")


def make_response_for_emission_by_me_load(p_result:list) -> dict:
  ret_dict:dict = {"all": dict(), "ballast": dict(), "laden": dict() }
  for index in range(0, 125, 5):
    ret_dict["all"][index] = {"co2": 0.0, "ch4": 0.0, "co2e": 0.0}
    ret_dict["ballast"][index] = {"co2": 0.0, "ch4": 0.0, "co2e": 0.0}
    ret_dict["laden"][index] = {"co2": 0.0, "ch4": 0.0, "co2e": 0.0}
  for row in p_result:
    logging.debug(row)
    # load_index, sum(ballast_tot_co2), sum(ballast_tot_ch4), sum(ballast_tot_co2e), sum(laden_tot_co2), sum(laden_tot_ch4), sum(laden_tot_co2e)
    load_index = int(row[0])
    ret_dict["all"][load_index] = {"co2": row[1] + row[4], "ch4": row[2] + row[5], "co2e": row[3] + row[6]}
    ret_dict["ballast"][load_index] = {"co2": row[1], "ch4": row[2], "co2e": row[3]}
    ret_dict["laden"][load_index] = {"co2": row[4], "ch4": row[5], "co2e": row[6]}
  return ret_dict


@app.get(
    "/api/v1/ships/<string:shipid>/emission/bymeload",
    summary="Get emission by me load",
    tags=[TAGS["emission"]],
    description="Get emission by me load",
    responses={200: ResponseBodyEmission, 400:ResponseBodyError})
def api_emission_by_me_load(path:RequestPathShipID, query:RequestQueryDuration):
  with G_CONFIG["db_conn_pool"].connection() as conn:
    emissiondb = Voyage.EmissionByMELoadDB(conn)
    result = emissiondb.select_total_in_period(path.shipid, query.start, query.end)
  if result is None:
    ret_str = json.dumps({"result": "failed", "reason": "data is null"})
    status = 400
  else:
    ret_dict = make_response_for_emission_by_me_load(result)
    ret_str = json.dumps({"result": "success", "rows": ret_dict})
    status = 200
  access_log(status, len(ret_str))
  return ret_str, status



def make_response_for_cii_by_me_load(p_result:list) -> dict:
  ret_dict:dict = dict()
  for index in range(0, 125, 5):
    ret_dict[index] = {"actual_cii_value": 0.0, "actual_cii_grade": 0.0, "predict_cii_value": 0.0, "predict_cii_grade": 0.0}
  for row in p_result:
    # ship_id, year, load_index, actual_cii_value, actual_cii_grade, predict_cii_value, predict_cii_grade, modified_at
    load_index = int(row[2])
    ret_dict[load_index] = {"actual_cii_value": row[3] if row[3] is not None else 0.0,
                            "actual_cii_grade": row[4] if row[4] is not None else 0.0,
                            "predict_cii_value": row[5] if row[5] is not None else 0.0,
                            "predict_cii_grade": row[6] if row[6] is not None else 0.0}
  return ret_dict


class RequestQueryYear(BaseModel):
  year:int = Field(None, description="연도 ex) 2024")


@app.get(
    "/api/v1/ships/<string:shipid>/cii/bymeload",
    summary="Get cii by me load",
    tags=[TAGS["emission"]],
    description="Get cii by me load",
    responses={200: ResponseBodyEmission, 400:ResponseBodyError})
def api_cii_by_me_load(path:RequestPathShipID, query:RequestQueryYear):
  with G_CONFIG["db_conn_pool"].connection() as conn:
    ciidb = Voyage.CIIByMELoadDB(conn)
    result = ciidb.select(path.shipid, query.year)
  if result is None:
    ret_str = json.dumps({"result": "failed", "reason": "data is null"})
    status = 400
  else:
    ret_dict = make_response_for_cii_by_me_load(result)
    ret_str = json.dumps({"result": "success", "rows": ret_dict})
    status = 200
  access_log(status, len(ret_str))
  return ret_str, status


@app.get(
    "/api/v1/ships/<string:shipid>/cii/period",
    summary="Get cii values and grade on duration",
    tags=[TAGS["emission"]],
    description="Get cii values and grade on duration",
    responses={200: ResponseBodyEmission, 400:ResponseBodyError})
def api_get_cii_value_and_grade_on_period(path:RequestPathShipID, query:RequestQueryDuration):
  with G_CONFIG["db_conn_pool"].connection() as conn:
    update_cii_base_data(path.shipid, conn)
  client = HS4Wrapper.ComplexQueryAPIClient( G_CONFIG["api-gateway"]["hostname"], int(G_CONFIG["api-gateway"]["port"]), float(G_CONFIG["api-gateway"]["timeout"]),
    G_CONFIG["authentication"]["token-storage"], G_CONFIG["authentication"]["loginname"], G_CONFIG["authentication"]["password"])
  result = client.get_stat(path.shipid, query.start, query.end, ["/hs4rda_v1/ship////kg/co2", "/hs4rda_v1/vdr////km/dist"])

  if result is not None and result["result"] == "success":
    status = 200
    temp_dict:dict = dict()
    for row in result["rows"]:
      temp_dict[row["dataChannelId"]] = row
    co2_sum = temp_dict["/hs4rda_v1/ship////kg/co2"]["sum"]
    dist_sum = temp_dict["/hs4rda_v1/vdr////km/dist"]["sum"]
    cii_value, cii_grade = Voyage.calc_cii_value_and_grade(
      co2_sum, dist_sum, G_CONFIG["ship_data"]["cii_grade_lines"], G_CONFIG["ship_data"]["dwt"])
    ret_str = json.dumps({"result": "success", "rows": {"cii_value": cii_value, "cii_grade": cii_grade}})
    status = 200
  else:
    ret_str = json.dumps({"result": "failed", "reason": "data is null"})
    status = 400
  access_log(status, len(ret_str))
  return ret_str, status


@app.get(
    "/api/v1/ships/<string:shipid>/cii/current",
    summary="Get cii values and grade current year and others",
    tags=[TAGS["emission"]],
    description="Get cii values and grade current year and previous years",
    responses={200: ResponseBodyEmission, 400:ResponseBodyError})
def api_get_cii_value_and_grade_in_current(path:RequestPathShipID):
  ret_dict:dict = dict()
  now = datetime.datetime.now(datetime.timezone.utc)
  with G_CONFIG["db_conn_pool"].connection() as conn:
    update_cii_base_data(path.shipid, conn)
    ciidb = Voyage.CIIAnnualValuesDB(conn)
    result = ciidb.select_all(path.ship_id)

  if result is not None and len(result) > 0:
    temp_dict:dict = dict()
    for row in result:
      # ship_id, year, actual_cii_value, actual_cii_grade, predict_cii_value, predict_cii_grade, modified_at
      temp_dict[row[1]] = {"actual_cii_value": row[2], "actual_cii_grade": row[3], "predict_cii_value": row[4], "predict_cii_grade": row[5]}
    cii_lines:list = list()
    for item in G_CONFIG["ship_data"]["cii_grade_lines"]:
      cii_lines.append(float(item))
    ret_dict = { "result": "success", "rows": dict() }
    ret_dict["rows"]["cii_lines"] = cii_lines
    ret_dict["rows"]["cii_annual"] = temp_dict

    ret_str = json.dumps(ret_dict)
    status = 200
  else:
    ret_str = json.dumps({"result": "failed", "reason": "data is null"})
    status = 400
  access_log(status, len(ret_str))
  return ret_str, status


@app.get(
    "/api/v1/ships/<string:shipid>/cii/inyear",
    summary="Get cii values and grade on duration",
    tags=[TAGS["emission"]],
    description="Get cii values and grade on duration",
    responses={200: ResponseBodyEmission, 400:ResponseBodyError})
def api_get_cii_value_and_grade_in_year(path:RequestPathShipID, query:RequestQueryYear):
  start_of_year = datetime.datetime(query.year, 1, 1, 0, 0, 0, 0, datetime.timezone.utc)
  end_of_year = datetime.datetime(query.year, 12, 31, 23, 59, 59, 999999, datetime.timezone.utc)
  with G_CONFIG["db_conn_pool"].connection() as conn:
    update_cii_base_data(path.shipid, conn)
    statdb = DataAccess.ResultStatDB(conn)
    result = statdb.get_hourly_stat(path.shipid, ["/hs4rda_v1/ship////kg/co2", "/hs4rda_v1/vdr////km/dist"], start_of_year, end_of_year)

  if result is not None and len(result) > 0:
    status = 200
    temp_dict:dict = dict()
    for row in result:
      # data_channel_id, created_time, min, max, sum, avg, count
      if row[0] not in temp_dict:
        temp_dict[row[0]] = row[4]
      else:
        temp_dict[row[0]] += row[4]
    co2_sum = temp_dict["/hs4rda_v1/ship////kg/co2"]
    dist_sum = temp_dict["/hs4rda_v1/vdr////km/dist"]
    cii_value, cii_grade = Voyage.calc_cii_value_and_grade(
      co2_sum, dist_sum, G_CONFIG["ship_data"]["cii_grade_lines"], G_CONFIG["ship_data"]["dwt"])
    ret_str = json.dumps({"result": "success", "rows": {"cii_value": cii_value, "cii_grade": cii_grade}})
    status = 200
  else:
    ret_str = json.dumps({"result": "failed", "reason": "data is null"})
    status = 400
  access_log(status, len(ret_str))
  return ret_str, status


def get_time(p_15s_tick:int) -> str:
  total_time = 15 * p_15s_tick
  total_hour = int(total_time / 3600)
  days = int(total_hour / 24)
  hours = total_hour % 24
  return f"{days}D-{hours}H"


@app.get(
    "/api/v1/ships/<string:shipid>/cii/trend",
    summary="Get cii values and grade in voyage trend",
    tags=[TAGS["emission"]],
    description="Get cii values and grade in voyage trend",
    responses={200: ResponseBodyEmission, 400:ResponseBodyError})
def api_get_cii_value_and_grade_in_voyage_trend(path:RequestPathShipID):
  with G_CONFIG["db_conn_pool"].connection() as conn:
    update_cii_base_data(path.shipid, conn)
    vsdb = Voyage.VoyageServiceDB(conn)
    result = vsdb.select_voyage_trend_all(path.shipid)

  if result is not None and len(result) > 0:
    status = 200
    row_list:list = list()
    for row in result:
      # ship_id, voyage_seq, loading, voyage_id, voyage_name, departure_time, arrival_time, insert_count, all_tot_dist, all_tot_sog, all_tot_foc, all_tot_fgc, all_tot_co2
      voyage_name = row[4] if row[3] != -1 else f"{row[4]} ({row[1]})"
      cii_value, cii_grade = Voyage.calc_cii_value_and_grade(row[12], row[8], G_CONFIG["ship_data"]["cii_grade_lines"], G_CONFIG["ship_data"]["dwt"])
      sog = row[9] / row[7] if row[7] != 0 else 0.0
      row_list.append({"loading": row[2], "voyage_name": voyage_name, "departure_time": row[5].isoformat(), "arrival_time": row[6].isoformat(),
                       "time": get_time(row[7]), "distance": row[8] * 0.539957, "sog_avg": sog,
                       "foc_sum": row[9], "fgc_sum": row[10], "co2_sum": row[11], "cii_value": cii_value, "cii_grade": cii_grade })
    ret_str = json.dumps({"result": "success", "rows": row_list })
    status = 200
  else:
    ret_str = json.dumps({"result": "failed", "reason": "data is null"})
    status = 400
  access_log(status, len(ret_str))
  return ret_str, status


# -------------------------------------------------------------------------------------------------
# API - bog (Smart BoG)
# -------------------------------------------------------------------------------------------------
@app.get(
    "/api/v1/ships/<string:shipid>/bog",
    summary="Get cii values and grade in voyage trend",
    tags=[TAGS["bog"]],
    description="Get cii values and grade in voyage trend",
    responses={200: ResponseBodyEmission, 400:ResponseBodyError})
def api_get_actual_and_simulation_bog(path:RequestPathShipID):
  with G_CONFIG["db_conn_pool"].connection() as conn:
    bogdb = SmartBogSupport.BogDataDB(conn)
    bogdb.begin_transaction()
    actual = bogdb.select_by_type(path.shipid, SmartBogSupport.BogDataDB.DATA_TYPE["Actual"])
    past_simulation = bogdb.select_by_type(path.shipid,SmartBogSupport.BogDataDB.DATA_TYPE["Past Simulation"])
    future_simulation = bogdb.select_by_type(path.shipid,SmartBogSupport.BogDataDB.DATA_TYPE["Future Simulation"])

  if (actual is not None and len(actual) > 0) or (past_simulation is not None and len(past_simulation) > 0) or (future_simulation is not None and len(future_simulation) > 0):
    status = 200
    ret_str = json.dumps(
      {
        "result": "success",
        "rows": {
          "actual": actual,
          "past_simulation": past_simulation,
          "future": future_simulation
        }
      }
    )
    status = 200
  else:
    ret_str = json.dumps({"result": "failed", "reason": "data is null"})
    status = 400
  access_log(status, len(ret_str))
  return ret_str, status


# -------------------------------------------------------------------------------------------------
# Main routines - argument parser and main
# -------------------------------------------------------------------------------------------------
def parse_argument(p_argv: list) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Test API Server")

  parser.add_argument("-d", "--debug", action="store_true", help="debug flag")
  parser.add_argument("-c", "--config", type=str, help="application configuration file")
  parser.add_argument("-l", "--log-config", type=str, help="log configuration file")

  return parser.parse_args(p_argv)


if __name__ == '__main__':
  args = parse_argument(sys.argv[1:])
  app_cfg = SFWBase.read_config_with_env(args.config)
  SFWBase.setup_logger(args.log_config)
  setup_api_server(app_cfg)
  G_CONFIG["db_conn_pool"] = psycopg_pool.ConnectionPool(G_CONFIG["databases"]["service-framework"], open=True)

  G_CONFIG["graph_data"] = load_graph_data(app_cfg["voyage"]["graph-data"])

  if args.debug:
    app.run(app_cfg["api-server"]["bind-address"], port=int(app_cfg["api-server"]["port"]), debug=args.debug)
  else:
    waitress.serve(app=app, host=app_cfg["api-server"]["bind-address"], port=int(app_cfg["api-server"]["port"]))
  G_CONFIG["db_conn_pool"].close()
