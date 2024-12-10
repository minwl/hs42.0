import datetime
import json

import psycopg

from hs4da import SFWBase, DataAccess

import HS4Wrapper

"""
p_base_time:str, : 기준 시각

[playback_base select]
p_latitude:float, p_longitude:float, : select latitude, longitude from da.playback_base where base_time = '2024-11-25 00:00:00';

[data platform stat 1h]
p_all_avg_sog:float : "/hs4sd_v1/vdr////kn/sog"
p_all_avg_stw:float : "/hs4sd_v1/vdr////kn/stw"

/platform-stat/stat/devicedata/ships/{shipId}/hourly


[service framework stat 1h]
p_all_tot_dist:float : "/hs4rda_v1/vdr////km/dist"
p_all_avg_foc:float, p_all_tot_foc:float  : "/hs4rda_v1/ship/fuel/oil//equiv_kg/foc"
p_all_avg_fgc:float, p_all_tot_fgc:float  : "/hs4rda_v1/ship/fuel/gas//equiv_kg/foc"
p_all_avg_co2:float, p_all_tot_co2:float  : "/hs4rda_v1/ship////kgph/co2", "/hs4rda_v1/ship////kgph/co2" / 1000
p_all_avg_ch4:float, p_all_tot_ch4:float  : "/hs4rda_v1/ship////kgph/ch4", "/hs4rda_v1/ship////kgph/ch4" / 1000
p_all_avg_co2e:float, p_all_tot_co2e:float : "/hs4rda_v1/ship////kgph/co2e", "/hs4rda_v1/ship////kgph/co2e" / 1000
p_me_avg_load:float : "/hs4rda_v1/me////per/load"
p_me_avg_foc:float, p_me_tot_foc:float : "/hs4rda_v1/me/fuel/oil//kgph/foc", "/hs4rda_v1/me/fuel/oil//kgph/foc" / 1000
p_me_avg_fgc:float, p_me_tot_fgc:float : "/hs4rda_v1/me/fuel/gas//kgph/foc", "/hs4rda_v1/me/fuel/gas//kgph/foc" / 1000
p_me_avg_co2:float : "/hs4rda_v1/me////kgph/co2"
p_me_avg_ch4:float : "/hs4rda_v1/me////kgph/ch4"
p_me_avg_co2e:float : "/hs4rda_v1/me////kgph/co2e"
p_ge_avg_foc:float, p_ge_tot_foc:float : "/hs4rda_v1/ge/fuel/oil//kgph/foc", "/hs4rda_v1/ge/fuel/oil//kgph/foc" / 1000
p_ge_avg_fgc:float, p_ge_tot_fgc:float : "/hs4rda_v1/ge/fuel/gas//kgph/foc", "/hs4rda_v1/ge/fuel/gas//kgph/foc" / 1000
p_ge_avg_co2:float : "/hs4rda_v1/ge////kgph/co2"
p_ge_avg_ch4:float : "/hs4rda_v1/ge////kgph/cg4"
p_ge_avg_co2e:float : "/hs4rda_v1/ge////kgph/co2e"
p_ab_avg_co2:float : "/hs4rda_v1/ab////kgph/co2"
p_ab_avg_ch4:float : "/hs4rda_v1/ab////kgph/ch4"
p_ab_avg_co2e:float : "/hs4rda_v1/ab////kgph/co2e"
p_cb_avg_co2:float : "/hs4rda_v1/cb////kgph/co2"
p_cb_avg_ch4:float : "/hs4rda_v1/cb////kgph/ch4"
p_cb_avg_co2e:float : "/hs4rda_v1/cb////kgph/co2e"
p_gcu_avg_co2:float : "/hs4rda_v1/gcu////kgph/co2"
p_gcu_avg_ch4:float : "/hs4rda_v1/gcu////kgph/ch4"
p_gcu_avg_co2e:float : "/hs4rda_v1/gcu////kgph/co2e"

->
select data_channel_id, sum, avg, count from stat.result_timeseries_stat_1h where data_channel_id = ANY( <data channel ids> ) AND created_time = '2024-11-25 00:00:00';
"""

SVCFW_STAT_CHANNEL_ID_DICT:dict = {
  "/hs4rda_v1/vdr////km/dist": {"all_tot_dist": "sum"},
  "/hs4rda_v1/ship/fuel/oil//equiv_kg/foc" : {"all_avg_foc": "avg", "all_tot_foc": "sum/1000"},
  "/hs4rda_v1/ship/fuel/gas//equiv_kg/foc" : {"all_avg_fgc": "avg", "all_tot_fgc": "sum/1000"},
  "/hs4rda_v1/ship////kgph/co2": {"all_avg_co2": "avg", "all_tot_co2": "sum/1000"},
  "/hs4rda_v1/ship////kgph/ch4": {"all_avg_ch4": "avg", "all_tot_ch4": "sum/1000"},
  "/hs4rda_v1/ship////kgph/co2e": {"all_avg_co2e": "avg", "all_tot_co2e": "sum/1000"},
  "/hs4rda_v1/me////per/load": {"me_avg_load": "avg"},
  "/hs4rda_v1/me/fuel/oil//kgph/foc": {"me_avg_foc": "avg", "me_tot_foc": "sum/1000"},
  "/hs4rda_v1/me/fuel/gas//kgph/foc": {"me_avg_fgc": "avg", "me_tot_fgc": "sum/1000"},
  "/hs4rda_v1/me////kgph/co2": {"me_avg_co2": "avg"},
  "/hs4rda_v1/me////kgph/ch4": {"me_avg_ch4": "avg"},
  "/hs4rda_v1/me////kgph/co2e": {"me_avg_co2e": "avg"},
  "/hs4rda_v1/ge/fuel/oil//kgph/foc": {"ge_avg_foc": "avg", "ge_tot_foc": "sum/1000"},
  "/hs4rda_v1/ge/fuel/gas//kgph/foc": {"ge_avg_fgc": "avg", "ge_tot_fgc": "sum/1000"},
  "/hs4rda_v1/ge////kgph/co2": {"ge_avg_co2": "avg"},
  "/hs4rda_v1/ge////kgph/cg4": {"ge_avg_ch4": "avg"},
  "/hs4rda_v1/ge////kgph/co2e": {"ge_avg_co2e": "avg"},
  "/hs4rda_v1/ab////kgph/co2": {"ab_avg_co2": "avg"},
  "/hs4rda_v1/ab////kgph/ch4": {"ab_avg_ch4": "avg"},
  "/hs4rda_v1/ab////kgph/co2e": {"ab_avg_co2e": "avg"},
  "/hs4rda_v1/cb////kgph/co2": {"cb_avg_co2": "avg"},
  "/hs4rda_v1/cb////kgph/ch4": {"cb_avg_ch4": "avg"},
  "/hs4rda_v1/cb////kgph/co2e": {"cb_avg_co2e": "avg"},
  "/hs4rda_v1/gcu////kgph/co2": {"gcu_avg_co2": "avg"},
  "/hs4rda_v1/gcu////kgph/ch4": {"gcu_avg_ch4": "avg"},
  "/hs4rda_v1/gcu////kgph/co2e": {"gcu_avg_co2e": "avg"}
}


DATAPF_STAT_CHANNEL_ID_DICT:dict = {
  "/hs4sd_v1/vdr////kn/sog": {"all_avg_sog": "avg"},
  "/hs4sd_v1/vdr////kn/stw": {"all_avg_stw": "avg"}
}


class VoyageStatDB(SFWBase.PostgreSQLTransactionalHandler):

  def __init__(self, p_conn: psycopg.Connection):
    """
    Construct VoyageServiceDB

    :param p_conn_str: connection string
    """
    super().__init__(p_conn)

  def insert(self, p_ship_id:str, p_base_time:str, p_latitude:float, p_longitude:float, p_all_avg_sog:float, p_all_avg_stw:float, p_all_tot_dist:float,
    p_all_avg_foc:float, p_all_tot_foc:float, p_all_avg_fgc:float, p_all_tot_fgc:float, p_all_avg_co2:float, p_all_tot_co2:float, p_all_avg_ch4:float, p_all_tot_ch4:float, p_all_avg_co2e:float, p_all_tot_co2e:float,
    p_me_avg_load:float, p_me_avg_foc:float, p_me_tot_foc:float, p_me_avg_fgc:float, p_me_tot_fgc:float, p_me_avg_co2:float, p_me_avg_ch4:float, p_me_avg_co2e:float,
    p_ge_avg_foc:float, p_ge_tot_foc:float, p_ge_avg_fgc:float, p_ge_tot_fgc: float, p_ge_avg_co2:float, p_ge_avg_ch4:float, p_ge_avg_co2e:float,
    p_ab_avg_co2:float, p_ab_avg_ch4:float, p_ab_avg_co2e:float,
    p_cb_avg_co2:float, p_cb_avg_ch4:float, p_cb_avg_co2e:float,
    p_gcu_avg_co2:float, p_gcu_avg_ch4:float, p_gcu_avg_co2e:float):
    self.check_transaction_before()
    self.cursor.execute(
      """
      INSERT INTO da.voyage_stat_1h
      ( ship_id, base_time, latitude, longitude, all_avg_sog, all_avg_stw, all_tot_dist,
        all_avg_foc, all_tot_foc, all_avg_fgc, all_tot_fgc, all_avg_co2, all_tot_co2, all_avg_ch4, all_tot_ch4, all_avg_co2e, all_tot_co2e,
        me_avg_load, me_avg_foc, me_tot_foc, me_avg_fgc, me_tot_fgc, me_avg_co2, me_avg_ch4, me_avg_co2e,
        ge_avg_foc, ge_tot_foc, ge_avg_fgc, ge_tot_fgc, ge_avg_co2, ge_avg_ch4, ge_avg_co2e,
        ab_avg_co2, ab_avg_ch4, ab_avg_co2e,
        cb_avg_co2, cb_avg_ch4, cb_avg_co2e,
        gcu_avg_co2, gcu_avg_ch4, gcu_avg_co2e)
      VALUES
      ( %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s,
        %s, %s, %s,
        %s, %s, %s)
      """, [
      p_ship_id, p_base_time, p_latitude, p_longitude, p_all_avg_sog, p_all_avg_stw, p_all_tot_dist,
      p_all_avg_foc, p_all_tot_foc, p_all_avg_fgc, p_all_tot_fgc, p_all_avg_co2, p_all_tot_co2, p_all_avg_ch4, p_all_tot_ch4, p_all_avg_co2e, p_all_tot_co2e,
      p_me_avg_load, p_me_avg_foc, p_me_tot_foc, p_me_avg_fgc, p_me_tot_fgc, p_me_avg_co2, p_me_avg_ch4, p_me_avg_co2e,
      p_ge_avg_foc, p_ge_tot_foc, p_ge_avg_fgc, p_ge_tot_fgc, p_ge_avg_co2, p_ge_avg_ch4, p_ge_avg_co2e,
      p_ab_avg_co2, p_ab_avg_ch4, p_ab_avg_co2e,
      p_cb_avg_co2, p_cb_avg_ch4, p_cb_avg_co2e,
      p_gcu_avg_co2, p_gcu_avg_ch4, p_gcu_avg_co2e]
    ) 
    self.result_code = 0
    self.check_transaction_after(True)

  def select(self, p_ship_id:str, p_start_time:str, p_end_time:str):
    self.check_transaction_before()
    self.cursor.execute(
      """
      SELECT ship_id, base_time, latitude, longitude, all_avg_sog, all_avg_stw, all_tot_dist,
        all_avg_foc, all_tot_foc, all_avg_fgc, all_tot_fgc, all_avg_co2, all_tot_co2, all_avg_ch4, all_tot_ch4, all_avg_co2e, all_tot_co2e,
        me_avg_load, me_avg_foc, me_tot_foc, me_avg_fgc, me_tot_fgc, me_avg_co2, me_avg_ch4, me_avg_co2e,
        ge_avg_foc, ge_tot_foc, ge_avg_fgc, ge_tot_fgc, ge_avg_co2, ge_avg_ch4, ge_avg_co2e,
        ab_avg_co2, ab_avg_ch4, ab_avg_co2e,
        cb_avg_co2, cb_avg_ch4, cb_avg_co2e,
        gcu_avg_co2, gcu_avg_ch4, gcu_avg_co2e
      FROM da.voyage_stat_1h
      WHERE ship_id = %s AND base_time BETWEEN %s AND %s
      ORDER BY base_time DESC
      """, [p_ship_id, p_start_time, p_end_time])
    ret_list = self.cursor.fetchall()
    self.result_code = 0
    self.check_transaction_after(False)
    return ret_list


class ServiceFrameworkStatDB(SFWBase.PostgreSQLTransactionalHandler):

  def __init__(self, p_conn: psycopg.Connection):
    """
    Construct VoyageServiceDB

    :param p_conn_str: connection string
    """
    super().__init__(p_conn)

  def get_playback_base(self, p_ship_id:str, p_basetime: datetime.datetime):
    self.check_transaction_before()
    self.cursor.execute("SELECT latitude, longitude FROM da.playback_base WHERE ship_id = %s AND base_time = %s", [p_ship_id, p_basetime])
    result = self.cursor.fetchone()
    self.check_transaction_after(False)
    return result

  def get_rda_stat(self, p_ship_id:str, p_basetime: datetime.datetime, p_data_channel_id_list:list):
    self.check_transaction_before()
    self.cursor.execute(
      """
      SELECT data_channel_id, sum, avg, count
      FROM stat.result_timeseries_stat_1h
      WHERE ship_id = %s AND created_time = %s AND data_channel_id = ANY(%s)
      """, [p_ship_id, p_basetime, p_data_channel_id_list])
    result = self.cursor.fetchall()
    self.check_transaction_after(False)
    return result


class VoyagePlanDB(SFWBase.PostgreSQLTransactionalHandler):

  def __init__(self, p_conn: psycopg.Connection):
    """
    Construct VoyageServiceDB

    :param p_conn_str: connection string
    """
    super().__init__(p_conn)

  def get_approved_plan(self, p_ship_id:str):
    self.check_transaction_before()
    self.cursor.execute("SELECT name, waypoints, status, type FROM vasc.voyage_plan WHERE ship_id = %s AND status = 'Approved'", [p_ship_id])
    result = self.cursor.fetchone()
    self.check_transaction_after(False)
    return result

  def get_smart_bog_config(self, p_ship_id:str):
    self.check_transaction_before()
    self.cursor.execute(
      """
      SELECT target_pressure, hh,  h , l, ll, ship_id
      FROM equip.smart_bog_chart
      WHERE ship_id = %s
      """, [p_ship_id])
    result = self.cursor.fetchone()
    self.check_transaction_after(False)
    return result

  def get_port_point(self, p_port_code:str):
    self.check_transaction_before()
    self.cursor.execute(
      """
      SELECT latitude, longitude
      FROM framework.port
      WHERE code = %s
      """, [p_port_code])
    result = self.cursor.fetchone()
    self.check_transaction_after(False)
    return result


def get_basetime() -> datetime.datetime:
  one_hour_before = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1)
  basetime = datetime.datetime(one_hour_before.year, one_hour_before.month, one_hour_before.day, one_hour_before.hour, 0, 0, 0, datetime.timezone.utc)
  return basetime


def get_voyage_stat_hourly(p_basetime:datetime.datetime, p_ship_id:str, p_db:ServiceFrameworkStatDB, p_client:HS4Wrapper.DataPlatformStatClient) -> dict:
  global SVCFW_STAT_CHANNEL_ID_DICT, DATAPF_STAT_CHANNEL_ID_DICT
  ret_dict:dict = dict()
  p_db.begin_transaction()
  result = p_db.get_playback_base(p_ship_id, p_basetime)
  if len(result) > 0:
    ret_dict["latitude"] = result[0]
    ret_dict["longitude"] = result[1]
  result = p_db.get_rda_stat(p_ship_id, p_basetime, list(SVCFW_STAT_CHANNEL_ID_DICT.keys()))
  p_db.end_transaction()
  print(result)
  for row in result:
    # data_channel_id, sum, avg, count
    for column, field in SVCFW_STAT_CHANNEL_ID_DICT[row[0]].items():
      if field == "sum":
        value = row[1]
      elif field == "avg":
        value = row[2]
      elif field == "sum/1000":
        value = row[1] / 1000
      ret_dict[column] = value
  result = p_client.request_stat_hourly(p_ship_id, list(DATAPF_STAT_CHANNEL_ID_DICT.keys()), p_basetime.isoformat(), p_basetime.isoformat())
  if result["result"] == "success":
    for channel_id, data in result["rows"].items():
      for column, field in DATAPF_STAT_CHANNEL_ID_DICT[channel_id].items():
        if field in ("sum", "avg"):
          value = data[0][field]
        elif field == "sum/1000":
          value = data[0]["sum"] / 1000
        ret_dict[column] = value
  return ret_dict


def null_check(p_dict:dict, p_key:str) -> float:
  if p_key in p_dict:
    if p_dict[p_key] is not None:
      return p_dict[p_key]
  return 0.0


def stat_voyage_hourly(p_ship_id:str, p_conn:psycopg.Connection, p_dp_client:DataAccess.DataPlatformAPIClient):
  svcfw_stat_db = ServiceFrameworkStatDB(p_conn)
  basetime = get_basetime()
  stat_data = get_voyage_stat_hourly(basetime, p_ship_id, svcfw_stat_db, p_dp_client)
  print(stat_data)
  voyage_stat_db = VoyageStatDB(p_conn)
  voyage_stat_db.insert(
    p_ship_id,
    basetime,
    null_check(stat_data, "latitude"),
    null_check(stat_data, "longitude"),
    null_check(stat_data, "all_avg_sog"),
    null_check(stat_data, "all_avg_stw"),
    null_check(stat_data, "all_tot_dist"),
    null_check(stat_data, "all_avg_foc"),
    null_check(stat_data, "all_tot_foc"),
    null_check(stat_data, "all_avg_fgc"),
    null_check(stat_data, "all_tot_fgc"),
    null_check(stat_data, "all_avg_co2"),
    null_check(stat_data, "all_tot_co2"),
    null_check(stat_data, "all_avg_ch4"),
    null_check(stat_data, "all_tot_ch4"),
    null_check(stat_data, "all_avg_co2e"),
    null_check(stat_data, "all_tot_co2e"),
    null_check(stat_data, "me_avg_load"),
    null_check(stat_data, "me_avg_foc"),
    null_check(stat_data, "me_tot_foc"),
    null_check(stat_data, "me_avg_fgc"),
    null_check(stat_data, "me_tot_fgc"),
    null_check(stat_data, "me_avg_co2"),
    null_check(stat_data, "me_avg_ch4"),
    null_check(stat_data, "me_avg_co2e"),
    null_check(stat_data, "ge_avg_foc"),
    null_check(stat_data, "ge_tot_foc"),
    null_check(stat_data, "ge_avg_fgc"),
    null_check(stat_data, "ge_tot_fgc"),
    null_check(stat_data, "ge_avg_co2"),
    null_check(stat_data, "ge_avg_ch4"),
    null_check(stat_data, "ge_avg_co2e"),
    null_check(stat_data, "ab_avg_co2"),
    null_check(stat_data, "ab_avg_ch4"),
    null_check(stat_data, "ab_avg_co2e"),
    null_check(stat_data, "cb_avg_co2"),
    null_check(stat_data, "cb_avg_ch4"),
    null_check(stat_data, "cb_avg_co2e"),
    null_check(stat_data, "gcu_avg_co2"),
    null_check(stat_data, "gcu_avg_ch4"),
    null_check(stat_data, "gcu_avg_co2e")
  )


def get_approved_voyage_plan(p_ship_id:str, p_conn:psycopg.Connection) -> dict:
  plandb = VoyagePlanDB(p_conn)
  result = plandb.get_approved_plan(p_ship_id)
  if len(result) > 0:
    return {"name": result[0], "waypoints": json.loads(result[1]), "status": result[2], "type": result[3]}
  return None


def get_smart_bog_chart_config(p_ship_id, p_conn:psycopg.Connection) -> list:
  plandb = VoyagePlanDB(p_conn)
  result = plandb.get_smart_bog_config(p_ship_id)
  if len(result) > 0:
    # target_pressure, hh,  h , l, ll, ship_id, id
    temp_list = list(result)
    ret_list = temp_list[0:5]
    ret_list.reverse()
    return ret_list
