import json
import urllib

from hs4da import SFWBase


class DataPlatformStatClient(SFWBase.HS4APIClient):
  """
  Class of DataPlatformAPIClient : Access class for device data
  """
  def __init__(self, p_host: str, p_port: int, p_timeout: float, p_token_store:str, p_username: str, p_password: str):
    """
    Construct DataPlatformAPIClient
    
    :param p_host: API hostname
    :param p_port: API port
    :param p_timeout: API Access timeout
    :param p_token_storage: JWT storage for cache
    :param p_username: login name
    :param p_password: login password
    """
    super().__init__(p_host, p_port, p_timeout, p_token_store)
    self.username = p_username
    self.password = p_password

  def request_stat_hourly(self, p_ship_id:str, p_data_channel_ids: list, p_start_time: str, p_end_time: str) -> dict:
    """
    Get timeseries data of devices

    :param p_shipid: Ship ID
    :param p_data_channel_ids: list of data channel ids
    :param p_start_time: start time of duration (ISO8601 type)
    :param p_end_time: end time of duration (ISO8601 type)
    :return: result dictionary
    """
    jwt = self.login(self.username, self.password)
    if jwt is not None:
      request_uri = f"/platform-stat/stat/devicedata/ships/{p_ship_id}/hourly"
      query_str:str = None
      if p_start_time is not None:
        if query_str is None:
          query_str = "?"
        query_str += "from={}".format(urllib.parse.quote_plus(p_start_time))
      if p_end_time is not None:
        if query_str is None:
          query_str = "?"
        else:
          query_str += "&"
        query_str += "to={}".format(urllib.parse.quote_plus(p_end_time))
      if query_str is not None:
        request_uri += query_str
      body = json.dumps({"channelIds": p_data_channel_ids})
      response = self.default_transaction(
        "POST",
        request_uri,
        {
	        "Content-Length": str(len(body)),
	        "Content-Type": "application/json",
	        "Authorization": "Bearer "+ jwt["access_token"]
        },
        body)
      self.logout(self.username)
      if response["status"] == 200:
        return response["body"]
    return None


class ComplexQueryAPIClient(SFWBase.HS4APIClient):
  """
  Class of DataSendAPIClient : Sender class for service data
  """
  def __init__(self, p_host: str, p_port: int, p_timeout: float, p_token_store:str, p_username: str, p_password: str):
    """
    Construct DataPlatformAPIClient
    
    :param p_host: API hostname
    :param p_port: API port
    :param p_timeout: API Access timeout
    :param p_token_storage: JWT storage for cache
    :param p_username: login name
    :param p_password: login password
    """
    super().__init__(p_host, p_port, p_timeout, p_token_store)
    self.username = p_username
    self.password = p_password


  def get_stat(self, p_ship_id: str, p_start_time: str, p_end_time: str, p_data_channel_id_list: list) -> dict:
    """
    Get stat with duration

    :param p_ship_id: Ship ID
    :param p_start_time: start of duration
    :param p_end_time: end of duration
    :param p_data_channel_id_list: data channel id list
    :return: result dictionary
    """
    jwt = self.login(self.username, self.password)
    if jwt is not None:
      request_uri:str = f"/service-stat/stat/complex/ships/{p_ship_id}/result?from={urllib.parse.quote_plus(p_start_time)}&to={urllib.parse.quote_plus(p_end_time)}"
      body = json.dumps(p_data_channel_id_list)
      response = self.default_transaction(
        "POST",
        request_uri,
        {
	        "Content-Length": str(len(body)),
	        "Content-Type": "application/json",
	        "Authorization": "Bearer "+ jwt["access_token"],
        },
        body)
      self.logout(self.username)
      if response["status"] == 200:
        return response["body"]
    return None

