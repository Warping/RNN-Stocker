import configparser
from sec_api import QueryApi

config = configparser.ConfigParser()
config.read('config.ini')

SEC_API_KEY = config.get('SEC', 'API_KEY')

queryApi = QueryApi(api_key=SEC_API_KEY)

ticker = input("Enter a ticker: ")


query = {
    "query": f"ticker:{ticker} AND  formType:\"8-K\"",
    "from": "0",
    "size": "1",
    "sort": [{ "filedAt": { "order": "desc" } }]
}

filings = queryApi.get_filings(query)

print(f"Filings for {ticker}:")
print(filings)