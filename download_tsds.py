#!/usr/bin/env python
from __future__ import print_function
import subprocess
import json
import os
import dateutil.parser
import requests
from datetime import datetime
from itertools import izip
import pytz
import argparse
parser = argparse.ArgumentParser(description="")
parser.add_argument('--all', action='store_true', dest='refresh_all')
args = parser.parse_args()
from tsd_tools import JWTWrapper


auto_token = JWTWrapper()

STORAGE_FORMAT = "./data/tsd.{trip_pk}.jsonl"

print("Fetching list of available TSDs ...")
items = json.loads(subprocess.check_output("ssh -Cqt rideserver@rideserver-backend-1 ./manage.py list_misclassifications --out=json", shell=True))
for item in items:
    item['created'] = dateutil.parser.parse(item['created'])
    path = STORAGE_FORMAT.format(**item)
    item['local_path'] = path
    if not args.refresh_all and os.path.isfile(path):
        mtime_epoch = os.path.getmtime(path)
        mdate = datetime.utcfromtimestamp(os.path.getmtime(path))
        filedate = pytz.utc.localize(mdate)
        try:
            with open(path) as f:
                json.load(f)
            valid = True
        except:
            valid = False

        if filedate > item['created'] and valid:
            item['load'] = False
print("Got {} TSDs from server".format(len(items)))
items = [item for item in items if item.get('load', True)]
print("Fetching {} TSDs".format(len(items)))

def futures(session, items):
    for item in items:
        url = 'https://ride.report/__tools/inspect/tripsensordata_raw/{trip_pk}'.format(**item)
        headers = {
            'Authorization': 'JWT {}'.format(auto_token),
        }
        # TODO: write session wrapper that auto-refreshes jwt every few minutes
        yield session.get(url, headers=headers)

from requests_futures.sessions import FuturesSession
with FuturesSession(max_workers=6) as session:
    for future, item in izip(list(futures(session, items)), items):
        response = future.result()
        try:
            response.raise_for_status()
        except:
            print('Failed on TSD {trip_pk}'.format(**item))
            print(response.text)
            continue

        with open(item['local_path'], 'wb') as outfile:
            outfile.write(response.text.encode('utf-8'))

        # cmd = "ssh -Cqt rideserver@rideserver-backend-1 ./manage.py export_tsd {trip_pk}".format(**item)
        # outfile = open(item['local_path'], 'wb')
        # subprocess.call(cmd, shell=True, stdout=outfile)
        print("Finished export TSD {trip_pk} {created}".format(**item))
