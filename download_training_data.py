#!/usr/bin/env python
from __future__ import print_function
import subprocess
import json
import os
import dateutil.parser
from datetime import datetime
import pytz
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument('--all', action='store_true', dest='refresh_all')
args = parser.parse_args()

STORAGE_FORMAT = "./data/classification_data.{pk}.jsonl"

items = json.loads(subprocess.check_output("ssh -Cqt rideserver@rideserver-backend-1 ./manage.py list_ios_classification_data --out=json", shell=True))
for item in items:
    item['modified'] = dateutil.parser.parse(item['modified'])
    path = STORAGE_FORMAT.format(**item)
    item['local_path'] = path
    if not args.refresh_all and os.path.isfile(path):
        mtime_epoch = os.path.getmtime(path)
        mdate = datetime.utcfromtimestamp(os.path.getmtime(path))
        filedate = pytz.utc.localize(mdate)
        if filedate > item['modified']:
            item['load'] = False

    if item['notes'] is not None and 'sync' in item['notes'].lower():
        print("Skipping item={} because of notes: {}".format(item['pk'], item['notes']))
        item['load'] = False

    if not item['included']:
        print("Skipping item={pk} because `included` is False; notes={notes}, admin_notes={admin_notes}".format(**item))
        item['load'] = False

for item in items:
    if item.get('load', True):
        cmd = "ssh -Cqt rideserver@rideserver-backend-1 ./manage.py export_ios_classification_data {pk}".format(**item)
        outfile = open(item['local_path'], 'wb')
        subprocess.call(cmd, shell=True, stdout=outfile)
        print("Finished export {}".format(item))
