import pipeline
import json
import subprocess
import logging
logger = logging.getLogger(__name__)
STORAGE_FORMAT = "./data/tsd.{trip_pk}.jsonl"


class JWTWrapper(object):
    def __init__(self):
        type(self)._instance = self
        self._token = None

    @property
    def token(self):
        if self._token is None:
            print("Refreshing token ...")
            response_body = subprocess.check_output("ssh -Cqt rideserver@rideserver-backend-1 ./manage.py get_admin_jwt", shell=True)
            data = json.loads(response_body.split('\n')[0])
            self._token = data['token']
            print("Got new token")
        return self._token

    def reset(self):
        self._token = None

    def __unicode__(self):
        return self.token

    def __str__(self):
        return unicode(self)

    @classmethod
    def getInstance(cls):
        if not hasattr(cls, '_instance'):
            cls._instance = JWTWrapper()
        return cls._instance


def get_tsd_stubs(sa=None):
    params = []
    if sa is not None:
        params.append('--sa {}'.format(sa))
    cmd_base = "ssh -Cqt rideserver@rideserver-backend-1 ./manage.py list_misclassifications --out=json"
    cmd = '{} {}'.format(cmd_base, ' '.join(params))
    items = json.loads(subprocess.check_output(cmd, shell=True))
    return items

def tsd_url(pk):
    return 'https://ride.report/__tools/inspect/tripsensordata_raw/{}'.format(pk)

def download_single_tsd(session, token, pk):
    url = tsd_url(pk)
    headers = {
        'Authorization': 'JWT {}'.format(token),
    }
    response = session.get(url, headers=headers)
    if hasattr(response, 'result'):
        response = response.result()
    response.raise_for_status()
    with open(STORAGE_FORMAT.format(trip_pk=pk), 'wb') as f:
        f.write(response.text.encode('utf-8'))

def load_tsd_by_pk(pk, force_update=False):
    filename = STORAGE_FORMAT.format(trip_pk=pk)
    try:
        return pipeline.loadTSD(filename, force_update=force_update)
    except (IOError, OSError):
        import requests
        download_single_tsd(requests, JWTWrapper.getInstance(), pk)
        return pipeline.loadTSD(filename, force_update=force_update)

def load_many_tsds(stubs, force_update=False):
    for stub in stubs:
        try:
            yield load_tsd_by_pk(stub['trip_pk'], force_update=force_update)
        except Exception as e:
            logger.error("Failed to load TSD trip_pk={}: {}".format(stub.get('trip_pk', None), repr(e)))
